"""
Vector database operations with optimized indexing and caching.

Combines basic FAISS operations with advanced HNSW indexing and query caching
for better performance on large document collections.
"""

import os
import pickle
import hashlib
import time
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# Get absolute path to project root and create vector_db path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
VECTOR_DB_PATH = os.path.join(_project_root, "assets", "vector_db")

INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"
HNSW_INDEX_FILE = "hnsw_index.bin"
CACHE_FILE = "query_cache.pkl"

class QueryCache:
    """Simple query result cache with TTL"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.cache_file = os.path.join(VECTOR_DB_PATH, CACHE_FILE)
        self._load_cache()
    
    def _get_query_hash(self, query: str, k: int) -> str:
        """Generate hash for query and parameters"""
        query_str = f"{query}_{k}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() - timestamp > self.ttl_seconds
    
    def get(self, query: str, k: int) -> Optional[List[Tuple[float, Dict]]]:
        """Get cached results for query"""
        query_hash = self._get_query_hash(query, k)
        
        if query_hash in self.cache:
            timestamp = self.access_times.get(query_hash, 0)
            if not self._is_expired(timestamp):
                # Update access time
                self.access_times[query_hash] = time.time()
                return self.cache[query_hash]
            else:
                # Remove expired entry
                del self.cache[query_hash]
                del self.access_times[query_hash]
        
        return None
    
    def set(self, query: str, k: int, results: List[Tuple[float, Dict]]):
        """Cache query results"""
        query_hash = self._get_query_hash(query, k)
        
        # Check if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[query_hash] = results
        self.access_times[query_hash] = time.time()
        self._save_cache()
    
    def _evict_oldest(self):
        """Remove oldest cache entry"""
        if not self.access_times:
            return
        
        oldest_hash = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_hash]
        del self.access_times[oldest_hash]
    
    def _load_cache(self):
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.access_times = data.get('access_times', {})
                    
                # Clean expired entries
                current_time = time.time()
                expired_keys = [
                    key for key, timestamp in self.access_times.items()
                    if current_time - timestamp > self.ttl_seconds
                ]
                for key in expired_keys:
                    self.cache.pop(key, None)
                    self.access_times.pop(key, None)
                    
            except Exception as e:
                print(f"Failed to load query cache: {e}")
                self.cache = {}
                self.access_times = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'cache': self.cache,
                    'access_times': self.access_times
                }, f)
        except Exception as e:
            print(f"Failed to save query cache: {e}")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

def load_embeddings_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDINGS_MODEL)

def get_embeddings(model: SentenceTransformer, text: str) -> np.ndarray:
    return np.asarray(model.encode(text))

def create_vector_db_dir():
    """Create vector database directory if it doesn't exist"""
    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        print(f"Created vector database directory: {VECTOR_DB_PATH}")

def faiss_index(embeddings: List[np.ndarray]) -> faiss.Index | None:
    if not embeddings:
        return None
    
    embeddings_array = np.vstack(embeddings)
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)
    return index

def create_hnsw_index(embeddings: List[np.ndarray]) -> faiss.Index:
    """Create HNSW index for better performance"""
    if not embeddings:
        return None
    
    embeddings_array = np.vstack(embeddings).astype(np.float32)
    dimension = embeddings_array.shape[1]
    
    # Create HNSW index
    # M: number of connections for every new element during construction
    # efConstruction: size of dynamic candidate list for construction
    index = faiss.IndexHNSWFlat(dimension, 32)  # M=32
    index.hnsw.efConstruction = 200  # Higher = better quality, slower construction
    index.hnsw.efSearch = 100        # Higher = better quality, slower search
    
    index.add(embeddings_array)
    return index

def save_vector_database(index: faiss.Index, metadata: List[Dict]):
    create_vector_db_dir()
    
    # Save HNSW index if available
    if isinstance(index, faiss.IndexHNSWFlat):
        hnsw_path = os.path.join(VECTOR_DB_PATH, HNSW_INDEX_FILE)
        faiss.write_index(index, hnsw_path)
        print("✓ Saved HNSW index for faster searches")
    
    # Save FAISS index
    index_path = os.path.join(VECTOR_DB_PATH, INDEX_FILE)
    faiss.write_index(index, index_path)
    
    # Save metadata
    metadata_path = os.path.join(VECTOR_DB_PATH, METADATA_FILE)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

def load_vector_database(use_hnsw: bool = True) -> Tuple[faiss.Index | None, List[Dict]]:
    """Load vector database from disk"""
    hnsw_path = os.path.join(VECTOR_DB_PATH, HNSW_INDEX_FILE)
    index_path = os.path.join(VECTOR_DB_PATH, INDEX_FILE)
    metadata_path = os.path.join(VECTOR_DB_PATH, METADATA_FILE)
    
    index = None
    
    # Try HNSW index first if available and requested
    if use_hnsw and os.path.exists(hnsw_path):
        try:
            index = faiss.read_index(hnsw_path)
            print("✓ Loaded HNSW index")
        except Exception as e:
            print(f"Failed to load HNSW index: {e}")
    
    # Fallback to regular index
    if index is None and os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            print("✓ Loaded flat index")
        except Exception as e:
            print(f"Error loading vector database: {e}")
            return None, []
    
    if not os.path.exists(metadata_path):
        print(f"Vector database not found at {VECTOR_DB_PATH}")
        print("Please run the vectorization process first")
        return None, []
    
    try:
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        if index is not None:
            print(f"✓ Loaded vector database with {len(metadata)} documents from {VECTOR_DB_PATH}")
        return index, metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None, []

def search_similar(query_embedding: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
    index, metadata = load_vector_database()
    if index is None:
        return []
    
    query_vector = query_embedding.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_vector, k)
    
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < len(metadata) and idx != -1:
            results.append((float(distance), metadata[idx]))
    
    return results

class OptimizedEmbeddings:
    """Optimized embeddings with HNSW and caching"""
    
    def __init__(self):
        self.model = self._load_model()
        self.cache = QueryCache()
        self.index = None
        self.metadata = []
        self.use_hnsw = True
        
    @lru_cache(maxsize=1)
    def _load_model(self) -> SentenceTransformer:
        """Load embeddings model with caching"""
        return SentenceTransformer(EMBEDDINGS_MODEL)
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for text"""
        return np.asarray(self.model.encode(text))
    
    def search_with_cache(self, query: str, k: int = 5) -> List[Tuple[float, Dict]]:
        """Search with caching support"""
        # Check cache first
        cached_results = self.cache.get(query, k)
        if cached_results is not None:
            return cached_results
        
        # Perform search
        if self.index is None:
            self.index, self.metadata = load_vector_database(self.use_hnsw)
        
        if self.index is None:
            return []
        
        # Get query embedding
        query_embedding = self.get_embeddings(query)
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Format results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata) and idx != -1:
                results.append((float(distance), self.metadata[idx]))
        
        # Cache results
        self.cache.set(query, k, results)
        
        return results
    
    def rebuild_with_optimization(self, embeddings: List[np.ndarray], metadata: List[Dict]):
        """Rebuild index with optimization"""
        print("Building optimized vector index...")
        
        # Try to create HNSW index
        if self.use_hnsw and len(embeddings) > 100:  # HNSW is better for larger datasets
            try:
                self.index = create_hnsw_index(embeddings)
                print(f"✓ Created HNSW index with {len(embeddings)} vectors")
            except Exception as e:
                print(f"HNSW creation failed: {e}, falling back to flat index")
                self.index = faiss_index(embeddings)
        else:
            self.index = faiss_index(embeddings)
            print(f"✓ Created flat index with {len(embeddings)} vectors")
        
        self.metadata = metadata
        
        # Save optimized database
        if self.index is not None:
            save_vector_database(self.index, metadata)
        
        # Clear cache after rebuild
        self.cache.clear()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {
            'index_type': 'HNSW' if isinstance(self.index, faiss.IndexHNSWFlat) else 'Flat',
            'total_vectors': self.index.ntotal if self.index else 0,
            'vector_dimension': self.index.d if self.index else 0,
            'cache_size': len(self.cache.cache),
            'cache_hit_rate': 'N/A'  # Could implement hit rate tracking
        }
        
        if isinstance(self.index, faiss.IndexHNSWFlat):
            stats['hnsw_M'] = self.index.hnsw.M
            stats['hnsw_efSearch'] = self.index.hnsw.efSearch
        
        return stats

# Global optimized embeddings instance
_optimized_embeddings = None

def get_optimized_embeddings() -> OptimizedEmbeddings:
    """Get or create optimized embeddings instance"""
    global _optimized_embeddings
    if _optimized_embeddings is None:
        _optimized_embeddings = OptimizedEmbeddings()
    return _optimized_embeddings

def search_optimized(query: str, k: int = 5) -> List[Tuple[float, Dict]]:
    """Convenience function for optimized search"""
    embeddings = get_optimized_embeddings()
    return embeddings.search_with_cache(query, k)