import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Dict, Tuple

EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = "../../assets/vector_db"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

def load_embeddings_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDINGS_MODEL)

def get_embeddings(model: SentenceTransformer, text: str) -> np.ndarray:
    return np.asarray(model.encode(text))

def create_vector_db_dir():
    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)

def save_vector_database(index: faiss.Index, metadata: List[Dict]):
    create_vector_db_dir()
    
    # Save FAISS index
    index_path = os.path.join(VECTOR_DB_PATH, INDEX_FILE)
    faiss.write_index(index, index_path)
    
    # Save metadata
    metadata_path = os.path.join(VECTOR_DB_PATH, METADATA_FILE)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

def load_vector_database() -> Tuple[faiss.Index | None, List[Dict]]:
    index_path = os.path.join(VECTOR_DB_PATH, INDEX_FILE)
    metadata_path = os.path.join(VECTOR_DB_PATH, METADATA_FILE)
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, []
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata

def faiss_index(embeddings: List[np.ndarray]) -> faiss.Index | None:
    if not embeddings:
        return None
    
    embeddings_array = np.vstack(embeddings)
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)
    return index

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