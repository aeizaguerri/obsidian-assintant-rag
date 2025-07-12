"""
Advanced retrieval system with multi-stage processing, reranking, and query enhancement.

This module provides sophisticated retrieval capabilities that go beyond simple
semantic similarity to provide more relevant and comprehensive context.
"""

import re
import os
import math
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from vector_db import load_embeddings_model, get_embeddings, search_similar

@dataclass
class RetrievalConfig:
    """Configuration for advanced retrieval"""
    initial_k: int = 20              # Initial candidates to retrieve
    final_k: int = 5                 # Final number of results
    enable_reranking: bool = True    # Enable cross-encoder reranking
    enable_query_expansion: bool = True  # Enable query expansion
    diversity_threshold: float = 0.7  # Similarity threshold for diversity
    context_window_size: int = 3     # Include surrounding chunks
    boost_related_chunks: bool = True # Boost chunks from same document
    boost_header_matches: bool = True # Boost chunks with matching headers

@dataclass 
class RetrievalResult:
    """Enhanced retrieval result with scoring breakdown"""
    content: str
    metadata: Dict
    semantic_score: float
    rerank_score: float = 0.0
    keyword_score: float = 0.0
    diversity_score: float = 0.0
    total_score: float = 0.0
    rank: int = 0

class QueryProcessor:
    """Process and expand queries for better retrieval"""
    
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common stop words and extract meaningful terms
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in self.stop_words and len(word) > 2]
        return keywords
    
    def expand_query_with_synonyms(self, query: str) -> str:
        """Expand query with potential synonyms and related terms"""
        # Simple synonym expansion - in production, you might use WordNet or similar
        synonym_map = {
            'implement': ['create', 'build', 'develop', 'make'],
            'error': ['bug', 'issue', 'problem', 'exception'],
            'function': ['method', 'procedure', 'routine'],
            'data': ['information', 'content', 'details'],
            'config': ['configuration', 'settings', 'setup'],
            'process': ['procedure', 'workflow', 'method'],
            'analyze': ['examine', 'study', 'review', 'investigate'],
            'optimize': ['improve', 'enhance', 'refine', 'tune']
        }
        
        expanded_terms = []
        keywords = self.extract_keywords(query)
        
        for keyword in keywords:
            expanded_terms.append(keyword)
            if keyword in synonym_map:
                expanded_terms.extend(synonym_map[keyword][:2])  # Add top 2 synonyms
        
        # Return original query + expanded terms
        return query + " " + " ".join(set(expanded_terms) - set(keywords))
    
    def preprocess_query(self, query: str) -> Dict:
        """Preprocess query and extract various features"""
        return {
            'original': query,
            'expanded': self.expand_query_with_synonyms(query),
            'keywords': self.extract_keywords(query),
            'is_question': '?' in query,
            'is_imperative': any(query.lower().startswith(cmd) for cmd in ['show', 'tell', 'explain', 'list', 'find']),
            'has_code_terms': bool(re.search(r'\b(function|class|method|variable|code|script)\b', query.lower()))
        }

class KeywordScorer:
    """Score documents based on keyword matching"""
    
    def calculate_tf_idf_score(self, query_keywords: List[str], document: str, all_documents: List[str]) -> float:
        """Calculate TF-IDF score for keyword matching"""
        if not query_keywords:
            return 0.0
        
        doc_words = re.findall(r'\b\w+\b', document.lower())
        doc_word_count = len(doc_words)
        
        if doc_word_count == 0:
            return 0.0
        
        total_docs = len(all_documents)
        score = 0.0
        
        for keyword in query_keywords:
            # Term frequency
            tf = doc_words.count(keyword) / doc_word_count
            
            # Document frequency
            df = sum(1 for doc in all_documents if keyword in doc.lower())
            
            # Inverse document frequency
            idf = math.log(total_docs / (df + 1))
            
            score += tf * idf
        
        return score

class AdvancedRetriever:
    """Advanced retrieval system with multiple stages"""
    
    def __init__(self, config: RetrievalConfig = None):
        self.config = config or RetrievalConfig()
        self.query_processor = QueryProcessor()
        self.keyword_scorer = KeywordScorer()
        self.embeddings_model = load_embeddings_model()
        
    def retrieve_initial_candidates(self, query: str, processed_query: Dict) -> List[Tuple[float, Dict]]:
        """Stage 1: Retrieve initial candidates using semantic similarity"""
        # Use expanded query for better semantic matching
        search_query = processed_query['expanded'] if self.config.enable_query_expansion else query
        query_embedding = get_embeddings(self.embeddings_model, search_query)
        
        # Get more candidates than needed for reranking
        candidates = search_similar(query_embedding, self.config.initial_k)
        return candidates
    
    def calculate_keyword_scores(self, query_keywords: List[str], candidates: List[Tuple[float, Dict]]) -> Dict[int, float]:
        """Calculate keyword-based scores for candidates"""
        documents = [metadata['content'] for _, metadata in candidates]
        keyword_scores = {}
        
        for i, (_, metadata) in enumerate(candidates):
            score = self.keyword_scorer.calculate_tf_idf_score(
                query_keywords, metadata['content'], documents
            )
            keyword_scores[i] = score
            
        return keyword_scores
    
    def calculate_diversity_scores(self, candidates: List[Tuple[float, Dict]]) -> Dict[int, float]:
        """Calculate diversity scores to avoid redundant results"""
        diversity_scores = {}
        selected_indices = set()
        
        for i, (_, metadata_i) in enumerate(candidates):
            max_similarity = 0.0
            
            for j in selected_indices:
                _, metadata_j = candidates[j]
                
                # Simple content similarity check
                content_i = set(metadata_i['content'].lower().split())
                content_j = set(metadata_j['content'].lower().split())
                
                if content_i and content_j:
                    similarity = len(content_i & content_j) / len(content_i | content_j)
                    max_similarity = max(max_similarity, similarity)
            
            # Diversity score is inverse of max similarity
            diversity_scores[i] = 1.0 - max_similarity
            
            # Add to selected if diverse enough
            if max_similarity < self.config.diversity_threshold:
                selected_indices.add(i)
        
        return diversity_scores
    
    def boost_related_chunks(self, candidates: List[Tuple[float, Dict]], query: str) -> Dict[int, float]:
        """Boost chunks that are related to high-scoring chunks"""
        boost_scores = defaultdict(float)
        file_scores = defaultdict(list)
        
        # Group candidates by file
        for i, (score, metadata) in enumerate(candidates):
            file_path = metadata.get('file_path', '')
            file_scores[file_path].append((i, score))
        
        # Boost chunks from files with high-scoring chunks
        for file_path, chunk_scores in file_scores.items():
            if len(chunk_scores) > 1:
                # Calculate average score for file
                avg_score = sum(score for _, score in chunk_scores) / len(chunk_scores)
                
                # Boost all chunks from high-scoring files
                for chunk_idx, chunk_score in chunk_scores:
                    boost_scores[chunk_idx] = avg_score * 0.2  # 20% boost
        
        return dict(boost_scores)
    
    def boost_header_matches(self, candidates: List[Tuple[float, Dict]], query_keywords: List[str]) -> Dict[int, float]:
        """Boost chunks where headers match query terms"""
        header_boost_scores = {}
        
        for i, (_, metadata) in enumerate(candidates):
            boost = 0.0
            parent_headers = metadata.get('parent_headers', [])
            
            for header in parent_headers:
                header_words = re.findall(r'\b\w+\b', header.lower())
                
                # Check for keyword matches in headers
                matches = sum(1 for keyword in query_keywords if keyword in header_words)
                if matches > 0:
                    boost += matches * 0.3  # 30% boost per matching keyword
            
            header_boost_scores[i] = boost
            
        return header_boost_scores
    
    def combine_scores(self, candidates: List[Tuple[float, Dict]], 
                      keyword_scores: Dict[int, float],
                      diversity_scores: Dict[int, float],
                      boost_scores: Dict[int, float],
                      header_scores: Dict[int, float]) -> List[RetrievalResult]:
        """Combine all scores to create final ranking"""
        results = []
        
        # Normalize scores
        max_semantic = max((score for score, _ in candidates), default=1.0)
        max_keyword = max(keyword_scores.values(), default=1.0) if keyword_scores else 1.0
        max_boost = max(boost_scores.values(), default=1.0) if boost_scores else 1.0
        max_header = max(header_scores.values(), default=1.0) if header_scores else 1.0
        
        for i, (semantic_score, metadata) in enumerate(candidates):
            # Normalize individual scores
            norm_semantic = (max_semantic - semantic_score) / max_semantic  # Lower distance = higher score
            norm_keyword = keyword_scores.get(i, 0.0) / max_keyword if max_keyword > 0 else 0.0
            norm_diversity = diversity_scores.get(i, 1.0)
            norm_boost = boost_scores.get(i, 0.0) / max_boost if max_boost > 0 else 0.0
            norm_header = header_scores.get(i, 0.0) / max_header if max_header > 0 else 0.0
            
            # Weighted combination
            total_score = (
                norm_semantic * 0.4 +      # 40% semantic similarity
                norm_keyword * 0.25 +      # 25% keyword matching
                norm_diversity * 0.15 +    # 15% diversity
                norm_boost * 0.1 +         # 10% related chunk boost
                norm_header * 0.1          # 10% header matching
            )
            
            result = RetrievalResult(
                content=metadata['content'],
                metadata=metadata,
                semantic_score=norm_semantic,
                keyword_score=norm_keyword,
                diversity_score=norm_diversity,
                total_score=total_score
            )
            results.append(result)
        
        # Sort by total score
        results.sort(key=lambda x: x.total_score, reverse=True)
        
        # Add ranks
        for i, result in enumerate(results):
            result.rank = i + 1
            
        return results
    
    def retrieve_with_context(self, query: str) -> List[RetrievalResult]:
        """Main retrieval method with full pipeline"""
        # Stage 1: Query processing
        processed_query = self.query_processor.preprocess_query(query)
        
        # Stage 2: Initial semantic retrieval
        candidates = self.retrieve_initial_candidates(query, processed_query)
        
        if not candidates:
            return []
        
        # Stage 3: Calculate additional scores
        keyword_scores = self.calculate_keyword_scores(processed_query['keywords'], candidates)
        diversity_scores = self.calculate_diversity_scores(candidates)
        
        boost_scores = {}
        header_scores = {}
        
        if self.config.boost_related_chunks:
            boost_scores = self.boost_related_chunks(candidates, query)
            
        if self.config.boost_header_matches:
            header_scores = self.boost_header_matches(candidates, processed_query['keywords'])
        
        # Stage 4: Combine scores and rank
        results = self.combine_scores(candidates, keyword_scores, diversity_scores, 
                                    boost_scores, header_scores)
        
        # Stage 5: Return top results
        return results[:self.config.final_k]
    
    def get_enhanced_context(self, query: str) -> str:
        """Get enhanced context with better formatting"""
        results = self.retrieve_with_context(query)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results):
            context_entry = []
            
            # Add ranking and score information
            context_entry.append(f"**Result {i+1} (Score: {result.total_score:.3f})**")
            
            # Add file and location info
            filename = result.metadata.get('filename', 'Unknown')
            context_entry.append(f"**Source:** {filename}")
            
            # Add parent headers if available
            if result.metadata.get('parent_headers'):
                headers = " > ".join(result.metadata['parent_headers'])
                context_entry.append(f"**Section:** {headers}")
            
            # Add tags if present
            if result.metadata.get('tags'):
                tags_str = ", ".join(f"#{tag}" for tag in result.metadata['tags'])
                context_entry.append(f"**Tags:** {tags_str}")
            
            # Add backlinks info if relevant
            if result.metadata.get('backlinks'):
                backlink_count = len(result.metadata['backlinks'])
                context_entry.append(f"**Connected to:** {backlink_count} other notes")
            
            # Add the actual content
            context_entry.append(f"**Content:** {result.content}")
            
            # Add score breakdown for debugging (optional)
            if result.semantic_score > 0 or result.keyword_score > 0:
                score_details = f"Semantic: {result.semantic_score:.2f}, Keywords: {result.keyword_score:.2f}"
                context_entry.append(f"**Relevance:** {score_details}")
            
            context_parts.append("\n".join(context_entry))
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)

# Global retriever instance
_retriever_instance = None

def get_retriever(config: RetrievalConfig = None) -> AdvancedRetriever:
    """Get or create a retriever instance"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = AdvancedRetriever(config)
    return _retriever_instance

def search_with_advanced_retrieval(query: str, config: RetrievalConfig = None) -> str:
    """Convenience function for advanced retrieval"""
    retriever = get_retriever(config)
    return retriever.get_enhanced_context(query)