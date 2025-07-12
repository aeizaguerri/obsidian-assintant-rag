"""
Advanced document chunking strategies for improved RAG performance.

This module provides sophisticated chunking methods that preserve context
and create optimal chunks for embedding and retrieval.
"""

import re
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from markdown_processing import clean_markdown_content
from metadata_extractor import extract_enhanced_metadata

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    ENCODING = None
    print("Tiktoken not available, using rough approximation for token count")

@dataclass
class ChunkConfig:
    """Configuration for chunking strategy"""
    max_chunk_size: int = 800  # Maximum tokens per chunk
    min_chunk_size: int = 100  # Minimum tokens per chunk
    overlap_size: int = 100    # Overlap between chunks in tokens
    preserve_structure: bool = True  # Preserve markdown structure
    split_on_sentences: bool = True  # Split on sentence boundaries
    include_headers: bool = True     # Include parent headers in chunks

@dataclass 
class DocumentChunk:
    """Represents a document chunk with metadata"""
    content: str
    chunk_id: str
    file_path: str
    chunk_index: int
    token_count: int
    parent_headers: List[str]
    metadata: Dict
    overlap_with_previous: bool = False
    overlap_with_next: bool = False

class AdvancedChunker:
    """Advanced document chunking with multiple strategies"""
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        if TIKTOKEN_AVAILABLE and ENCODING:
            return len(ENCODING.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
            
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns"""
        # Handle common sentence endings, but be careful with abbreviations
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_headers_hierarchy(self, content: str) -> List[Tuple[int, str, int]]:
        """Extract header hierarchy with levels, text, and positions"""
        headers = []
        lines = content.split('\n')
        position = 0
        
        for i, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                level = len(header_match.group(1))
                text = header_match.group(2).strip()
                headers.append((level, text, position))
            position += len(line) + 1  # +1 for newline
            
        return headers
        
    def get_parent_headers(self, headers: List[Tuple[int, str, int]], position: int) -> List[str]:
        """Get parent headers for a given position in the document"""
        parent_headers = []
        current_levels = {}
        
        for level, text, header_pos in headers:
            if header_pos <= position:
                # Update the hierarchy
                current_levels[level] = text
                # Remove deeper levels
                current_levels = {k: v for k, v in current_levels.items() if k <= level}
            else:
                break
                
        # Return headers in hierarchical order
        return [current_levels[level] for level in sorted(current_levels.keys())]
        
    def create_header_aware_chunks(self, content: str, file_path: str) -> List[DocumentChunk]:
        """Create chunks that respect markdown header structure"""
        headers = self.extract_headers_hierarchy(content)
        metadata = extract_enhanced_metadata(content, file_path)
        
        # Split content into sections by headers
        sections = []
        lines = content.split('\n')
        current_section = []
        current_header = None
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                # Save previous section
                if current_section:
                    sections.append((current_header, '\n'.join(current_section)))
                
                # Start new section
                current_header = line.strip()
                current_section = [line]
            else:
                current_section.append(line)
                
        # Don't forget the last section
        if current_section:
            sections.append((current_header, '\n'.join(current_section)))
            
        chunks = []
        chunk_index = 0
        
        for header, section_content in sections:
            section_chunks = self._chunk_section(
                section_content, file_path, headers, chunk_index, metadata
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
            
        return chunks
        
    def _chunk_section(self, content: str, file_path: str, headers: List[Tuple[int, str, int]], 
                      start_index: int, metadata: Dict) -> List[DocumentChunk]:
        """Chunk a section of content with overlap"""
        chunks = []
        cleaned_content = clean_markdown_content(content)
        
        if self.estimate_tokens(cleaned_content) <= self.config.max_chunk_size:
            # Section is small enough to be a single chunk
            chunk = DocumentChunk(
                content=cleaned_content,
                chunk_id=f"{os.path.basename(file_path)}_{start_index}",
                file_path=file_path,
                chunk_index=start_index,
                token_count=self.estimate_tokens(cleaned_content),
                parent_headers=self.get_parent_headers(headers, 0),
                metadata=metadata
            )
            chunks.append(chunk)
            return chunks
            
        # Split section into smaller chunks with overlap
        if self.config.split_on_sentences:
            sentences = self.split_into_sentences(cleaned_content)
            chunk_sentences = []
            current_tokens = 0
            chunk_index = start_index
            
            for i, sentence in enumerate(sentences):
                sentence_tokens = self.estimate_tokens(sentence)
                
                if (current_tokens + sentence_tokens > self.config.max_chunk_size and 
                    chunk_sentences and current_tokens >= self.config.min_chunk_size):
                    
                    # Create chunk from accumulated sentences
                    chunk_content = ' '.join(chunk_sentences)
                    chunk = DocumentChunk(
                        content=chunk_content,
                        chunk_id=f"{os.path.basename(file_path)}_{chunk_index}",
                        file_path=file_path,
                        chunk_index=chunk_index,
                        token_count=current_tokens,
                        parent_headers=self.get_parent_headers(headers, 0),
                        metadata=metadata,
                        overlap_with_next=True if i < len(sentences) - 1 else False
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_sentences = []
                    overlap_tokens = 0
                    
                    # Add sentences for overlap
                    for j in range(len(chunk_sentences) - 1, -1, -1):
                        overlap_sentence = chunk_sentences[j]
                        overlap_sentence_tokens = self.estimate_tokens(overlap_sentence)
                        
                        if overlap_tokens + overlap_sentence_tokens <= self.config.overlap_size:
                            overlap_sentences.insert(0, overlap_sentence)
                            overlap_tokens += overlap_sentence_tokens
                        else:
                            break
                            
                    chunk_sentences = overlap_sentences + [sentence]
                    current_tokens = overlap_tokens + sentence_tokens
                    chunk_index += 1
                    
                    if chunks:
                        chunks[-1].overlap_with_next = True
                        chunk.overlap_with_previous = True
                else:
                    chunk_sentences.append(sentence)
                    current_tokens += sentence_tokens
                    
            # Handle remaining sentences
            if chunk_sentences and current_tokens >= self.config.min_chunk_size:
                chunk_content = ' '.join(chunk_sentences)
                chunk = DocumentChunk(
                    content=chunk_content,
                    chunk_id=f"{os.path.basename(file_path)}_{chunk_index}",
                    file_path=file_path,
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                    parent_headers=self.get_parent_headers(headers, 0),
                    metadata=metadata,
                    overlap_with_previous=len(chunks) > 0
                )
                chunks.append(chunk)
                
        return chunks
        
    def chunk_document(self, file_path: str) -> List[DocumentChunk]:
        """Main method to chunk a document"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if self.config.preserve_structure:
            return self.create_header_aware_chunks(content, file_path)
        else:
            # Fallback to simple chunking
            return self._simple_chunk(content, file_path)
            
    def _simple_chunk(self, content: str, file_path: str) -> List[DocumentChunk]:
        """Simple chunking fallback method"""
        cleaned_content = clean_markdown_content(content)
        metadata = extract_enhanced_metadata(content, file_path)
        
        chunks = []
        if self.config.split_on_sentences:
            sentences = self.split_into_sentences(cleaned_content)
        else:
            sentences = [cleaned_content]
            
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            if (current_tokens + sentence_tokens > self.config.max_chunk_size and 
                current_chunk):
                
                # Create chunk
                chunk_content = ' '.join(current_chunk)
                chunk = DocumentChunk(
                    content=chunk_content,
                    chunk_id=f"{os.path.basename(file_path)}_{chunk_index}",
                    file_path=file_path,
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                    parent_headers=[],
                    metadata=metadata
                )
                chunks.append(chunk)
                
                current_chunk = [sentence]
                current_tokens = sentence_tokens
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                
        # Handle remaining content
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk = DocumentChunk(
                content=chunk_content,
                chunk_id=f"{os.path.basename(file_path)}_{chunk_index}",
                file_path=file_path,
                chunk_index=chunk_index,
                token_count=current_tokens,
                parent_headers=[],
                metadata=metadata
            )
            chunks.append(chunk)
            
        return chunks

def chunk_documents(file_paths: List[str], config: ChunkConfig = None) -> List[DocumentChunk]:
    """Chunk multiple documents with the advanced chunker"""
    chunker = AdvancedChunker(config)
    all_chunks = []
    
    for file_path in file_paths:
        try:
            chunks = chunker.chunk_document(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error chunking {file_path}: {e}")
            continue
            
    return all_chunks