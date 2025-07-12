import os
from dotenv import load_dotenv
from embeddings import load_embeddings_model, get_embeddings, faiss_index, save_vector_database, load_vector_database
from advanced_chunking import chunk_documents, ChunkConfig

# Load environment variables
load_dotenv()
VAULT_PATH = os.getenv("OBSIDIAN_FOLDER", "../..")

def validate_vault_path(path: str) -> bool:
    """Validate that the vault path exists and is accessible"""
    if not path:
        print("❌ No vault path provided")
        return False
    
    if not os.path.exists(path):
        print(f"❌ Vault path does not exist: {path}")
        return False
    
    if not os.path.isdir(path):
        print(f"❌ Vault path is not a directory: {path}")
        return False
    
    # Check if we can read the directory
    try:
        os.listdir(path)
    except PermissionError:
        print(f"❌ No permission to read vault directory: {path}")
        return False
    
    print(f"✓ Vault path validated: {path}")
    return True

def load_vault(path: str) -> list[str]:
    md_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))
    return md_files


def vectorize_docs(vault_path: str, force_rebuild: bool = False, chunk_config: ChunkConfig | None = None):
    # Validate vault path first
    if not validate_vault_path(vault_path):
        print("❌ Cannot proceed with invalid vault path")
        return None, []
    
    # Check if vector database already exists
    existing_index, existing_metadata = load_vector_database()
    if existing_index is not None and not force_rebuild:
        print(f"Loaded existing vector database with {len(existing_metadata)} documents")
        return existing_index, existing_metadata
    
    print("Building new vector database with advanced chunking...")
    model = load_embeddings_model()
    files = load_vault(vault_path)
    
    if not files:
        print(f"❌ No markdown files found in {vault_path}")
        return None, []
    
    print(f"Found {len(files)} markdown files")
    
    # Use advanced chunking system
    if chunk_config is None:
        chunk_config = ChunkConfig(
            max_chunk_size=800,
            min_chunk_size=100,
            overlap_size=100,
            preserve_structure=True,
            split_on_sentences=True,
            include_headers=True
        )
    
    print("Chunking documents with advanced strategy...")
    chunks = chunk_documents(files, chunk_config)
    
    if not chunks:
        print("❌ No chunks created after processing files")
        return None, []
    
    print(f"Created {len(chunks)} chunks from {len(files)} files")
    
    # Create embeddings and metadata
    embeddings = []
    metadata = []
    
    for i, chunk in enumerate(chunks):
        # Skip empty chunks
        if not chunk.content.strip():
            continue
            
        embedding = get_embeddings(model, chunk.content)
        embeddings.append(embedding)
        
        # Enhanced metadata with chunking information
        chunk_metadata = {
            'id': i,
            'content': chunk.content,
            'file_path': chunk.file_path,
            'chunk_id': chunk.chunk_id,
            'chunk_index': chunk.chunk_index,
            'token_count': chunk.token_count,
            'parent_headers': chunk.parent_headers,
            'overlap_with_previous': chunk.overlap_with_previous,
            'overlap_with_next': chunk.overlap_with_next,
            **chunk.metadata  # Include all the enhanced metadata
        }
        metadata.append(chunk_metadata)
    
    # Create and save the index
    index = faiss_index(embeddings)
    if index is not None:
        save_vector_database(index, metadata)
        print(f"Saved vector database with {len(metadata)} chunks")
        print(f"Average tokens per chunk: {sum(chunk.token_count for chunk in chunks) / len(chunks):.1f}")
    
    return index, metadata


if __name__ == "__main__":
    # Test the vector database functionality
    index, metadata = vectorize_docs(VAULT_PATH)
    
    if index is not None and metadata:
        print(f"\nVector database ready with {len(metadata)} documents")
        print(f"Vector dimension: {index.d}")
        print(f"Total vectors: {index.ntotal}")
        
        # Show sample metadata
        print("\nSample documents:")
        for i, meta in enumerate(metadata[:3]):
            print(f"\nDocument {i+1}:")
            print(f"File: {meta['file_path']}")
            print(f"Content: {meta['content'][:150]}...")
    else:
        print(f"No markdown files found in {VAULT_PATH}")

