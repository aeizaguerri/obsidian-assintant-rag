import os
import numpy as np
from markdown_processing import clean_markdown_content
from embeddings import load_embeddings_model, get_embeddings, faiss_index, save_vector_database, load_vector_database

VAULT_PATH = "../.."

def load_vault(path: str) -> list[str]:
    md_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))
    return md_files

def read_split_files(files: list[str]) -> list[str]:
    docs = []
    for file in files:
        cleaned_content = clean_md_file(file)
        fragments = cleaned_content.split("\n\n")
        for fragment in fragments:
            if fragment.strip():
                docs.append(fragment.strip())
    return docs

def clean_md_file(file: str) -> str:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        return clean_markdown_content(content)

def vectorize_docs(vault_path: str, force_rebuild: bool = False):
    # Check if vector database already exists
    existing_index, existing_metadata = load_vector_database()
    if existing_index is not None and not force_rebuild:
        print(f"Loaded existing vector database with {len(existing_metadata)} documents")
        return existing_index, existing_metadata
    
    print("Building new vector database...")
    model = load_embeddings_model()
    files = load_vault(vault_path)
    docs = read_split_files(files)
    cleaned_docs = [clean_markdown_content(doc) for doc in docs]
    
    # Create embeddings and metadata
    embeddings = []
    metadata = []
    
    for i, (doc, file_path) in enumerate(zip(cleaned_docs, get_file_paths_for_docs(files))):
        embedding = get_embeddings(model, doc)
        embeddings.append(embedding)
        metadata.append({
            'id': i,
            'content': doc,
            'file_path': file_path,
            'fragment_index': get_fragment_index(file_path, doc, files)
        })
    
    # Create and save the index
    index = faiss_index(embeddings)
    if index is not None:
        save_vector_database(index, metadata)
        print(f"Saved vector database with {len(metadata)} documents")
    
    return index, metadata

def get_file_paths_for_docs(files: list[str]) -> list[str]:
    file_paths = []
    for file in files:
        cleaned_content = clean_md_file(file)
        fragments = cleaned_content.split("\n\n")
        for fragment in fragments:
            if fragment.strip():
                file_paths.append(file)
    return file_paths

def get_fragment_index(file_path: str, doc: str, all_files: list[str]) -> int:
    cleaned_content = clean_md_file(file_path)
    fragments = cleaned_content.split("\n\n")
    for i, fragment in enumerate(fragments):
        if fragment.strip() == doc:
            return i
    return 0

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

