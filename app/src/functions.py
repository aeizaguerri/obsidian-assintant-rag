import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from text_processing import clean_markdown_content

EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
VAULT_PATH = "../.."

def load_embeddings_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDINGS_MODEL)

def get_embeddings(model: SentenceTransformer, text: str) -> np.ndarray:
    return model.encode(text)

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


if __name__ == "__main__":
    # Test the markdown cleaning function
    files = load_vault(VAULT_PATH)
    if files:
        cleaned_docs = read_split_files(files)
        print(f"Found {len(cleaned_docs)} document fragments after cleaning")
        for i, doc in enumerate(cleaned_docs[:3]):  # Show first 3 fragments
            print(f"\nFragment {i+1}:")
            print(doc[:200] + "..." if len(doc) > 200 else doc)
    else:
        print(f"No markdown files found in {VAULT_PATH}")

