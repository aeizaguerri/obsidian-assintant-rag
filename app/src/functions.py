import faiss
import numpy as np

embeddings_model = "all-MiniLM-L6-v2"

def get_embeddings(text):
    return embeddings_model.encode(text)




