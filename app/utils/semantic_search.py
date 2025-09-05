
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

from app.utils.embed import KnowledgeBaseEmbedder

class SemanticSearch:
    def __init__(self, index_path: str, metadata_path: str):
        embedder = KnowledgeBaseEmbedder()
        print("Generating and storing knowledge base embeddings...")
        embedder.generate_and_store_embeddings()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            self.knowledge_chunks = json.load(f)

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        results = [self.knowledge_chunks[i] for i in indices[0]]
        return results
