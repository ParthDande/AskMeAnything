# file: rag_searcher.py

import faiss
import pickle
from sentence_transformers import SentenceTransformer

class RAGSearcher:
    def __init__(self, index_path="my_vector.index", chunks_path="chunks.pkl", model_name='all-MiniLM-L6-v2'):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(model_name)

    def search(self, query, k=3):
        query_embedding = self.model.encode([query]).astype("float32")
        D, I = self.index.search(query_embedding, k)
        results = []
        for i in range(k):
            results.append({
                "chunk": self.chunks[I[0][i]],
                "score": D[0][i]
            })
        return results
