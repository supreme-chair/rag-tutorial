"""Retrieval module with hybrid search (TF-IDF + embeddings)."""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import chromadb
from scipy.sparse import csr_matrix, load_npz

from app.config import INDEX_DIR, RETRIEVAL_TOP_K, RETRIEVAL_SIMILARITY_THRESHOLD, EMBEDDING_MODEL


class HybridRetriever:
    """Hybrid retriever combining TF-IDF and embedding search."""
    
    def __init__(self):
        print("🔧 Initializing HybridRetriever...")
        
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunks = []
        
        # Load chunks
        chunks_path = INDEX_DIR / "chunks.jsonl"
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.chunks.append(json.loads(line))
            print(f"   ✓ Loaded {len(self.chunks)} chunks")
        else:
            print(f"   ✗ Chunks file not found: {chunks_path}")
        
        # Load TF-IDF
        vectorizer_path = INDEX_DIR / "vectorizer.pkl"
        matrix_path = INDEX_DIR / "matrix.npz"
        
        if vectorizer_path.exists() and matrix_path.exists():
            with open(vectorizer_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            self.tfidf_matrix = load_npz(matrix_path)
            print(f"   ✓ Loaded TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        else:
            print(f"   ✗ TF-IDF files not found")
        
        # Load embedding model and ChromaDB
        print(f"   Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        chroma_path = INDEX_DIR / "chroma_db"
        print(f"   Connecting to ChromaDB at: {chroma_path}")
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        
        try:
            self.embedding_collection = self.chroma_client.get_collection("rag_chunks")
            count = self.embedding_collection.count()
            print(f"   ✓ Loaded ChromaDB collection with {count} chunks")
        except Exception as e:
            print(f"   ✗ Could not load ChromaDB collection: {e}")
            self.embedding_collection = None
        
        print("✅ Retriever ready\n")
    
    def _tfidf_search(self, query: str, k: int) -> List[tuple]:
        """Search using TF-IDF."""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]
        
        return results
    
    def _embedding_search(self, query: str, k: int) -> List[tuple]:
        """Search using embeddings via ChromaDB."""
        if self.embedding_collection is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.embedding_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
        )
        
        if results['ids'] and results['distances']:
            # Convert distances to similarities (1 - normalized distance)
            distances = results['distances'][0]
            if distances and max(distances) > 0:
                similarities = [1 - (d / max(distances)) for d in distances]
            else:
                similarities = [1.0] * len(distances)
            return [(ids[0], sim) for ids, sim in zip(results['ids'][0], similarities)]
        
        return []
    
    def search(self, query: str, k: int = RETRIEVAL_TOP_K, 
               tfidf_weight: float = 0.3, embedding_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Hybrid search combining TF-IDF and embeddings."""
        print(f"\n🔍 Searching for: '{query}'")
        
        # Get results from both methods
        tfidf_results = self._tfidf_search(query, k * 2)
        embedding_results = self._embedding_search(query, k * 2)
        
        print(f"   TF-IDF found: {len(tfidf_results)}")
        print(f"   Embeddings found: {len(embedding_results)}")
        
        # Combine scores
        combined_scores = {}
        
        for idx, score in tfidf_results:
            if idx < len(self.chunks):
                chunk_id = self.chunks[idx]["chunk_id"]
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + score * tfidf_weight
        
        for chunk_id, score in embedding_results:
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + score * embedding_weight
        
        # Sort by combined score
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Build results
        results = []
        for chunk_id, score in sorted_chunks:
            if score < RETRIEVAL_SIMILARITY_THRESHOLD:
                continue
            
            # Find chunk data
            chunk_data = next((c for c in self.chunks if c["chunk_id"] == chunk_id), None)
            if chunk_data:
                results.append({
                    "chunk_id": chunk_id,
                    "doc_id": chunk_data["doc_id"],
                    "name": chunk_data["name"],
                    "text": chunk_data["text"],
                    "source": chunk_data.get("source", "unknown"),
                    "score": round(score, 4),
                })
        
        print(f"   Final results after threshold: {len(results)}")
        return results


# Singleton instance
_retriever = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def search(query: str, k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
    """Convenience function for search."""
    retriever = get_retriever()
    return retriever.search(query, k)