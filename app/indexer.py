"""Build TF-IDF and embedding index from chunks."""
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import chromadb
from scipy.sparse import csr_matrix, save_npz

from app.config import INDEX_DIR, PROCESSED_DATA_DIR, EMBEDDING_MODEL
from app.chunker import chunk_documents


def build_index():
    """Build index from documents."""
    # 1. First, create chunks if they don't exist
    chunks_file = PROCESSED_DATA_DIR / "chunks.jsonl"
    if not chunks_file.exists():
        print("Creating chunks from documents...")
        chunk_documents()
    
    # 2. Load chunks
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} chunks")
    
    # 3. Build TF-IDF index
    print("Building TF-IDF index...")
    texts = [chunk["text"] for chunk in chunks]
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=None,  # Can add Russian stop words if needed
        ngram_range=(1, 2),
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Save TF-IDF artifacts
    with open(INDEX_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save as CSR format components
    data = tfidf_matrix.data
    indices = tfidf_matrix.indices
    indptr = tfidf_matrix.indptr
    shape = tfidf_matrix.shape
    
    np.savez(INDEX_DIR / "matrix.npz", data=data, indices=indices, indptr=indptr, shape=shape)
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # 4. Build embedding index with ChromaDB
    print("Building embedding index...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Clear existing collection if any
    chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma_db"))
    try:
        chroma_client.delete_collection("rag_chunks")
    except:
        pass
    
    collection = chroma_client.create_collection("rag_chunks")
    
    # Add chunks in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        texts_batch = [chunk["text"] for chunk in batch]
        ids_batch = [chunk["chunk_id"] for chunk in batch]
        metadatas_batch = [
            {"doc_id": chunk["doc_id"], "name": chunk["name"], "source": chunk.get("source", "unknown")}
            for chunk in batch
        ]
        
        # Generate embeddings
        embeddings = embedding_model.encode(texts_batch, show_progress_bar=False)
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts_batch,
            ids=ids_batch,
            metadatas=metadatas_batch,
        )
        
        print(f"  Added {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
    
    # 5. Save chunks.jsonl for retriever
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    print(f"Index built successfully in {INDEX_DIR}")


if __name__ == "__main__":
    build_index()