"""Build TF-IDF index and embedding index."""
import json
import pickle
from pathlib import Path
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import PROCESSED_DATA_DIR, INDEX_DIR, EMBEDDING_MODEL


def build_tfidf_index():
    """Build and save TF-IDF index."""
    chunks_file = PROCESSED_DATA_DIR / "chunks.jsonl"
    if not chunks_file.exists():
        print("Error: chunks.jsonl not found. Run chunking first.")
        return False
    
    # Load chunks
    chunks = []
    texts = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)
            texts.append(chunk["text"])
    
    # Build TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=None,  # Russian stop words could be added
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)
    
    # Save artifacts
    with open(INDEX_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(INDEX_DIR / "matrix.npz", "wb") as f:
        np.savez_compressed(f, data=matrix.data, indices=matrix.indices,
                            indptr=matrix.indptr, shape=matrix.shape)
    
    # Save chunks metadata
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    print(f"TF-IDF index built: {matrix.shape[0]} chunks, {matrix.shape[1]} features")
    return True


def build_embedding_index():
    """Build and save embedding index using ChromaDB."""
    chunks_file = PROCESSED_DATA_DIR / "chunks.jsonl"
    if not chunks_file.exists():
        print("Error: chunks.jsonl not found. Run chunking first.")
        return False
    
    # Load chunks
    chunks = []
    texts = []
    metadatas = []
    ids = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)
            texts.append(chunk["text"])
            metadatas.append({
                "doc_id": chunk["doc_id"],
                "name": chunk["name"],
                "source": chunk["source"],
            })
            ids.append(chunk["chunk_id"])
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma_db"))
    
    # Delete existing collection if exists
    try:
        chroma_client.delete_collection("rag_chunks")
    except:
        pass
    
    collection = chroma_client.create_collection(name="rag_chunks")
    
    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Generate embeddings in batches
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )
        print(f"Added {len(batch_ids)} chunks to ChromaDB ({(i+len(batch_texts))}/{len(texts)})")
    
    print(f"Embedding index built with {len(texts)} chunks")
    return True


if __name__ == "__main__":
    print("Building TF-IDF index...")
    build_tfidf_index()
    print("\nBuilding embedding index...")
    build_embedding_index()
    print("\nAll indexes built successfully!")