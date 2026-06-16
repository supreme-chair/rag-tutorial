"""Standalone script to build the index without Streamlit."""
import json
import pickle
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import chromadb
from scipy.sparse import save_npz

# Paths
PROJECT_ROOT = Path(__file__).parent
INDEX_DIR = PROJECT_ROOT / "data" / "index"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Create directories
INDEX_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

print("=" * 60)
print("RAG INDEX BUILDER")
print("=" * 60)

# Step 1: Load datasets.json
print("\n[1/5] Loading datasets.json...")
datasets_file = RAW_DATA_DIR / "datasets.json"

if not datasets_file.exists():
    print(f"ERROR: {datasets_file} not found!")
    print("Creating directory and you need to place datasets.json there.")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    exit(1)

# FIXED: Using utf-8-sig to handle BOM
with open(datasets_file, "r", encoding="utf-8-sig") as f:
    data = json.load(f)

print(f"✓ Loaded {len(data['datasets'])} documents")

# Step 2: Convert to documents.jsonl
print("\n[2/5] Converting to documents.jsonl...")
docs_jsonl = PROCESSED_DATA_DIR / "documents.jsonl"

with open(docs_jsonl, "w", encoding="utf-8") as f:
    for i, doc in enumerate(data["datasets"]):
        doc_id = f"doc_{i:04d}"
        jsonl_doc = {
            "doc_id": doc_id,
            "name": doc["name"],
            "text": doc["text"],
            "source_file": doc.get("source", "unknown"),
        }
        f.write(json.dumps(jsonl_doc, ensure_ascii=False) + "\n")

print(f"✓ Created {docs_jsonl}")

# Step 3: Create chunks
print("\n[3/5] Creating chunks...")
from app.chunker import chunk_documents

chunks = chunk_documents(
    input_file=docs_jsonl,
    output_file=PROCESSED_DATA_DIR / "chunks.jsonl"
)

print(f"✓ Created {len(chunks)} chunks")

# Step 4: Load chunks for indexing
print("\n[4/5] Loading chunks...")
chunks_list = []
with open(PROCESSED_DATA_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks_list.append(json.loads(line))

texts = [chunk["text"] for chunk in chunks_list]
print(f"✓ Loaded {len(chunks_list)} chunks")

# Step 5: Build TF-IDF index
print("\n[5/5] Building TF-IDF and embedding indexes...")
print("   Building TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(texts)

# Save TF-IDF
with open(INDEX_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

save_npz(INDEX_DIR / "matrix.npz", tfidf_matrix)
print(f"   ✓ TF-IDF matrix shape: {tfidf_matrix.shape}")

# Step 6: Build embedding index
print("   Building embedding index with ChromaDB...")
print(f"   Loading model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Create ChromaDB
chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma_db"))
try:
    chroma_client.delete_collection("rag_chunks")
    print("   Removed existing collection")
except:
    pass

collection = chroma_client.create_collection("rag_chunks")

# Add in batches
batch_size = 50
for i in range(0, len(chunks_list), batch_size):
    batch = chunks_list[i:i+batch_size]
    
    texts_batch = [chunk["text"] for chunk in batch]
    ids_batch = [chunk["chunk_id"] for chunk in batch]
    metadatas_batch = [
        {"doc_id": chunk["doc_id"], "name": chunk["name"], "source": chunk.get("source", "unknown")}
        for chunk in batch
    ]
    
    embeddings = embedding_model.encode(texts_batch, show_progress_bar=False)
    
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts_batch,
        ids=ids_batch,
        metadatas=metadatas_batch,
    )
    
    print(f"   Added {min(i+batch_size, len(chunks_list))}/{len(chunks_list)} chunks")

# Save chunks for retriever
with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
    for chunk in chunks_list:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print("\n" + "=" * 60)
print("✅ INDEX BUILD COMPLETE!")
print(f"   Index location: {INDEX_DIR}")
print("=" * 60)
print("\n🎉 Now run: streamlit run app/main.py")