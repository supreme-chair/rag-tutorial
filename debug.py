import json
from pathlib import Path

print("=== DIAGNOSTICS ===\n")

# 1. Check processed chunks
chunks_file = Path('data/processed/chunks.jsonl')
if chunks_file.exists():
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]
    print(f'1. Chunks in processed: {len(chunks)}')
    for chunk in chunks[:3]:
        print(f'   - {chunk["name"]}: {chunk["text"][:80]}...')
else:
    print('1. chunks.jsonl NOT FOUND in processed/')

print()

# 2. Check index chunks
index_chunks_file = Path('data/index/chunks.jsonl')
if index_chunks_file.exists():
    with open(index_chunks_file, 'r', encoding='utf-8') as f:
        index_chunks = [json.loads(line) for line in f]
    print(f'2. Chunks in index: {len(index_chunks)}')
    for chunk in index_chunks[:3]:
        print(f'   - {chunk["name"]}: {chunk["text"][:80]}...')
else:
    print('2. index chunks NOT FOUND')

print()

# 3. Check TF-IDF
import pickle
import numpy as np
vectorizer_file = Path('data/index/vectorizer.pkl')
matrix_file = Path('data/index/matrix.npz')
if vectorizer_file.exists() and matrix_file.exists():
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    loaded = np.load(matrix_file)
    print(f'3. TF-IDF loaded: vocab size={len(vectorizer.vocabulary_)}')
else:
    print('3. TF-IDF files missing')

print()

# 4. Test similarity
if index_chunks_file.exists() and vectorizer_file.exists():
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import csr_matrix
    
    query = 'python'
    query_vec = vectorizer.transform([query])
    matrix = csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
    sim = cosine_similarity(query_vec, matrix).flatten()
    print(f'4. Similarity scores for "{query}":')
    print(f'   min={sim.min():.6f}, max={sim.max():.6f}, mean={sim.mean():.6f}')
    print(f'   Non-zero scores: {(sim > 0).sum()} out of {len(sim)}')
    
    # Top results
    top_indices = sim.argsort()[-5:][::-1]
    print(f'   Top 5 scores:')
    for idx in top_indices:
        if idx < len(index_chunks):
            print(f'     {sim[idx]:.4f} - {index_chunks[idx]["name"]}')

print()

# 5. Check retriever
print('5. Testing HybridRetriever:')
try:
    from app.retriever import HybridRetriever
    r = HybridRetriever()
    print(f'   TF-IDF ready: {r.tfidf_vectorizer is not None}')
    print(f'   Chunks loaded: {len(r.chunks)}')
    print(f'   Embedding ready: {r.embedding_collection is not None}')
    
    results = r.search('python', k=3)
    print(f'   Search results: {len(results)}')
    for res in results:
        print(f'     score={res["score"]} - {res["name"]}')
except Exception as e:
    print(f'   ERROR: {e}')
