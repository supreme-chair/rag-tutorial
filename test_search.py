from app.retriever import search

question = "Как работает Docker?"
results = search(question, k=5)

print(f"Найдено результатов: {len(results)}")
for r in results:
    print(f"\n--- {r['name']} (score: {r['score']}) ---")
    print(r['text'][:200])