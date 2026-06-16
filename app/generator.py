"""Answer generator that uses retrieved chunks as context."""
from typing import List, Dict, Any


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string."""
    if not chunks:
        return "Нет релевантных документов."
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] Источник: {chunk['name']}\n{chunk['text']}")
    
    return "\n\n---\n\n".join(context_parts)


def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate an answer based on retrieved chunks.
    In MVP, this is a demo answer without external LLM.
    """
    if not chunks:
        return {
            "answer": "Извините, я не могу ответить на этот вопрос, так как не нашёл релевантной информации в базе знаний. Пожалуйста, задайте вопрос по темам, которые есть в документации.",
            "sources": [],
            "has_context": False,
        }
    
    # Demo answer: select the best chunk and provide it as context
    best_chunk = chunks[0]
    
    # Create a simple answer based on the retrieved content
    answer = f"На основе найденной информации из документа «{best_chunk['name']}»:\n\n{best_chunk['text']}"
    
    # If multiple chunks, add note
    if len(chunks) > 1:
        answer += f"\n\n(Также найдено ещё {len(chunks) - 1} релевантных фрагментов.)"
    
    return {
        "answer": answer,
        "sources": chunks,
        "has_context": True,
    }


def generate_with_llm(question: str, chunks: List[Dict[str, Any]], api_key: str = None) -> Dict[str, Any]:
    """
    Generate answer using an LLM (OpenAI-compatible API).
    This is an improvement over the demo answer.
    """
    if not api_key:
        return generate_answer(question, chunks)
    
    try:
        import openai
        
        if not chunks:
            return {
                "answer": "Извините, я не могу ответить на этот вопрос, так как не нашёл релевантной информации.",
                "sources": [],
                "has_context": False,
            }
        
        context = format_context(chunks)
        
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты полезный ассистент. Отвечай только на основе предоставленного контекста. Если ответа нет в контексте, скажи, что не знаешь."},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:"}
            ],
            temperature=0.3,
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": chunks,
            "has_context": True,
        }
    except Exception as e:
        print(f"LLM error: {e}")
        return generate_answer(question, chunks)