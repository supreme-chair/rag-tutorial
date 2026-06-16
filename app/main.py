"""Streamlit UI for RAG system."""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.retriever import search
from app.generator import generate_answer
from app.config import UI_TITLE, UI_ICON, RETRIEVAL_TOP_K


st.set_page_config(page_title=UI_TITLE, page_icon=UI_ICON, layout="wide")

st.title(f"{UI_ICON} {UI_TITLE}")

# Sidebar
with st.sidebar:
    st.header("⚙️ Настройки")
    top_k = st.slider("Количество фрагментов (top-k)", 1, 10, RETRIEVAL_TOP_K)
    show_scores = st.checkbox("Показывать оценки релевантности", value=True)
    
    st.header("📊 Статистика")
    st.info("База знаний содержит статьи по IT-тематике: Python, ML, DevOps, базы данных, веб-разработка и др.")
    
    st.header("❓ Примеры вопросов")
    st.markdown("""
    - Что такое Python?
    - Как работает Docker?
    - Объясните REST API
    - Что такое машинное обучение?
    - Какие бывают базы данных?
    """)

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area("💬 Задайте вопрос:", placeholder="Например: Что такое RAG?", height=100)
    
    if st.button("🔍 Найти ответ", type="primary"):
        if question.strip():
            with st.spinner("Поиск в базе знаний..."):
                # Retrieve chunks
                chunks = search(question, k=top_k)
                
                # Generate answer
                result = generate_answer(question, chunks)
                
                # Display answer
                st.markdown("### 📝 Ответ")
                st.markdown(result["answer"])
                
                # Store in session state for sources display
                st.session_state.last_chunks = result["sources"]
                st.session_state.last_question = question
        else:
            st.warning("Пожалуйста, введите вопрос.")

with col2:
    st.markdown("### 📚 Источники")
    if "last_chunks" in st.session_state and st.session_state.last_chunks:
        for i, chunk in enumerate(st.session_state.last_chunks, 1):
            with st.expander(f"📄 {chunk['name']}"):
                st.markdown(f"**Фрагмент {i}**")
                st.markdown(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                if show_scores:
                    st.caption(f"*Оценка релевантности: {chunk['score']}*")
                st.caption(f"🔗 Источник: {chunk.get('source', 'unknown')} | ID: {chunk['doc_id']}")
    else:
        st.info("Ответ на вопрос появится здесь")

# Footer
st.markdown("---")
st.caption("RAG система на TF-IDF + эмбеддингах. Данные: IT статьи (35+ документов).")