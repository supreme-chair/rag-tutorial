# RAG Tutorial Project

RAG (Retrieval-Augmented Generation) система для ответов на вопросы по IT-статьям на русском языке.

## 🚀 Возможности

- Поиск по двум методам: TF-IDF + эмбеддинги (гибридный поиск)
- Веб-интерфейс на Streamlit
- Поддержка 35+ IT-документов (расширяется)
- Интерактивный поиск с источниками

## 📦 Требования

- Python 3.10+
- uv (быстрый установщик пакетов)

## 🛠️ Установка и запуск

```bash
# 1. Клонировать репозиторий
git clone <your-repo-url>
cd rag-project

# 2. Установить зависимости
uv sync

# 3. Построить индексы (TF-IDF + эмбеддинги)
uv run python scripts/build_index.py

# 4. Запустить веб-интерфейс
uv run streamlit run app/main.py