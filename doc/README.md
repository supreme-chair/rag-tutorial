# RAG Tutorial — учебный проект

> **Статус:** 📋 Этап планирования завершён. Реализация — после подтверждения плана.

Учебный пример RAG-приложения: учимся **пользоваться RAG** на практике.  
Данные — описания экономических датасетов с Kaggle.  
Pipeline: от сырых файлов до ответа с источниками.

## Зачем этот проект

- Понять RAG не как «магию промпта», а как **инженерный pipeline**.
- Пройти путь: идея → workflow → данные → чанки → индекс → ответ.
- Получить шаблон для самостоятельного задания на своих данных.

## Документы

| Документ | Содержание |
|----------|------------|
| [00_project_idea.md](00_project_idea.md) | Идея, цель, данные |
| [vision.md](vision.md) | Техническое видение, стек |
| [conventions.md](conventions.md) | Правила для code-ассистента |
| [tasklist.md](tasklist.md) | Итерационный план разработки |
| [workflow.md](workflow.md) | Правила работы code-ассистента по tasklist |

## Технологии (план MVP)

| Компонент | Выбор |
|-----------|-------|
| Язык | Python 3.10+ |
| UI | Streamlit |
| Vector store | ChromaDB (локально) |
| Embeddings | sentence-transformers (MiniLM multilingual) |
| Данные | Описания экономических датасетов Kaggle (JSON/txt) |
| LLM | Demo mode (без API) + опционально OpenAI-compatible |

## Будущие шаги (после одобрения плана)

1. Создать структуру `app/`, `scripts/`, `data/`, `tests/`.
2. Подготовить демо-данные (10+ описаний датасетов).
3. Реализовать ingestion → chunking → index → retrieval → answer.
4. Собрать Streamlit UI с отображением источников.
5. Написать тесты, README, EXPLAIN_FOR_STUDENTS, ASSIGNMENT_TEMPLATE.

## Быстрый старт (будет после реализации)

```bash
# Установка
pip install -r requirements.txt

# Сборка индекса
python scripts/build_index.py

# Запуск приложения
streamlit run app/main.py
```

## Ограничения MVP (план)

- Без reranking, hybrid search, auto-reindex.
- Только текстовые описания, не анализ CSV.
- Локальные embeddings (не SOTA для узкой лексики).
- Demo mode без LLM API — для работы без ключей.
- Статические файлы, без Kaggle API в runtime.

## Что улучшить дальше

- Reranking (cross-encoder).
- Eval-фреймворк (RAGAS).
- Hybrid search (BM25 + vectors).
- Подключение Kaggle API для автообновления.
- Guardrails и prompt injection filtering.
