# RAG Tutorial

Учебный пример RAG: от идеи до ответа с источниками на данных Kaggle.

**Документация и ход разработки:** [doc/README.md](doc/README.md)

## Структура

```
rag-tutorial/
├── doc/          ← планирование (идея, workflow, data, …)
├── app/          ← код (после реализации)
├── scripts/
├── data/
└── tests/
```

## Быстрый старт (после реализации)

```bash
uv venv
uv sync
python scripts/build_index.py
uv run streamlit run app/main.py
```
