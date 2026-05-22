# Tasklist — итерационный план разработки

Опирается на: @vision.md · @conventions.md · @00_project_idea.md · @workflow.md

**Правило:** одна итерация = один проверяемый результат. Не переходить дальше, пока текущий шаг не протестирован.

---

## 📊 Отчёт по прогрессу

| Итерация | Название | Статус | Проверка |
|:--------:|----------|:------:|----------|
| 0 | Каркас проекта | ✅ | `uv sync` без ошибок |
| 1 | Демо-данные Kaggle | ⬜ | JSON читается, ≥5 датасетов |
| 2 | Ingestion | ⬜ | `documents.jsonl` создан |
| 3 | Chunking | ⬜ | `chunks.jsonl`, тест chunking |
| 4 | Индекс TF-IDF | ⬜ | файлы в `data/index/` |
| 5 | Retrieval | ⬜ | top-k + score в консоли |
| 6 | Demo-ответ | ⬜ | ответ + источники без UI |
| 7 | Streamlit UI | ⬜ | 3 demo-вопроса в браузере |
| 8 | Тесты и README | ⬜ | `pytest` green, README воспроизводим |

**Легенда:** ⬜ не начато · 🔄 в работе · ✅ готово · ❌ блокер

**Текущая итерация:** 1  
**Готовность MVP:** 1 / 9

---

## Итерация 0 — Каркас проекта

- [x] `pyproject.toml` — зависимости: streamlit, scikit-learn, pytest
- [x] `.gitignore` — `.venv/`, `data/index/`, `__pycache__/`
- [x] Папки: `app/`, `scripts/`, `data/raw/`, `data/processed/`, `data/index/`, `tests/`
- [x] `app/config.py` — пути, `top_k`, размер чанка

**Проверка:**
```bash
uv venv && uv sync
uv run python -c "import app.config"
```

---

## Итерация 1 — Демо-данные Kaggle

- [ ] `data/raw/datasets.json` — 5–10 описаний (безработица, инфляция, ВВП…)
- [ ] Поля: `id`, `name`, `text` (+ опционально `tags`, `kaggle_url`)

**Проверка:**
```bash
uv run python -c "import json; d=json.load(open('data/raw/datasets.json')); print(len(d['datasets']))"
```

---

## Итерация 2 — Ingestion

- [ ] `scripts/ingest.py` — JSON → `data/processed/documents.jsonl`
- [ ] Очистка текста, метаданные (`doc_id`, `name`, `source_file`)

**Проверка:**
```bash
uv run python scripts/ingest.py
# → documents.jsonl, строк = числу датасетов
```

---

## Итерация 3 — Chunking

- [ ] `app/chunker.py` — нарезка по абзацам, max 400, overlap 50
- [ ] Выход: `data/processed/chunks.jsonl`

**Проверка:**
```bash
uv run python -c "from app.chunker import ..."  # или через build_index позже
uv run pytest tests/test_chunking.py -v
```

---

## Итерация 4 — Индекс TF-IDF

- [ ] `scripts/build_index.py` — ingest + chunk + TF-IDF fit
- [ ] Сохранение: `data/index/vectorizer.pkl`, `matrix.npz`, `chunks.jsonl`

**Проверка:**
```bash
uv run python scripts/build_index.py
# → три файла в data/index/
```

---

## Итерация 5 — Retrieval

- [ ] `app/retriever.py` — загрузка индекса, cosine top-k
- [ ] Возврат: `text`, `doc_id`, `score`

**Проверка:**
```bash
uv run python -c "
from app.retriever import Retriever
r = Retriever()
print(r.search('безработица переменные', k=3))
"
```

---

## Итерация 6 — Demo-ответ

- [ ] `app/prompts.py` — system-правила (только по контексту, отказ без данных)
- [ ] `app/generator.py` — ответ из top-k + список источников

**Проверка:**
```bash
uv run python -c "
from app.generator import ask
print(ask('Какие переменные в датасете про безработицу?'))
"
# → текст + sources с doc_id
```

---

## Итерация 7 — Streamlit UI

- [ ] `app/main.py` — поле вопроса, top-k фрагменты, ответ, источники
- [ ] Сообщение, если индекс не собран

**Проверка:**
```bash
uv run streamlit run app/main.py
```
Demo-вопросы:
1. «Какие переменные в датасете про безработицу?»
2. «За какой период данные об инфляции?»
3. «Как приготовить борщ?» → отказ

---

## Итерация 8 — Тесты и README

- [ ] `tests/test_chunking.py`, `tests/test_retrieval.py` — 3–5 тестов
- [ ] Корневой `README.md` — uv, build_index, streamlit, demo-вопросы

**Проверка:**
```bash
uv run pytest tests/ -v
# README: запуск с нуля на чистой машине
```

---

## Критерий «MVP готов»

- [ ] Все итерации ✅ в таблице прогресса
- [ ] 3 demo-вопроса — ответ с источником; 1 negative — отказ
- [ ] Студент может пройти путь: `data/raw/` → index → Streamlit
