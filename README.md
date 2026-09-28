
# Avito Candidate Generation

Решение этапа кандидатогенерации для поиска услуг Авито: по короткому
поисковому запросу отобрать до 50 объявлений-кандидатов из корпуса,
максимизируя Recall@50.

## Подход

Решение — гибридный поиск, каскад из трёх этапов:

1. **Фильтрация по категории.** EDA показал, что `search_category` в
   бенчмарке почти не информативен (только два значения: `0` и `114`,
   тогда как `item_category_id` имеет 47 значений), поэтому основной
   сигнал — маппинг **текст запроса → категория объявления**, обученный
   на `train.parquet` (для >99.99% уникальных запросов одна категория
   покрывает ≥80% реальных выборов пользователей). `search_category`
   используется только когда он валиден и встречается среди категорий
   объявлений; иначе включается fallback без фильтра.

2. **Лексический + семантический поиск внутри отфильтрованного
   подмножества:**
   - **BM25** (`rank_bm25`) — по полному тексту объявления
     (`title + params + description`).
   - **Bi-encoder** (`intfloat/multilingual-e5-base` по умолчанию,
     `sentence-transformers`) + **FAISS** (`IndexFlatIP` по
     нормализованным эмбеддингам = cosine similarity) — по обрезанному
     тексту объявления (title + params + первые 250 символов
     description; `max_seq_length=192` токенов).

   Оба индекса строятся **один раз на весь корпус**; категорийная
   фильтрация на этапе поиска реализована через маскирование скоров
   (BM25) и `faiss.IDSelectorArray` (dense) — без пересборки индексов
   под каждую категорию.

3. **Слияние результатов — Reciprocal Rank Fusion (RRF)** с настраиваемым
   весом `alpha` между BM25 и dense-списками:

   ```
   score(item) = alpha * 1/(k + rank_bm25) + (1 - alpha) * 1/(k + rank_dense)
   ```

   RRF выбран вместо простого union или взвешенной суммы сырых скоров,
   потому что не требует нормализации скоров разной природы (BM25 vs
   cosine) и явно поддаётся тюнингу одним параметром. `alpha` подобран
   через grid search на отложенной валидации (см. ниже).

### Почему не переранжирование, а именно этот подход к отбору

Задача — только кандидатогенерация (Recall@50, без учёта порядка
внутри топ-50), поэтому решение сфокусировано на полноте: категорийный
фильтр резко сужает пространство поиска без потери релевантных
объявлений (высокая точность маппинга запрос→категория), а RRF на
уровне рангов, а не скоров, даёт устойчивое объединение двух
непохожих по своей природе списков кандидатов.

## Используемые open-source модели и библиотеки

- `sentence-transformers` + `intfloat/multilingual-e5-base` — bi-encoder
  для семантического поиска (разворачивается локально, без обращений к
  внешним API).
- `rank_bm25` — лексический поиск (BM25Okapi).
- `faiss-cpu` — векторный индекс и поиск по эмбеддингам.
- `transformers` — токенизатор для EDA длин и для bi-encoder.

Все модели скачиваются один раз с HuggingFace Hub при первом запуске
и далее кешируются локально (`~/.cache/huggingface`) — сам инференс
не обращается к внешним API.

## Структура проекта

```
avito-candidate-generation/
│
├── data/
│   ├── raw/            # train.parquet, benchmark_queries.parquet, benchmark_items.parquet
│   ├── processed/       # (не используется в текущей версии; зарезервировано)
│   └── indices/         # Сохранённые индексы (bm25.pkl, dense.index, dense_meta.parquet)
│
├── notebooks/
│   ├── EDA.ipynb            # Анализ длин текстов, категорий, пустых полей, дубликатов
│   └── tune_alpha.ipynb     # Подбор alpha (и top_n_categories) на отложенной валидации
│
├── src/avito_retrieval/
│   ├── config.py             # Пути, константы, параметры по умолчанию
│   ├── utils.py               # get_device (CUDA/MPS/CPU), set_seed
│   ├── metrics.py             # Recall@K
│   ├── build_index.py         # Скрипт построения индексов (отдельно от main.py)
│   ├── main.py                 # Финальный прогон: benchmark -> answer.csv
│   │
│   ├── data/
│   │   ├── loader.py          # Чтение parquet, дедуп item_id
│   │   └── preprocess.py      # Нормализация текста, сборка текстов для BM25/dense
│   │
│   └── retrieval/
│       ├── category_filter.py # Маппинг запрос -> допустимые категории
│       ├── lexical.py          # BM25Retriever (единый индекс + маска по категории)
│       ├── dense.py            # DenseRetriever (единый FAISS-индекс + маска по категории)
│       └── hybrid.py           # HybridRetriever + Reciprocal Rank Fusion
│
├── answer.csv                  # Результат (генерируется main.py)
├── requirements.txt
├── README.md
└── .gitignore
```

## Установка

Требуется Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[notebooks]"   # для полного набора, включая EDA/tune_alpha
# или
pip install -e .                # только для запуска main.py / build_index.py
```

Скачай данные по ссылке из условия задания и положи файлы в `data/raw/`:

```
data/raw/train.parquet
data/raw/benchmark_queries.parquet
data/raw/benchmark_items.parquet
```

## Воспроизведение результата

### Шаг 1. (Опционально) Подбор alpha и top_n_categories

Подбор проводится на отдельной валидации, построенной из `train.parquet`
(train/val сплит по уникальным `search_query`, чтобы избежать утечки),
т.к. `benchmark_items.parquet` не размечен.

```bash
jupyter notebook notebooks/tune_alpha.ipynb
```

Ноутбук выводит Recall@50 для сетки значений `alpha` (и опционально
`top_n_categories`) и печатает лучшую комбинацию. Уже подобранные по
умолчанию значения (`alpha=0.5`, `top_n_categories=2`) — разумная
стартовая точка, если пропустить этот шаг.

### Шаг 2. Построение индексов

Строит BM25 и Dense индексы один раз на полном корпусе
`benchmark_items.parquet` (189 212 объявлений) и сохраняет их в
`data/indices/`.

```bash
python -m avito_retrieval.build_index \
    --max-seq-length 192 \
    --batch-size 128 \
    --top-n-categories 2
```

**Время выполнения** зависит от устройства:
- CUDA GPU: ~10–20 минут
- Apple Silicon (MPS): ~40–60 минут при `max_seq_length=192`
- CPU: несколько часов — рекомендуется по возможности использовать
  MPS/CUDA или уменьшить `--max-seq-length`

Автоматически определяется лучшее доступное устройство: CUDA → MPS →
CPU (см. `avito_retrieval/utils.py: get_device`).

### Шаг 3. Финальный прогон и генерация answer.csv

```bash
python -m avito_retrieval.main \
    --alpha 0.5 \
    --top-n-categories 2 \
    --max-seq-length 192
```

Если индексы ещё не построены (шаг 2 пропущен), добавь флаг
`--rebuild-index` — `main.py` построит их сам перед прогоном:

```bash
python -m avito_retrieval.main --alpha 0.5 --rebuild-index
```

Скрипт:
1. Загружает (или строит) индексы.
2. Прогоняет кандидатогенерацию по всем 2452 запросам из
   `benchmark_queries.parquet`.
3. Валидирует формат ответа (≤50 уникальных `item_id` на запрос, все
   `item_id` существуют в корпусе, покрыты все `query_id`, без дублей
   строк).
4. Сохраняет `answer.csv` в корне проекта.

Время выполнения шага 3 (при готовых индексах) — единицы минут:
основная вычислительная нагрузка приходится на построение индексов
(шаг 2), а сам поиск по 2452 запросам — быстрые операции над уже
построенными структурами (BM25 scoring, FAISS search).

### Параметры main.py

| Флаг | По умолчанию                    | Описание |
|---|---------------------------------|---|
| `--alpha` | 0.5                             | Вес BM25 в RRF (1.0 = только BM25, 0.0 = только dense) |
| `--top-n-categories` | 2                               | Сколько категорий на запрос допускать в фильтре |
| `--top-k-each` | 100                             | Сколько кандидатов брать из каждого метода до слияния |
| `--final-top-k` | 50                              | Финальное число кандидатов на запрос |
| `--max-seq-length` | 192                             | Макс. длина в токенах для bi-encoder |
| `--model-name` | `intfloat/multilingual-e5-base` | Модель bi-encoder |
| `--rebuild-index` | False                           | Построить индексы заново вместо загрузки |
| `--index-dir` | `data/indices`                  | Куда сохранять/откуда загружать индексы |
| `--output-path` | `answer.csv`                    | Путь для сохранения результата |

## Ограничения решения

- Решение полностью локально: модели скачиваются один раз с
  HuggingFace Hub и кешируются, инференс не обращается к внешним API.
- Random seed фиксирован (`config.SEED = 42`) для воспроизводимости.
- Категорийный фильтр — эвристика на основе `train.parquet`; для
  запросов, не встречавшихся в train, включается fallback без
  фильтрации по категории (поиск по всему корпусу).
- `max_seq_length=192` для bi-encoder — компромисс между качеством и
  скоростью, обоснован в `notebooks/EDA.ipynb`: truncation с конца
  отсекает в основном "хвост" description, а title + params (наиболее
  информативная часть) почти всегда помещается целиком.