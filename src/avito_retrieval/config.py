from pathlib import Path

# Пути к данным
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_INDICES = ROOT_DIR / "data" / "indices"

TRAIN_PATH = DATA_RAW / "train.parquet"
BENCH_QUERIES_PATH = DATA_RAW / "benchmark_queries.parquet"
BENCH_ITEMS_PATH = DATA_RAW / "benchmark_items.parquet"

# Random seed
SEED = 42

# Кандидатогенерация
TOP_K_BM25 = 100        # сколько кандидатов брать из BM25 до слияния
TOP_K_DENSE = 100       # сколько кандидатов брать из bi-encoder до слияния
FINAL_TOP_K = 50        # финальное число кандидатов (требование задачи)

# Модель bi-encoder (кандидаты для сравнения в EDA/baseline)
BIENCODER_CANDIDATES = [
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "ai-forever/sbert_large_nlu_ru",
]
BIENCODER_MODEL_NAME = BIENCODER_CANDIDATES[0]  # выбор после сравнения
MAX_SEQ_LENGTH = 512     # уточнить после EDA длин