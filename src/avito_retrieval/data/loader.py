import pandas as pd
from avito_retrieval import config


def load_train() -> pd.DataFrame:
    """Загружает обучающую выборку (пары запрос-объявление)."""
    return pd.read_parquet(config.TRAIN_PATH)


def load_benchmark_queries() -> pd.DataFrame:
    """Загружает запросы бенчмарка без разметки."""
    return pd.read_parquet(config.BENCH_QUERIES_PATH)


def load_benchmark_items() -> pd.DataFrame:
    """Загружает корпус объявлений бенчмарка."""
    return pd.read_parquet(config.BENCH_ITEMS_PATH)


def build_item_text(df: pd.DataFrame) -> pd.Series:
    """
    Собирает единый текст объявления для индексации (BM25 и bi-encoder).
    Порядок важен: title обычно самый информативный сигнал, ставим первым.
    Пустые поля безопасно заменяются на пустую строку.
    """
    title = df["item_title_raw"].fillna("")
    params = df["item_infm_params_text"].fillna("")
    desc = df["item_description_raw"].fillna("")
    return (title + ". " + params + ". " + desc).str.strip()


def build_query_text(df: pd.DataFrame) -> pd.Series:
    """Собирает текст запроса вместе с текстовыми фильтрами поиска."""
    query = df["search_query"].fillna("")
    params = df["search_infm_params_text"].fillna("")
    return (query + ". " + params).str.strip()