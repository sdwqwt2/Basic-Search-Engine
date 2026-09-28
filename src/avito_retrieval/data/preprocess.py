import re
import pandas as pd


def normalize_text(text: str) -> str:
    """Базовая нормализация текста: нижний регистр, схлопывание пробелов."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_dense_item_text(df: pd.DataFrame, max_params_chars: int = 200, max_desc_chars: int = 250) -> pd.Series:
    """
    Текст объявления для bi-encoder: title + params + description, каждая часть обрезана по символам.
    params обрезаются, потому что у части объявлений это длинные прайс-листы, которые иначе
    съедают окно 128 токенов до description. Используется и в обучении, и в индексации.
    """
    title = df["item_title_raw"].fillna("")
    params = df["item_infm_params_text"].fillna("").str.slice(0, max_params_chars)
    desc = df["item_description_raw"].fillna("").str.slice(0, max_desc_chars)
    return (title + ". " + params + ". " + desc).map(normalize_text)


def build_dense_query_text(query: pd.Series, params: pd.Series, use_params: bool = True) -> pd.Series:
    """Текст запроса для bi-encoder (без префикса 'query: ', его добавляет вызывающий код)."""
    query = query.fillna("")
    if not use_params:
        return query
    params = params.fillna("")
    return query.where(params == "", query + ". " + params)


def build_bm25_item_text(df: pd.DataFrame) -> pd.Series:
    """Текст объявления для BM25: полный текст, т.к. это просто мешок токенов."""
    title = df["item_title_raw"].fillna("")
    params = df["item_infm_params_text"].fillna("")
    desc = df["item_description_raw"].fillna("")
    text = title + ". " + params + ". " + desc
    return text.map(normalize_text)


def build_query_text(df: pd.DataFrame) -> pd.Series:
    query = df["search_query"].fillna("")
    params = df["search_infm_params_text"].fillna("")
    return (query + ". " + params).map(normalize_text)


def tokenize_simple(text: str) -> list[str]:
    """Простая токенизация для BM25 (rank_bm25 сам эмбеддинги не строит)."""
    return re.findall(r"\w+", text.lower())
