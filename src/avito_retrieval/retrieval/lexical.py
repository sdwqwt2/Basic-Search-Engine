import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from avito_retrieval.data.preprocess import tokenize_simple


class BM25Retriever:
    """
    Один BM25-индекс на весь корпус объявлений.
    Категорийная фильтрация — маскированием скоров (-inf для запрещённых
    категорий), без пересборки индекса под каждую категорию.
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.item_ids: list[str] = []
        self.item_categories: np.ndarray | None = None

    def fit(self, item_ids: pd.Series, item_texts: pd.Series, item_categories: pd.Series):
        self.item_ids = item_ids.tolist()
        self.item_categories = item_categories.to_numpy()
        tokenized_corpus = [tokenize_simple(t) for t in item_texts.tolist()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return self

    def search(self, query_text: str, top_k: int = 100, mask: np.ndarray | None = None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize_simple(query_text))
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        # scores > 0: документы без единого совпадения токенов только шумят в RRF
        return [(self.item_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "item_ids": self.item_ids, "item_categories": self.item_categories}, f
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.item_ids = data["item_ids"]
        self.item_categories = data["item_categories"]
        return self
