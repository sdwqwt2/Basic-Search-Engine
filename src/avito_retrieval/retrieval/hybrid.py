import pandas as pd

from avito_retrieval.retrieval.category_filter import CategoryFilter
from avito_retrieval.retrieval.lexical import BM25Retriever
from avito_retrieval.retrieval.dense import DenseRetriever
from avito_retrieval.data.preprocess import build_bm25_item_text, build_dense_item_text
from avito_retrieval import config


def reciprocal_rank_fusion(
        bm25_results: list[tuple[str, float]],
        dense_results: list[tuple[str, float]],
        alpha: float = 0.5,
        k: int = 60,
) -> list[str]:
    """
    RRF со смешивающим весом alpha между лексическим и семантическим списками.
    alpha=1.0 -> только BM25, alpha=0.0 -> только dense, alpha=0.5 -> классический RRF.

    score(item) = alpha * 1/(k + rank_bm25) + (1 - alpha) * 1/(k + rank_dense)

    Отсутствие item в одном из списков не штрафуется явно большим рангом —
    его вклад от этого списка просто равен 0, что не наказывает находки,
    обнаруженные только одним из двух методов.
    """
    bm25_ranks = {item_id: rank for rank, (item_id, _) in enumerate(bm25_results, start=1)}
    dense_ranks = {item_id: rank for rank, (item_id, _) in enumerate(dense_results, start=1)}

    all_items = set(bm25_ranks) | set(dense_ranks)
    scores = {}
    for item_id in all_items:
        s = 0.0
        if item_id in bm25_ranks:
            s += alpha * (1.0 / (k + bm25_ranks[item_id]))
        if item_id in dense_ranks:
            s += (1 - alpha) * (1.0 / (k + dense_ranks[item_id]))
        scores[item_id] = s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in ranked]


class HybridRetriever:
    """
    Один BM25-индекс + один Dense-индекс на весь корпус (построены один раз
    в build()). Категорийная фильтрация — маска на этапе поиска.
    Слияние результатов — RRF с настраиваемым alpha.
    """

    def __init__(
            self,
            category_filter: CategoryFilter,
            model_name: str = config.BIENCODER_MODEL_NAME,
            max_seq_length: int = 128,
            batch_size: int = 128,
    ):
        self.category_filter = category_filter
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.bm25: BM25Retriever | None = None
        self.dense: DenseRetriever | None = None

    def build(self, items_df: pd.DataFrame):
        """Строит оба индекса ОДИН РАЗ на весь корпус."""
        bm25_texts = build_bm25_item_text(items_df)
        dense_texts = build_dense_item_text(items_df)

        print(f"Строю BM25-индекс: {len(items_df)} объявлений")
        self.bm25 = BM25Retriever().fit(items_df["item_id"], bm25_texts, items_df["item_category_id"])

        print(f"Строю Dense-индекс: {len(items_df)} объявлений")
        self.dense = DenseRetriever(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
        ).fit(items_df["item_id"], dense_texts, items_df["item_category_id"])
        return self

    def retrieve_for_query(
            self,
            search_query: str,
            search_category=None,
            alpha: float = 0.5,
            top_k_each: int = config.TOP_K_BM25,
            final_top_k: int = config.FINAL_TOP_K,
    ) -> list[str]:
        allowed = self.category_filter.get_allowed_categories(search_query, search_category)

        bm25_results = self.bm25.search(search_query, top_k=top_k_each, allowed_categories=allowed)
        dense_results = self.dense.search(search_query, top_k=top_k_each, allowed_categories=allowed)

        fused = reciprocal_rank_fusion(bm25_results, dense_results, alpha=alpha)
        return fused[:final_top_k]

    def save(self, dir_path: str):
        self.bm25.save(f"{dir_path}/bm25.pkl")
        self.dense.save(f"{dir_path}/dense.index", f"{dir_path}/dense_meta.parquet")

    def load(self, dir_path: str):
        self.bm25 = BM25Retriever().load(f"{dir_path}/bm25.pkl") if self.bm25 is None else self.bm25
        self.bm25 = BM25Retriever()
        self.bm25.load(f"{dir_path}/bm25.pkl")

        self.dense = DenseRetriever(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
        )
        self.dense.load(f"{dir_path}/dense.index", f"{dir_path}/dense_meta.parquet")
        return self
