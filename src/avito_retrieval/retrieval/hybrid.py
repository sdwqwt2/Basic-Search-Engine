import numpy as np
import pandas as pd
from tqdm import tqdm

from avito_retrieval.retrieval.category_filter import CategoryFilter
from avito_retrieval.retrieval.location_filter import LocationFilter
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
    RRF с весом alpha: alpha=1.0 -> только BM25, alpha=0.0 -> только dense.
    score(item) = alpha * 1/(k + rank_bm25) + (1 - alpha) * 1/(k + rank_dense)
    Отсутствие item в одном из списков даёт вклад 0 от этого списка.
    """
    bm25_ranks = {i: r for r, (i, _) in enumerate(bm25_results, start=1)}
    dense_ranks = {i: r for r, (i, _) in enumerate(dense_results, start=1)}

    scores = {}
    for item_id in set(bm25_ranks) | set(dense_ranks):
        s = 0.0
        if item_id in bm25_ranks:
            s += alpha / (k + bm25_ranks[item_id])
        if item_id in dense_ranks:
            s += (1 - alpha) / (k + dense_ranks[item_id])
        scores[item_id] = s
    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def _col(df: pd.DataFrame, name: str, default=None) -> list:
    return df[name].tolist() if name in df.columns else [default] * len(df)


class HybridRetriever:
    """
    Один BM25 + один Dense индекс на весь корпус. Для каждого запроса строится булева
    маска допустимых объявлений: категория (из train) И локация (обученный маппинг).
    Обе части — мягкие: если маска оставляет меньше `min_mask_items` объявлений,
    соответствующее ограничение снимается (защита от пустых выдач).
    """

    def __init__(
            self,
            category_filter: CategoryFilter,
            location_filter: LocationFilter | None = None,
            model_name: str = config.BIENCODER_MODEL_NAME,
            max_seq_length: int = 128,
            batch_size: int = 128,
            min_mask_items: int = 200,
            use_query_params: bool = True,  # добавлять фильтры поиска в текст dense-запроса
            encode_devices: list[str] | None = None,
    ):
        self.category_filter = category_filter
        self.location_filter = location_filter
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.min_mask_items = min_mask_items
        self.use_query_params = use_query_params
        self.encode_devices = encode_devices

        self.bm25: BM25Retriever | None = None
        self.dense: DenseRetriever | None = None
        self.item_categories: np.ndarray | None = None
        self.item_locations: np.ndarray | None = None

    def _new_dense(self) -> DenseRetriever:
        return DenseRetriever(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            encode_devices=self.encode_devices,
        )

    def build(self, items_df: pd.DataFrame):
        """Строит оба индекса один раз на весь корпус."""
        self.item_categories = items_df["item_category_id"].to_numpy()
        self.item_locations = items_df["item_location_id"].to_numpy()

        print(f"Строю BM25-индекс: {len(items_df)} объявлений")
        self.bm25 = BM25Retriever().fit(
            items_df["item_id"], build_bm25_item_text(items_df), items_df["item_category_id"]
        )
        print(f"Строю Dense-индекс: {len(items_df)} объявлений")
        self.dense = self._new_dense().fit(items_df["item_id"], build_dense_item_text(items_df))
        return self

    def _build_mask(self, search_query, search_category, search_location_id, cache: dict) -> np.ndarray:
        cats = self.category_filter.get_allowed_categories(search_query, search_category)
        key = (frozenset(cats) if cats else None, search_location_id)
        if key in cache:
            return cache[key]

        n = len(self.item_categories)
        mask = np.isin(self.item_categories, list(cats)) if cats else np.ones(n, dtype=bool)
        if mask.sum() < self.min_mask_items:
            mask = np.ones(n, dtype=bool)  # категория дала слишком мало -> без неё

        if self.location_filter is not None and search_location_id is not None and not pd.isna(search_location_id):
            locs = self.location_filter.get_allowed_locations(search_location_id)
            with_loc = mask & np.isin(self.item_locations, list(locs))
            if with_loc.sum() >= self.min_mask_items:
                mask = with_loc  # локация применяется, только если оставляет достаточно

        cache[key] = mask
        return mask

    def search_candidates(
            self,
            queries: pd.DataFrame,
            top_k_each: int = config.TOP_K_BM25,
            chunk_size: int = 256,
    ) -> tuple[list[list], list[list]]:
        """
        Возвращает (bm25_lists, dense_lists) по каждому запросу. Списки НЕ зависят от alpha,
        поэтому для подбора alpha их достаточно посчитать один раз и затем вызывать fuse().
        Ожидаемые колонки queries: search_query, search_category, search_location_id, search_infm_params_text
        (отсутствующие заменяются на None / "").
        """
        bm25_all, dense_all = [], []
        for start in tqdm(range(0, len(queries), chunk_size), desc="Кандидаты"):
            chunk = queries.iloc[start:start + chunk_size]
            q_text = _col(chunk, "search_query", "")
            q_cat = _col(chunk, "search_category")
            q_loc = _col(chunk, "search_location_id")
            q_par = _col(chunk, "search_infm_params_text", "")

            cache: dict = {}  # кэш масок живёт в пределах чанка, чтобы не копить память
            masks = np.stack([self._build_mask(q, c, l, cache) for q, c, l in zip(q_text, q_cat, q_loc)])

            # BM25 — только по тексту запроса: слова фильтров ("Вид услуги ...") встречаются везде и шумят
            for q, m in zip(q_text, masks):
                bm25_all.append(self.bm25.search(q, top_k=top_k_each, mask=m))

            dense_q = [
                f"{q}. {p}".strip() if (self.use_query_params and p) else q
                for q, p in zip(q_text, q_par)
            ]
            dense_all.extend(self.dense.search_batch(dense_q, masks, top_k=top_k_each))
        return bm25_all, dense_all

    @staticmethod
    def fuse(bm25_all, dense_all, alpha: float = 0.5, final_top_k: int = config.FINAL_TOP_K) -> list[list[str]]:
        return [
            reciprocal_rank_fusion(b, d, alpha=alpha)[:final_top_k]
            for b, d in zip(bm25_all, dense_all)
        ]

    def retrieve_batch(
            self,
            queries: pd.DataFrame,
            alpha: float = 0.5,
            top_k_each: int = config.TOP_K_BM25,
            final_top_k: int = config.FINAL_TOP_K,
            chunk_size: int = 256,
    ) -> list[list[str]]:
        bm25_all, dense_all = self.search_candidates(queries, top_k_each, chunk_size)
        return self.fuse(bm25_all, dense_all, alpha, final_top_k)

    def retrieve_for_query(
            self,
            search_query: str,
            search_category=None,
            alpha: float = 0.5,
            top_k_each: int = config.TOP_K_BM25,
            final_top_k: int = config.FINAL_TOP_K,
            search_location_id=None,
            search_params: str = "",
    ) -> list[str]:
        """Совместимость со старым интерфейсом (один запрос)."""
        q = pd.DataFrame([{
            "search_query": search_query, "search_category": search_category,
            "search_location_id": search_location_id, "search_infm_params_text": search_params,
        }])
        return self.retrieve_batch(q, alpha, top_k_each, final_top_k)[0]

    def save(self, dir_path: str):
        self.bm25.save(f"{dir_path}/bm25.pkl")
        self.dense.save(f"{dir_path}/dense_emb.npy")
        pd.DataFrame({
            "item_id": self.dense.item_ids,
            "item_category_id": self.item_categories,
            "item_location_id": self.item_locations,
        }).to_parquet(f"{dir_path}/dense_meta.parquet")

    def load(self, dir_path: str):
        meta = pd.read_parquet(f"{dir_path}/dense_meta.parquet")
        self.item_categories = meta["item_category_id"].to_numpy()
        self.item_locations = meta["item_location_id"].to_numpy()

        self.bm25 = BM25Retriever().load(f"{dir_path}/bm25.pkl")
        assert self.bm25.item_ids == meta["item_id"].tolist(), "BM25 и dense построены на разном порядке объявлений"

        self.dense = self._new_dense().load(f"{dir_path}/dense_emb.npy", meta["item_id"].tolist())
        return self
