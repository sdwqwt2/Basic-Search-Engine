import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from avito_retrieval import config
from avito_retrieval.utils import get_device


class DenseRetriever:
    """
    Один общий FAISS-индекс на весь корпус объявлений.
    Категорийная фильтрация делается через IDSelectorArray на этапе поиска —
    без пересборки индекса и без повторного энкодинга под каждую категорию.

    max_seq_length ограничен (по умолчанию 128) осознанно: EDA показал, что
    truncation с конца отсекает в основном "хвост" description, а самая
    информативная часть текста (title + params) идёт первой и почти всегда
    попадает в окно целиком. Ограничение длины даёт кратное ускорение
    энкодинга (512->128 токенов: ~4x) ценой минимальной потери контекста.
    """

    def __init__(
            self,
            model_name: str = config.BIENCODER_MODEL_NAME,
            max_seq_length: int = 128,
            batch_size: int = 128,
    ):
        self.device = get_device()
        print(f"DenseRetriever использует device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

        self.index: faiss.Index | None = None
        self.item_ids: list[str] = []
        self.item_categories: np.ndarray | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # нормализация -> inner product = cosine similarity
        )
        return embeddings.astype("float32")

    def fit(self, item_ids: pd.Series, item_texts: pd.Series, item_categories: pd.Series, is_e5: bool = True):
        """Энкодит ВЕСЬ корпус один раз и строит один плоский индекс."""
        self.item_ids = item_ids.tolist()
        self.item_categories = item_categories.to_numpy()

        texts = item_texts.tolist()
        if is_e5:
            texts = [f"passage: {t}" for t in texts]  # E5-модели требуют этот префикс для документов

        embeddings = self._encode(texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        return self

    def search(
            self,
            query_text: str,
            top_k: int = 100,
            allowed_categories: set | None = None,
            is_e5: bool = True,
    ) -> list[tuple[str, float]]:
        text = f"query: {query_text}" if is_e5 else query_text
        query_emb = self._encode([text])

        if allowed_categories:
            mask = np.isin(self.item_categories, list(allowed_categories))
            allowed_idx = np.where(mask)[0].astype("int64")
            if len(allowed_idx) == 0:
                return []
            selector = faiss.IDSelectorArray(allowed_idx)
            search_params = faiss.SearchParameters(sel=selector)
            k = min(top_k, len(allowed_idx))
            scores, idx = self.index.search(query_emb, k, params=search_params)
        else:
            scores, idx = self.index.search(query_emb, top_k)

        return [(self.item_ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)
        pd.DataFrame({"item_id": self.item_ids, "item_category_id": self.item_categories}).to_parquet(meta_path)

    def load(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(index_path)
        meta = pd.read_parquet(meta_path)
        self.item_ids = meta["item_id"].tolist()
        self.item_categories = meta["item_category_id"].to_numpy()
        return self
