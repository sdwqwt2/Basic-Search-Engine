import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from avito_retrieval import config


class DenseRetriever:
    """Семантический поиск через bi-encoder + FAISS (inner product по нормализованным векторам = cosine)."""

    def __init__(self, model_name: str = config.BIENCODER_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.item_ids: list[str] = []

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # E5-модели требуют префиксы "query: " / "passage: " — учти это,
        # если используешь intfloat/multilingual-e5-*
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # нормализация -> inner product = cosine similarity
        )
        return embeddings.astype("float32")

    def fit(self, item_ids: pd.Series, item_texts: pd.Series, is_e5: bool = True):
        self.item_ids = item_ids.tolist()
        texts = item_texts.tolist()
        if is_e5:
            texts = [f"passage: {t}" for t in texts]
        embeddings = self._encode(texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        return self

    def search(self, query_text: str, top_k: int = 100, is_e5: bool = True) -> list[tuple[str, float]]:
        text = f"query: {query_text}" if is_e5 else query_text
        query_emb = self._encode([text])
        scores, idx = self.index.search(query_emb, top_k)
        return [(self.item_ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

    def save(self, index_path: str, ids_path: str):
        faiss.write_index(self.index, index_path)
        with open(ids_path, "w") as f:
            f.write("\n".join(self.item_ids))

    def load(self, index_path: str, ids_path: str):
        self.index = faiss.read_index(index_path)
        with open(ids_path) as f:
            self.item_ids = f.read().splitlines()
        return self