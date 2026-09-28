import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from avito_retrieval import config
from avito_retrieval.utils import get_device


class DenseRetriever:
    """
    Эмбеддинги всего корпуса лежат одним тензором на устройстве (GPU/MPS/CPU).
    Поиск: батч запросов @ эмбеддинги.T, затем маска допустимых объявлений
    (-inf для запрещённых) и topk. Маска у каждого запроса своя (категория + локация),
    поэтому FAISS IDSelector не нужен: матмул по 189k x 768 занимает миллисекунды.
    """

    def __init__(
            self,
            model_name: str = config.BIENCODER_MODEL_NAME,
            max_seq_length: int = 128,
            batch_size: int = 128,
            encode_devices: list[str] | None = None,  # напр. ["cuda:0", "cuda:1"] для энкодинга корпуса
    ):
        self.device = get_device()
        print(f"DenseRetriever использует device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.encode_devices = encode_devices
        # fp16 для матмула на GPU/MPS; на CPU остаёмся в fp32
        self.dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        self.emb: torch.Tensor | None = None
        self.item_ids: list[str] = []

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        # нормализуем вручную: так результат не зависит от версии sentence-transformers
        # и от того, использовался ли multi-process pool -> inner product = cosine
        return (x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)).astype("float32")

    def _encode(self, texts: list[str], show_progress: bool = False, pool=None) -> np.ndarray:
        if pool is not None:
            emb = self.model.encode(texts, pool=pool, batch_size=self.batch_size)
        else:
            emb = self.model.encode(
                texts, batch_size=self.batch_size, show_progress_bar=show_progress, convert_to_numpy=True
            )
        return self._normalize(emb)

    def fit(self, item_ids: pd.Series, item_texts: pd.Series, is_e5: bool = True):
        """Энкодит весь корпус один раз (опционально на нескольких GPU)."""
        self.item_ids = item_ids.tolist()
        texts = item_texts.tolist()
        if is_e5:
            texts = [f"passage: {t}" for t in texts]  # E5 требует этот префикс

        pool = self.model.start_multi_process_pool(self.encode_devices) if self.encode_devices else None
        try:
            emb = self._encode(texts, show_progress=True, pool=pool)
        finally:
            if pool is not None:
                self.model.stop_multi_process_pool(pool)
        self._set_embeddings(emb)
        return self

    def _set_embeddings(self, emb: np.ndarray):
        self.emb = torch.from_numpy(emb).to(self.device, self.dtype)

    def search_batch(
            self,
            query_texts: list[str],
            masks: np.ndarray,  # bool [B, N]: какие объявления допустимы для каждого запроса
            top_k: int = 100,
            is_e5: bool = True,
    ) -> list[list[tuple[str, float]]]:
        if is_e5:
            query_texts = [f"query: {t}" for t in query_texts]
        q = torch.from_numpy(self._encode(query_texts)).to(self.device, self.dtype)

        scores = q @ self.emb.T  # [B, N]
        m = torch.from_numpy(masks).to(self.device)
        scores = scores.masked_fill(~m, float("-inf"))
        vals, idx = scores.topk(min(top_k, scores.shape[1]), dim=1)
        vals, idx = vals.float().cpu().numpy(), idx.cpu().numpy()

        return [
            [(self.item_ids[j], float(v)) for j, v in zip(row_i, row_v) if np.isfinite(v)]
            for row_i, row_v in zip(idx, vals)
        ]

    def save(self, emb_path: str):
        np.save(emb_path, self.emb.float().cpu().numpy().astype("float16"))

    def load(self, emb_path: str, item_ids: list[str]):
        self.item_ids = item_ids
        self._set_embeddings(np.load(emb_path).astype("float32"))
        return self
