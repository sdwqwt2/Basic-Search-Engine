"""
Контрастивное дообучение bi-encoder на train.parquet (пары "запрос -> выбранное объявление").

Идея: у готового e5 нет сигнала "что именно кликают" — только семантическая близость.
MultipleNegativesRankingLoss учит модель поднимать выбранное объявление над остальными
объявлениями батча (in-batch negatives).

ВАЖНО для Kaggle с 2 GPU: запускать с CUDA_VISIBLE_DEVICES=0, иначе Trainer включит
DataParallel и in-batch негативы будут считаться на половинках батча.

    CUDA_VISIBLE_DEVICES=0 python -m avito_retrieval.train_biencoder \
        --train-path /kaggle/input/avito/train.parquet \
        --bench-items-path /kaggle/input/avito/benchmark_items.parquet \
        --output-dir /kaggle/working/e5-avito

Используются: sentence-transformers (MultipleNegativesRankingLoss в кэшированном варианте),
базовая модель intfloat/multilingual-e5-base (open-source, локально).
"""
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses,
)
from sentence_transformers.training_args import BatchSamplers
from sklearn.model_selection import train_test_split

from avito_retrieval import config
from avito_retrieval.data.preprocess import build_dense_item_text, build_dense_query_text
from avito_retrieval.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", default=str(config.TRAIN_PATH))
    p.add_argument("--bench-items-path", default=str(config.BENCH_ITEMS_PATH))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default=config.BIENCODER_MODEL_NAME)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--split", choices=["query", "random"], default="query")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-pairs-per-query", type=int, default=20)
    p.add_argument("--n-train-pairs", type=int, default=150_000, help="0 = все пары")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256, help="Эффективный батч = число in-batch негативов")
    p.add_argument("--mini-batch-size", type=int, default=64, help="Размер куска для gradient caching (память)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--scale", type=float, default=20.0,
                   help="1/температура лосса. e5 предобучался с температурой 0.01 (scale=100) — можно попробовать")
    p.add_argument("--n-eval-queries", type=int, default=3000)
    p.add_argument("--n-distractors", type=int, default=60_000)
    p.add_argument("--skip-baseline-eval", action="store_true")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def make_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Пара (anchor, positive) с префиксами e5 и метаданными для оценки."""
    return pd.DataFrame({
        "search_query": df["search_query"].values,
        "item_id": df["item_id"].values,
        "anchor": ("query: " + build_dense_query_text(df["search_query"], df["search_infm_params_text"])).values,
        "positive": ("passage: " + build_dense_item_text(df)).values,
    })


def split_pairs(pairs: pd.DataFrame, mode: str, val_frac: float, seed: int):
    if mode == "query":
        # по уникальным запросам: запрос из val не встречается в train
        tr_q, va_q = train_test_split(pairs["search_query"].unique(), test_size=val_frac, random_state=seed)
        return pairs[pairs["search_query"].isin(tr_q)], pairs[pairs["search_query"].isin(va_q)]
    return train_test_split(pairs, test_size=val_frac, random_state=seed)


@torch.no_grad()
def eval_recall(model, val_pairs, bench_items, n_queries, n_distractors, seed):
    """
    Recall@50/100 одной dense-модели БЕЗ фильтров и без BM25 — нужен для честного сравнения
    "базовый e5 vs дообученный". Корпус: положительные из val + дистракторы из benchmark_items
    (их модель в обучении не видела, и это ближе к реальному корпусу, чем только положительные).
    """
    pos = val_pairs.drop_duplicates("item_id")
    dis = bench_items[~bench_items["item_id"].isin(pos["item_id"])]
    dis = dis.sample(min(n_distractors, len(dis)), random_state=seed)

    corpus_ids = np.concatenate([pos["item_id"].values, dis["item_id"].values])
    corpus_texts = pos["positive"].tolist() + ("passage: " + build_dense_item_text(dis)).tolist()

    gt = val_pairs.groupby("anchor")["item_id"].apply(set)
    gt = gt.sample(min(n_queries, len(gt)), random_state=seed)

    with torch.autocast("cuda", dtype=torch.float16):
        c = model.encode(corpus_texts, batch_size=256, convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=True).half()
        q = model.encode(gt.index.tolist(), batch_size=256, convert_to_tensor=True,
                         normalize_embeddings=True).half()

    hits = {50: [], 100: []}
    for s in range(0, len(q), 256):
        top = (q[s:s + 256] @ c.T).topk(100, dim=1).indices.cpu().numpy()
        for row, rel in zip(top, gt.iloc[s:s + 256]):
            ids = corpus_ids[row]
            for k in hits:
                hits[k].append(len(set(ids[:k]) & rel) / len(rel))
    res = {f"recall@{k}": float(np.mean(v)) for k, v in hits.items()}
    logger.info(f"Корпус {len(corpus_ids)}, запросов {len(gt)}: {res}")
    return res


def main():
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.device_count() > 1:
        logger.warning("Видно >1 GPU: запусти с CUDA_VISIBLE_DEVICES=0, иначе включится DataParallel!")

    train = pd.read_parquet(args.train_path)
    bench_items = pd.read_parquet(args.bench_items_path).drop_duplicates("item_id")

    pairs = make_pairs(train)
    tr, va = split_pairs(pairs, args.split, args.val_frac, args.seed)
    logger.info(f"Пар: train {len(tr)}, val {len(va)}")

    # Дедуп + лимит на запрос: частые запросы ("маникюр") иначе доминируют в обучении.
    tr = tr.drop_duplicates(["anchor", "positive"]).groupby("search_query").head(args.max_pairs_per_query)
    if args.n_train_pairs and len(tr) > args.n_train_pairs:
        tr = tr.sample(args.n_train_pairs, random_state=args.seed)
    logger.info(f"Пар для обучения после дедупа/лимитов: {len(tr)}")

    model = SentenceTransformer(args.model_name, device="cuda")
    model.max_seq_length = args.max_seq_length

    if not args.skip_baseline_eval:
        logger.info("Оценка базовой модели...")
        base = eval_recall(model, va, bench_items, args.n_eval_queries, args.n_distractors, args.seed)

    loss = losses.CachedMultipleNegativesRankingLoss(
        model, scale=args.scale, mini_batch_size=args.mini_batch_size
    )
    train_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=True,  # T4 не поддерживает bf16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # без одинаковых текстов в батче (иначе ложные негативы)
        logging_steps=25,
        save_strategy="steps", save_steps=200, save_total_limit=2,  # чекпойнты на случай обрыва сессии
        report_to="none",
        seed=args.seed,
    )
    ds = Dataset.from_pandas(tr[["anchor", "positive"]].reset_index(drop=True))
    SentenceTransformerTrainer(model=model, args=train_args, train_dataset=ds, loss=loss).train()

    final_dir = f"{args.output_dir}/final"
    model.save(final_dir)
    logger.info(f"Модель сохранена: {final_dir}")

    tuned = eval_recall(model, va, bench_items, args.n_eval_queries, args.n_distractors, args.seed)
    if not args.skip_baseline_eval:
        logger.info(f"База: {base}")
    logger.info(f"Дообученная: {tuned}")


if __name__ == "__main__":
    main()
