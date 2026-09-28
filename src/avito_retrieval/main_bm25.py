"""
Бейзлайн: чистый BM25 + категорийный фильтр, без dense-части.
Загружает только bm25.pkl (модель bi-encoder не поднимается).

Запуск:
    python -m avito_retrieval.main_bm25
"""
import argparse
import logging
import time

from tqdm import tqdm
import pandas as pd

from avito_retrieval.data.loader import load_benchmark_items, load_benchmark_queries, load_train
from avito_retrieval.retrieval.category_filter import CategoryFilter
from avito_retrieval.retrieval.lexical import BM25Retriever
from avito_retrieval.main import validate_answer
from avito_retrieval.utils import set_seed
from avito_retrieval import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Чистый BM25 бейзлайн")
    p.add_argument("--top-n-categories", type=int, default=2)
    p.add_argument("--no-category-filter", action="store_true",
                   help="Искать по всему корпусу, без фильтра по категории")
    p.add_argument("--index-dir", type=str, default=str(config.DATA_INDICES))
    p.add_argument("--output-path", type=str, default=str(config.ROOT_DIR / "answer.csv"))
    p.add_argument("--limit", type=int, default=None, help="Только первые N запросов (для замера времени)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)

    items = load_benchmark_items()
    queries = load_benchmark_queries()
    if args.limit:
        queries = queries.head(args.limit)

    known_categories = set(items["item_category_id"].unique())
    cat_filter = None
    if not args.no_category_filter:
        train = load_train()
        cat_filter = CategoryFilter(top_n_categories=args.top_n_categories).fit(train, known_categories)

    # Грузим только BM25 — dense-индекс и модель не нужны
    bm25 = BM25Retriever().load(f"{args.index_dir}/bm25.pkl")

    t0 = time.time()
    results = []
    for _, row in tqdm(queries.iterrows(), total=len(queries), desc="BM25"):
        allowed = None
        if cat_filter is not None:
            allowed = cat_filter.get_allowed_categories(row["search_query"], row.get("search_category"))
        hits = bm25.search(row["search_query"], top_k=config.FINAL_TOP_K, allowed_categories=allowed)
        results.append({"query_id": row["query_id"], "answer": " ".join(i for i, _ in hits)})
    logger.info(f"Обработка заняла {time.time() - t0:.1f} с")

    answer_df = pd.DataFrame(results)
    if not args.limit:
        validate_answer(answer_df, queries, items)
    answer_df.to_csv(args.output_path, index=False, encoding="utf-8")
    logger.info(f"Сохранено: {args.output_path}")


if __name__ == "__main__":
    main()
