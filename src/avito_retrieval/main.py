"""
Финальный пайплайн: загружает построенные индексы (или строит заново, если их нет),
прогоняет кандидатогенерацию по всем запросам benchmark_queries.parquet
и сохраняет answer.csv в корне проекта.

Запуск:
    python -m avito_retrieval.main --alpha 0.5
    python -m avito_retrieval.main --alpha 0.5 --rebuild-index   # если индексов ещё нет
"""
import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from avito_retrieval.data.loader import load_benchmark_items, load_benchmark_queries, load_train
from avito_retrieval.retrieval.category_filter import CategoryFilter
from avito_retrieval.retrieval.hybrid import HybridRetriever
from avito_retrieval.utils import set_seed
from avito_retrieval import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Кандидатогенерация: финальный прогон на бенчмарке")
    parser.add_argument("--alpha", type=float, default=0.5, help="Вес BM25 в RRF (1.0=только BM25, 0.0=только dense)")
    parser.add_argument("--top-n-categories", type=int, default=2)
    parser.add_argument("--top-k-each", type=int, default=config.TOP_K_BM25)
    parser.add_argument("--final-top-k", type=int, default=config.FINAL_TOP_K)
    parser.add_argument("--index-dir", type=str, default=str(config.DATA_INDICES))
    parser.add_argument("--output-path", type=str, default=str(config.ROOT_DIR / "answer.csv"))
    parser.add_argument(
        "--rebuild-index", action="store_true",
        help="Построить индексы заново вместо загрузки с диска (медленно, ~несколько часов)",
    )
    parser.add_argument("--model-name", type=str, default=config.BIENCODER_MODEL_NAME)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def build_or_load_retriever(args, items: pd.DataFrame, train: pd.DataFrame) -> HybridRetriever:
    known_categories = set(items["item_category_id"].unique())
    cat_filter = CategoryFilter(top_n_categories=args.top_n_categories).fit(train, known_categories)

    retriever = HybridRetriever(
        cat_filter,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )

    index_dir = Path(args.index_dir)
    index_files_exist = (index_dir / "bm25.pkl").exists() and (index_dir / "dense.index").exists()

    if args.rebuild_index or not index_files_exist:
        logger.info("Строю индексы заново...")
        t0 = time.time()
        retriever.build(items)
        logger.info(f"Индексы построены за {(time.time() - t0) / 60:.1f} минут")
        index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(index_dir))
        logger.info(f"Индексы сохранены в {index_dir}")
    else:
        logger.info(f"Загружаю готовые индексы из {index_dir}...")
        retriever.load(str(index_dir))

    return retriever


def run_retrieval(retriever: HybridRetriever, queries: pd.DataFrame, args) -> pd.DataFrame:
    """Прогоняет кандидатогенерацию по всем запросам и валидирует результат перед сохранением."""
    results = []
    for _, row in tqdm(queries.iterrows(), total=len(queries), desc="Обрабатываю запросы"):
        query_id = row["query_id"]
        search_query = row["search_query"]
        search_category = row.get("search_category")

        candidates = retriever.retrieve_for_query(
            search_query=search_query,
            search_category=search_category,
            alpha=args.alpha,
            top_k_each=args.top_k_each,
            final_top_k=args.final_top_k,
        )

        # Защита от пустого результата — не должно происходить, но на всякий случай логируем
        if not candidates:
            logger.warning(f"query_id={query_id}: пустой список кандидатов!")

        results.append({"query_id": query_id, "answer": " ".join(candidates)})

    return pd.DataFrame(results)


def validate_answer(answer_df: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame):
    """Проверяет формат ответа согласно требованиям задания перед сохранением."""
    assert set(answer_df["query_id"]) == set(queries["query_id"]), \
        "Не все query_id из benchmark_queries присутствуют в ответе (или есть лишние)"
    assert answer_df["query_id"].is_unique, "Есть повторяющиеся query_id"

    valid_item_ids = set(items["item_id"])
    max_candidates = 0
    for _, row in answer_df.iterrows():
        item_ids = row["answer"].split() if row["answer"] else []
        assert len(item_ids) <= 50, f"query_id={row['query_id']}: больше 50 item_id ({len(item_ids)})"
        assert len(item_ids) == len(set(item_ids)), f"query_id={row['query_id']}: есть повторы item_id внутри строки"
        assert all(iid in valid_item_ids for iid in item_ids), \
            f"query_id={row['query_id']}: есть item_id не из benchmark_items"
        max_candidates = max(max_candidates, len(item_ids))

    logger.info(f"Валидация пройдена. Максимум кандидатов в одной строке: {max_candidates}")


def main():
    args = parse_args()
    set_seed(config.SEED)

    logger.info("Загружаю данные...")
    items = load_benchmark_items()
    queries = load_benchmark_queries()
    train = load_train()
    logger.info(f"Объявлений: {len(items)}, запросов: {len(queries)}")

    retriever = build_or_load_retriever(args, items, train)

    logger.info(f"Запускаю кандидатогенерацию (alpha={args.alpha})...")
    t0 = time.time()
    answer_df = run_retrieval(retriever, queries, args)
    logger.info(f"Обработка запросов заняла {time.time() - t0:.1f} секунд")

    validate_answer(answer_df, queries, items)

    answer_df.to_csv(args.output_path, index=False, encoding="utf-8")
    logger.info(f"Ответ сохранён: {args.output_path}")


if __name__ == "__main__":
    main()
