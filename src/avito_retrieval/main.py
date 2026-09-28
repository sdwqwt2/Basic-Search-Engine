"""
Финальный пайплайн: загружает построенные индексы (или строит заново, если их нет),
прогоняет кандидатогенерацию по всем запросам benchmark_queries.parquet
и сохраняет answer_hybrid.csv в корне проекта.

Запуск:
    python -m avito_retrieval.main --alpha 0.5
    python -m avito_retrieval.main --alpha 0.5 --rebuild-index   # если индексов ещё нет
    python -m avito_retrieval.main --limit 100                   # быстрый смоук-тест

Порядок и содержимое эмбеддингов в индексе зависят от --model-name и --max-seq-length,
поэтому при загрузке готовых индексов они должны совпадать с теми, что были при построении.
"""
import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from avito_retrieval.data.loader import load_benchmark_items, load_benchmark_queries, load_train
from avito_retrieval.retrieval.category_filter import CategoryFilter
from avito_retrieval.retrieval.hybrid import HybridRetriever
from avito_retrieval.retrieval.location_filter import LocationFilter
from avito_retrieval.utils import set_seed
from avito_retrieval import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Кандидатогенерация: финальный прогон на бенчмарке")
    p.add_argument("--alpha", type=float, default=0.5, help="Вес BM25 в RRF (1.0=только BM25, 0.0=только dense)")
    p.add_argument("--top-n-categories", type=int, default=2)
    p.add_argument("--location-coverage", type=float, default=0.95,
                   help="Доля выборов в train, которую должен покрывать маппинг локаций")
    p.add_argument("--no-location-filter", action="store_true", help="Отключить фильтр по локации")
    p.add_argument("--min-mask-items", type=int, default=200,
                   help="Если фильтр оставляет меньше объявлений, он снимается")
    p.add_argument("--no-query-params", action="store_true",
                   help="Не добавлять search_infm_params_text в текст dense-запроса")
    p.add_argument("--top-k-each", type=int, default=config.TOP_K_BM25)
    p.add_argument("--final-top-k", type=int, default=config.FINAL_TOP_K)
    p.add_argument("--index-dir", type=str, default=str(config.DATA_INDICES))
    p.add_argument("--output-path", type=str, default=str(config.ROOT_DIR / "answer_hybrid.csv"))
    p.add_argument("--rebuild-index", action="store_true",
                   help="Построить индексы заново вместо загрузки с диска")
    p.add_argument("--model-name", type=str, default=config.BIENCODER_MODEL_NAME)
    p.add_argument("--max-seq-length", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--encode-devices", type=str, default=None,
                   help="Устройства для энкодинга корпуса при --rebuild-index, например cuda:0,cuda:1")
    p.add_argument("--limit", type=int, default=None, help="Только первые N запросов (смоук-тест)")
    return p.parse_args()


def build_or_load_retriever(args, items: pd.DataFrame, train: pd.DataFrame) -> HybridRetriever:
    known_categories = set(items["item_category_id"].unique())
    cat_filter = CategoryFilter(top_n_categories=args.top_n_categories).fit(train, known_categories)

    # Маппинг search_location_id -> item_location_id обучается на train (id локаций иерархичны)
    loc_filter = None
    if not args.no_location_filter:
        loc_filter = LocationFilter(coverage=args.location_coverage).fit(train)

    retriever = HybridRetriever(
        cat_filter,
        loc_filter,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        min_mask_items=args.min_mask_items,
        use_query_params=not args.no_query_params,
        encode_devices=args.encode_devices.split(",") if args.encode_devices else None,
    )

    index_dir = Path(args.index_dir)
    index_files_exist = all((index_dir / f).exists() for f in ("bm25.pkl", "dense_emb.npy", "dense_meta.parquet"))

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
    """Батчевая кандидатогенерация по всем запросам (маски по категории и локации строятся внутри)."""
    predictions = retriever.retrieve_batch(
        queries,
        alpha=args.alpha,
        top_k_each=args.top_k_each,
        final_top_k=args.final_top_k,
    )

    empty = sum(1 for c in predictions if not c)
    if empty:
        logger.warning(f"Запросов с пустым списком кандидатов: {empty}")

    return pd.DataFrame({
        "query_id": queries["query_id"].tolist(),
        "answer": [" ".join(c) for c in predictions],
    })


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
    if args.limit:
        queries = queries.head(args.limit)
    logger.info(f"Объявлений: {len(items)}, запросов: {len(queries)}")

    retriever = build_or_load_retriever(args, items, train)

    logger.info(f"Запускаю кандидатогенерацию (alpha={args.alpha})...")
    t0 = time.time()
    answer_df = run_retrieval(retriever, queries, args)
    logger.info(f"Обработка запросов заняла {time.time() - t0:.1f} секунд")

    # Полная валидация формата имеет смысл только на всех запросах
    if not args.limit:
        validate_answer(answer_df, queries, items)

    answer_df.to_csv(args.output_path, index=False, encoding="utf-8")
    logger.info(f"Ответ сохранён: {args.output_path}")


if __name__ == "__main__":
    main()
