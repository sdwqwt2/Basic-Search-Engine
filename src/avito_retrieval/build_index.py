"""
Скрипт построения индексов кандидатогенерации.
Запуск: python -m avito_retrieval.build_index
Строит BM25 + Dense индексы один раз на весь корпус benchmark_items.parquet
и сохраняет их в data/indices/ для последующего использования в main.py
и в экспериментах с alpha без пересчёта.
"""
import argparse
import time
import logging

from avito_retrieval.data.loader import load_benchmark_items, load_train
from avito_retrieval.retrieval.category_filter import CategoryFilter
from avito_retrieval.retrieval.hybrid import HybridRetriever
from avito_retrieval.utils import set_seed
from avito_retrieval import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Построение индексов кандидатогенерации")
    parser.add_argument("--model-name", type=str, default=config.BIENCODER_MODEL_NAME)
    parser.add_argument("--max-seq-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-n-categories", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=str(config.DATA_INDICES))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)

    logger.info("Загружаю корпус объявлений и train...")
    items = load_benchmark_items()
    train = load_train()
    logger.info(f"Корпус: {len(items)} объявлений (после дедупа)")

    known_categories = set(items["item_category_id"].unique())
    cat_filter = CategoryFilter(top_n_categories=args.top_n_categories).fit(train, known_categories)

    retriever = HybridRetriever(
        cat_filter,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )

    t0 = time.time()
    retriever.build(items)
    elapsed = time.time() - t0
    logger.info(f"Индексы построены за {elapsed / 60:.1f} минут")

    retriever.save(args.output_dir)
    logger.info(f"Индексы сохранены в {args.output_dir}")


if __name__ == "__main__":
    main()
