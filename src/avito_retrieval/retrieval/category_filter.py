import pandas as pd


class CategoryFilter:
    """
    Определяет допустимые item_category_id для запроса.

    Важно: search_category в бенчмарке почти всегда 0 (неизвестно) —
    unique-значения [114, 0], тогда как item_category_id имеет 47 значений.
    Поэтому search_category используется, только если он попадает в множество
    реальных категорий объявлений; основной источник — маппинг текста запроса
    к категориям, обученный на train.parquet.
    """

    def __init__(self, top_n_categories: int = 2, unknown_category_value=0):
        self.top_n_categories = top_n_categories
        self.unknown_category_value = unknown_category_value
        self.query_to_categories: dict[str, set] = {}
        self.known_item_categories: set = set()

    def fit(self, train_df: pd.DataFrame, known_item_categories: set):
        self.known_item_categories = known_item_categories
        grouped = (
            train_df.groupby("search_query")["item_category_id"]
            .apply(lambda s: set(s.value_counts().head(self.top_n_categories).index))
        )
        self.query_to_categories = grouped.to_dict()
        return self

    def get_allowed_categories(self, search_query: str, search_category=None) -> set | None:
        """
        Приоритет:
        1) search_category, если он валиден (не unknown) и реально встречается
           среди item_category_id — прямой, самый надёжный сигнал.
        2) маппинг по тексту запроса из train.
        3) None -> используется полный (fallback) индекс.
        """
        if (
                search_category is not None
                and not pd.isna(search_category)
                and search_category != self.unknown_category_value
                and search_category in self.known_item_categories
        ):
            return {search_category}

        if search_query in self.query_to_categories:
            return self.query_to_categories[search_query]

        return None
