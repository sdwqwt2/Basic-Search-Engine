import pandas as pd
import numpy as np


class CategoryFilter:
    """
    Фильтрует корпус объявлений по категории запроса перед текстовым поиском.

    EDA показал: у >99.99% запросов одна категория объявлений покрывает
    >=80% реальных выборов пользователей. Поэтому используем search_category
    (или, если его нет / он не совпадает по значениям с item_category_id,
    строим маппинг запрос -> топ категорий по train) как первичный фильтр.
    """

    def __init__(self, top_n_categories: int = 1):
        self.top_n_categories = top_n_categories
        self.query_to_categories: dict[str, set] = {}

    def fit(self, train_df: pd.DataFrame):
        """
        Строит маппинг нормализованный search_query -> набор наиболее частых
        item_category_id, на случай если search_category недостаточно надёжен
        или отсутствует в бенчмарке в сопоставимом виде с item_category_id.
        """
        grouped = (
            train_df.groupby("search_query")["item_category_id"]
            .apply(lambda s: set(s.value_counts().head(self.top_n_categories).index))
        )
        self.query_to_categories = grouped.to_dict()
        return self

    def get_allowed_categories(self, search_query: str, search_category=None) -> set | None:
        """
        Возвращает множество допустимых item_category_id для запроса.
        Приоритет: 1) прямое поле search_category, если оно валидно,
                   2) обученный маппинг по тексту запроса,
                   3) None (фильтр не применяется — fallback на полный корпус).
        """
        if search_category is not None and not pd.isna(search_category):
            return {search_category}
        if search_query in self.query_to_categories:
            return self.query_to_categories[search_query]
        return None

    def filter_items(self, items_df: pd.DataFrame, allowed_categories: set | None) -> pd.DataFrame:
        """Возвращает подмножество items_df, если фильтр применим, иначе весь items_df."""
        if not allowed_categories:
            return items_df
        mask = items_df["item_category_id"].isin(allowed_categories)
        filtered = items_df[mask]
        # Safety net: если фильтр слишком агрессивный и почти ничего не оставил
        # (например, категория в бенчмарке отличается от train), откатываемся к полному корпусу
        if len(filtered) < 20:
            return items_df
        return filtered
