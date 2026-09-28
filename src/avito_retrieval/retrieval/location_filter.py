import pandas as pd


class LocationFilter:
    """
    Маппинг search_location_id -> допустимые item_location_id, обученный на train.
    Нужен, потому что id локаций иерархичны и совпадают лишь в ~83% случаев
    (например, 107620 в поиске -> 637640 в объявлениях).
    Берём наименьший набор item-локаций, покрывающий `coverage` выборов пользователей.
    """

    def __init__(self, coverage: float = 0.95, max_locations: int = 50):
        self.coverage = coverage
        self.max_locations = max_locations
        self.mapping: dict = {}

    def fit(self, train_df: pd.DataFrame):
        counts = (
            train_df.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        )
        for loc, g in counts.groupby("search_location_id"):
            g = g.sort_values("n", ascending=False)
            share = g["n"].cumsum() / g["n"].sum()
            k = min(int((share < self.coverage).sum()) + 1, self.max_locations)
            self.mapping[loc] = set(g["item_location_id"].iloc[:k])
        return self

    def get_allowed_locations(self, search_location_id) -> set:
        allowed = set(self.mapping.get(search_location_id, set()))
        allowed.add(search_location_id)  # собственный id всегда допустим
        return allowed
