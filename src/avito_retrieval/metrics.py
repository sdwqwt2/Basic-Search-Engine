def recall_at_k(predictions: dict[str, list[str]], ground_truth: dict[str, set[str]], k: int = 50) -> float:
    """
    predictions: query_id -> список item_id (уже обрезанный до top-k)
    ground_truth: query_id -> множество релевантных item_id
    """
    recalls = []
    for query_id, relevant in ground_truth.items():
        if not relevant:
            continue
        predicted_top_k = set(predictions.get(query_id, [])[:k])
        recalls.append(len(predicted_top_k & relevant) / len(relevant))
    return sum(recalls) / len(recalls) if recalls else 0.0
