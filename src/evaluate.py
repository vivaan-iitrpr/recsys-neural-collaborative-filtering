"""
evaluate.py — Evaluation metrics for recommendation models.

Implements:
- RMSE (for explicit feedback comparison)
- Precision@K, Recall@K, NDCG@K (ranking metrics)
- Hit Rate@K
- MAP@K (Mean Average Precision)
"""

import numpy as np
import torch
from typing import Dict, List
from collections import defaultdict


def compute_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((predictions - actuals) ** 2))


def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Precision@K: fraction of recommended items that are relevant.
    """
    recommended_k = recommended[:k]
    if len(recommended_k) == 0:
        return 0.0
    hits = len(set(recommended_k) & relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Recall@K: fraction of relevant items that are recommended.
    """
    if len(relevant) == 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @K.
    Rewards placing relevant items higher in the ranking.
    """
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: all relevant items ranked at top
    ideal_length = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Hit Rate@K: 1 if any relevant item appears in top-K, else 0."""
    recommended_k = recommended[:k]
    return 1.0 if len(set(recommended_k) & relevant) > 0 else 0.0


def average_precision(recommended: List[int], relevant: set, k: int) -> float:
    """Average Precision for a single user."""
    recommended_k = recommended[:k]
    hits = 0
    sum_precision = 0.0

    for i, item in enumerate(recommended_k):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / min(len(relevant), k) if len(relevant) > 0 else 0.0


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate model on test set using ranking metrics.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test interactions
        device: torch device
        k_values: List of K values for ranking metrics

    Returns:
        Dictionary of metric_name -> value
    """
    model.eval()

    all_users, all_items, all_labels, all_preds = [], [], [], []

    with torch.no_grad():
        for users, items, labels in test_loader:
            users, items = users.to(device), items.to(device)
            predictions = torch.sigmoid(model(users, items))

            all_users.extend(users.cpu().numpy())
            all_items.extend(items.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

    all_users = np.array(all_users)
    all_items = np.array(all_items)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # RMSE on predicted scores vs actual labels
    rmse = compute_rmse(all_preds, all_labels)

    # Group by user for ranking metrics
    user_items = defaultdict(list)
    user_relevant = defaultdict(set)

    for user, item, label, pred in zip(all_users, all_items, all_labels, all_preds):
        user_items[user].append((item, pred))
        if label == 1:
            user_relevant[user].add(item)

    # Compute ranking metrics per user, then average
    metrics = {"rmse": rmse}

    for k in k_values:
        precisions, recalls, ndcgs, hit_rates, aps = [], [], [], [], []

        for user in user_items:
            if len(user_relevant[user]) == 0:
                continue

            # Sort items by predicted score (descending)
            sorted_items = sorted(user_items[user], key=lambda x: x[1], reverse=True)
            recommended = [int(item) for item, _ in sorted_items]
            relevant = user_relevant[user]

            precisions.append(precision_at_k(recommended, relevant, k))
            recalls.append(recall_at_k(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k))
            hit_rates.append(hit_rate_at_k(recommended, relevant, k))
            aps.append(average_precision(recommended, relevant, k))

        metrics[f"precision@{k}"] = np.mean(precisions)
        metrics[f"recall@{k}"] = np.mean(recalls)
        metrics[f"ndcg@{k}"] = np.mean(ndcgs)
        metrics[f"hit_rate@{k}"] = np.mean(hit_rates)
        metrics[f"map@{k}"] = np.mean(aps)

    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*55}")
    print(f"  Evaluation Results: {model_name}")
    print(f"{'='*55}")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*35}")
    for name, value in sorted(metrics.items()):
        print(f"  {name:<20} {value:>10.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    # Unit test with synthetic data
    recommended = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    relevant = {3, 5, 8}

    print("Unit Test — Ranking Metrics:")
    print(f"  Recommended: {recommended}")
    print(f"  Relevant:    {relevant}")
    print(f"  Precision@5: {precision_at_k(recommended, relevant, 5):.4f}")
    print(f"  Recall@5:    {recall_at_k(recommended, relevant, 5):.4f}")
    print(f"  NDCG@5:      {ndcg_at_k(recommended, relevant, 5):.4f}")
    print(f"  Hit Rate@5:  {hit_rate_at_k(recommended, relevant, 5):.4f}")
    print(f"  AP@5:        {average_precision(recommended, relevant, 5):.4f}")
