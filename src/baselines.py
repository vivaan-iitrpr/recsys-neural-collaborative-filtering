"""
baselines.py — Classical recommendation baselines for comparison.

Models:
1. Popularity Baseline — recommends globally popular items
2. SVD (Singular Value Decomposition) — via Surprise library
3. ALS (Alternating Least Squares) — via scipy sparse matrices
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from evaluate import (
    precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, compute_rmse,
)


class PopularityBaseline:
    """Recommend the most popular (most rated) items to everyone."""

    def __init__(self):
        self.popular_items = []

    def fit(self, train_df: pd.DataFrame):
        """Compute item popularity from training data."""
        item_counts = train_df.groupby("movieId").size().reset_index(name="count")
        item_counts = item_counts.sort_values("count", ascending=False)
        self.popular_items = item_counts["movieId"].tolist()
        print(f"PopularityBaseline: fitted on {len(self.popular_items)} items")

    def recommend(self, user_id: int, k: int = 10, exclude: set = None) -> List[int]:
        """Return top-K popular items, excluding items user already interacted with."""
        exclude = exclude or set()
        recs = [item for item in self.popular_items if item not in exclude]
        return recs[:k]


class SVDBaseline:
    """SVD-based collaborative filtering using scipy sparse SVD."""

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.sigma = None
        self.global_mean = 0

    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int):
        """Fit SVD on the user-item interaction matrix."""
        self.global_mean = train_df["rating"].mean()

        # Build sparse user-item matrix
        row = train_df["userId"].values
        col = train_df["movieId"].values
        data = train_df["rating"].values - self.global_mean

        matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

        # Compute truncated SVD
        n_factors = min(self.n_factors, min(num_users, num_items) - 1)
        U, sigma, Vt = svds(matrix, k=n_factors)

        self.user_factors = U
        self.sigma = np.diag(sigma)
        self.item_factors = Vt.T

        print(f"SVDBaseline: fitted with {n_factors} factors on {num_users}x{num_items} matrix")

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a single user-item pair."""
        pred = (
            self.user_factors[user_id] @ self.sigma @ self.item_factors[item_id]
            + self.global_mean
        )
        return np.clip(pred, 0.5, 5.0)

    def predict_user(self, user_id: int) -> np.ndarray:
        """Predict ratings for all items for a given user."""
        preds = self.user_factors[user_id] @ self.sigma @ self.item_factors.T + self.global_mean
        return np.clip(preds, 0.5, 5.0)

    def recommend(self, user_id: int, k: int = 10, exclude: set = None) -> List[int]:
        """Get top-K recommendations for a user."""
        exclude = exclude or set()
        scores = self.predict_user(user_id)
        for item in exclude:
            scores[item] = -np.inf
        top_k = np.argsort(scores)[::-1][:k]
        return top_k.tolist()


class ALSBaseline:
    """
    Alternating Least Squares for implicit feedback.
    Minimizes: sum ||R_ui - X_u * Y_i^T||^2 + lambda * (||X||^2 + ||Y||^2)
    """

    def __init__(self, n_factors: int = 50, reg: float = 0.01, n_iterations: int = 15):
        self.n_factors = n_factors
        self.reg = reg
        self.n_iterations = n_iterations
        self.user_factors = None
        self.item_factors = None

    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int):
        """Fit ALS on implicit interaction matrix."""
        # Build confidence matrix: C_ui = 1 + alpha * r_ui
        alpha = 40
        row = train_df["userId"].values
        col = train_df["movieId"].values
        data = np.ones(len(train_df))

        R = csr_matrix((data, (row, col)), shape=(num_users, num_items))
        C = R.multiply(alpha) + csr_matrix(np.ones((num_users, num_items)))

        # Initialize factors randomly
        np.random.seed(42)
        X = np.random.normal(0, 0.01, (num_users, self.n_factors))
        Y = np.random.normal(0, 0.01, (num_items, self.n_factors))
        I_reg = self.reg * np.eye(self.n_factors)

        print(f"ALSBaseline: fitting with {self.n_factors} factors, {self.n_iterations} iterations")

        for iteration in range(self.n_iterations):
            # Fix Y, solve for X
            YtY = Y.T @ Y
            for u in range(num_users):
                Cu = np.array(C[u].todense()).flatten()
                Cu_diag = np.diag(Cu)
                Pu = (R[u].toarray().flatten() > 0).astype(float)
                X[u] = np.linalg.solve(YtY + Y.T @ Cu_diag @ Y + I_reg, Y.T @ Cu_diag @ Pu)

            # Fix X, solve for Y
            XtX = X.T @ X
            for i in range(num_items):
                Ci = np.array(C[:, i].todense()).flatten()
                Ci_diag = np.diag(Ci)
                Pi = (R[:, i].toarray().flatten() > 0).astype(float)
                Y[i] = np.linalg.solve(XtX + X.T @ Ci_diag @ X + I_reg, X.T @ Ci_diag @ Pi)

            if (iteration + 1) % 5 == 0:
                print(f"  ALS iteration {iteration + 1}/{self.n_iterations} complete")

        self.user_factors = X
        self.item_factors = Y
        print(f"ALSBaseline: fitting complete")

    def predict_user(self, user_id: int) -> np.ndarray:
        """Predict scores for all items for a user."""
        return self.user_factors[user_id] @ self.item_factors.T

    def recommend(self, user_id: int, k: int = 10, exclude: set = None) -> List[int]:
        """Get top-K recommendations for a user."""
        exclude = exclude or set()
        scores = self.predict_user(user_id)
        for item in exclude:
            scores[item] = -np.inf
        top_k = np.argsort(scores)[::-1][:k]
        return top_k.tolist()


def evaluate_baseline(
    model,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k_values: List[int] = [5, 10, 20],
    threshold: float = 3.5,
    max_users: int = 5000,
) -> Dict[str, float]:
    """
    Evaluate a baseline model using ranking metrics.

    Args:
        model: Baseline model with .recommend() method
        test_df: Test set DataFrame
        train_df: Train set DataFrame (for excluding seen items)
        k_values: List of K values
        threshold: Rating threshold for relevance
        max_users: Max users to evaluate (for speed)
    """
    # Build user -> seen items from training
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    # Build user -> relevant items from test
    test_relevant = (
        test_df[test_df["rating"] >= threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    users_to_eval = [u for u in test_relevant if len(test_relevant[u]) > 0]
    if len(users_to_eval) > max_users:
        np.random.seed(42)
        users_to_eval = np.random.choice(users_to_eval, max_users, replace=False).tolist()

    metrics = {}
    max_k = max(k_values)

    all_precisions = {k: [] for k in k_values}
    all_recalls = {k: [] for k in k_values}
    all_ndcgs = {k: [] for k in k_values}
    all_hit_rates = {k: [] for k in k_values}

    for user_id in users_to_eval:
        relevant = test_relevant.get(user_id, set())
        exclude = user_seen.get(user_id, set())

        recommended = model.recommend(user_id, k=max_k, exclude=exclude)

        for k in k_values:
            all_precisions[k].append(precision_at_k(recommended, relevant, k))
            all_recalls[k].append(recall_at_k(recommended, relevant, k))
            all_ndcgs[k].append(ndcg_at_k(recommended, relevant, k))
            all_hit_rates[k].append(hit_rate_at_k(recommended, relevant, k))

    for k in k_values:
        metrics[f"precision@{k}"] = np.mean(all_precisions[k])
        metrics[f"recall@{k}"] = np.mean(all_recalls[k])
        metrics[f"ndcg@{k}"] = np.mean(all_ndcgs[k])
        metrics[f"hit_rate@{k}"] = np.mean(all_hit_rates[k])

    return metrics


if __name__ == "__main__":
    # Quick test
    print("Baseline models module loaded successfully.")
    print("Available models: PopularityBaseline, SVDBaseline, ALSBaseline")
