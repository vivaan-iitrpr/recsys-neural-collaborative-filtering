"""
data_loader.py — Load and preprocess MovieLens 25M dataset.

Handles:
- Loading raw CSV data
- Train/Validation/Test splitting (80/10/10)
- Negative sampling for implicit feedback
- PyTorch Dataset/DataLoader creation
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class MovieLensDataset(Dataset):
    """PyTorch Dataset for user-item interactions with negative sampling."""

    def __init__(self, user_ids, item_ids, labels):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


def load_movielens(data_dir: str = "data/ml-25m", sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load MovieLens 25M ratings data.

    Args:
        data_dir: Path to extracted ml-25m directory
        sample_size: If set, randomly sample this many ratings (for faster iteration)

    Returns:
        DataFrame with columns: [userId, movieId, rating, timestamp]
    """
    ratings_path = os.path.join(data_dir, "ratings.csv")

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"ratings.csv not found at {ratings_path}. "
            f"Download MovieLens 25M from https://grouplens.org/datasets/movielens/25m/"
        )

    print(f"Loading ratings from {ratings_path}...")
    df = pd.read_csv(ratings_path)
    print(f"Loaded {len(df):,} ratings from {df['userId'].nunique():,} users on {df['movieId'].nunique():,} movies")

    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size:,} ratings...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return df


def load_movies(data_dir: str = "data/ml-25m") -> pd.DataFrame:
    """Load movie metadata (movieId, title, genres)."""
    movies_path = os.path.join(data_dir, "movies.csv")
    return pd.read_csv(movies_path)


def encode_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Re-encode userId and movieId to contiguous integers starting from 0.

    Returns:
        df: DataFrame with encoded IDs
        user_map: {original_userId: encoded_id}
        item_map: {original_movieId: encoded_id}
    """
    user_map = {uid: idx for idx, uid in enumerate(df["userId"].unique())}
    item_map = {iid: idx for idx, iid in enumerate(df["movieId"].unique())}

    df = df.copy()
    df["userId"] = df["userId"].map(user_map)
    df["movieId"] = df["movieId"].map(item_map)

    print(f"Encoded {len(user_map):,} users and {len(item_map):,} items to contiguous IDs")
    return df, user_map, item_map


def binarize_ratings(df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """
    Convert explicit ratings to implicit feedback.
    Ratings >= threshold -> 1 (positive), else 0 (negative).
    """
    df = df.copy()
    df["label"] = (df["rating"] >= threshold).astype(int)
    return df


def create_negative_samples(
    df: pd.DataFrame,
    num_items: int,
    neg_ratio: int = 4,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate negative samples (items user hasn't interacted with).

    Args:
        df: DataFrame with positive interactions
        num_items: Total number of unique items
        neg_ratio: Number of negatives per positive sample

    Returns:
        DataFrame with both positive and negative samples
    """
    np.random.seed(seed)

    # Build set of positive interactions per user
    user_positive_items = df.groupby("userId")["movieId"].apply(set).to_dict()

    negative_users, negative_items, negative_labels = [], [], []

    for user_id, pos_items in user_positive_items.items():
        num_neg = len(pos_items) * neg_ratio
        neg_candidates = list(set(range(num_items)) - pos_items)

        if len(neg_candidates) < num_neg:
            neg_samples = np.random.choice(neg_candidates, size=num_neg, replace=True)
        else:
            neg_samples = np.random.choice(neg_candidates, size=num_neg, replace=False)

        negative_users.extend([user_id] * num_neg)
        negative_items.extend(neg_samples.tolist())
        negative_labels.extend([0] * num_neg)

    neg_df = pd.DataFrame({
        "userId": negative_users,
        "movieId": negative_items,
        "label": negative_labels,
    })

    # Positive samples
    pos_df = df[["userId", "movieId", "label"]].copy()

    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Created {len(neg_df):,} negative samples ({neg_ratio}:1 ratio)")
    print(f"Total samples: {len(combined):,} (Pos: {len(pos_df):,}, Neg: {len(neg_df):,})")

    return combined


def prepare_data(
    data_dir: str = "data/ml-25m",
    sample_size: Optional[int] = 10_000_000,
    neg_ratio: int = 4,
    threshold: float = 3.5,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Full data preparation pipeline.

    Returns:
        Dictionary with train/val/test DataLoaders and metadata.
    """
    # Step 1: Load
    df = load_movielens(data_dir, sample_size=sample_size)

    # Step 2: Encode IDs
    df, user_map, item_map = encode_ids(df)
    num_users = len(user_map)
    num_items = len(item_map)

    # Step 3: Binarize ratings for implicit feedback
    df = binarize_ratings(df, threshold=threshold)

    # Step 4: Train/Val/Test split (before negative sampling to prevent leakage)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=val_size / (1 - test_size), random_state=seed)

    print(f"\nSplit sizes — Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Step 5: Negative sampling (only on train)
    train_samples = create_negative_samples(
        train_df[train_df["label"] == 1], num_items, neg_ratio=neg_ratio, seed=seed
    )

    # Val/Test: add negatives with lower ratio for faster eval
    val_samples = create_negative_samples(
        val_df[val_df["label"] == 1], num_items, neg_ratio=1, seed=seed + 1
    )
    test_samples = create_negative_samples(
        test_df[test_df["label"] == 1], num_items, neg_ratio=1, seed=seed + 2
    )

    # Step 6: Create datasets
    train_dataset = MovieLensDataset(
        train_samples["userId"].values,
        train_samples["movieId"].values,
        train_samples["label"].values,
    )
    val_dataset = MovieLensDataset(
        val_samples["userId"].values,
        val_samples["movieId"].values,
        val_samples["label"].values,
    )
    test_dataset = MovieLensDataset(
        test_samples["userId"].values,
        test_samples["movieId"].values,
        test_samples["label"].values,
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "num_users": num_users,
        "num_items": num_items,
        "user_map": user_map,
        "item_map": item_map,
        "test_df": test_df,
    }


def get_dataloaders(data: dict, batch_size: int = 1024, num_workers: int = 4) -> dict:
    """Create DataLoaders from prepared datasets."""
    data["train_loader"] = DataLoader(
        data["train_dataset"], batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    data["val_loader"] = DataLoader(
        data["val_dataset"], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    data["test_loader"] = DataLoader(
        data["test_dataset"], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return data


if __name__ == "__main__":
    data = prepare_data(sample_size=1_000_000)
    data = get_dataloaders(data)
    print(f"\nDataLoaders ready:")
    print(f"  Train batches: {len(data['train_loader'])}")
    print(f"  Val batches:   {len(data['val_loader'])}")
    print(f"  Test batches:  {len(data['test_loader'])}")
