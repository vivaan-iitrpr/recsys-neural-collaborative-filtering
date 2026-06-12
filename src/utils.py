"""
utils.py — Helper functions for the recommendation system project.

Includes:
- Seed setting for reproducibility
- Model parameter counting
- Plotting training curves
- Results comparison table
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: str = "figures/training_curves.png",
):
    """Plot training and validation loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, val_losses, "r-o", label="Val Loss", markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (BCEWithLogits)", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_metrics_comparison(
    results: dict,
    k: int = 10,
    save_path: str = "figures/model_comparison.png",
):
    """
    Bar chart comparing models across metrics.

    Args:
        results: {model_name: {metric_name: value, ...}, ...}
        k: K value for @K metrics
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metrics_to_plot = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]
    model_names = list(results.keys())
    n_metrics = len(metrics_to_plot)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    colors = sns.color_palette("viridis", len(model_names))

    for idx, metric in enumerate(metrics_to_plot):
        values = [results[m].get(metric, 0) for m in model_names]
        bars = axes[idx].bar(model_names, values, color=colors)
        axes[idx].set_title(metric.upper(), fontsize=13)
        axes[idx].set_ylim(0, 1)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10
            )

    plt.suptitle(f"Model Comparison (K={k})", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison chart to {save_path}")


def print_results_table(results: dict):
    """Print a formatted comparison table of all models."""
    print(f"\n{'Model':<25} {'RMSE':>8} {'P@10':>8} {'R@10':>8} {'NDCG@10':>8} {'HR@10':>8}")
    print("-" * 75)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<25} "
            f"{metrics.get('rmse', 0):>8.4f} "
            f"{metrics.get('precision@10', 0):>8.4f} "
            f"{metrics.get('recall@10', 0):>8.4f} "
            f"{metrics.get('ndcg@10', 0):>8.4f} "
            f"{metrics.get('hit_rate@10', 0):>8.4f}"
        )
    print()
