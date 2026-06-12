"""
train.py — Training pipeline for NCF models.

Features:
- BCEWithLogitsLoss for implicit feedback
- MLflow experiment tracking
- Optuna hyperparameter optimization
- Early stopping
- Model checkpointing
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from data_loader import prepare_data, get_dataloaders
from model import get_model
from evaluate import evaluate_model, print_metrics


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "min", delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_worse(score):
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def _is_worse(self, score):
        if self.mode == "min":
            return score > self.best_score - self.delta
        return score < self.best_score + self.delta


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for users, items, labels in tqdm(train_loader, desc="Training", leave=False):
        users, items, labels = users.to(device), items.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(users, items)
        loss = criterion(predictions, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for users, items, labels in val_loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            predictions = model(users, items)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train(
    model_name: str = "neumf",
    embed_dim: int = 64,
    mlp_layers: list = [128, 64, 32],
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    optimizer_name: str = "adam",
    epochs: int = 20,
    batch_size: int = 1024,
    patience: int = 5,
    sample_size: int = 10_000_000,
    neg_ratio: int = 4,
    data_dir: str = "data/ml-25m",
    save_dir: str = "checkpoints",
    use_mlflow: bool = True,
    use_wandb: bool = False,
    run_name: str = None,
):
    """
    Full training pipeline.

    Args:
        model_name: 'gmf', 'mlp', or 'neumf'
        embed_dim: Embedding dimension for user/item vectors
        mlp_layers: Hidden layer sizes for MLP
        dropout: Dropout rate
        lr: Learning rate
        weight_decay: L2 regularization strength
        optimizer_name: 'adam' or 'adamw'
        epochs: Max training epochs
        batch_size: Training batch size
        patience: Early stopping patience
        sample_size: Number of ratings to sample (None for full 25M)
        neg_ratio: Negative samples per positive
        data_dir: Path to MovieLens 25M data
        save_dir: Directory to save model checkpoints
        use_mlflow: Whether to log to MLflow
        use_wandb: Whether to log to W&B
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ---- Data Preparation ----
    print("\n" + "=" * 55)
    print("  Step 1: Preparing Data")
    print("=" * 55)
    data = prepare_data(
        data_dir=data_dir,
        sample_size=sample_size,
        neg_ratio=neg_ratio,
    )
    data = get_dataloaders(data, batch_size=batch_size)

    # ---- Model ----
    print("\n" + "=" * 55)
    print("  Step 2: Building Model")
    print("=" * 55)
    model = get_model(
        model_name=model_name,
        num_users=data["num_users"],
        num_items=data["num_items"],
        embed_dim=embed_dim,
        mlp_layers=mlp_layers,
        dropout=dropout,
    ).to(device)

    # ---- Optimizer & Loss ----
    optimizers = {
        "adam": Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        "adamw": AdamW(model.parameters(), lr=lr, weight_decay=weight_decay),
    }
    optimizer = optimizers[optimizer_name.lower()]
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=patience, mode="min")

    # ---- Logging Setup ----
    config = {
        "model": model_name,
        "embed_dim": embed_dim,
        "mlp_layers": str(mlp_layers),
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "batch_size": batch_size,
        "neg_ratio": neg_ratio,
        "sample_size": sample_size,
        "num_users": data["num_users"],
        "num_items": data["num_items"],
    }

    if use_mlflow and HAS_MLFLOW:
        mlflow.set_experiment("NCF-MovieLens-25M")
        mlflow.start_run(run_name=run_name or f"{model_name}-dim{embed_dim}-lr{lr}")
        mlflow.log_params(config)

    if use_wandb and HAS_WANDB:
        wandb.init(project="recsys-ncf", name=run_name, config=config)

    # ---- Training Loop ----
    print("\n" + "=" * 55)
    print("  Step 3: Training")
    print("=" * 55)

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss = train_epoch(model, data["train_loader"], optimizer, criterion, device)
        val_loss = validate(model, data["val_loader"], criterion, device)

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Logging
        log_data = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "epoch": epoch,
        }

        if use_mlflow and HAS_MLFLOW:
            mlflow.log_metrics(log_data, step=epoch)

        if use_wandb and HAS_WANDB:
            wandb.log(log_data)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, f"best_{model_name}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, save_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if early_stopping.should_stop:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

    # ---- Evaluation ----
    print("\n" + "=" * 55)
    print("  Step 4: Evaluating on Test Set")
    print("=" * 55)

    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, f"best_{model_name}.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_model(model, data["test_loader"], device, k_values=[5, 10, 20])
    print_metrics(metrics, model_name.upper())

    # Log final metrics
    if use_mlflow and HAS_MLFLOW:
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.end_run()

    if use_wandb and HAS_WANDB:
        wandb.log({f"test/{k}": v for k, v in metrics.items()})
        wandb.finish()

    return model, metrics


def hyperparameter_search(n_trials: int = 20, data_dir: str = "data/ml-25m"):
    """
    Optuna-based hyperparameter optimization.
    Searches over embed_dim, lr, dropout, mlp_layers, weight_decay.
    """
    try:
        import optuna
    except ImportError:
        print("Install optuna: pip install optuna")
        return

    def objective(trial):
        embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        n_layers = trial.suggest_int("n_layers", 2, 4)

        layer_sizes = []
        dim = embed_dim * 2
        for _ in range(n_layers):
            dim = dim // 2
            layer_sizes.append(max(dim, 16))

        _, metrics = train(
            model_name="neumf",
            embed_dim=embed_dim,
            mlp_layers=layer_sizes,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=10,
            sample_size=2_000_000,  # Smaller sample for faster search
            data_dir=data_dir,
            use_mlflow=True,
            run_name=f"optuna-trial-{trial.number}",
        )

        return metrics["rmse"]

    study = optuna.create_study(direction="minimize", study_name="NCF-HPO")
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 55)
    print("  Best Hyperparameters")
    print("=" * 55)
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best RMSE: {study.best_value:.4f}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NCF model on MovieLens 25M")
    parser.add_argument("--model", type=str, default="neumf", choices=["gmf", "mlp", "neumf"])
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=10_000_000)
    parser.add_argument("--neg_ratio", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data/ml-25m")
    parser.add_argument("--hpo", action="store_true", help="Run hyperparameter search")
    parser.add_argument("--no_mlflow", action="store_true")
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    if args.hpo:
        hyperparameter_search(data_dir=args.data_dir)
    else:
        train(
            model_name=args.model,
            embed_dim=args.embed_dim,
            lr=args.lr,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            sample_size=args.sample_size,
            neg_ratio=args.neg_ratio,
            data_dir=args.data_dir,
            use_mlflow=not args.no_mlflow,
            use_wandb=args.wandb,
        )
