"""
model.py — Neural Collaborative Filtering (NCF) model.

Implements three variants:
1. GMF (Generalized Matrix Factorization) — linear user-item interactions
2. MLP — non-linear interactions via deep layers
3. NeuMF — fuses GMF + MLP for best performance

Reference: He et al., 2017 — "Neural Collaborative Filtering" (https://arxiv.org/abs/1708.05031)
"""

import torch
import torch.nn as nn


class GMF(nn.Module):
    """
    Generalized Matrix Factorization.
    Element-wise product of user and item embeddings, followed by a linear output layer.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.output_layer = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.kaiming_uniform_(self.output_layer.weight)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        element_product = user_emb * item_emb  # Element-wise product
        logits = self.output_layer(element_product)
        return logits.squeeze(-1)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for collaborative filtering.
    Concatenates user and item embeddings, then passes through MLP layers.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        layers: list = [128, 64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # Build MLP layers
        mlp_layers = []
        input_dim = embed_dim * 2  # Concatenated embeddings
        for hidden_dim in layers:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=-1)
        mlp_out = self.mlp(concat)
        logits = self.output_layer(mlp_out)
        return logits.squeeze(-1)


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization — fuses GMF and MLP pathways.

    Architecture:
        User/Item -> GMF Embeddings -> Element-wise product ─┐
                                                              ├─> Concat -> Output
        User/Item -> MLP Embeddings -> MLP layers ───────────┘

    This is the main model used in the project.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        mlp_layers: list = [128, 64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()

        # GMF path — separate embeddings for GMF
        self.user_embed_gmf = nn.Embedding(num_users, embed_dim)
        self.item_embed_gmf = nn.Embedding(num_items, embed_dim)

        # MLP path — separate embeddings for MLP
        self.user_embed_mlp = nn.Embedding(num_users, embed_dim)
        self.item_embed_mlp = nn.Embedding(num_items, embed_dim)

        # MLP layers
        layers = []
        input_dim = embed_dim * 2
        for hidden_dim in mlp_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # NeuMF output: GMF output (embed_dim) + MLP output (last layer dim)
        self.output_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for embedding in [
            self.user_embed_gmf, self.item_embed_gmf,
            self.user_embed_mlp, self.item_embed_mlp,
        ]:
            nn.init.normal_(embedding.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.kaiming_uniform_(self.output_layer.weight)

    def forward(self, user_ids, item_ids):
        # GMF path
        user_gmf = self.user_embed_gmf(user_ids)
        item_gmf = self.item_embed_gmf(item_ids)
        gmf_out = user_gmf * item_gmf  # Element-wise product

        # MLP path
        user_mlp = self.user_embed_mlp(user_ids)
        item_mlp = self.item_embed_mlp(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # Fusion
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        logits = self.output_layer(concat)

        return logits.squeeze(-1)


def get_model(
    model_name: str,
    num_users: int,
    num_items: int,
    embed_dim: int = 64,
    mlp_layers: list = [128, 64, 32],
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        model_name: One of 'gmf', 'mlp', 'neumf'
    """
    models = {
        "gmf": GMF(num_users, num_items, embed_dim),
        "mlp": MLP(num_users, num_items, embed_dim, mlp_layers, dropout),
        "neumf": NeuMF(num_users, num_items, embed_dim, mlp_layers, dropout),
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    model = models[model_name.lower()]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_name.upper()}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


if __name__ == "__main__":
    # Quick test
    model = get_model("neumf", num_users=1000, num_items=5000, embed_dim=64)
    users = torch.LongTensor([0, 1, 2])
    items = torch.LongTensor([10, 20, 30])
    output = model(users, items)
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.detach().numpy()}")
