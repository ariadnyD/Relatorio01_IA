from typing import Sequence
import torch
import torch.nn as nn


ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
}


class MLP(nn.Module):
    """
    Perceptron multicamadas (MLP) parametrizável em PyTorch.
    A camada de saída é sempre linear (sem ativação final), para:
    - Regressão: usar com nn.MSELoss()
    - Classificação binária: usar com nn.BCEWithLogitsLoss()
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden_units: Sequence[int] = (64, 32),
        activation: str = "ReLU",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if activation not in ACTIVATIONS:
            opcoes = ", ".join(ACTIVATIONS.keys())
            raise ValueError(f"Ativação '{activation}' inválida. Escolha entre: {opcoes}")

        act_cls = ACTIVATIONS[activation]
        layers = []
        prev_dim = in_features

        for h_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # Camada de saída linear (sem ativação)
        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)