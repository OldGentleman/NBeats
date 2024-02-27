from typing import List, Union, Tuple, Callable
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanAbsolutePercentageError

from src.util.config import NBeatsConfig
from src.base.biase import Biase
from src.data.time_series_dataset import TimeSeriesDataset


class Block(nn.Module):
    def __init__(
        self,
        id: str,
        backcast_size: int,
        forecast_size: int,
        mlpes: List,
        activation: Union[str, List],
        biase: Biase,
        lr: float = 0.001,
    ) -> None:
        super(Block, self).__init__()
        self.id = id
        self.config = NBeatsConfig()
        self.biase = biase

        if isinstance(activation, str):
            activ = self.__activation(activation)
            activ = [activ] * (len(mlpes) + 1)
        elif isinstance(activation, list):
            # assert len(mlp_sizes) == len(activation), f"Layers and activations haven't same length"
            activ = [self.__activation(act) for act in activation]

        self.layers = nn.Sequential()
        # Input layer
        self.layers.append(
            nn.Linear(
                in_features=backcast_size, out_features=mlpes[0], dtype=torch.float64
            )
        )
        self.layers.append(activ[0])
        for i in range(1, len(mlpes)):
            self.layers.append(
                nn.Linear(
                    in_features=mlpes[i - 1], out_features=mlpes[i], dtype=torch.float64
                )
            )
            self.layers.append(activ[i])
        # Output layer
        self.layers.append(
            nn.Linear(
                in_features=mlpes[-1],
                out_features=backcast_size + forecast_size,
                dtype=torch.float64,
            )
        )

        # optimazer choise
        self.optimazer = optim.Adam(self.parameters(), lr=lr)
        # criterion choise
        self.loss = MeanAbsolutePercentageError()

    def __activation(self, activation: str) -> Callable:
        assert (
            activation in self.config.ACTIVATIONS
        ), f"{activation} is not in {self.config.ACTIVATIONS}"
        return getattr(nn, activation)()

    def parameters(self):
        params = []
        for layer in self.layers:
            params += list(layer.parameters())
        return params

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = self.layers(x)
        forecast, backcast = self.biase.forward(theta)
        return forecast, backcast

    # Rewrite
    def train(
        self,
        train: TimeSeriesDataset,
        num_epochs: int,
        batch_size: int = 1,
        shuffle: bool = False,
    ) -> None:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
        for epoch in range(num_epochs):
            for x, y in train_loader:
                self.optimazer.zero_grad()
                forecast, backcast = self.forward(x)
                cast = torch.cat([forecast, backcast])
                # logger.info(f"X: {x.size()}, Y: {y.size()}")
                seq = torch.cat([x, y], dim=1)
                loss = self.loss(cast, seq)
                loss.backward()
                self.optimazer.step()
            logger.info(f"Block: {self.id}, Epoch {epoch}, Loss: {loss.item():.4f}")
