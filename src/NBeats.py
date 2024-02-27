from typing import Union, List, Tuple
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim

from src.base.block import Block
from src.base.stack import Stack
from src.data.time_series_dataset import TimeSeriesDataset
from src.base.biase import GenericBiase


class NBeats(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        num_stacks: int,
        num_blocks: int,
        activation: Union[str, List],
        mlpes: List,
        batch_size: int = 10,
        lr: float = 0.001,
    ) -> None:
        super(NBeats, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.activation = activation
        self.mlpes = mlpes
        # Traning params
        self.batch_size = None
        self.lr = lr
        self.epochs = 10
        self.num_workers = 0
        self.shuffle = False
        # Stacks
        self.stacks: torch.nn.ModuleList = torch.nn.ModuleList(self.create_stacks())
        # Train
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def parameters(self):
        params = []
        for stack in self.stacks:
            params += list(stack.parameters())
        return params

    def create_stacks(self) -> List:
        stacks = []
        for i in range(self.num_stacks):
            stacks.append(
                Stack(
                    id=f"{i}",
                    backcast_size=self.backcast_size,
                    forecast_size=self.forecast_size,
                    num_blocks=self.num_blocks,
                    mlpes=self.mlpes,
                    activation=self.activation,
                    lr=self.lr,
                )
            )
        return stacks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecast, backcast = self.stacks[0].forward(x)
        beats_forecast = forecast
        for stack in self.stacks[1:]:
            forecast, backcast = stack.forward(backcast)
            beats_forecast += forecast
        return beats_forecast

    def train(self, trainset: TimeSeriesDataset):
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, outputs = data
                self.optimizer.zero_grad()
                predictions = self(inputs)
                loss = self.criterion(outputs, predictions)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            logger.info(f"Epoch {epoch + 1}, loss: {running_loss / len(trainloader)}")
