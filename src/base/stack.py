from typing import Union, List, Tuple
from loguru import logger

import torch
import torch.nn as nn

from src.base.block import Block
from src.base.biase import GenericBiase


class Stack(nn.Module):
    def __init__(
        self,
        id: str,
        backcast_size: int,
        forecast_size: int,
        num_blocks: int,
        activation: Union[str, List],
        mlpes: List,
        lr: float = 0.001,
    ) -> None:
        super(Stack, self).__init__()
        self.id = id
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.num_blocks = num_blocks
        self.activation = activation
        self.mlpes = mlpes
        self.lr = lr
        self.blocks: torch.nn.ModuleList = torch.nn.ModuleList(self.create_blocks())

    def parameters(self):
        params = []
        for block in self.blocks:
            params += list(block.parameters())
        return params

    def create_blocks(self) -> List:
        blocks = []
        for i in range(self.num_blocks):
            base_biase = GenericBiase(self.backcast_size, self.forecast_size)
            blocks.append(
                Block(
                    id=f"{self.id}_{i}",
                    backcast_size=self.backcast_size,
                    forecast_size=self.forecast_size,
                    mlpes=self.mlpes,
                    activation=self.activation,
                    biase=base_biase,
                    lr=self.lr,
                )
            )
        return blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecast, backcast = self.blocks[0].forward(x)
        residual_backcast = x - backcast
        stack_forecast: torch.Tensor = forecast
        for block in self.blocks[1:]:
            forecast, backcast = block.forward(residual_backcast)
            residual_backcast = residual_backcast - backcast
            stack_forecast += forecast
        return stack_forecast, residual_backcast
