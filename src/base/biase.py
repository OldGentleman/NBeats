from typing import Tuple
import torch
import torch.nn as nn

from src.util.config import NBeatsConfig


class Biase(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int) -> None:
        super(Biase, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class GenericBiase(Biase):
    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        forecast, backcast = theta[self.backcast_size :], theta[: self.backcast_size]
        return forecast, backcast
