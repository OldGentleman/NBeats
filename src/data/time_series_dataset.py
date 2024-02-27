from typing import Tuple
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self, data: torch.Tensor, backcast_size: int, forecast_size: int
    ) -> None:
        # 1D support
        if data.dim() != 1:
            raise NotImplementedError("Only supports 1D tensors")
        self.backcast_size: int = backcast_size
        self.forecast_size: int = forecast_size
        self.data: torch.Tensor = data

    def __len__(self) -> int:
        return len(self.data) - self.backcast_size - self.forecast_size + 1

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.data[idx : idx + self.backcast_size],
            self.data[
                idx + self.backcast_size : idx + self.backcast_size + self.forecast_size
            ],
        )
