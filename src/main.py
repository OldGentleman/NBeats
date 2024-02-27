import numpy as np
import pandas as pd
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.util.paths import Paths
from src.util.utils import setup_logger
from src.NBeats import NBeats
from src.data.time_series_dataset import TimeSeriesDataset

pathes = Paths()
df = pd.read_csv(pathes.warsaw_weather_filepath, sep=",")
shift = 20
data = torch.tensor(df["TAVG"].values + shift)
data_sample = data[:1000]

setup_logger()
logger.info("Start")
logger.info(f"Is the GPU available? {torch.cuda.is_available()}")

backcast_size = 7
forecast_size = 3
dataset = TimeSeriesDataset(data_sample, backcast_size, forecast_size)
nbeats = NBeats(
    backcast_size=backcast_size,
    forecast_size=forecast_size,
    num_stacks=5,
    num_blocks=5,
    activation="ReLU",
    mlpes=[10, 7],
    batch_size=10,
    lr=0.001,
)
logger.info(f"Actual: {dataset[0][1]}")
logger.info(f"Untrained NN predict: {nbeats.forward(dataset[0][0])}")
nbeats.train(dataset)
logger.info(f"Trained NN predict: {nbeats.forward(dataset[0][0])}")
logger.info("Finish")
