from typing import List
from pydantic import BaseModel


class NBeatsConfig(BaseModel):
    ACTIVATIONS: List = [
        "ReLU",
        "Softplus",
        "Tanh",
        "SELU",
        "LeakyReLU",
        "PReLU",
        "Sigmoid",
    ]
