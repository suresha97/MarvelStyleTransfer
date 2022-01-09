import torch
from torch import Tensor


def squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.sum(
        torch.square(prediction - target)
    )
