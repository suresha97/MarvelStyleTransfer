from typing import Tuple

import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms


def local_image_loader(file_path: str, image_shape: Tuple[int]) -> Tensor:
    raw_image = Image.open(file_path)
    image_processor = _get_image_processor(image_shape)

    return image_processor(raw_image)


def white_noise_image_loader(image_shape: Tuple[int]) -> Tensor:
    return torch.rand(image_shape)


def _get_image_processor(image_shape: Tuple[int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor()
    ])
