from typing import Dict, Any
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from utils import squared_error


class LossCalculator(ABC):
    def __init__(self, layer_activations: Dict[int, Dict[str, Tensor]]) -> None:
        self._layer_activations = layer_activations

    @abstractmethod
    def get_loss(self, loss_parameters: Dict[str, Any]) -> None:
        pass


class StyleTransferLossCalculator(LossCalculator):
    def __init__(self, layer_activations: Dict[int, Dict[str, Tensor]]) -> None:
        super().__init__(layer_activations)

    def get_loss(self, loss_parameters: Dict[str, Any]) -> float:
        content_loss = self._content_loss_calculator.get_loss(loss_parameters["content"])
        style_loss = self._style_loss_calculator.get_loss(loss_parameters["style"])

        return loss_parameters["alpha"] * content_loss + loss_parameters["beta"] * style_loss

    @property
    def _content_loss_calculator(self) -> LossCalculator:
        return ContentLossCalculator(self._layer_activations)

    @property
    def _style_loss_calculator(self) -> LossCalculator:
        return StyleLossCalculator(self._layer_activations)


class ContentLossCalculator(LossCalculator):
    def __init__(self, layer_activations: Dict[int, Dict[str, Tensor]]) -> None:
        super().__init__(layer_activations)

    def get_loss(self, loss_parameters: Dict[str, Any]) -> Tensor:
        content_loss_layer_num = loss_parameters["content_loss_layer_num"]
        content_image_features = self._layer_activations[content_loss_layer_num]["content"]
        generated_image_features = self._layer_activations[content_loss_layer_num]["generated"]

        return 0.5 * squared_error(content_image_features, generated_image_features)


class StyleLossCalculator(LossCalculator):
    def __init__(self, layer_activations: Dict[int, Dict[str, Tensor]]) -> None:
        super().__init__(layer_activations)
        self._layer_activations = layer_activations

    def get_loss(self, loss_parameters: Dict[str, Any]) -> float:
        total_style_loss = 0
        layer_weights = loss_parameters["layer_weights"]
        for idx, (layer, activations) in enumerate(self._layer_activations.items()):
            layer_style_loss = layer_weights[idx] * self.get_layer_style_loss(
                activations["style"], activations["generated"]
            )

            total_style_loss += layer_style_loss

        return total_style_loss

    def get_layer_style_loss(self, style_image_features: Tensor, generated_image_features: Tensor) -> Tensor:
        scaling_constant = 1 / (
                    4 * (style_image_features.size()[0] ** 2) * (style_image_features.size()[1] ** 2)
        )
        style_gram_matrix = self._gram_matrix(style_image_features)
        gen_gram_matrix = self._gram_matrix(generated_image_features)

        return scaling_constant * squared_error(gen_gram_matrix, style_gram_matrix)

    @staticmethod
    def _gram_matrix(input_features: Tensor) -> Tensor:
        return torch.mm(input_features, torch.transpose(input_features, 0, 1))
