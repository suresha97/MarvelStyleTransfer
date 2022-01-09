import torch
from torch import Tensor


class PretrainedVGGNet:
    def __init__(self, variant: int) -> None:
        self._model = torch.hub.load('pytorch/vision:v0.10.0', f'vgg{variant}', pretrained=True).eval()

    def forward(self, input_tensor: Tensor, extraction_layer: int) -> Tensor:
        if input_tensor.size() != 4:
            print(f"Tensor of size, {input_tensor.size()}, passed to model did not have batch dimension, "
                  f"so one has been created.")
            input_tensor = input_tensor.unsqueeze(0)

        return self._model.features[:extraction_layer](input_tensor).squeeze()
