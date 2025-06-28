import os
from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Linear, Module
import torch.hub

from algorithms.model_factory_registry import ModelFactoryRegistry


def make_dinov2_model(model_name: str, pretrained: bool = True, **kwargs):
    num_classes = kwargs.pop("num_classes", 1)
    is_tune = model_name.endswith("_tune")
    if is_tune:
        model_name = model_name.removesuffix("_tune")
    else:
        assert model_name.endswith("_probe")
        model_name = model_name.removesuffix("_probe")

    if is_tune:
        return DINOv2ModelTune(model_name, num_classes=num_classes)
    else:
        return DINOv2ModelProbe(model_name, num_classes=num_classes)


class DINOv2ModelProbe(torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int = 1, shape=(3, 224, 224)):
        super().__init__()

        # Get local rank from env
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Load DINOv2
        backbone: Module = torch.hub.load("facebookresearch/dinov2", model_name)
        backbone.to(device)
        backbone.eval()
        backbone.requires_grad_(False)

        self._bb: Tuple[Module] = (backbone,)

        with torch.no_grad():
            dummy_input = torch.zeros((1, *shape)).to(device)
            features: Tensor = self.backbone(dummy_input)
            self.intermediate_size: int = features.shape[-1]

        self.fc = Linear(self.intermediate_size, num_classes)

    def forward(self, x: Tensor, return_feature=False) -> Tensor:
        features = self.forward_features(x)
        if return_feature:
            return features
        return self.forward_head(features)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.backbone.to(*args, **kwargs)
        return self

    def forward_features(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return features

    def forward_head(self, x: Tensor) -> Tensor:
        return self.fc(x)

    @property
    def backbone(self) -> Module:
        return self._bb[0]


class DINOv2ModelTune(torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int = 1, shape=(3, 224, 224)):
        super().__init__()

        # Get local rank from env
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Load DINOv2
        backbone: Module = torch.hub.load("facebookresearch/dinov2", model_name)
        backbone.to(device)
        backbone.eval()
        backbone.requires_grad_(False)

        self.backbone: Module = backbone

        with torch.no_grad():
            dummy_input = torch.zeros((1, *shape)).to(device)
            features: Tensor = self.backbone(dummy_input)
            self.intermediate_size: int = features.shape[-1]

        self.fc = Linear(self.intermediate_size, num_classes)

    def forward(self, x: Tensor, return_feature=False) -> Tensor:
        features = self.forward_features(x)
        if return_feature:
            return features
        return self.forward_head(features)

    def forward_features(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return features

    def forward_head(self, x: Tensor) -> Tensor:
        return self.fc(x)


ModelFactoryRegistry().register_model_factory("dinov2", make_dinov2_model)

__all__ = [
    "make_dinov2_model",
    "DINOv2ModelProbe",
    "DINOv2ModelTune",
]


if __name__ == "__main__":
    model = make_dinov2_model("dinov2_vitl14_tune")
    print(model)
    model = make_dinov2_model("dinov2_vitl14_probe")
    print(model)
