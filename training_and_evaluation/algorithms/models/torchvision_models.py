import warnings
from algorithms.model_factory_registry import ModelFactoryRegistry
from torch.nn import Identity, Linear

from algorithms.models.generic_probe import GenericModelProbe


def make_torchvision_model(model_name: str, pretrained: bool = True, **kwargs):
    from torchvision.models import get_model

    model_plain_name = model_name.removesuffix("_probe").removesuffix("_tune")

    num_classes = kwargs.pop("num_classes", 1)  # Can't pass it to torchvision
    model = get_model(model_plain_name, pretrained=pretrained, **kwargs)

    embedding_size = None
    recognized_fc_names = ["classifier", "head", "fc"]
    for fc_name in recognized_fc_names:
        if hasattr(model, fc_name):
            embedding_size = getattr(model, fc_name).in_features
            setattr(model, fc_name, Identity())
            break
    else:
        raise ValueError(f"Unrecognized model head name: {model_name}")

    if model_name.endswith("_probe"):
        model = GenericModelProbe(
            backbone=model,
            embedding_size=embedding_size,
            num_classes=num_classes,
        )
    else:
        if not model_name.endswith("_tune"):
            warnings.warn(
                f"Model name should end with either '_probe' or '_tune'. Using the model as-is (tune)."
            )
        setattr(model, fc_name, Linear(embedding_size, num_classes))

    return model


ModelFactoryRegistry().register_model_factory("torchvision", make_torchvision_model)


__all__ = ["make_torchvision_model"]


if __name__ == "__main__":
    import torch

    model_to_try = "resnet50"
    for mode in ("_probe", "_tune"):
        uut_model = make_torchvision_model(
            f"{model_to_try}{mode}", pretrained=True, num_classes=1
        )
        print(uut_model)
        fake_image = torch.randn(1, 3, 256, 256)
        model_out = uut_model(fake_image)
        print(model_out)
        assert model_out.shape == (1, 1), f"Output shape mismatch: {model_out.shape}"
        uut_model = None
