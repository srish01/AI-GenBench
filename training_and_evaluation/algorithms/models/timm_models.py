import warnings
from algorithms.model_factory_registry import ModelFactoryRegistry
from torch.nn import Identity

from algorithms.models.generic_probe import GenericModelProbe


def make_timm_model(model_name: str, pretrained: bool = True, **kwargs):
    import timm

    model_plain_name = model_name.removesuffix("_probe").removesuffix("_tune")
    num_classes = kwargs.get("num_classes", 1)

    model = timm.create_model(
        model_plain_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if model_name.endswith("_probe"):
        embedding_size = None
        head = None
        recognized_fc_names = ["classifier", "head", "fc"]
        for fc_name in recognized_fc_names:
            if hasattr(model, fc_name):
                embedding_size = getattr(model, fc_name).in_features
                head = getattr(model, fc_name)
                setattr(model, fc_name, Identity())
                break
        else:
            raise ValueError(f"Unrecognized model head name: {model_name}")

        model = GenericModelProbe(
            backbone=model,
            embedding_size=embedding_size,
            num_classes=num_classes,
            new_head=head,
        )
    elif not model_name.endswith("_tune"):
        warnings.warn(
            f"Model name should end with either '_probe' or '_tune'. Using the model as-is (tune)."
        )

    return model


ModelFactoryRegistry().register_model_factory("timm", make_timm_model)


__all__ = ["make_timm_model"]


if __name__ == "__main__":
    import torch

    model_to_try = "convnext_xxlarge.clip_laion2b_soup_ft_in1k"
    for mode in ("_probe", "_tune"):
        uut_model = make_timm_model(f"{model_to_try}{mode}", pretrained=True)
        print(uut_model)
        fake_image = torch.randn(1, 3, 256, 256)
        model_out = uut_model(fake_image)
        print(model_out)
        assert model_out.shape == (1, 1), f"Output shape mismatch: {model_out.shape}"
        uut_model = None
