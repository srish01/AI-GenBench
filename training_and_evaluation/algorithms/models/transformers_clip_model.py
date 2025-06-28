from pathlib import Path
from typing import Union
from algorithms.model_factory_registry import ModelFactoryRegistry


def make_transformers_clip(
    model_name: str,
    pretrain_config: Union[str, Path],
    pretrained: bool = True,
    **kwargs
):
    try:
        from transformers import CLIPProcessor, CLIPModel

        clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(pretrain_config)
        return model, clip_processor
    except Exception:
        return None


ModelFactoryRegistry().register_model_factory(
    "transformers_clip", make_transformers_clip
)


__all__ = ["make_transformers_clip"]
