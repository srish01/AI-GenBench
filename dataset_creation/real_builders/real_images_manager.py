import random
from typing import Dict, Iterable, List, Optional

from dataset_utils.common_utils import RowDictPath, saturating_balanced_choice
from real_builders.real_images_builder import RealImagesBuilder


class RealImagesManager:
    def __init__(self, real_images_builders: List[RealImagesBuilder]):
        self.real_images_builders = real_images_builders

        self._builder_map: Dict[str, RealImagesBuilder] = dict()
        self._make_builder_map()

    def keep_builders(self, builder_prefixes: Iterable[str]) -> "RealImagesManager":
        """
        Return a new version of the manager with only thew specified builders.
        This does not modify the current instance.
        :param builder_prefixes: Iterable of builder prefixes to keep.
        :return: A new RealImagesManager instance with the specified builders.
        """
        builder_prefixes = set(builder_prefixes)
        new_builders = [
            builder
            for builder in self.real_images_builders
            if builder.get_prefix() in builder_prefixes
        ]
        return RealImagesManager(new_builders)

    def available_images(
        self,
        builder_prefix: Optional[str] = None,
        among: Optional[Iterable[str]] = None,
    ) -> Iterable[str]:
        among = set(among) if among is not None else None
        if builder_prefix is not None:
            return [
                image_id
                for image_id in self._builder_map[builder_prefix].available_images()
                if among is None or image_id in among
            ]
        else:
            if among is not None:
                all_available_images = set()
                for builder in self.real_images_builders:
                    all_available_images.update(builder.available_images())

                return all_available_images.intersection(among)
            else:
                return [
                    image_id
                    for builder in self.real_images_builders
                    for image_id in builder.available_images()
                ]

    def get_image(
        self,
        image_id: str,
    ) -> RowDictPath:
        builder_prefix = image_id.split("/")[0]
        return self._builder_map[builder_prefix].get_image(
            image_id,
        )

    def available_builder_prefixes(self):
        return list(self._builder_map.keys())

    def _make_builder_map(self):
        for builder in self.real_images_builders:
            builder_prefix = builder.get_prefix()
            self._builder_map[builder_prefix] = builder

    def select_random_images(
        self,
        num_images: int,
        seed: int,
        excluding: Optional[Iterable[str]] = None,
        allowed_prefixes: Optional[Iterable[str]] = None,
    ) -> List[str]:
        allowed_prefixes = (
            set(allowed_prefixes)
            if allowed_prefixes is not None
            else set(self._builder_map.keys())
        )
        excluding = set(excluding) if excluding is not None else set()

        allowed_builders = [
            builder
            for prefix, builder in self._builder_map.items()
            if prefix in allowed_prefixes
        ]
        allowed_images: Dict[RealImagesBuilder, List[str]] = dict()
        for builder in allowed_builders:
            allowed_images[builder] = [
                x for x in builder.available_images() if x not in excluding
            ]
            allowed_images[builder].sort()

        possible_choices: List[str] = []
        associated_builders: List[RealImagesBuilder] = []
        for builder, images in allowed_images.items():
            possible_choices.extend(images)
            associated_builders.extend([builder] * len(images))

        return saturating_balanced_choice(
            num_images,
            possible_choices,
            associated_builders,
            seed,
            choice_fn=_random_choice_fn,
        )[0]


def _random_choice_fn(
    real_images_manager: RealImagesBuilder,
    available_images: List[str],
    num_images: int,
) -> List[str]:
    builder_specific_choice = real_images_manager.select_random_images(
        num_images, allowed=available_images
    )
    if builder_specific_choice is None:
        return random.sample(available_images, num_images)
    return builder_specific_choice


__all__ = ["RealImagesManager"]
