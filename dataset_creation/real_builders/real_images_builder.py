from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from dataset_utils.common_utils import RowDictPath


class RealImagesBuilder(ABC):

    @abstractmethod
    def get_image(
        self,
        image_id: str,
    ) -> RowDictPath:
        pass

    @abstractmethod
    def available_images(self) -> Iterable[str]:
        pass

    @abstractmethod
    def get_prefix(self) -> str:
        pass

    @abstractmethod
    def get_builder_name(self) -> str:
        pass

    def select_random_images(
        self,
        num_images: int,
        seed: Optional[int] = None,
        excluding: Optional[Iterable[str]] = None,
        allowed: Optional[Iterable[str]] = None,
    ) -> Optional[List[str]]:
        return None


__all__ = ["RealImagesBuilder"]
