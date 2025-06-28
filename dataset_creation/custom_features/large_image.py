import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Union

import numpy as np
import pyarrow as pa

from datasets import config
from datasets.download.download_config import DownloadConfig
from datasets.table import array_cast
from datasets.utils.file_utils import is_local_path, xopen
from datasets.utils.py_utils import no_op_if_value_is_null, string_to_dict
from datasets.features.image import encode_np_array, encode_pil_image
from datasets.features.image import Image as BaseImageFeature


if TYPE_CHECKING:
    import PIL.Image

    from datasets.features import FeatureType


@dataclass
class LargeImage(BaseImageFeature):
    """
    LargeImage [`Feature`] to read image data from an image file.

    This is a custom feature that extends the `Image` feature to support datasets of big images.
    """

    mode: Optional[str] = None
    decode: bool = True
    id: Optional[str] = None
    # Automatically constructed
    dtype: ClassVar[str] = "PIL.Image.Image"
    pa_type: ClassVar[Any] = pa.struct(
        {"bytes": pa.large_binary(), "path": pa.large_string()}
    )
    _type: str = field(default="LargeImage", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(
        self, value: Union[str, bytes, dict, np.ndarray, "PIL.Image.Image"]
    ) -> dict:
        """Encode example into a format for Arrow.

        Args:
            value (`str`, `np.ndarray`, `PIL.Image.Image` or `dict`):
                Data passed as input to Image feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """
        if config.PIL_AVAILABLE:
            import PIL.Image
        else:
            raise ImportError("To support encoding images, please install 'Pillow'.")

        if isinstance(value, list):
            value = np.array(value)

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, bytes):
            return {"path": None, "bytes": value}
        elif isinstance(value, np.ndarray):
            # convert the image array to PNG/TIFF bytes
            return encode_np_array(value)
        elif isinstance(value, PIL.Image.Image):
            # convert the PIL image to bytes (default format is PNG/TIFF)
            return encode_pil_image(value)
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data if they're already available locally
            return {"bytes": None, "path": value.get("path")}
        elif value.get("bytes") is not None or value.get("path") is not None:
            # store the image bytes, and path is used to infer the image format using the file extension
            return {"bytes": value.get("bytes"), "path": value.get("path")}
        else:
            raise ValueError(
                f"An image sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def decode_example(self, value: dict, token_per_repo_id=None) -> "PIL.Image.Image":
        """Decode example image file into image data.

        Args:
            value (`str` or `dict`):
                A string with the absolute image file path, a dictionary with
                keys:

                - `path`: String with absolute or relative image file path.
                - `bytes`: The bytes of the image file.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                image files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `PIL.Image.Image`
        """
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use Image(decode=True) instead."
            )

        if config.PIL_AVAILABLE:
            import PIL.Image
            import PIL.ImageOps
        else:
            raise ImportError("To support decoding images, please install 'Pillow'.")

        if token_per_repo_id is None:
            token_per_repo_id = {}

        path, bytes_ = value["path"], value["bytes"]
        if bytes_ is None:
            if path is None:
                raise ValueError(
                    f"An image should have one of 'path' or 'bytes' but both are None in {value}."
                )
            else:
                if is_local_path(path):
                    image = PIL.Image.open(path)
                else:
                    source_url = path.split("::")[-1]
                    pattern = (
                        config.HUB_DATASETS_URL
                        if source_url.startswith(config.HF_ENDPOINT)
                        else config.HUB_DATASETS_HFFS_URL
                    )
                    try:
                        repo_id = string_to_dict(source_url, pattern)["repo_id"]
                        token = token_per_repo_id.get(repo_id)
                    except ValueError:
                        token = None
                    download_config = DownloadConfig(token=token)
                    with xopen(path, "rb", download_config=download_config) as f:
                        bytes_ = BytesIO(f.read())
                    image = PIL.Image.open(bytes_)
        else:
            image = PIL.Image.open(BytesIO(bytes_))
        image.load()  # to avoid "Too many open files" errors
        if image.getexif().get(PIL.Image.ExifTags.Base.Orientation) is not None:
            image = PIL.ImageOps.exif_transpose(image)
        if self.mode and self.mode != image.mode:
            image = image.convert(self.mode)
        return image

    def flatten(self) -> Union["FeatureType", Dict[str, "FeatureType"]]:
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        from datasets.features import Value

        return (
            self
            if self.decode
            else {
                "bytes": Value("large_binary"),
                "path": Value("large_string"),
            }
        )

    def cast_storage(
        self, storage: Union[pa.LargeStringArray, pa.StructArray, pa.ListArray]
    ) -> pa.StructArray:
        """Cast an Arrow array to the Image arrow storage type.
        The Arrow types that can be converted to the Image pyarrow storage type are:

        - `pa.large_string()` - it must contain the "path" data
        - `pa.large_binary()` - it must contain the image bytes
        - `pa.struct({"bytes": pa.large_binary()})`
        - `pa.struct({"path": pa.large_string()})`
        - `pa.struct({"bytes": pa.large_binary(), "path": pa.large_string()})`  - order doesn't matter
        - `pa.list(*)` - it must contain the image array data

        Args:
            storage (`Union[pa.large_stringArray, pa.StructArray, pa.ListArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Image arrow storage type, that is
                `pa.struct({"bytes": pa.large_binary(), "path": pa.large_string()})`.
        """
        if pa.types.is_large_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.large_binary())
            storage = pa.StructArray.from_arrays(
                [bytes_array, storage], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_large_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.large_string())
            storage = pa.StructArray.from_arrays(
                [storage, path_array], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.large_binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.large_string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_list(storage.type):
            bytes_array = pa.array(
                [
                    encode_np_array(np.array(arr))["bytes"] if arr is not None else None
                    for arr in storage.to_pylist()
                ],
                type=pa.large_binary(),
            )
            path_array = pa.array([None] * len(storage), type=pa.large_string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null()
            )
        return array_cast(storage, self.pa_type)

    def embed_storage(self, storage: pa.StructArray) -> pa.StructArray:
        """Embed image files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.

        Returns:
            `pa.StructArray`: Array in the Image arrow storage type, that is
                `pa.struct({"bytes": pa.large_binary(), "path": pa.large_string()})`.
        """

        @no_op_if_value_is_null
        def path_to_bytes(path):
            with xopen(path, "rb") as f:
                bytes_ = f.read()
            return bytes_

        bytes_array = pa.array(
            [
                (
                    (path_to_bytes(x["path"]) if x["bytes"] is None else x["bytes"])
                    if x is not None
                    else None
                )
                for x in storage.to_pylist()
            ],
            type=pa.large_binary(),
        )
        path_array = pa.array(
            [
                os.path.basename(path) if path is not None else None
                for path in storage.field("path").to_pylist()
            ],
            type=pa.large_string(),
        )
        storage = pa.StructArray.from_arrays(
            [bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null()
        )
        return array_cast(storage, self.pa_type)
