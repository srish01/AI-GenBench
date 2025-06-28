from collections import defaultdict
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
from collections import defaultdict
import threading

PathALike = Union[str, Path]


class BatchedExtractionContext:
    """
    A context that does the following:
    - Given an archive, allows for the iteration (in the form of obtaining a temp path for each file) of multiple files from an archive
    - It uses 7z under the hood for all archives
    - Files are extracted in batches to a tmp directory and are cleaned up after use
    - Batch extraction is quite faster than extracting files one by one (and is less memory intensive than extracting all files at once)
    """

    def __init__(
        self,
        archive: PathALike,
        files: Iterable[str],
        batch_size: int = 128,
        extractor_executable: str = "7zz",
    ):
        self.archive = Path(archive)
        self.files = [Path(x) for x in files]
        self.batch_size = batch_size
        self.current_file = 0
        self.current_batch = -1
        self.batch_files = []
        self.next_batch_files = []
        self.tmp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.next_tmp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.extractor_executable = extractor_executable
        self.next_batch_thread = None
        self._process_result = None

    def __enter__(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.next_tmp_dir = tempfile.TemporaryDirectory()
        self.current_file = -1
        self.current_batch = -1
        self._prepare_next_batch()
        return self

    def __iter__(self):
        if self.tmp_dir is None:
            raise ValueError("Context not entered")

        extract_folder = Path(self.tmp_dir.name)
        self.current_file = 0
        while self.current_file < len(self.files):

            if (self.current_file % self.batch_size) == 0:
                self.next_batch_thread.join()
                self._use_next_batch()
                extract_folder = self._make_extraction_folder(self.tmp_dir.name)
                self._prepare_next_batch()

            next_file = (
                extract_folder / self.batch_files[self.current_file % self.batch_size]
            )
            yield next_file

            next_file.unlink()

            self.current_file += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.next_batch_thread:
            self.next_batch_thread.join()
        self._process_result = None
        self.tmp_dir.cleanup()
        self.next_tmp_dir.cleanup()
        self.tmp_dir = None
        self.next_tmp_dir = None

    def _prepare_next_batch(self):
        if self.current_file == -1:
            base_file = 0  # First batch
        else:
            base_file = self.current_file + self.batch_size

        self.next_batch_thread = threading.Thread(
            target=self._extract_next_batch, args=(base_file,)
        )
        self.next_batch_thread.start()

    def _use_next_batch(self):
        if self._process_result is not None and self._process_result.returncode != 0:
            raise RuntimeError(f"Extraction failed: {self._process_result.stderr}")
        self._process_result = None

        self.current_batch += 1
        self.batch_files = self.next_batch_files
        self.tmp_dir, self.next_tmp_dir = self.next_tmp_dir, self.tmp_dir

    def _extract_next_batch(self, start_index: int):
        end_index = min(start_index + self.batch_size, len(self.files))
        self.next_batch_files = self.files[start_index:end_index]

        if len(self.next_batch_files) == 0:
            return

        extract_folder = self._make_extraction_folder(self.next_tmp_dir.name)

        cmd_args = [
            self.extractor_executable,
            "x",
            str(self.archive),
            "-o" + str(extract_folder),
        ]
        cmd_args.extend([str(x) for x in self.next_batch_files])

        self._process_result = subprocess.run(cmd_args, capture_output=True, text=True)

    def _make_extraction_folder(self, tmp_dir: str):
        rel_folder = self.archive.parent.resolve()
        extract_folder = Path(tmp_dir) / rel_folder.relative_to(rel_folder.anchor)
        return extract_folder


class MultiSourceFilesIterator:
    """
    A context that allows for the iteration of files from multiple sources (archives and regular files)

    The files are iterated in the order they are provided using a series of internal optimizations
    aimed at reducing the number of times the archives are opened and closed (and in general to imporve performance).
    """

    def __init__(
        self,
        files: Iterable[PathALike],
        batch_size: int = 128,
        extractor_executable: str = "7zz",
    ):
        self.files = list(files)
        self.current_file = 0
        self.batch_size = batch_size
        self.file_order: Optional[List[Tuple[Optional[str], PathALike]]] = None
        self.open_archives: Dict[str, BatchedExtractionContext] = dict()
        self.archive_iterators: Dict[str, Iterator[Path]] = dict()
        self.extractor_executable = extractor_executable

    def _make_plan(self):
        self.file_order = []
        archive_files = defaultdict(list)
        for file_path in self.files:
            if "!" in str(file_path):
                file_path = str(file_path)
                zip_path = file_path.split("!", maxsplit=1)[0]
                internal_path = file_path.split("!", maxsplit=1)[1]
                self.file_order.append((zip_path, internal_path))
                archive_files[zip_path].append(internal_path)
            else:
                self.file_order.append((None, file_path))

        # Open archives
        for archive, files in archive_files.items():
            archive_context = BatchedExtractionContext(
                archive,
                files,
                batch_size=self.batch_size,
                extractor_executable=self.extractor_executable,
            )
            self.open_archives[archive] = archive_context
            archive_context.__enter__()

    def __iter__(self):
        if self.file_order is None:
            raise ValueError("Context not entered")

        for archive, file_path in self.file_order:
            if archive is None:
                yield Path(file_path)
            else:
                if archive not in self.archive_iterators:
                    self.archive_iterators[archive] = iter(self.open_archives[archive])
                yield next(self.archive_iterators[archive])

    def __enter__(self):
        self._make_plan()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_order = None
        for archive_context in self.open_archives.values():
            try:
                archive_context.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print(f"Error closing archive: {e}")
        self.open_archives.clear()


def list_archive_files(
    archive: PathALike,
    extractor_executable: str = "7zz",
    include_directories: bool = False,
    include_archive_path: bool = False,
) -> List[str]:
    """
    List the files in an archive
    """
    cmd_args = [extractor_executable, "l", "-ba", str(archive)]
    result = subprocess.run(cmd_args, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Listing files failed: {result.stderr}")

    lines = result.stdout.split("\n")
    files = []
    for line in lines:
        if len(line) == 0:
            continue

        attributes = line[20:25]
        if len(attributes) != 5:
            raise RuntimeError(f"Error parsing line: {line}")

        is_dir = attributes[0] == "D"

        if is_dir and not include_directories:
            continue

        filename = line[53:]
        if filename == "":
            raise RuntimeError(f"Error parsing line: {line}")

        files.append(filename)

    if include_archive_path:
        files = [f"{archive}!{x}" for x in files]

    return files


__all__ = [
    "BatchedExtractionContext",
    "MultiSourceFilesIterator",
    "list_archive_files",
]

if __name__ == "__main__":
    import time
    import random
    from PIL import Image

    files_to_extract = []

    # Optional: add a file that is not in an archive
    # files_to_extract.append("/deepfake/000000000139.jpg")

    for archive_path in [
        "/deepfake/GenImage/ADM/imagenet_ai_0508_adm.zip",
        "/deepfake/GenImage/BigGAN/imagenet_ai_0419_biggan.zip",
    ]:
        filelist = list_archive_files(archive_path)
        filelist = filelist[:1000]  # Just a part of the dataset (for testing)

        # Make file list in the format of <archive_path>!<internal_path>
        for file in filelist:
            if not file.startswith("/"):
                files_to_extract.append(f"{archive_path}!{file}")

    # Shuffle
    random.shuffle(files_to_extract)

    # Time the extraction
    start_time = time.time()
    print("Start time:", start_time)

    images = []
    with MultiSourceFilesIterator(files_to_extract, batch_size=128) as context:
        for file in context:
            image = Image.open(file)
            image.load()
            images.append(image)
    end_time = time.time()
    print(f"Extraction completed in {end_time - start_time} seconds")
    print(images)
