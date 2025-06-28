import json
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # To avoid a warning
from datasets import load_from_disk, Dataset
from img2dataset import download
from pathlib import Path
from PIL import Image
import random


def _make_filelist(dataset, split: str, filelist_path: str):
    dataset_split = dataset[split]
    urls = dataset_split["url"]
    ids = dataset_split["id"]

    # Each row must be {'url': 'http://example.com/image.jpg'}
    urls = [{"url": url, "id": str(file_id)} for url, file_id in zip(urls, ids)]

    if False:
        # Only for testing purposes
        # Select only 100 images (random)
        urls = random.sample(urls, 100)

    # Make json of urls
    Path(filelist_path).parent.mkdir(parents=True, exist_ok=True)
    with open(filelist_path, "w") as f:
        json.dump(urls, f)

    return urls


def main():
    # !!! ADAPT THESE PATHS !!!

    # Where you are placing all the datasets
    overall_output_path = "/deepfake"

    # Where you want to download the datasets (raw files, will be converted to arrow later)
    # Doesn't need to be a folder inside overall_output_path!
    download_path = "/deepfake/laion400m_elsad3_subset_download"

    # ELSA_D3 must already be downloaded and saved offline using "save_to_disk"
    # Note: it should already be a in a subdirectory of overall_output_path
    # (if you are doing things correctly...)
    elsa_d3_path = "/deepfake/ELSA_D3_offline"

    # --- END OF PATHS ---

    train_output_path = os.path.join(
        overall_output_path, "laion400m_elsad3_real_train_arrow"
    )
    validation_output_path = os.path.join(
        overall_output_path, "laion400m_elsad3_real_validation_arrow"
    )

    # Intermediate paths (tmp folders)
    train_download_dir = os.path.join(download_path, "subset_train")
    validation_download_dir = os.path.join(download_path, "subset_validation")
    train_filelist_path = os.path.join(download_path, "train_filelist.json")
    validation_filelist_path = os.path.join(download_path, "validation_filelist.json")

    elsa_d3_dataset = None

    if not os.path.exists(train_download_dir):
        elsa_d3_dataset = load_from_disk(elsa_d3_path)
        _make_filelist(elsa_d3_dataset, "train", train_filelist_path)

    if not os.path.exists(validation_download_dir):
        if elsa_d3_dataset is None:
            elsa_d3_dataset = load_from_disk(elsa_d3_path)
        _make_filelist(elsa_d3_dataset, "validation", validation_filelist_path)

    del elsa_d3_dataset

    if os.path.exists(train_download_dir):
        print("Training set already downloaded. Skipping download")
        print(
            "Note: if you want to redownload the training set, please remove the directory manually."
        )
    else:
        print(f"Downloading training set")
        with open(train_filelist_path, "r") as f:
            urls = json.load(f)
        print(f"Found {len(urls)} images to download in the training set.")

        download(
            url_list=train_filelist_path,
            image_size=-1,
            output_folder=train_download_dir,
            processes_count=16,
            thread_count=32,
            resize_mode="no",
            encode_quality=9,
            encode_format="png",
            skip_reencode=True,
            output_format="files",
            input_format="json",
            number_sample_per_shard=1000,
            save_additional_columns=["id", "description"],
        )

    if os.path.exists(validation_download_dir):
        print("Validation set already downloaded. Skipping download")
        print(
            "Note: if you want to redownload the validation set, please remove the directory manually."
        )
    else:
        print("Downloading validation set")
        with open(validation_filelist_path, "r") as f:
            urls = json.load(f)
        print(f"Found {len(urls)} images to download in the validation set.")

        download(
            url_list=validation_filelist_path,
            image_size=-1,
            output_folder=validation_download_dir,
            processes_count=16,
            thread_count=32,
            resize_mode="no",
            encode_quality=9,
            encode_format="png",
            skip_reencode=True,
            output_format="files",
            input_format="json",
            number_sample_per_shard=1000,
            save_additional_columns=["id", "description"],
        )

    print("Datasets downloaded from internet")
    print("Starting arrow conversion")

    for split, download_dir, result_path in zip(
        ["train", "validation"],
        [train_download_dir, validation_download_dir],
        [train_output_path, validation_output_path],
    ):
        download_dir = Path(download_dir)
        result_path = Path(result_path)

        if result_path.exists():
            print("Dataset already exists", download_dir.name)
            dataset = load_from_disk(str(result_path))  # Just to check if it worked
            print(f"It contains {len(dataset)} images.")
        elif not download_dir.exists():
            print("Dataset not found", download_dir.name)
        else:
            images_metadata = []
            images_id = []
            for dirpath, dirnames, filenames in os.walk(str(download_dir)):
                for filename in filenames:
                    if filename.endswith(".png"):

                        # Load the json file with the same name and obtain the id
                        json_path = os.path.join(
                            dirpath, filename.removesuffix(".png") + ".json"
                        )
                        assert os.path.exists(
                            json_path
                        ), f"JSON file not found: {json_path}"
                        with open(json_path, "r") as f:
                            json_data = json.load(f)
                            # Assuming the id is stored under the key 'id'
                            if "id" in json_data:
                                images_id.append(int(json_data["id"]))
                                images_metadata.append(
                                    (
                                        os.path.join(dirpath, filename),
                                        json_data["description"],
                                    )
                                )
                            else:
                                raise KeyError(f"'id' key not found in {json_path}")

            images_id, images_metadata = zip(*sorted(zip(images_id, images_metadata)))
            images_id = list(images_id)
            images_metadata = list(images_metadata)

            dataset = Dataset.from_generator(
                generate_dataset,
                gen_kwargs={"image_ids": images_id, "images_list": images_metadata},
                num_proc=8,
            )

            dataset.save_to_disk(str(result_path))
            dataset = load_from_disk(str(result_path))  # Just to check if it worked
            print(f"Saved dataset to {result_path}. It contains {len(dataset)} images.")
            dataset = None


def generate_dataset(image_ids, images_list):
    for img_id, (img_path, img_description) in zip(image_ids, images_list):
        # NOTE: here we assume that all images are already in the expected format
        # If you need to convert images to a different format, you can do that here
        img = Image.open(img_path)
        img.load()
        yield {
            "image": img,
            "file_id": f"LAION-400M/{img_id}",
            "description": img_description,
        }


if __name__ == "__main__":
    main()  # Paths are hardcoded in the function (change them!)
