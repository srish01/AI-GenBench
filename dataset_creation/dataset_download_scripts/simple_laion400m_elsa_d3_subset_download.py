import json
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # To avoid a warning
from datasets import load_from_disk, Dataset
from img2dataset import download
from pathlib import Path
from PIL import Image
import random


def main():
    # !!! ADAPT THESE PATHS !!!

    # Where you are placing all the datasets
    # overall_output_path = "/deepfake"
    overall_output_path = "/home/datasets/images"

    # Where you want to download the datasets (raw files, will be converted to arrow later)
    # Doesn't need to be a folder inside overall_output_path!
    
    # download_path = "/deepfake/simple_laion400m_elsad3_subset_download"
    download_path = "/home/datasets/images/laion400m_elsad3_subset_download"

    train_filelist_path = "../resources/train_laion400m_filelist.json"
    train_filelist_zip_path = "../resources/train_laion400m_filelist.zip"
    validation_filelist_path = "../resources/validation_laion400m_filelist.json"
    validation_filelist_zip_path = "../resources/validation_laion400m_filelist.zip"

    # --- END OF PATHS ---

    if not os.path.exists(train_filelist_path):
        # Unzip it
        if os.path.exists(train_filelist_zip_path):
            print(f"Unzipping {train_filelist_zip_path}")
            import zipfile
            
            with zipfile.ZipFile(train_filelist_zip_path, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(train_filelist_path))
        else:
            raise FileNotFoundError(
                f"File {train_filelist_zip_path} not found. Please download it first."
            )

    if not os.path.exists(validation_filelist_path):
        # Unzip it
        if os.path.exists(validation_filelist_zip_path):
            print(f"Unzipping {validation_filelist_zip_path}")
            import zipfile
            
            with zipfile.ZipFile(validation_filelist_zip_path, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(validation_filelist_path))
        else:
            raise FileNotFoundError(
                f"File {validation_filelist_zip_path} not found. Please download it first."
            )
        
    assert os.path.exists(train_filelist_path), f"File {train_filelist_path} not found."
    assert os.path.exists(validation_filelist_path), f"File {validation_filelist_path} not found."

    train_output_path = os.path.join(
        overall_output_path, "simple_laion400m_elsad3_real_train_arrow"
    )
    validation_output_path = os.path.join(
        overall_output_path, "simple_laion400m_elsad3_real_validation_arrow"
    )

    # Intermediate paths (tmp folders)
    train_download_dir = os.path.join(download_path, "subset_train")
    validation_download_dir = os.path.join(download_path, "subset_validation")

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
        print("Finished downloading training set")

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
        print("Finished downloading training set")
        
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
