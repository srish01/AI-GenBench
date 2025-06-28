from typing import Dict, List, Set
from img2dataset import download
import shutil
import os
import pandas as pd


def modify_csv(input_file: str, output_file: str, columns: Set[str], prefix: str):

    df = pd.read_csv(input_file)

    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).apply(lambda x: prefix + x if pd.notnull(x) else x)
            )
        else:
            print(f"Column '{col}' not found in {input_file}")

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    print(
        "IMPORTANT: you will need the database.csv file from https://github.com/qbammey/polardiffshield"
    )

    columns_names: Dict[str, List[str]] = {
        "URL firefly": ["ff modelVersion"],
        "URL midjourney": ["md ver"],
        "URL Dall.e 2": [],
        "URL Dall.e 3": [],
        "URL Glide": [],
        "URL Stable Diffusion 1.*": ["sd 1.x model"],
        "URL Stable Diffusion 2.1": [],
        "URL Stable Diffusion XL": [],
    }
    modify_csv("database.csv", "new_database.csv", columns_names.keys(), "http://")
    for col, data in columns_names.items():
        output_dir = os.path.abspath(
            col.replace("URL", "")[1:]
            .replace(" ", "-")
            .replace(".", "-")
            .replace("-1-*", "")
        )
        if os.path.exists(output_dir):
            raise ValueError(
                "Output directory already exists. Please remove it manually."
            )

        data.append("prompt")
        download(
            url_list="new_database.csv",
            image_size=-1,
            output_folder=output_dir,
            processes_count=16,
            thread_count=32,
            resize_mode="no",
            encode_quality=9,
            encode_format="png",
            skip_reencode=True,
            output_format="files",
            input_format="csv",
            url_col=col,
            save_additional_columns=data,
            number_sample_per_shard=1000,
        )
