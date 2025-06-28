from img2dataset import download
import os

if __name__ == "__main__":
    print(
        "You will need the RAISE_all.csv file from http://loki.disi.unitn.it/RAISE/confirm.php?package=all"
    )
    print('Note: use the "Get the images!" link')

    output_dir = os.path.abspath("RAISE_all_TIF")

    if os.path.exists(output_dir):
        raise ValueError("Output directory already exists. Please remove it manually.")

    download(
        url_list="RAISE_all.csv",
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
        url_col="TIFF",
        number_sample_per_shard=1000,
    )
