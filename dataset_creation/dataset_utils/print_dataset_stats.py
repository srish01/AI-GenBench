# %%
import argparse
from datasets import load_from_disk
import datasets
from datasets.features import Image
from collections import Counter, defaultdict
from custom_features.large_image import LargeImage
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL.Image import Image
from multiprocessing import Pool, cpu_count

# Register the LargeImage type
datasets.features.features.register_feature(LargeImage, "LargeImage")


def print_stats(dataset_path, split="train"):
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset[split]

    print("Available columns:", dataset.column_names)

    # dataset = dataset.cast_column("image", Image())
    dataset = dataset.select_columns(["generator", "label", "origin_dataset"])

    generator_counter = Counter()
    label_counter = Counter()
    origin_counter = Counter()
    generator_origin_counter = defaultdict(Counter)

    for item in tqdm(dataset):
        generator = item["generator"]
        label = item["label"]
        origin = item["origin_dataset"]

        generator_counter[generator] += 1
        label_counter[label] += 1
        origin_counter[origin] += 1

        if label != 0 and generator != "":
            generator_origin_counter[generator][origin] += 1

    print("Number of images per generator:")
    for generator in sorted(generator_counter.keys()):
        count = generator_counter[generator]
        if generator == "":
            generator = "(Real)"
        print(f"{generator}: {count}")

    print("\nNumber of images per label:")
    for label in sorted(label_counter.keys()):
        count = label_counter[label]
        print(f"{label}: {count}")

    print("\nNumber of images per origin dataset:")
    for origin in sorted(origin_counter.keys()):
        count = origin_counter[origin]
        print(f"{origin}: {count}")

    print(
        "\nNumber of images per origin dataset for each generator (excluding real images):"
    )
    for generator in sorted(generator_origin_counter.keys()):
        origin_counts = generator_origin_counter[generator]
        print(f"{generator}:")
        for origin in sorted(origin_counts.keys()):
            count = origin_counts[origin]
            print(f"  {origin}: {count}")


def process_image(item):
    image: Image = item["image"]
    generator = item["generator"]

    return (image.size, image.format, image.mode, generator)


def process_image_by_index(args):
    index, dataset = args
    item = dataset[index]
    return index, *process_image(item)


def print_images_stats(dataset_path, split="train", show_some_images: bool = False):
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset[split]

    dataset = dataset.select_columns(["generator", "label", "origin_dataset", "image"])
    # dataset = dataset.cast_column("image", Image())

    max_pixels = 0
    min_pxiels = float("inf")
    max_image = None
    min_image = None
    max_width = 0
    max_height = 0
    min_width = float("inf")
    min_height = float("inf")

    by_format = Counter()
    by_mode = Counter()

    format_for_each_generator = defaultdict(Counter)
    mode_for_each_generator = defaultdict(Counter)

    indices = list(range(len(dataset)))
    args = [(index, dataset) for index in indices]

    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap(process_image_by_index, args), total=len(indices)):
            row_index, image_size, image_format, image_mode, generator = result
            image_pixels = image_size[0] * image_size[1]

            max_width = max(max_width, image_size[0])
            max_height = max(max_height, image_size[1])
            min_width = min(min_width, image_size[0])
            min_height = min(min_height, image_size[1])

            if image_pixels > max_pixels:
                max_pixels = image_pixels
                max_image = dataset[row_index]["image"]
            if image_pixels < min_pxiels:
                min_pxiels = image_pixels
                min_image = dataset[row_index]["image"]

            by_format[image_format] += 1
            by_mode[image_mode] += 1

            format_for_each_generator[generator][image_format] += 1
            mode_for_each_generator[generator][image_mode] += 1

    print(f"Largest image size: {max_pixels} pixels")
    print(f"Smallest image size: {min_pxiels} pixels")
    print(f"Max width: {max_width}, Max height: {max_height}")
    print(f"Min width: {min_width}, Min height: {min_height}")

    print("\nNumber of images by format:")
    for format in sorted(by_format.keys()):
        print(f"{format}: {by_format[format]}")

    print("\nNumber of images by mode:")
    for mode in sorted(by_mode.keys()):
        print(f"{mode}: {by_mode[mode]}")

    print("\nNumber of images by format for each generator:")
    for generator in sorted(format_for_each_generator.keys()):
        print(f"{generator}:")
        for format in sorted(format_for_each_generator[generator].keys()):
            print(f"  {format}: {format_for_each_generator[generator][format]}")

    print("\nNumber of images by mode for each generator:")
    for generator in sorted(mode_for_each_generator.keys()):
        print(f"{generator}:")
        for mode in sorted(mode_for_each_generator[generator].keys()):
            print(f"  {mode}: {mode_for_each_generator[generator][mode]}")

    if show_some_images:
        if max_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(max_image)
            plt.title(f"Largest Image: {max_image.size[0]}x{max_image.size[1]}")
            plt.savefig("largest_image.png")
            plt.show()

        if min_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(min_image)
            plt.title(f"Smallest Image: {min_image.size[0]}x{min_image.size[1]}")
            plt.savefig("smallest_image.png")
            plt.show()

        # Show 4 random images
        for i in range(4):
            plt.figure(figsize=(10, 10))
            plt.imshow(dataset[i]["image"])
            plt.title(
                f"Generator: {dataset[i]['generator']}, Label: {dataset[i]['label']}, Origin: {dataset[i]['origin_dataset']}"
            )
            plt.savefig(f"random_image_{i}.png")
            plt.show()


def print_images_stats_for_generator(dataset_path, generator: str, split="train"):
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset[split]
    dataset = dataset.select_columns(["generator", "label", "origin_dataset", "image"])
    generator_column = dataset["generator"]
    generator_indices = [i for i, g in enumerate(generator_column) if g == generator]
    dataset = dataset.select(generator_indices)
    dataset = dataset.select_columns(["image", "origin_dataset"])

    image_sizes = [image.size for image in dataset["image"]]
    image_sizes = [image[0] * image[1] for image in image_sizes]
    image_sizes = sorted(image_sizes)
    plt.hist(image_sizes, bins=100)
    plt.title(f"Image sizes for {generator}")
    plt.xlabel("Image size (pixels)")
    plt.ylabel("Number of images")

    for origin in dataset.unique("origin_dataset"):
        origin_dataset_indices = [
            i for i, o in enumerate(dataset["origin_dataset"]) if o == origin
        ]
        origin_dataset = dataset.select(origin_dataset_indices)
        image_sizes = [image.size for image in origin_dataset["image"]]
        image_sizes = [image[0] * image[1] for image in image_sizes]
        image_sizes = sorted(image_sizes)
        plt.hist(image_sizes, range=(0, max(image_sizes)), bins=10)
        plt.title(f"Image sizes for {generator}")
        plt.xlabel("Image size (pixels)")
        plt.ylabel("Number of images")
        plt.savefig(f"image_sizes_{generator}_{origin.replace('/', '_')}.png")
        plt.close()

        n_small = sum(1 for size in image_sizes if size < 40000)
        print(f"Origin {origin}, number of images < 40000 pixels: {n_small}")


def print_overlaps(dataset_path):
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    train_images = (
        train_dataset["file_id"]
        if "file_id" in train_dataset.column_names
        else train_dataset["filename"]
    )
    validation_images = (
        validation_dataset["file_id"]
        if "file_id" in validation_dataset.column_names
        else validation_dataset["filename"]
    )

    train_set = set(train_images)
    validation_set = set(validation_images)

    if len(train_set) == len(train_images):
        print("Train set is unique")
    else:
        print("Train set has duplicates")

    if len(validation_set) == len(validation_images):
        print("Validation set is unique")
    else:
        print("Validation set has duplicates")

    overlap = train_set.intersection(validation_set)
    if len(overlap) == 0:
        print("No overlapping images between train and validation splits")
    else:
        print("Overlapping images found between train and validation splits")
        print(f"Number of overlapping images: {len(overlap)}")
        print(f"Overlapping images: {overlap}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print dataset statistics.")
    parser.add_argument(
        "dataset_path", type=str, help="Path to the HuggingFace dataset"
    )
    args = parser.parse_args()

    print("-" * 20, "Train split stats", "-" * 20)
    print_stats(args.dataset_path, "train")
    # print_images_stats(args.dataset_path, "train", show_some_images=False)

    print()
    print("\n", "-" * 20, "Validation split stats", "-" * 20)
    print_stats(args.dataset_path, "validation")
    # print_images_stats(args.dataset_path, "validation", show_some_images=False)

    # print("\n", "-" * 20, "Images stats for BigGAN, train split", "-" * 20)
    # print_images_stats_for_generator(args.dataset_path, "BigGAN", "train")

    print("\n", "-" * 20, "Overlap between train and validation splits", "-" * 20)
    print_overlaps(args.dataset_path)
