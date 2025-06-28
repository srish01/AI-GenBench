import random
from typing import Union

import numpy as np

from dataset_utils.common_utils import PathAlike, check_dataset_format
from datasets import Dataset, load_from_disk
import matplotlib.pyplot as plt
import textwrap


def visualize_dataset_rows(dataset: Union[PathAlike, Dataset]):
    if not isinstance(dataset, Dataset):
        dataset = load_from_disk(dataset)

    assert isinstance(dataset, Dataset)

    check_dataset_format(dataset)

    # Create a grid of images containing rows 0, 1, -2, -1
    # with their respecitve labels and descriptions
    # Use matplotlib to display the grid

    columns = 2
    rows = 2

    fig = plt.figure(figsize=(10, 10))

    for i, idx in zip(range(1, columns * rows + 1), [0, 1, -2, -1]):
        row = dataset[idx]
        fig.add_subplot(rows, columns, i)
        plt.imshow(row["image"])
        plt.title(f"[{idx}] label={row['label']}: {make_caption(row)}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(10, 10))

    rng_indices = random.sample(range(len(dataset)), 4)
    for i, idx in zip(range(1, columns * rows + 1), rng_indices):
        row = dataset[idx]
        fig.add_subplot(rows, columns, i)
        plt.imshow(row["image"])
        plt.title(f"[{idx}] label={row['label']}: {make_caption(row)}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()

    # Generators histogram
    generators = dataset["generator"]
    generator_names = list(set(generators))
    if "" in generator_names:
        generator_names.remove("")
    generator_counts = [generators.count(name) for name in generator_names]

    n_real_images = dataset["label"].count(0)

    # Sort by name
    generator_names, generator_counts = zip(
        *sorted(zip(generator_names, generator_counts))
    )

    generator_names = list(generator_names) + ["Real images"]
    generator_counts = list(generator_counts) + [n_real_images]

    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(10, 10))
    y_pos = np.arange(len(generator_names))
    ax.barh(y_pos, generator_counts)
    ax.set_yticks(y_pos, labels=generator_names)
    ax.invert_yaxis()
    ax.set_xlabel("Number of samples")
    plt.show()
    plt.close()


def wrap_text(text, width=30):
    """Wrap text to fit into the plot."""
    return "\n".join(textwrap.wrap(text, width=width))


def make_caption(row):
    """Create a caption for the image."""
    description = row["description"] if len(row["description"]) > 0 else ""

    if len(row["positive_prompt"]) > 0 or len(row["negative_prompt"]) > 0:
        if len(description) > 0:
            description += "\n"
        negative_prompt = row["negative_prompt"]
        if len(negative_prompt) == 0:
            negative_prompt = f"<no negative prompt>"
        description += f"+{row['positive_prompt']} - {negative_prompt}"

    description += "\nGenerator: " + row["generator"]

    if len(description) > 0:
        description = wrap_text(description)  # Wrap text
    else:
        description = "No description"

    return description


__all__ = ["visualize_dataset_rows"]
