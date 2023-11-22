import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from typing import Tuple, List

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "pencity")
PROJECT_LABELS = [
    "house",
    "bicycle",
    "bridge",
    "bus",
    "car",
    "church",
    "firetruck",
    "garden",
    "hospital",
    "motorbike",
    "palm tree",
    "pickup truck",
    "police car",
    "river",
    "roller coaster",
    "rollerskates",
    "school bus",
    "skyscraper",
    "tent",
    "The Eiffel Tower",
    "tractor",
    "traffic light",
    "train",
    "tree",
    "van",
]


def build_dataset() -> Tuple[Dataset, Dataset, List[str]]:
    try:
        dataset = load_from_disk(DATASET_PATH)
        print("Dataset exists")
    except:
        print("Dataset not found, creating it")
        preprocess_dataset()
        dataset = load_from_disk(DATASET_PATH)
    return dataset["train"], dataset["test"], PROJECT_LABELS


def preprocess_dataset() -> None:
    dataset = load_dataset("Xenova/quickdraw", split="train")
    list_labels = dataset.features["label"].names

    print("Before filtering: ", len(dataset))
    labels_keep_index = [list_labels.index(label) for label in PROJECT_LABELS]
    dataset = dataset.filter(lambda example: example["label"] in labels_keep_index)
    print("After filtering: ", len(dataset))

    # replace label index by index in labels_keep
    dataset = dataset.map(
        lambda example: {"label": PROJECT_LABELS.index(list_labels[example["label"]])}
    )

    # split in train and test
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    # save dataset
    dataset.save_to_disk(DATASET_PATH)
    print("Dataset saved to disk")


def viz_dataset(dataset):
    # plot some images with their labels
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            index = np.random.randint(len(dataset))
            img = dataset[index]["image"]
            label = dataset[index]["label"]
            axs[i, j].imshow(img, cmap="gray")
            axs[i, j].set_title(labels[label])
            axs[i, j].axis("off")
    plt.show()


def values_count(train, test, labels):
    for name, dataset in [("train", train), ("test", test)]:
        print(f"There is {len(dataset)} in the {name}ing dataset")
        # show repartition of labels
        label_count = {}
        for example in tqdm(dataset, desc=f"Counting labels in {name}ing dataset"):
            label = example["label"]
            label_count[label] = label_count.get(label, 0) + 1
        print("Repartition of labels:")
        for label in label_count:
            print(f"{labels[label]}: {label_count[label]}")
        print("=================")


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--count", action="store_true")
parser.add_argument("-v", "--vizualise", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    train_set, test_set, labels = build_dataset()
    print("Labels :")
    print(labels)
    if args.count:
        print("=================")
        values_count(train_set, test_set, labels)
    if args.vizualise:
        viz_dataset(train_set)
