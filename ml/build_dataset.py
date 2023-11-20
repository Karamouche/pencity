from datasets import load_dataset, load_from_disk, Dataset
from typing import Tuple, List

DATASET_PATH = "dataset/pencity/quickdraw"


def build_dataset() -> Tuple[Dataset, List[str]]:
    dataset = load_from_disk(DATASET_PATH)
    list_labels = [
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
        "stairs",
        "tent",
        "The Eiffel Tower",
        "tractor",
        "traffic light",
        "train",
        "tree",
        "van",
    ]
    return dataset.shuffle(), list_labels


def preprocess_dataset() -> None:
    try:
        dataset = load_from_disk(DATASET_PATH)
        print("Dataset already exists")
        return
    except:
        print("Dataset not found, creating it")
    dataset = load_dataset("Xenova/quickdraw", split="train")
    list_labels = dataset.features["label"].names
    labels_keep = [
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
        "stairs",
        "tent",
        "The Eiffel Tower",
        "tractor",
        "traffic light",
        "train",
        "tree",
        "van",
    ]

    print("Before filtering: ", len(dataset))
    labels_keep_index = [list_labels.index(label) for label in labels_keep]
    dataset = dataset.filter(lambda example: example["label"] in labels_keep_index)
    print("After filtering: ", len(dataset))

    # replace label index by index in labels_keep
    dataset = dataset.map(
        lambda example: {"label": labels_keep.index(list_labels[example["label"]])}
    )
    dataset.features["label"].names = labels_keep

    # save dataset
    dataset.save_to_disk(DATASET_PATH)
    print("Dataset saved to disk")


def test_dataset(dataset):
    # plot some images with their labels
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            index = np.random.randint(len(dataset))
            img = dataset[index]["image"]
            label = dataset[index]["label"]
            axs[i, j].imshow(img)
            axs[i, j].set_title(labels[label])
            axs[i, j].axis("off")
    plt.show()


if __name__ == "__main__":
    preprocess_dataset()
    dataset, labels = build_dataset()
    print("Labels :")
    print(labels)
    test_dataset(dataset)
