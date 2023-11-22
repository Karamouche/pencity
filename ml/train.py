from datasets import Dataset
from typing import List, Tuple
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import random as rd
import numpy as np

from build_dataset import build_dataset


def train(
    train_set: Dataset,
    test_set: Dataset,
    class_names: List[str],
    model_name: str = "yolov8n",
    eval_every_n_steps=1024,
):
    output_dir = (f"./{model_name}",)

    def build_batch(
        dataset: Dataset, batchsize: int = 8
    ) -> Tuple[np.ndarray, torch.Tensor]:
        N_ITEMS = 16
        IMAGE_SIZE = 416
        batch_images = []
        batch_tensors = []
        for i in tqdm(range(batchsize), desc="Building batches"):
            images = dataset.shuffle().select(range(N_ITEMS))
            background = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
            list_of_tensors = []
            for img in images:
                label = img["label"]
                img = np.array(img["image"])
                x = rd.randint(0, IMAGE_SIZE - img.shape[0])
                y = rd.randint(0, IMAGE_SIZE - img.shape[1])
                background[x : x + img.shape[0], y : y + img.shape[1]] = img
                # Create YOLO tensor values
                Bx = (x + img.shape[0] / 2) / IMAGE_SIZE
                By = (y + img.shape[1] / 2) / IMAGE_SIZE
                Bh = img.shape[0] / IMAGE_SIZE
                Bw = img.shape[1] / IMAGE_SIZE
                # Create YOLO tensor
                yolo_tensor = torch.zeros(len(class_names) + 5)
                yolo_tensor[0] = 1  # Set Pc to 1
                yolo_tensor[1] = Bx
                yolo_tensor[2] = By
                yolo_tensor[3] = Bh
                yolo_tensor[4] = Bw
                yolo_tensor[label + 5] = 1  # Set class label to 1
                list_of_tensors.append(yolo_tensor)
            batch_images.append(background)
            batch_tensors.append(torch.stack(list_of_tensors))
        print("Batch built")
        return batch_images, batch_tensors

    batch_images, batch_tensors = build_batch(train_set, batchsize=4)
    # saves images in ./images folder
    for i, img in enumerate(batch_images):
        plt.imsave(f"ml/images/{i}.png", img, cmap="gray")
    # save tensors in tensor.yml folder
    with open("ml/tensors.yml", "w") as f:
        f.write("tensors:\n")
        for i, tensor in enumerate(batch_tensors):
            f.write(f"  - tensor_{i}:\n")
            f.write(f"      - {tensor.tolist()}\n")


if __name__ == "__main__":
    train_set, test_set, labels = build_dataset()
    train(train_set, test_set, labels)
