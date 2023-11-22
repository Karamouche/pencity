from datasets import Dataset
from typing import List, Tuple
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import cv2

from build_dataset import build_dataset
from utils import from_coords_to_yolo, from_yolo_to_coords


def train(
    train_set: Dataset,
    test_set: Dataset,
    class_names: List[str],
    model_name: str = "yolov8n",
    eval_every_n_steps=1024,
):
    output_dir = (f"./{model_name}",)
    IMAGE_SIZE = 416

    def build_batch(
        dataset: Dataset, batchsize: int = 8
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        N_ITEMS = 16
        batch_images = []
        batch_tensors = []
        for i in tqdm(range(batchsize), desc="Building batches"):
            images = dataset.shuffle().select(range(N_ITEMS))
            background = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
            list_of_tensors = []
            for img in images:
                label = img["label"]
                img = np.array(img["image"])
                H, W = img.shape
                x = rd.randint(0, IMAGE_SIZE - W)
                y = rd.randint(0, IMAGE_SIZE - H)
                background[x : x + img.shape[0], y : y + img.shape[1]] = img
                yolo_tensor = from_coords_to_yolo(
                    img, x, y, label, IMAGE_SIZE, class_names
                )
                list_of_tensors.append(yolo_tensor)
            batch_images.append(background)
            batch_tensors.append(torch.stack(list_of_tensors))
        print("Batch built")
        return batch_images, batch_tensors

    def save_tensor(
        batch_images: List[np.ndarray], batch_tensors: List[torch.Tensor]
    ) -> None:
        batch_folder = os.path.join(os.path.dirname(__file__), "batch")
        # check if batch folder exists
        if not os.path.exists(batch_folder):
            os.makedirs(batch_folder)
        # save batch images in batch/images folder
        if not os.path.exists(os.path.join(batch_folder, "images")):
            os.makedirs(os.path.join(batch_folder, "images"))
        for i, img in enumerate(batch_images):
            plt.imsave(
                os.path.join(batch_folder, "images", f"image{i}.png"), img, cmap="gray"
            )
        with open(os.path.join(batch_folder, "tensor.yml"), "w") as f:
            f.write("tensors:\n")
            for i, tensor in enumerate(batch_tensors):
                f.write(f"  - tensor_{i}:\n")
                f.write(f"      - {tensor.tolist()}\n")
        print("Batch saved")

    batch_images, batch_tensors = build_batch(train_set, batchsize=4)
    new_batch_images = []
    # draw bounding boxes on each image in batch_images
    for tensor, img in zip(batch_tensors, batch_images):
        for i in range(len(tensor)):
            if tensor[i][0] == 1:
                x, y, h, w = from_yolo_to_coords(tensor[i], IMAGE_SIZE)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        new_batch_images.append(img)
    save_tensor(new_batch_images, batch_tensors)


if __name__ == "__main__":
    train_set, test_set, labels = build_dataset()
    train(train_set, test_set, labels)
