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
    BACKGROUND_SIZE = 640

    def build_batch(
        dataset: Dataset, batchsize: int = 8
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        N_ITEMS = 16
        batch_images = []
        batch_tensors = []
        for i in tqdm(range(batchsize), desc="Building batches"):
            images = dataset.shuffle().select(range(N_ITEMS))
            background = np.zeros((BACKGROUND_SIZE, BACKGROUND_SIZE))
            list_of_tensors = []
            for img in images:
                label = img["label"]
                img = np.array(img["image"])
                # resize the image randomly between x1 and x4
                H, W = img.shape
                img_ratio = float(W) / float(H)
                W = rd.randint(W, W * 4)
                H = int(W / img_ratio)
                img = cv2.resize(
                    img,
                    (W, H),
                    interpolation=cv2.INTER_CUBIC,
                )
                assert img.shape == (H, W)
                x = rd.randint(0, BACKGROUND_SIZE - W)
                y = rd.randint(0, BACKGROUND_SIZE - H)
                # be sure that the image is not overlapping another one
                while np.any(background[y : y + H, x : x + W]):
                    x = rd.randint(0, BACKGROUND_SIZE - W)
                    y = rd.randint(0, BACKGROUND_SIZE - H)
                background[y : y + H, x : x + W] = img
                yolo_tensor = from_coords_to_yolo(
                    img, x, y, label, BACKGROUND_SIZE, class_names
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
        # save batch tensors in batch/tensors folder
        if not os.path.exists(os.path.join(batch_folder, "tensors")):
            os.makedirs(os.path.join(batch_folder, "tensors"))
        for i, tensors in enumerate(batch_tensors):
            with open(
                os.path.join(batch_folder, "tensors", f"tensor{i}.yml"), "w"
            ) as f:
                for element in tensors:
                    # save in format Pc x y w h c1 c2 ... cn
                    for propertie in element.tolist():
                        f.write(f"{propertie} ")
                    f.write("\n")
        print("Batch saved")

    batch_images, batch_tensors = build_batch(train_set, batchsize=4)
    save_tensor(batch_images, batch_tensors)


if __name__ == "__main__":
    train_set, test_set, labels = build_dataset()
    train(train_set, test_set, labels)
