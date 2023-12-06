from datasets import Dataset
from typing import List, Tuple
import os
import torch
from tqdm import tqdm
from PIL import Image
import random as rd
import numpy as np
import cv2

from build_dataset import build_dataset
from misc import from_coords_to_yolo, save_tensors


def train_processor(
    train_set: Dataset,
    test_set: Dataset,
    class_names: List[str],
    BACKGROUND_SIZE: int = 416,
    dataset_size=512,
    seed=0,
):
    def build_batch(
        dataset: Dataset, amount: int = 8, seed: int = 0
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        N_ITEMS = 16
        batch_images = []
        batch_tensors = []
        # seed to reproduce the same batch
        if seed != 0:
            rd.seed(seed)
        for _ in tqdm(range(amount), desc="Building batch"):
            images = dataset.shuffle().select(range(N_ITEMS))
            # if there not at list one item per class, reshuffle and retry
            while len(set([img["label"] for img in images])) < len(class_names):
                images = dataset.shuffle().select(range(N_ITEMS))
            background = np.zeros((BACKGROUND_SIZE, BACKGROUND_SIZE))
            list_of_tensors = []
            for img in images:
                label = img["label"]
                img = np.array(img["image"])
                # resize the image randomly between x1 and x2.5
                H, W = img.shape
                img_ratio = float(W) / float(H)
                W = rd.randint(W, W * 2.5)
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
        return batch_images, batch_tensors

    # build and save dataset
    print("Building train data")
    train_batch_images, train_batch_tensors = build_batch(
        train_set, amount=dataset_size, seed=seed
    )
    save_tensors(train_batch_images, train_batch_tensors, split="train")
    print("Building test data")
    test_batch_images, test_batch_tensors = build_batch(
        test_set, amount=int(dataset_size / 4), seed=seed
    )
    save_tensors(test_batch_images, test_batch_tensors, split="test")
    print("Dataset built")


if __name__ == "__main__":
    train_set, test_set, labels = build_dataset()
    train_processor(train_set, test_set, labels, dataset_size=4096, seed=42)
