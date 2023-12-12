from datasets import Dataset
from typing import List, Tuple
import os
import torch
from tqdm import tqdm
from PIL import Image
import random as rd
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import gc
import argparse

from build_dataset import build_dataset
from misc import from_coords_to_yolo, save_tensors


def train_processor(
    train_set: Dataset,
    test_set: Dataset,
    class_names: List[str],
    BACKGROUND_SIZE: int = 640,
    amount_per_label=3000,
    seed=0,
):
    def build_batch(
        dataset: Dataset, amount_per_label: int, seed: int = 0
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        batch_images = []
        batch_tensors = []
        # seed to reproduce the same batch
        if seed != 0:
            rd.seed(seed)

        def create_image_groups(max_images_per_group=16, min_images_per_group=8):
            label_images = {i: [] for i in range(len(class_names))}
            print(f"Taille du dataset: {len(dataset)}")
            picking_pbar = tqdm(
                range(amount_per_label * len(class_names)),
                desc=f"Picking {amount_per_label} drawings per label",
            )
            for item in dataset:
                if len(label_images[item["label"]]) < amount_per_label:
                    label_images[item["label"]].append(item)
                    picking_pbar.update(1)
                # Stop if all images have been picked
                if all(
                    len(images) >= amount_per_label for images in label_images.values()
                ):
                    break
            picking_pbar.close()

            # Flatten the list and shuffle
            print("Shuffling images")
            all_images = [image for images in label_images.values() for image in images]
            rd.shuffle(all_images)

            # Create groups of images
            print("Creating groups of images")
            image_groups = []
            groups_pbar = tqdm(range(len(all_images)))
            while len(all_images) >= min_images_per_group:
                group_size = rd.randint(
                    min_images_per_group, min(max_images_per_group, len(all_images))
                )
                group = []
                # pop images from the list and add them to the group
                for _ in range(group_size):
                    group.append(all_images.pop())

                # add the group to the list of groups
                image_groups.append(group)
                groups_pbar.update(group_size)
            groups_pbar.close()

            return image_groups

        def create_image(images) -> np.ndarray:
            background = np.zeros((BACKGROUND_SIZE, BACKGROUND_SIZE))
            list_of_tensors = []
            for img in images:
                label = img["label"]
                img = np.array(img["image"])
                # resize the image randomly between x1.5 and x3
                H, W = img.shape
                img_ratio = float(W) / float(H)
                W = rd.randint(W * 1.5, W * 3)
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
            return background, torch.stack(list_of_tensors)

        image_groups = create_image_groups()
        print(f"Nombre de groupes : {len(image_groups)}")
        for groups in tqdm(image_groups, desc="Creating images"):
            background, list_of_tensors = create_image(groups)
            batch_images.append(background)
            batch_tensors.append(list_of_tensors)
        return batch_images, batch_tensors

    # build and save dataset
    print("Building train data")
    train_batch_images, train_batch_tensors = build_batch(
        train_set, amount_per_label=amount_per_label, seed=seed
    )
    save_tensors(train_batch_images, train_batch_tensors, split="train")
    print("\nBuilding test data")
    test_batch_images, test_batch_tensors = build_batch(
        test_set, amount_per_label=int(amount_per_label / 4), seed=seed
    )
    save_tensors(test_batch_images, test_batch_tensors, split="test")
    print("\nDataset built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",  # --dataset_size
        "--size_per_label",
        type=int,
        default=512,
        help="How many images per label in the dataset",
    )
    parser.add_argument(
        "-r",  # --dataset_size
        "--random_seed",
        type=int,
        default=0,
        help="Seed to reproduce the same dataset",
    )
    args = parser.parse_args()
    train_set, test_set, labels = build_dataset()
    train_processor(
        train_set,
        test_set,
        labels,
        amount_per_label=args.size_per_label,
        seed=args.random_seed,
    )
