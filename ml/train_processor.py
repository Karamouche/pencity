from datasets import Dataset
from typing import List, Tuple
import torch
from tqdm import tqdm
from PIL import Image
import random as rd
import numpy as np
import cv2
import argparse

from build_dataset import build_dataset
from misc import from_coords_to_yolo, save_data, build_save_folders


def train_processor(
    train_set: Dataset,
    test_set: Dataset,
    class_names: List[str],
    BACKGROUND_SIZE: int = 640,
    amount_per_label=3000,
    seed=0,
):
    def build_batch(
        dataset: Dataset, split: str, amount_per_label: int, seed: int = 0
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """
        Create a batch of images (canvas) and save it
        """
        batch_images = []
        batch_tensors = []
        # seed to reproduce the same batch
        if seed != 0:
            rd.seed(seed)

        def create_image_groups(min_images_per_group=8, max_images_per_group=16):
            """
            Create groups of images to create a batch

            Args:
                min_images_per_group (int, optional): Minimum number of images per group. Defaults to 8.
                max_images_per_group (int, optional): Maximum number of images per group. Defaults to 16.
            """
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
            """
            Create an image (canvas) from a list of images
            """
            background = np.zeros((BACKGROUND_SIZE, BACKGROUND_SIZE))
            list_of_tensors = []
            for img in images:
                label = img["label"]
                img = np.array(img["image"])
                # resize the image randomly between x2.5 and x4.5
                H, W = img.shape
                img_ratio = float(W) / float(H)
                W = rd.randint(W * 2.5, W * 4.5)
                H = int(W / img_ratio)
                img = cv2.resize(
                    img,
                    (W, H),
                    interpolation=cv2.INTER_CUBIC,
                )
                # rotate the image randomly between -20° and 20°
                angle = rd.randint(-20, 20)
                M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1)
                img = cv2.warpAffine(img, M, (W, H))
                # binarize the image
                img = np.where(img > 178, 255, 0).astype(np.uint8)
                assert img.shape == (H, W)
                x = rd.randint(0, BACKGROUND_SIZE - W)
                y = rd.randint(0, BACKGROUND_SIZE - H)
                # be sure that the image is not overlapping another one
                overlap_count = 0
                do_past = True
                while np.any(background[y : y + H, x : x + W]):
                    x = rd.randint(0, BACKGROUND_SIZE - W)
                    y = rd.randint(0, BACKGROUND_SIZE - H)
                    overlap_count += 1
                    if overlap_count > 50:
                        do_past = False
                        break
                if do_past:
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
            save_data(background, list_of_tensors, split=split)

    # build and save dataset
    print("Building train data")
    build_save_folders(split="train")
    build_batch(train_set, split="train", amount_per_label=amount_per_label, seed=seed)

    print(f"Batch for train saved")
    print("\nBuilding test data")
    build_save_folders(split="test")
    build_batch(
        test_set, split="test", amount_per_label=int(amount_per_label / 4), seed=seed
    )

    print(f"Batch for test saved")
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
