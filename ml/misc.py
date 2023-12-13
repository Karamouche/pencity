from typing import List
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def from_coords_to_yolo(
    img: np.ndarray,
    x: int,
    y: int,
    label: int,
    BACKGROUND_SIZE: int,
    class_names: List[str],
) -> torch.Tensor:
    """
    Creates a YOLO tensor representing bounding box information and class labels.

    Parameters:
    - img (np.ndarray): Image array
    - label (int): Class label
    - BACKGROUND_SIZE (int): Size of the input image
    - class_names (List[str]): List of class names

    Returns:
    - torch.Tensor: YOLO tensor [Cn, Bx, By, Bh, Bw]
    """

    H, W = img.shape

    # Calculate class number
    Cn = label

    # Calculate YOLO tensor values
    Bx = (x + W / 2) / BACKGROUND_SIZE
    By = (y + H / 2) / BACKGROUND_SIZE
    Bh = H / BACKGROUND_SIZE
    Bw = W / BACKGROUND_SIZE

    # Create YOLO tensor
    yolo_tensor = torch.zeros(5)
    yolo_tensor[0] = Cn
    yolo_tensor[1] = Bx
    yolo_tensor[2] = By
    yolo_tensor[3] = Bh
    yolo_tensor[4] = Bw

    return yolo_tensor


def from_yolo_to_coords(tensor: torch.Tensor, BACKGROUND_SIZE: int) -> List[int]:
    """
    Converts a YOLO tensor to bounding box coordinates.

    Parameters:
    - tensor (torch.Tensor): YOLO tensor [Cn, Bx, By, Bh, Bw]
    - BACKGROUND_SIZE (int): Size of the input image

    Returns:
    - List[int]: Bounding box coordinates [x, y, h, w]
    """

    # Unpack tensor values for better readability
    bx = tensor[1]
    by = tensor[2]
    bh = tensor[3]
    bw = tensor[4]

    # Calculate bounding box coordinates
    x = int((bx - bw / 2) * BACKGROUND_SIZE)
    y = int((by - bh / 2) * BACKGROUND_SIZE)
    h = int(bh * BACKGROUND_SIZE)
    w = int(bw * BACKGROUND_SIZE)

    return x, y, h, w


def build_save_folders(split: str) -> None:
    batch_folder = os.path.join(os.path.dirname(__file__), "data", "batch", split)
    # check if batch folder exists
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
    # save batch images in batch/images folder
    if not os.path.exists(os.path.join(batch_folder, "images")):
        os.makedirs(os.path.join(batch_folder, "images"))
    else:
        for file in os.listdir(os.path.join(batch_folder, "images")):
            os.remove(os.path.join(batch_folder, "images", file))
    # save batch labels in batch/labels folder
    if not os.path.exists(os.path.join(batch_folder, "labels")):
        os.makedirs(os.path.join(batch_folder, "labels"))
    else:
        for file in os.listdir(os.path.join(batch_folder, "labels")):
            os.remove(os.path.join(batch_folder, "labels", file))


def save_data(
    image: np.ndarray,
    tensors: torch.Tensor,
    split: str = "train",
) -> None:
    batch_folder = os.path.join(os.path.dirname(__file__), "data", "batch", split)
    index = len(os.listdir(os.path.join(batch_folder, "images")))
    assert len(os.listdir(os.path.join(batch_folder, "images"))) == len(
        os.listdir(os.path.join(batch_folder, "labels"))
    )
    plt.imsave(os.path.join(batch_folder, "images", f"{index}.png"), image, cmap="gray")
    with open(os.path.join(batch_folder, "labels", f"{index}.txt"), "w") as f:
        for element in tensors:
            # save in format [Cn, Bx, By, Bh, Bw]
            for j, propertie in enumerate(element.tolist()):
                if j == 0:
                    f.write(f"{int(propertie)} ")
                else:
                    f.write(f"{propertie} ")
            f.write("\n")
