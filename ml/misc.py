from typing import List
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


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
    - torch.Tensor: YOLO tensor [Pc, Bx, By, Bh, Bw, c1, c2, ..., cn]
    """

    H, W = img.shape

    # Calculate YOLO tensor values
    Bx = (x + W / 2) / BACKGROUND_SIZE
    By = (y + H / 2) / BACKGROUND_SIZE
    Bh = H / BACKGROUND_SIZE
    Bw = W / BACKGROUND_SIZE

    # Create YOLO tensor
    yolo_tensor = torch.zeros(len(class_names) + 5)
    yolo_tensor[0] = 1  # Set Pc to 1
    yolo_tensor[1] = Bx
    yolo_tensor[2] = By
    yolo_tensor[3] = Bh
    yolo_tensor[4] = Bw
    yolo_tensor[label + 5] = 1  # Set class label to 1

    return yolo_tensor


def from_yolo_to_coords(tensor: torch.Tensor, BACKGROUND_SIZE: int) -> List[int]:
    """
    Converts a YOLO tensor to bounding box coordinates.

    Parameters:
    - tensor (torch.Tensor): YOLO tensor [Pc, Bx, By, Bh, Bw, c1, c2, ..., cn]
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


def save_tensors(
    batch_images: List[np.ndarray], batch_tensors: List[torch.Tensor]
) -> None:
    batch_folder = os.path.join(os.path.dirname(__file__), "batch")
    # check if batch folder exists
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
    # save batch images in batch/images folder
    if not os.path.exists(os.path.join(batch_folder, "images")):
        os.makedirs(os.path.join(batch_folder, "images"))
    else:
        for file in os.listdir(os.path.join(batch_folder, "images")):
            os.remove(os.path.join(batch_folder, "images", file))
    for i, img in enumerate(batch_images):
        plt.imsave(
            os.path.join(batch_folder, "images", f"image{i}.png"), img, cmap="gray"
        )
    # save batch tensors in batch/tensors folder
    if not os.path.exists(os.path.join(batch_folder, "tensors")):
        os.makedirs(os.path.join(batch_folder, "tensors"))
    else:
        for file in os.listdir(os.path.join(batch_folder, "tensors")):
            os.remove(os.path.join(batch_folder, "tensors", file))
    for i, tensors in enumerate(batch_tensors):
        with open(os.path.join(batch_folder, "tensors", f"tensor{i}.yml"), "w") as f:
            for element in tensors:
                # save in format Pc x y w h c1 c2 ... cn
                for propertie in element.tolist():
                    f.write(f"{propertie} ")
                f.write("\n")
    print("Batch saved")
