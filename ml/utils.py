from typing import List
import torch
import numpy as np


def from_yolo_to_coords(tensor: torch.Tensor, image_size: int) -> List[int]:
    """
        Converts a YOLO tensor to bounding box coordinates.
    import random as rd
    i
        Parameters:
        - tensor (torch.Tensor): YOLO tensor [Pc, Bx, By, Bh, Bw, c1, c2, ..., cn]
        - image_size (int): Size of the input image

        Returns:
        - List[int]: Bounding box coordinates [x, y, h, w]
    """

    # Unpack tensor values for better readability
    bx = tensor[1]
    by = tensor[2]
    bh = tensor[3]
    bw = tensor[4]

    # Calculate bounding box coordinates
    x = int((bx - bw / 2) * image_size)
    y = int((by - bh / 2) * image_size)
    h = int(bh * image_size)
    w = int(bw * image_size)

    return x, y, h, w


def from_coords_to_yolo(
    img: np.ndarray, x: int, y: int, label: int, image_size: int, class_names: List[str]
) -> torch.Tensor:
    """
    Creates a YOLO tensor representing bounding box information and class labels.

    Parameters:
    - img (np.ndarray): Image array
    - label (int): Class label
    - image_size (int): Size of the input image
    - class_names (List[str]): List of class names

    Returns:
    - torch.Tensor: YOLO tensor [Pc, Bx, By, Bh, Bw, c1, c2, ..., cn]
    """

    H, W = img.shape

    # Calculate YOLO tensor values
    Bx = (x + W / 2) / image_size
    By = (y + H / 2) / image_size
    Bh = H / image_size
    Bw = W / image_size

    # Create YOLO tensor
    yolo_tensor = torch.zeros(len(class_names) + 5)
    yolo_tensor[0] = 1  # Set Pc to 1
    yolo_tensor[1] = Bx
    yolo_tensor[2] = By
    yolo_tensor[3] = Bh
    yolo_tensor[4] = Bw
    yolo_tensor[label + 5] = 1  # Set class label to 1

    return yolo_tensor
