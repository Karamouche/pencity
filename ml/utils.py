from typing import List
import torch
import numpy as np


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
