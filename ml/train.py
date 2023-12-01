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


def train(
    train_set: Dataset,
    test_set: Dataset,
    class_names: List[str],
    model_name: str = "yolov5n6",
    eval_every_n_steps=1024,
    batch_size=8,
):
    output_dir = (f"./{model_name}",)
    BACKGROUND_SIZE = 416

    def build_batch(
        dataset: Dataset, batchsize: int = 8
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        N_ITEMS = 16
        batch_images = []
        batch_tensors = []
        for i in tqdm(range(batchsize), desc="Building batch"):
            images = dataset.shuffle().select(range(N_ITEMS))
            background = np.zeros((BACKGROUND_SIZE, BACKGROUND_SIZE))
            list_of_tensors = []
            for img in images:
                label = img["label"]
                img = np.array(img["image"])
                # resize the image randomly between x1 and x4
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

    # build a batch to check if everything is working
    # batch_images, batch_tensors = build_batch(train_set, batchsize=batch_size)
    # save_tensors(batch_images, batch_tensors)

    def load_model(model_name: str) -> torch.hub:
        # cd to ml
        os.chdir(os.path.dirname(__file__))
        model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
        return model

    model = load_model(model_name)
    print("Model loaded")

    def train_model(model: torch.hub, train_set: Dataset, num_epochs: int):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        for epoch in range(num_epochs):
            running_loss = 0.0
            images, targets = build_batch(train_set, batchsize=batch_size)
            for i, (image, target) in enumerate(zip(images, targets)):
                # convert image to torch tensor
                image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
                optimizer.zero_grad()
                outputs = model(image)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % eval_every_n_steps == eval_every_n_steps - 1:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_set)}], Loss: {running_loss/eval_every_n_steps}"
                    )
                    running_loss = 0.0
        # save the model
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
        print("Finished Training")

    train_model(model, train_set, num_epochs=10)


if __name__ == "__main__":
    train_set, test_set, labels = build_dataset()
    train(train_set, test_set, labels, batch_size=2)
