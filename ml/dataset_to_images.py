import random
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk
import os
import json
import cv2
import numpy as np
from build_dataset import DATASET_PATH

IMAGES_PATH = os.path.join(os.path.dirname(__file__), "dataset", "images_dataset")

"""
REQUIREMENTS:
- space on your disk : 1.5 GB

in the dataset folder, create folders following this structure:

dataset
├── images_dataset
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── test
│       ├── images
│       └── labels

"""

def check_overlap(new_box, existing_boxes):
    for box in existing_boxes:
        if not (new_box[2] < box[0] or new_box[0] > box[2] or new_box[3] < box[1] or new_box[1] > box[3]):
            return True
    return False


def create_white_image(width, height):
    return Image.new("RGB", (width, height), "white")

def place_images(base_image, images_to_place, image_size=82):
    base_w, base_h = base_image.size
    annotations = []
    placed_boxes = []
    
    image_data = images_to_place["image"]
    image_label = images_to_place["label"]
    
    #list of data and label as tuple
    image_data_list = list(zip(image_data, image_label))

    for img, img_label in image_data_list:
        
        # works around the image to improve sharpness while enlarging it
        
        # probleme des traits qui ont des trous lors du resize
        img = img.resize((image_size, image_size), Image.LANCZOS) 
        img_np = np.array(img)

        threshold = 128
        _, binary_np = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)

        binary = Image.fromarray(binary_np)

        while True:
            x = random.randint(0, base_w - image_size)
            y = random.randint(0, base_h - image_size)
            new_box = [x, y, x + image_size, y + image_size]
            
            if not check_overlap(new_box, placed_boxes):
                break

        base_image.paste(img, (x, y), img)
        placed_boxes.append(new_box)

        x_center, y_center = (x + image_size / 2) / base_w, (y + image_size / 2) / base_h
        norm_width, norm_height = image_size / base_w, image_size / base_h

        annotations.append(f"{img_label} {x_center} {y_center} {norm_width} {norm_height}")

    return base_image, annotations


if __name__ == "__main__":
    base_width, base_height = 416, 416

    dataset = load_from_disk(DATASET_PATH)
    
    for type_data in ["train", "test"]:
        all_images = dataset[type_data]
        
        image_groups = [all_images[i:i+random.randint(4, 8)] for i in range(0, len(all_images), random.randint(4, 8))]

        for index, image_group in enumerate(tqdm(image_groups)):
            base_img = create_white_image(base_width, base_height)
            result_img, annotations = place_images(base_img, image_group, image_size=82)

            result_img.save(IMAGES_PATH + f"/{type_data}/images/image_{index}.png")

            with open(IMAGES_PATH + f"/{type_data}/labels/image_{index}.txt", 'w') as f:
                f.write("\n".join(annotations))