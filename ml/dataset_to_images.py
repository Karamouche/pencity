import random
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
import os
import json
import cv2
import numpy as np
from build_dataset import DATASET_PATH, PROJECT_LABELS


IMAGES_PER_LABELS = 3000

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

def create_dataset_folders():
    base_path = os.getcwd()  # Get current working directory
    dataset_path = os.path.join(base_path, 'dataset')
    subfolders = ['images_dataset/train/images',
                  'images_dataset/train/labels',
                  'images_dataset/test/images',
                  'images_dataset/test/labels']

    for subfolder in subfolders:
        path = os.path.join(dataset_path, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")


def load_dataset(path):
    try:
        dataset = load_from_disk(path)
        return dataset
    except Exception as e:
        raise FileNotFoundError(f"Dataset not found at {path}. Error: {e}")


def create_image_sublists(dataset, max_images_per_label=3000, max_images_per_group=8, min_images_per_group=4):
    label_images = defaultdict(list)
    labels_keep_index = [PROJECT_LABELS.index(label) for label in PROJECT_LABELS]
    print(labels_keep_index)
    print(len(dataset))
    for item in tqdm(dataset, desc='Create dict for each label : '):
        if item['label'] in labels_keep_index:
            if label_images[item['label']] != max_images_per_label :
                label_images[item['label']].append(item)

    # Flatten the list and shuffle
    all_images = [image for images in label_images.values() for image in images]
    random.shuffle(all_images)

    image_groups = []
    pbar = tqdm()
    while len(all_images) >= min_images_per_group:

        group_size = random.randint(min_images_per_group, min(max_images_per_group, len(all_images)))
        group = []

        for _ in range(group_size):
            for i, image in enumerate(all_images):
                group.append(image)
                del all_images[i]
                break

        image_groups.append(group)
        pbar.update(1) 

    return image_groups


def check_overlap(new_box, existing_boxes):
    for box in existing_boxes:
        if not (new_box[2] < box[0] or new_box[0] > box[2] or new_box[3] < box[1] or new_box[1] > box[3]):
            return True
    return False


def create_white_image(width, height):
    return Image.new("RGB", (width, height), "black")

def place_images(base_image, images_to_place, image_size=82):
    base_w, base_h = base_image.size
    annotations = []
    placed_boxes = []
    
    image_data = images_to_place["image"]
    image_label = images_to_place["label"]
    
    #list of data and label as tuple
    image_data_list = list(zip(image_data, image_label))

    for img, img_label in image_data_list:

        img_np = np.array(img)
        resized_img_np = cv2.resize(img_np, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(resized_img_np)

        while True:
            x = random.randint(0, base_w - image_size)
            y = random.randint(0, base_h - image_size)
            new_box = [x, y, x + image_size, y + image_size]
            
            if not check_overlap(new_box, placed_boxes):
                break

        base_image.paste(img, (x, y))
        placed_boxes.append(new_box)

        x_center, y_center = (x + image_size / 2) / base_w, (y + image_size / 2) / base_h
        norm_width, norm_height = image_size / base_w, image_size / base_h

        annotations.append(f"{img_label} {x_center} {y_center} {norm_width} {norm_height}")

    return base_image, annotations


if __name__ == "__main__":
    create_dataset_folders()

    #canvas size
    base_width, base_height = 416, 416

    try:
        dataset = load_dataset(DATASET_PATH)
    except FileNotFoundError as error:
        print(error)

    
    for type_data in ["train", "test"]:
        all_images = dataset[type_data]

        
        image_groups = create_image_sublists(all_images, max_images_per_label=3000, max_images_per_group=8, min_images_per_group=4)

        for index, image_group in enumerate(tqdm(image_groups)):
            base_img = create_white_image(base_width, base_height)
            result_img, annotations = place_images(base_img, image_group, image_size=82)

            result_img.save(IMAGES_PATH + f"/{type_data}/images/image_{index}.png")

            with open(IMAGES_PATH + f"/{type_data}/labels/image_{index}.txt", 'w') as f:
                f.write("\n".join(annotations))