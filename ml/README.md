# Machine Learning

Here is the machine learning folder for the PenCity project. The python version used is 3.11. \
To get the repo to train YOLOv5, you need to clone the repo `https://github.com/ultralytics/yolov5` and install the requirements.txt file.

for training, you need to run the following command : \
`python ml/yolov5/train.py --img 416 --batch 64 --epochs 3 --data ml/data/pencity.yaml --weights "" --cfg ml/yolov5/models/yolov5n.yaml`
