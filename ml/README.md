# Machine Learning

Here is the machine learning folder for the PenCity project. The python version used is 3.11. \
To get the repo to train YOLOv5, you need to clone the repo `https://github.com/ultralytics/yolov5` and install the requirements.txt file.

for training, you need to run the following command : \
`python ml/yolov5/train.py --data ml/data/pencity.yaml --img 416 --batch 128 --epochs 5 --weights "" --cfg ml/yolov5/models/yolov5n.yaml --cache ram`
