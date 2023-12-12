# Machine Learning

Here is the machine learning folder for the PenCity project. The python version used is 3.11. \
To get the repo to train YOLOv5, you need to clone the repo `https://github.com/ultralytics/yolov5` and install the requirements.txt file.

For training, you need to run the following command : \

-   For YOLOv5n :
    `python ml/yolov5/train.py --data ml/data/pencity.yaml --project "pencity" --img 640 --batch 128 --epochs 100 --weights "" --cfg ml/yolov5/models/yolov5n.yaml --cache ram`

-   For YOLOv8n :
    `python yolo detect train data=ml/data/pencity.yaml model=yolov8n.yaml epochs=100 imgsz=640`
