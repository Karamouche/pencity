# Machine Learning

Here is the machine learning folder for the PenCity project. The python version used is 3.11.

- Download the github yolo in the ml folder : https://github.com/ultralytics/yolov5/tree/master

in your terminal in the ml directory : 

```
python yolov5/train.py --img 416 --batch 64 --epochs 3 --data data_yolov5s.yaml --weights "" --cfg yolov5/models/yolov5s.yaml
```

or if you are using powershell : 
```
python yolov5/train.py --img 416 --batch 64 --epochs 3 --data data_yolov5s.yaml --weights [string]::Empty --cfg yolov5/models/yolov5s.yaml
```

#Train using Yolov8 
 - use the file train_yolov8.py to train