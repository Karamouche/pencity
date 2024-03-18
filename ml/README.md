# Machine Learning

Here is the machine learning folder for the PenCity project. The python version used is 3.11. \
To get the repo to train YOLOv5, you need to clone the repo `https://github.com/ultralytics/yolov5` and install the requirements.txt file.

To build the dataset canvas, you need to execute the following command : `python ml/train_processor.py -s 16 -r 42` where s is the number of images per labels and r is the random seed.

For training, you need to run the following command : \

-   For YOLOv3-tiny :

    Clone repo and install requirements.txt in a Python>=3.7.0 environment, including PyTorch>=1.7.

    ```bash
    git clone https://github.com/ultralytics/yolov3  # clone
    cd yolov3
    pip install -r requirements.txt  # install
    ```

    Then, in ./ml, execute `python ml/yolov3/train.py --data ml/data/pencity.yaml --epochs 3 --img 640 --weights '' --cfg ml/yolov3/models/yolov3-tiny.yaml  --batch-size 16`

-   For YOLOv5n :
    `python ml/yolov5/train.py --data ml/data/pencity.yaml --project "pencity" --img 640 --batch 128 --epochs 100 --weights "" --cfg ml/yolov5/models/yolov5n.yaml --cache ram`

-   For YOLOv8n :
    `yolo detect train data=ml/data/pencity.yaml model=yolov8n.yaml epochs=100 imgsz=640`

### Inference

To run inference, your weights must be in the `ml/weights/v{8 or 5}` folder. \
Then, you can run the following command : `python ml/inference.py -w [weight_path]`
