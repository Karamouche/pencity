
from ultralytics import YOLO

def train_model():
    # Load the model.
    model = YOLO('yolov8n.pt')
    
    # Training.
    results = model.train(
       data='data/pencity.yaml',
       imgsz=416,
       epochs=50,
       batch=16,
       name='yolov8n_custom')

if __name__ == '__main__':
    train_model()
