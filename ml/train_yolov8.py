from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='data_yolov5s.yaml',
   imgsz=416,
   epochs=10,
   batch=16,
   name='yolov8n_custom')