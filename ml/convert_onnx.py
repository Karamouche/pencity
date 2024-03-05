from ultralytics import YOLO


model = YOLO('runs/detect/test_90000/weights/best.pt')

model.export(format="onnx", imgsz=[640, 640], opset=10)

#launch the code from the racine folder
# get the onnx file in the folder of the model