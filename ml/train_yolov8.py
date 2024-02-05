from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name_folder", help="Add repository name for yolo v8 results training.", default="yolov8n_custom")
    args = parser.parse_args()
 
    # Load the model.
    model = YOLO('yolov8n.pt')
 
    # Training.
    results = model.train(
       data='ml/data/pencity.yaml',
       imgsz=640,
       epochs=3,
       batch=16,
       name=args.name_folder)

if __name__ == '__main__':
    main()
