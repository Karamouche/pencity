import cv2
import torch
import argparse
import random as rd
from ultralytics import YOLO as YOLOv8

from build_dataset import PROJECT_LABELS

# set confidence threshold
CONFIDENCE_TRESHOLD = 0.5
# assign a random color for each label
COLOR_PER_LABEL = [
    (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
    for _ in range(len(PROJECT_LABELS))
]


def load_model(weight_path: str, model_version: str = "v5") -> torch.hub or YOLOv8:
    print(f"Loading YOLO{model_version} model from {weight_path.split('/')[-1]}")
    if model_version == "v5":
        model = torch.hub.load("ultralytics/yolov5", "custom", path=weight_path)
    elif model_version == "v8":
        # Load a pretrained YOLOv8n model
        model = YOLOv8(weight_path)
    else:
        raise ValueError(f"Model version {model_version} not supported")
    return model


def preprocess_cam(frame: cv2.Mat) -> cv2.Mat:
    # to grayscaleYOLOv5
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # reverse color
    frame = cv2.bitwise_not(frame)
    # to black and white
    _, frame = cv2.threshold(frame, 160, 255, cv2.THRESH_BINARY)
    # remove black noise
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, (5, 5))
    # dilate to make curves thicker
    struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # frame = cv2.dilate(frame, struct_element)

    return frame


def draw_pred(frame: cv2.Mat, results, model_version: str = "v5") -> cv2.Mat:
    for element in results.pred[0].tolist() if model_version == "v5" else results[0]:
        # draw rectangle on frame
        if model_version == "v5":
            x1, y1, x2, y2 = [int(value) for value in element[0:4]]
        elif model_version == "v8":
            x1, y1, x2, y2 = [int(value) for value in element.boxes.xyxy[0]]
        else:
            raise ValueError(f"Model version {model_version} not supported")

        # get boxes properties
        if model_version == "v5":
            label = int(element[5])
            conf_score = element[4]
        elif model_version == "v8":
            label = int(element.boxes.cls.tolist()[0])
            conf_score = element.boxes.conf.tolist()[0]
        label_name = f"{PROJECT_LABELS[label]} {conf_score:.2f}"

        # draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PER_LABEL[label], 2)

        # draw a background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - 15),
            (x1 + len(label_name) * 7, y1),
            COLOR_PER_LABEL[label],
            -1,
        )
        # Write label
        cv2.putText(
            frame,
            label_name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    return frame


def get_results_v5(model: torch.hub, frame: cv2.Mat):
    # apply preprocessing on frame
    frame_bin = preprocess_cam(frame)
    if model_version == "v8":
        # convert to rgb
        frame_bin = cv2.cvtColor(frame_bin, cv2.COLOR_GRAY2RGB)

    # Perform inference on the frame
    results = model(frame_bin)
    # Apply confidence threshold
    confiend_elements = [
        good for good in results.pred[0].tolist() if good[4] > CONFIDENCE_TRESHOLD
    ]
    results.pred[0] = torch.tensor(confiend_elements)
    return results


def get_results_v8(model: YOLOv8, frame: cv2.Mat):
    # apply preprocessing on frame
    frame_bin = preprocess_cam(frame)
    frame_bin = cv2.cvtColor(frame_bin, cv2.COLOR_GRAY2RGB)

    # Perform inference on the frame
    results = model(frame_bin, conf=CONFIDENCE_TRESHOLD)
    return results


def show_result(frame: cv2.Mat, results, model_version) -> None:
    if model_version == "v5":
        cv2.imshow("Drawlo-bin", results.render()[0])
    elif model_version == "v8":
        cv2.imshow("Drawlo-bin", results[0].plot())
    cv2.imshow("Drawlo", draw_pred(frame, results, model_version=model_version))


def yolo_inference(weight_path: str, model_version: str) -> None:
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Load the model
    model = load_model(weight_path, model_version=model_version)

    while True:
        # Read frames from the webcam
        ret, frame = cap.read()
        # Check if frame is empty
        if not ret:
            break

        # reverse the image
        frame = cv2.flip(frame, 1)

        # apply preprocessing on frame
        frame_bin = preprocess_cam(frame)

        # Perform inference on the frame
        if model_version == "v5":
            results = get_results_v5(model, frame)
        elif model_version == "v8":
            results = get_results_v8(model, frame)
        else:
            raise ValueError(f"Model version {model_version} not supported")

        # Display the results
        show_result(frame, results, model_version)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) == ord("q"):
            break

        # Delay so it works 24 fps
        # time.sleep(0.05)

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--weight_path", type=str, default="ml/weights/v5/hugo_10epoch_failes.pt"
    )
    args = parser.parse_args()

    model_version = args.weight_path.split("/")[-2]

    yolo_inference(args.weight_path, model_version=model_version)
