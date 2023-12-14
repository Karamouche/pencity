import cv2
import torch
import argparse
import time

from build_dataset import PROJECT_LABELS

CONFIDENCE_TRESHOLD = 0.75


def load_model(weight_path, name="ultralytics/yolov5"):
    model = torch.hub.load(name, "custom", path=weight_path)
    return model


def preprocess_cam(frame):
    # to grayscaleYOLOv5
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # reverse color
    frame = cv2.bitwise_not(frame)
    # to black and white
    _, frame = cv2.threshold(frame, 135, 255, cv2.THRESH_BINARY)
    # remove black noise
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, (6, 6))
    # dilate to make curves thicker
    frame = cv2.dilate(frame, (5, 5))
    # lisser les contours
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return frame


def draw_pred(frame, results):
    for element in results.pred[0].tolist():
        # draw rectangle on frame
        x1, y1, x2, y2 = [int(value) for value in element[0:4]]

        # draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # draw a red background rectangle for text
        label = int(element[5])
        conf_score = element[4]
        label_name = f"{PROJECT_LABELS[label]} {conf_score:.2f}"
        cv2.rectangle(
            frame,
            (x1, y1 - 15),
            (x1 + len(label_name) * 7, y1),
            (255, 0, 0),
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


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--weight_path", type=str, default="ml/weights/v5/lorelia_1.pt"
    )
    args = parser.parse_args()

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Load the model
    model = load_model(args.weight_path)

    while True:
        # Read frames from the webcam
        ret, frame = cap.read()

        # apply preprocessing on frame
        frame_bin = preprocess_cam(frame)

        # Perform inference on the frame
        results = model(frame_bin)
        # Apply confidence threshold
        confiend_elements = [
            good for good in results.pred[0].tolist() if good[4] > CONFIDENCE_TRESHOLD
        ]
        results.pred[0] = torch.tensor(confiend_elements)

        # Display the results
        cv2.imshow("Drawlo-bin", results.render()[0])
        cv2.imshow("Drawlo", draw_pred(frame, results))

        # Check for 'q' key press to exit
        if cv2.waitKey(1) == ord("q"):
            break

        # Delay so it works 24 fps
        # time.sleep(0.05)

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()
