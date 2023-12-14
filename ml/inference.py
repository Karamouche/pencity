import cv2
import torch
import argparse
import time

CONFIDENCE_TRESHOLD = 0.7


def load_model(weight_path, name="ultralytics/yolov5"):
    model = torch.hub.load(name, "custom", path=weight_path)
    return model


def preprocess_cam(frame):
    # resize to 640x640
    frame = cv2.resize(frame, (640, 640))
    # to grayscaleYOLOv5
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # to black and white
    _, frame = cv2.threshold(frame, 115, 255, cv2.THRESH_BINARY)
    # reverse color
    frame = cv2.bitwise_not(frame)
    # remove black noise
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, (3, 3))

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
        frame = preprocess_cam(frame)

        # Perform inference on the frame
        results = model(frame)
        print(results.pred)
        # Apply confidence threshold
        confiend_elements = [
            good for good in results.pred[0].tolist() if good[4] > CONFIDENCE_TRESHOLD
        ]
        results.pred[0] = torch.tensor(confiend_elements)

        # Display the results
        cv2.imshow("Drawlo", results.render()[0])

        # Check for 'q' key press to exit
        if cv2.waitKey(1) == ord("q"):
            break

        # Delay so it works 24 fps
        time.sleep(0.05)

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()
