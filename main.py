import os
import time

import cv2
from PIL import Image, ImageOps
import numpy as np
import torch

from src.model import load_model, init_buffer
from src.image_processing import get_transform, get_crop
from src.gestures import Gestures
from src.prediction_smoothing import PredictionSmoothing

def is_quit_key(key):
    return key & 0xFF == ord('q') or key == 27


def is_fullscreen_key(key):
    return key == ord('F') or key == ord('f')


def switch_fullscreen_mode(full_screen: bool, window_name: str):
    if full_screen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)


def setup_window(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowTitle(window_name, window_name)


def get_camera_capture(camera: int, width: int, height: int):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def add_label(img: np.array, gesture_name: str, frames_per_second: float):
    height, width, _ = img.shape
    label = np.zeros([height // 10, width, 3]).astype('uint8') + 255
    cv2.putText(label, 'Prediction: ' + gesture_name,
                (0, int(height / 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)
    cv2.putText(label, '{:.1f} Frames/s'.format(frames_per_second),
                (width - 170, int(height / 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)
    img = np.concatenate((img, label), axis=0)
    return img

if __name__ == "__main__":
    model = load_model()
    model.eval()

    cap = get_camera_capture(0, 320, 240)

    full_screen = False
    window_name = 'Video Gesture Recognition'
    setup_window(window_name)

    transform = get_transform()
    shift_buffer = init_buffer()
    gestures = Gestures()
    prediction_smoothing = PredictionSmoothing(7)

    while True:
        time_start = time.time()
        _, img = cap.read()

        with torch.no_grad():
            pil_image = [Image.fromarray(img).convert('RGB')]
            image_transformed = transform(pil_image)
            input_transformed = image_transformed.view(1, 3, image_transformed.size(1), image_transformed.size(2))
            predictions, *shift_buffer = model(input_transformed, *shift_buffer)

            _, prediction = predictions.max(1)
            prediction = prediction.item()
            prediction_smoothing.add_prediction(prediction)
            smooth_prediction = prediction_smoothing.get_most_common_prediction()

        time_end = time.time()
        frames_per_second = 1 / (time_end - time_start)

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        img = add_label(img, gestures.get_name(smooth_prediction), frames_per_second)
        cv2.imshow(window_name, img)

        key = cv2.waitKey(1)
        if is_quit_key(key):  # exit
            break
        elif is_fullscreen_key(key):  # full screen
            full_screen = not full_screen
            switch_fullscreen_mode(full_screen, window_name)

    cap.release()
    cv2.destroyAllWindows()

