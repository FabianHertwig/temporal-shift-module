import time

import cv2
import torch
from PIL import Image
from torch.nn import Softmax

from src.camera import get_camera_capture
from src.display import setup_window, add_label, is_quit_key, is_fullscreen_key, switch_fullscreen_mode
from src.gestures import Gestures
from src.image_processing import get_transform
from src.model import load_model, init_buffer
from src.prediction_smoothing import PredictionSmoothing

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
    softmax = Softmax(1)


    while True:
        time_start = time.time()
        _, img = cap.read()

        with torch.no_grad():
            pil_image = [Image.fromarray(img).convert('RGB')]
            image_transformed = transform(pil_image)
            input_transformed = image_transformed.view(1, 3, image_transformed.size(1), image_transformed.size(2))
            predictions, *shift_buffer = model(input_transformed, *shift_buffer)
            predictions = softmax(predictions)
            certainty, prediction = predictions.max(1)
            prediction = prediction.item()
            certainty = certainty.item()
            prediction_smoothing.add_prediction(prediction)
            smooth_prediction = prediction_smoothing.get_most_common_prediction()

        time_end = time.time()
        frames_per_second = 1 / (time_end - time_start)

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        img = add_label(img, gestures.get_name(smooth_prediction), certainty, frames_per_second)
        cv2.imshow(window_name, img)

        key = cv2.waitKey(1)
        if is_quit_key(key):  # exit
            break
        elif is_fullscreen_key(key):  # full screen
            full_screen = not full_screen
            switch_fullscreen_mode(full_screen, window_name)

    cap.release()
    cv2.destroyAllWindows()

