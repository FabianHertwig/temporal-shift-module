import torch
import torchvision
import os
import cv2
import time
import numpy as np
from PIL import Image, ImageOps


from mobilenet_v2_tsm import MobileNetV2
from image_processing import get_transform
from gestures import Gestures
from prediction_smoothing import PredictionSmoothing


def load_model():
    torch_module = MobileNetV2(n_class=27)
    # checkpoint not downloaded
    if not os.path.exists("mobilenetv2_jester_online.pth.tar"):
        print('Downloading PyTorch checkpoint...')
        import urllib.request
        url = 'https://hanlab.mit.edu/projects/tsm/models/mobilenetv2_jester_online.pth.tar'
        urllib.request.urlretrieve(url, './mobilenetv2_jester_online.pth.tar')
    torch_module.load_state_dict(torch.load(
        "mobilenetv2_jester_online.pth.tar"))

    return torch_module


def init_buffer():
    return [torch.zeros([1, 3, 56, 56]),
            torch.zeros([1, 4, 28, 28]),
            torch.zeros([1, 4, 28, 28]),
            torch.zeros([1, 8, 14, 14]),
            torch.zeros([1, 8, 14, 14]), 
            torch.zeros([1, 8, 14, 14]), 
            torch.zeros([1, 12, 14, 14]), 
            torch.zeros([1, 12, 14, 14]), 
            torch.zeros([1, 20, 7, 7]), 
            torch.zeros([1, 20, 7, 7])]


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

    input_transformed = torch.rand(1, 3, 224, 224)
    shift_buffer = init_buffer()
    with torch.no_grad():
        predictions, *shift_buffer = model(input_transformed, *shift_buffer)
        print(predictions)
        print([s.shape for s in shift_buffer])
    
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
            image_transformed = transform([Image.fromarray(img).convert('RGB')])
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

