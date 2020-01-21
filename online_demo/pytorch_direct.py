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

    while True:
        time_start = time.time()
        _, img = cap.read() 

        with torch.no_grad():
            image_transformed = transform([Image.fromarray(img).convert('RGB')])
            input_transformed = image_transformed.view(1, 3, image_transformed.size(1), image_transformed.size(2))
            predictions, *shift_buffer = model(input_transformed, *shift_buffer)

            _, prediction = predictions.max(1)
            prediction = prediction.item()

            print(gestures.get_name(prediction))

        time_end = time.time()

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1)
        if is_quit_key(key):  # exit
            break
        elif is_fullscreen_key(key):  # full screen
            full_screen = not full_screen
            switch_fullscreen_mode(full_screen, window_name)

    cap.release()
    cv2.destroyAllWindows()

