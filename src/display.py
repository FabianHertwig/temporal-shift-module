import numpy as np
import cv2

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


def add_label(img: np.array, gesture_name: str, certainty:float, frames_per_second: float):
    height, width, _ = img.shape
    label = np.zeros([height // 7, width, 3]).astype('uint8') + 255
    cv2.putText(label, 'Prediction: ' + gesture_name,
                (0, int(height / 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)
    cv2.putText(label, 'Certainty: {:.2f} '.format(certainty),
                (0, int(height / 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)
    cv2.putText(label, '{:.1f} Frames/s'.format(frames_per_second),
                (width - 170, int(height / 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)
    img = np.concatenate((img, label), axis=0)
    return img
