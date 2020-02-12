
from src.mobilenet_v2_tsm import MobileNetV2
import os
import torch
import torchvision

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

