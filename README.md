# Pytorch Gesture Recognition with the Temporal Shift Module for Efficient Video Understanding

This repository contains code to run the Temporal Shift Module for gesture recognition with pytorch.
It is based on: https://github.com/mit-han-lab/temporal-shift-module

```
@inproceedings{lin2019tsm,
  title={TSM: Temporal Shift Module for Efficient Video Understanding},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
} 
```
![tsm-demo](https://hanlab.mit.edu/projects/tsm/external/tsm-demo2.gif)


## Getting Started

This was developed with python 3.7

You can use a virtual environment, eg. conda

    conda create -n temporal-shift-module python=3.7
    conda activate temporal-shift-module
    
Install the requirements:

    pip install -r requirements.txt

Run the main.py file

    python main.py

If the model has not been downlaoded yet, it will be automatically downloaded. Then a window should open and show the camara feed of the first camera device that is found on you system. At the bottom of the feed the recognized gesture gets displayed.

You can use the f key to switch to fullscreen mode and q to quit the application.