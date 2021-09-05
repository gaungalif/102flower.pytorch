
import os
import sys
import json

curr_dir = os.getcwd()
sys.path.append(curr_dir)

import PIL.Image

import torchvision.transforms as transforms

import torch
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.datasets.utils as utils

from flower.models.residual import ResidualFlowerNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


curr_dir = Path(curr_dir)
root = curr_dir.joinpath('flower/weights')
utils.download_url('https://drive.google.com/file/d/1c6Dz5QVESdPPAvW0NUWuSnToVkT9fQ3Q/view?usp=sharing', root=root, filename='model_best.pth')
utils.download_url('https://drive.google.com/file/d/16doe5f4YTLlGpFR9_0WQtjAvsPMlwxxJ/view?usp=sharing', root=root, filename='checkpoint.pth')

def load_flower_network(filename):
    if os.path.isfile(filename): 
        checkpoint = torch.load(filename, map_location=device)
        resnet = torchvision.models.resnet34(pretrained=True)
        clazz = checkpoint['total_clazz']
        model = ResidualFlowerNetwork(resnet, clazz)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    else:
        return None
    

def load_checkpoint(filename):
    if os.path.isfile(filename): 
        checkpoint = torch.load(filename, map_location=device)
        return checkpoint
    else:
        return None

# model = load_flower_network(root.joinpath('model_best.pth'))
# checkpoint = load_checkpoint(root.joinpath('checkpoint.pth'))