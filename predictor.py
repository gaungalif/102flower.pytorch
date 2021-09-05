
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

def get_all_flower_names():
    with open('./dataset/cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name

def flower_name(val, array_index=False):
    labels = get_all_flower_names()
    if array_index:
        val = val + 1
    return labels[str(val)]

def getFlowerClassIndex(classes, class_to_idx):
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    class_to_flower_class_idx = [idx_to_class[lab] for lab in classes.squeeze().numpy().tolist()]
    flower_class_to_name = [flower_name(cls_idx) for cls_idx in class_to_flower_class_idx]
    return class_to_flower_class_idx, flower_class_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = PIL.Image.open(image)
    mean_val, std_val = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    do_transforms =  transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_val,std_val)
    ])
    im_tfmt = do_transforms(im)
    im_add_batch = im_tfmt.view(1, im_tfmt.shape[0], im_tfmt.shape[1], im_tfmt.shape[2])
    return im_add_batch

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax