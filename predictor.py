
import os
import sys
import json

curr_dir = os.getcwd()
sys.path.append(curr_dir)

import torch
import torch.nn.functional as F

import PIL.Image
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.datasets.utils as utils
import torchvision.transforms as transforms

from flower.models.residual import ResidualFlowerNetwork

DIR = False
if DIR:
    path = './dataset/cat_to_name.json'
else:
    path = '../input/pytorch-challange-flower-dataset/cat_to_name.json'
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


curr_dir = Path(curr_dir)
root = curr_dir.joinpath('102flower.pytorch/flower/weights')
# utils.download_url('https://drive.google.com/file/d/1c6Dz5QVESdPPAvW0NUWuSnToVkT9fQ3Q/view?usp=sharing', root=root, filename='model_best.pth')
# utils.download_url('https://drive.google.com/file/d/16doe5f4YTLlGpFR9_0WQtjAvsPMlwxxJ/view?usp=sharing', root=root, filename='checkpoint.pth')

def load_flower_network(filename):
    if os.path.isfile(filename): 
        checkpoint = torch.load(filename, map_location='cuda')
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

def get_all_flower_names():
    with open(path, 'r') as f:
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

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    model.eval()
    model = model.cpu()
    with torch.no_grad():
        output = model.forward(image)
        output = F.log_softmax(output, dim=1)
        ps = torch.exp(output)
        result = ps.topk(topk, dim=1, largest=True, sorted=True)
        
    return result

def view_classify(img_path, label_idx, prob, classes, class_to_idx):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img = np.asarray(PIL.Image.open(img_path))
    ps = prob.data.numpy().squeeze().tolist()
    fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
    ax1.imshow(img.squeeze())
    ax1.set_title(flower_name(label_idx))
    ax1.axis('off')
    
    ax2.barh(np.arange(5), ps)
    ax2.set_aspect(0.2)
    ax2.set_yticks(np.arange(5))
    
    
    class_idx, class_name = getFlowerClassIndex(classes, class_to_idx)
    ax2.set_yticklabels(class_name, size='large')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()