import os
import sys
import torch
from torch.cuda import is_available
curr_dir = os.getcwd()
sys.path.append(curr_dir)
import fire
from predictor import *

from pathlib import Path
base_dir = Path(curr_dir)

def predictors(image_path='dataset/valid/2/image_05094.jpg', weight_path='weights/model_best.pth', use_gpu=False):

  device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')

  images = image_path
  # download my last weights trained at 98.566% accuracy  
  download = True
  model = load_flower_network(base_dir.joinpath(weight_path), device, download=download)
  probs, classes = predict(images, model)
  print(probs)

if __name__ == '__main__':
  fire.Fire(component=predictors)