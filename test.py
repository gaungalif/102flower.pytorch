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
  
  print(type(images))
  print(type(weight_path))
  model = load_flower_network(base_dir.joinpath(weight_path), device)
  print(base_dir.joinpath(weight_path))
  probs, classes = predict(images, model)
  print(probs)

def main():
  fire.Fire(component=predictors)

if __name__ == '__main__':
  main()