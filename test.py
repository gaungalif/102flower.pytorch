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

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# validset = base_dir.joinpath('dataset/valid')
# image_file = validset.joinpath('2/image_05136.jpg')
# a = base_dir.joinpath('weights/model_best.pth')
# model = load_flower_network(a, device)
# checkpoint = load_checkpoint(root.joinpath('checkpoint.pth'),device)

# probs, classes = predict(image_file, model)

# print(type(model))
# print(type(image_file))
# print(probs)


def predictors(image_path='dataset/valid/2/image_05094.jpg', weight_path='weights/model_best.pth', use_gpu=False):

  device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')

  images = image_path
  
  print(type(images))
  print(type(weight_path))
  model = load_flower_network(base_dir.joinpath(weight_path), device)
  print(type(model))
  probs, classes = predict(images, model)
  
  return probs

def main():
  fire.Fire(component=predictors)
# a = predictors(image_path='dataset/valid/2/image_05094.jpg', weight_path='weights/model_best.pth')
# print(a)

if __name__ == '__main__':
  main()