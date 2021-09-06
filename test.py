import os
import sys
import torch
from torch.cuda import is_available
curr_dir = os.getcwd()
sys.path.append(curr_dir)
import fire
from predictor import *

import matplotlib.pyplot as plt
from pathlib import Path
base_dir = Path(curr_dir)

def predictors(image_path='dataset/valid/2/image_05094.jpg', weight_path='weights/model_best.pth', use_gpu=False):

  device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')

  images = Path(image_path)
  # download my last weights trained at 98.566% accuracy  
  download = False
  model = load_flower_network(base_dir.joinpath(weight_path), device, download=download)
  checkpoint = load_checkpoint(base_dir.joinpath('weights/checkpoint.pth'), device, download=download)
  prob, classes = predict(images, model)
  flower = Path(image_path)
  print('Flower category: {}'.format(flower_name(flower.parent.name)))
  class_idx, class_name = getFlowerClassIndex(classes, checkpoint['class_to_idx'])
  output = prob.cpu().detach().numpy()
  a = output.argsort()
  a = a[0]
  size = len(a)
  a = np.flip(a[-1*size:])
  prediction = list()
  clas = list()
  for i in a:
    prediction.append(float(output[:,i]*100))
  clas.append(str(i))
  print('Prediction')
  for i in a:
      print('Class: {} , confidence: {}'.format(class_name[int(i)],float(output[:,i]*100)))
  plt.bar(clas,prediction)

if __name__ == '__main__':
  fire.Fire(component=predictors)