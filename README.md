# 102flower.pytorch

This repository contains the code to train data taken from Oxford 102 Category Flower Dataset as an image classification using pytorch. I have used resnet34 pretrained model from torchvision to train the model and made some codes to train the module. 

## Requirment:
- Python=3.6.8
- pytorch=1.9.0
- Numpy=1.16.6
- fire=0.4.0
- torchvision=0.10.0

i have also save the package-list.txt into the repository.

## Hardware Requirment:
- Computer with decend RAM and CPU
- GPU (optional)

## How to Use:
### Dataset:
- Download the dataset manually from here: https://www.kaggle.com/nunenuh/pytorch-challange-flower-dataset/download
- Store dataset at `dataset/`

### Training:
- Use `train.py` to train the model.
- Change `dataset` path to the appropriate path if needed
- You can modify the Hyperparameter and Augmentation if needed
- Use this command 'python train.py --help' for help

example command: 
- python train.py  --lrate 0.05 --epoch 100 --lfreq 20 --bsize 128 --num_worker 64 


### Test:
- Use 'test.py' to test the model that you have trained.
- modify the image_path(default: dataset/test/image_*) and weight_path(default: weiights/model_best.pth) to the specific path location to test the image
- set the download variable to 'True' if you want yo use my last trained weights or you could train it yourself (follow 'Training' steps)
- if your device have gpu set the use_gpu to 'True'
- the program will predict the class (or classes) of an image using a trained deep learning model

example command: 
- python test.py --image_path=dataset/test/image_05166.jpg --weight_path=weights/model_best.pth --use_gpu=False
- output: tensor([[9.9872e-01, 4.8233e-04, 2.6316e-04, 1.7123e-04, 1.2501e-04]])


## Reference:
- Oxford 102 Flower Dataset, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Nunenuh, Oxford 102 Flower Dataset, https://www.kaggle.com/nunenuh/pytorch-challange-flower-dataset
- ResNet, Pytorch, https://pytorch.org/hub/pytorch_vision_resnet/
- Torchvision Documentation, Pytorch, https://pytorch.org/vision/stable/index.html

