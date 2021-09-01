import os
import sys

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

from pathlib import Path

from flower.datasets import loader
from flower.models.residual import ResidualFlowerNetwork
from flower.trainer import task

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr','--lrate', type=float, help='learning rate', default=0.005)
    parser.add_argument('-ep','--epoch',type=int, help='epoch', required=True)
    parser.add_argument('-lf','--lfreq',type=int, help='log frequency', default=20)
    parser.add_argument('-m','--momentum',type=float, help='momentum', default=0.9)
    parser.add_argument('-b','--bsize',type=int, help='batch size', default=32)
    parser.add_argument('-n','--num_worker',type=int, help='num worker', default=8)
    parser.add_argument('-o','--optimizer',type=str, help='optimizer', required=True)
    parser.add_argument('--net',type=str, help='net', required=True)
    parser.add_argument('-tr','--train_resized',type=int, help='train resized', default=64)
    parser.add_argument('-vr','--valid_resized',type=int, help='valid resized', default=64)
    parser.add_argument('-ro','--train_rotate',type=int, help='train rotate', default=30)
    args = parser.parse_args()

    LR = args.lrate
    EPOCHS = args.epoch
    LOG_FREQ = args.lfreq
    MOMENTUM = args.momentum
    BSIZE = args.bsize
    NUM_WORKER = args.num_worker
    TR = args.train_resized
    VR = args.valid_resized
    RO = args.train_rotate
    resnet = torchvision.models.resnet34(pretrained=True)
    # print(resnet) 
    NET = args.net
    if NET == 'res':
        NET = ResidualFlowerNetwork(resnet, 102)
    if torch.cuda.is_available():
        NET.to(device)
    NET.freeze()
    
    criterion = nn.CrossEntropyLoss()

    OPTIMIZER = args.optimizer
    if OPTIMIZER == 'sgd':
        OPTIMIZER = optim.SGD(NET.resnet.fc.parameters(), LR, MOMENTUM)
    else:
        print("error cuy")
    scheduler = optim.lr_scheduler.StepLR(OPTIMIZER, step_size=6, gamma=0.1)
       
    base_dir = Path('../input/pytorch-challange-flower-dataset/')
    cat2name_path = ('cat_to_name.json') 
    train_path = base_dir.joinpath('dataset/train')
    valid_path = base_dir.joinpath('dataset/valid')

    train_loader, trainset = loader.train_loader(root=train_path, jsfile=cat2name_path, batch_size=BSIZE, num_worker=NUM_WORKER, ro=RO, tr=TR)
    valid_loader, validset = loader.valid_loader(root=valid_path, jsfile=cat2name_path, batch_size=BSIZE, num_worker=NUM_WORKER, vr=VR)

    task.train_network(EPOCHS, train_loader, valid_loader, NET, criterion, OPTIMIZER, scheduler, BSIZE, LR, trainset, LOG_FREQ)  