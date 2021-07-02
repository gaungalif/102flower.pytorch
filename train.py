import os
import sys

curr_dir = os.getcwd()
sys.path.append(curr_dir)

import argparse

from pathlib import Path

from flower.datasets import loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--bsize',type=int, help='batch size', default=32)
    parser.add_argument('-n','--num_worker',type=int, help='num worker', default=8)
    parser.add_argument('-tr','--train_resized',type=int, help='train resized', default=64)
    parser.add_argument('-vr','--valid_resized',type=int, help='valid resized', default=64)
    parser.add_argument('-ro','--train_rotate',type=int, help='train rotate', default=30)
    args = parser.parse_args()

    BSIZE = args.bsize
    NUM_WORKER = args.num_worker
    TR = args.train_resized
    VR = args.valid_resized
    RO = args.train_rotate

    curr_dir = Path(curr_dir)
    base_dir = curr_dir.joinpath('/kaggle/input/pytorch-challange-flower-dataset/')
    cat2name_path = base_dir.joinpath('cat_to_name.json') 
    train_path = base_dir.joinpath('dataset/train')
    valid_path = base_dir.joinpath('dataset/valid')


    train_loader, trainset = loader.train_loader(root=train_path, jsfile=cat2name_path, batch_size=BSIZE, num_worker=NUM_WORKER, ro=RO, tr=TR)
    valid_loader, validset = loader.valid_loader(root=valid_path, jsfile=cat2name_path, batch_size=BSIZE, num_worker=NUM_WORKER, vr=VR)
