import os
import sys
import torch
import shutil

curr_dir = os.getcwd()
sys.path.append(curr_dir)

import torch
import numpy as np
import flower.metrics.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AccuracyTopK(object):
    def __init__(self, topk:tuple=(1,)):
        self.topk = topk
    
    def __call__(self, output , target):
        return F.accuracy(output, target, self.topk)

class SaveCheckpoint(object):
    def __init__(self, state, is_best):
        self.state = state
        self.is_best = is_best
    
    def __call__(self, filename='checkpoint.pth'):
        return F.save_checkpoint(self.state, self.is_best, filename)

def adjust_learning_rate(optimizer, epoch, decay, lrate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrate * (0.1 ** (epoch // decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


