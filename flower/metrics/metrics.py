import os
import sys
import torch
import shutil

curr_dir = os.getcwd()
sys.path.append(curr_dir)

import torch
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

class adjust_lr(object):
    def __init__(self, lrate):
        self.lrate = lrate

    def __call__(self, optimizer, epoch, decay,):
        return F.adjust_learning_rate(optimizer, epoch, decay, self.lrate) 

class getstep_lr(object):
    def __init__(self, base_lr=0.001, max_lr=0.1, step=4):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step = step
    def __call__(self):
        return F.getsteplr(self.base_lr, self.max_lr, self.step)


