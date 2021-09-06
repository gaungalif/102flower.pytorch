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

def getsteplr(base_lr=0.001, max_lr=0.1, step=4):
    lr = base_lr
    hlr = max_lr
    step = hlr/(step-1)
    step_lr = np.arange(lr, hlr+step, step).tolist()
    return step_lr

def save_checkpoint(state, is_best, filename='weights/checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'weights/model_best.pth')

def adjust_learning_rate(optimizer, epoch, decay, lrate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrate * (0.1 ** (epoch // decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


