import os
import time
from typing import Tuple

import torch
from torch.functional import Tensor
import torch.optim
from tqdm.notebook import tqdm 
from flower.metrics.metrics import *
from progress import ProgressMeter 


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

best_acc1 = 0

def train_batch(epoch, dataloader, net, criterion, optimizer, log_freq=2000):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    net.train()

    end = time.time()  

    for i, data in tqdm(enumerate(dataloader, 0)):
        data_time.update(time.time() - end)

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        output = net(inputs)
        output = torch.softmax(output, dim=1)
        loss = criterion(output,labels)

        acc1, acc5 = AccuracyTopK(topk=(1,5))(output=output, target=labels)
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % log_freq == 0:
            progress.display(i)

    
            
    return top1.avg
            
def valid_batch(dataloader, net, criterion, log_freq=2000):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    net.eval()
  
    with torch.no_grad():
        end = time.time()
        for i, data in tqdm(enumerate(dataloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            output = net(inputs)
            output = torch.softmax(output, dim=1)

            loss = criterion(output,labels)
            
            acc1, acc5 = AccuracyTopK(topk=(1,5))(output=output, target=labels)
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % log_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg


def train_network(epoch, tloader, vloader, net, criterion, optimizer, log_freq=2000):
    global best_acc1
    for ep in tqdm(range(epoch)):
        train_batch(ep, tloader, net, criterion, optimizer, log_freq=log_freq)
        valid_batch(vloader, net, criterion,  log_freq=log_freq)
        
        acc1 = valid_batch(vloader, net, criterion, log_freq=log_freq)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        SaveCheckpoint(filename='checkpoint.pth.tar')({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)