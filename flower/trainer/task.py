import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import time
import json



import torch
import torch.optim as optim
from torch.functional import Tensor
from metrics.metrics import *
from tqdm.notebook import tqdm 

from progress import ProgressMeter 


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def getsteplr(base_lr=0.001, max_lr=0.1, step=4):
    lr = base_lr
    hlr = max_lr
    step = hlr/(step-1)
    step_lr = np.arange(lr, hlr+step, step).tolist()
    return step_lr

def get_all_flower_names():
    # with open('./data/cat_to_name.json', 'r') as f:
    with open('../input/pytorch-challange-flower-dataset/cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name

def flower_name(val, array_index=False):
    labels = get_all_flower_names()
    if array_index:
        val = val + 1
    return labels[str(val)]

FLOWER_LABELS = get_all_flower_names()

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
    for i, data in enumerate(dataloader, 0):
        data_time.update(time.time() - end)

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        output = net(inputs)
        loss = criterion(output, labels)

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
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            output = net(inputs)

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



def train_network(epoch, tloader, vloader, net, criterion, optimizer, scheduler, bsize, lr, trainset, log_freq=2000):
    global best_acc1
    # print('test')
    for ep in range(epoch):
        if ep == 13:
            net.unfreeze()
            step_lr = getsteplr(base_lr=lr/100, max_lr=lr, step=6)
            optimizer = optim.SGD(
                [
                    {'params': net.resnet.conv1.parameters()},
                    {'params': net.resnet.bn1.parameters()},
                    {'params': net.resnet.relu.parameters()},
                    {'params': net.resnet.maxpool.parameters()},
                    {'params': net.resnet.layer1.parameters(), 'lr':step_lr[1]},
                    {'params': net.resnet.layer2.parameters(), 'lr':step_lr[2]},
                    {'params': net.resnet.layer3.parameters(), 'lr':step_lr[3]},
                    {'params': net.resnet.layer4.parameters(), 'lr':step_lr[4]},
                    {'params': net.resnet.avgpool.parameters(), 'lr':step_lr[4]},
                    {'params': net.resnet.fc.parameters(), 'lr': step_lr[4]}
                ],
                lr=step_lr[0])
                print('test')
        train_batch(ep, tloader, net, criterion, optimizer, log_freq=log_freq)
        valid_batch(vloader, net, criterion,  log_freq=log_freq)
        scheduler.step()
        acc1 = valid_batch(vloader, net, criterion, log_freq=log_freq)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        SaveCheckpoint({
            'epoch': epoch + 1,
            'batch_size': bsize,
            'learning_rate': lr,
            'total_clazz': 102,
            'class_to_idx': trainset.class_to_idx,
            'labels': FLOWER_LABELS,
            'arch': 'resnet101',
            'state_dict': net.state_dict(),
            'best_acc_topk1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)