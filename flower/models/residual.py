import torch
import torch.nn as nn

import torchvision

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz:int=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    
class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)
    
def Flatten()->torch.Tensor:
    return Lambda(lambda x: x.view((x.size(0), -1)))

class ResidualFlowerNetwork(nn.Module):
    def __init__(self, resnet, clazz):
        super(ResidualFlowerNetwork, self).__init__()
        self.pool_size = 1
        self.resnet = resnet
        
        # out_channels multiple by pool size and multiply by 2
        # multiply by 2 is get from torch cat of AdaptiveAvgPool2d and AdaptiveMaxPool2d
        in_features = self.get_last_layer_out_channels() * self.pool_size*self.pool_size*2
        
        self.resnet.avgpool = nn.Sequential(
            AdaptiveConcatPool2d(self.pool_size),
            Flatten()
        )
        
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, in_features//2),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm1d(in_features//2),
            nn.Dropout(0.3),
            nn.Linear(in_features//2, in_features//4),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm1d(in_features//4),
            nn.Dropout(0.2),
            nn.Linear(in_features//4, clazz+1),
            # nn.ReLU(inplace=True)
            # nn.Linear(512, clazz)
        )
    def get_last_layer_out_channels(self):
        if type(self.resnet.layer4[2]) == torchvision.models.resnet.BasicBlock:
            return self.resnet.layer4[2].conv2.out_channels
        elif type(self.resnet.layer4[2]) == torchvision.models.resnet.Bottleneck:
            return self.resnet.layer4[2].conv3.out_channels
        else:
            return 0
        
    def freeze(self):
        for param in self.resnet.parameters():
            param.require_grad = False
        for param in self.resnet.fc.parameters():
            param.require_grad= True
            
    def unfreeze(self):
        for param in self.resnet.parameters():
            param.require_grad = True
    
    def forward(self, x):
        x = self.resnet(x)
        return x

# resnet = torchvision.models.resnet34(pretrained=True)
# print(resnet) 
# model = ResidualFlowerNetwork(resnet, 102)