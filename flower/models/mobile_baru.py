import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding=padding, stride=stride)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn_point = nn.BatchNorm2d(out_channels)
        self.relu_point = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.relu(out)
        
        out = self.pointwise(out)
        out = self.bn_point(out)
        out = self.relu_point(out)
        return out

class ThreeConv(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=2, padding=1, bias=False):
        super(ThreeConv, self).__init__()
        self.three_conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out = self.three_conv(x)
        return out

class MobileNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNew, self).__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            ThreeConv(3,32),
            SeparableConv2d(32, 64, kernel_size=3, stride=1),
            SeparableConv2d(64, 128, kernel_size=3, stride=2),
            SeparableConv2d(128, 128, kernel_size=3, stride=1),
            SeparableConv2d(128, 256, kernel_size=3, stride=2),
            SeparableConv2d(256, 256, kernel_size=3, stride=1),
            SeparableConv2d(256, 512, kernel_size=3, stride=2),
            # 5
            SeparableConv2d(512, 512, kernel_size=3, stride=1),
            SeparableConv2d(512, 512, kernel_size=3, stride=1),
            SeparableConv2d(512, 512, kernel_size=3, stride=1),
            SeparableConv2d(512, 512, kernel_size=3, stride=1),
            SeparableConv2d(512, 512, kernel_size=3, stride=1),
            # end 5
            SeparableConv2d(512, 1024, kernel_size=3, stride=2),
            SeparableConv2d(1024, 1024, kernel_size=3, stride=2, padding=4),
            nn.AvgPool2d(7,7),
        )

        self.classifier = nn.Linear(1024,1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

# x = torch.rand(3,3,224,224)
# net = MobileNet()
# x = net(x)
# print(x.shape)