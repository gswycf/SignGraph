import time

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

from modules.gcn_lib.torch_vertex import Grapher, act_layer
from modules.gcn_lib.temgraph import TemporalGraph


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.localG = Grapher(in_channels=256, kernel_size=3, dilation=1, conv='edge', #mr
                              act='relu', norm="batch", bias=True, stochastic=False,
                              epsilon=0.0, r=1, n=14 * 14, drop_path=0.0, relative_pos=True)  # kernel_size=2
        self.localG2 = Grapher(in_channels=512, kernel_size=4, dilation=1, conv='edge',
                               act='relu', norm="batch", bias=True, stochastic=False,
                               epsilon=0.0, r=1, n=7 * 7, drop_path=0.0, relative_pos=True)  # kernel_size=2
        self.temporalG = TemporalGraph(k=14 * 14 // 4, in_channels=256, drop_path=0)
        self.temporalG2 = TemporalGraph(k=7 * 7, in_channels=512, drop_path=0)
        self.alpha = nn.Parameter(torch.ones(4), requires_grad=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        N, C, T, H, W = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # torch.Size([1, 64, 100, 56, 56])
        x = self.layer1(x)  # ([1, 64, 100, 56, 56])
        x = self.layer2(x)  # ize([1, 128, 100, 28, 28])
        x = self.layer3(x)  # e([1, 256, 100, 14, 14])
        #
        N, C, T, H, W = x.size()
        x = rearrange(x, 'N C T H W -> (N T) C H W')  # [78, 256, 14, 14])
        x = x + self.localG(x) * self.alpha[0]
        x = x + self.temporalG(x, N) * self.alpha[1]
        x = x.view(N, T, C, H, W).permute(0, 2, 1, 3, 4)
        # #
        x = self.layer4(x)  # [1, 512, 100, 7, 7])
        # #
        N, C, T, H, W = x.size()
        x = rearrange(x, 'N C T H W -> (N T) C H W')  # [78, 256, 14, 14])
        x = x + self.localG2(x) * self.alpha[2]
        x = x + self.temporalG2(x, N) * self.alpha[3]
        x = x.view(N, T, C, H, W).permute(0, 2, 1, 3, 4)
        #

        x = x.transpose(1, 2).contiguous()  # debug5= torch.Size([1, 100, 512, 7, 7])
        x = x.view((-1,) + x.size()[2:])  # bt,c,h,w  #ze([100, 512, 7, 7])
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # bt,c
        x = self.fc(x)  # bt,c

        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

 
