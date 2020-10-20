import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.resnet import BasicBlock, Bottleneck
from .modules import View, Bottleneck

__all__ = ['CResNet', 'cresnet14', 'cresnet20', 'cresnet32', 'cresnet44', 'resnet101',
           'resnet152']


class CResNet(nn.Module):

    def __init__(self, n, block, num_classes=10, lwf=False, num_source_cls=200, growing=False):
        self.inplanes = 16
        self.growing = growing
        super(CResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = BasicBlock(16, 16)
        self.blocks2 = []
        for i in range(n):
            self.blocks2.append(block(16))
        self.blocks2 = nn.Sequential(*self.blocks2)
        downsample = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                                    nn.BatchNorm2d(32))
        self.block3 = BasicBlock(16, 32, 2, downsample)
        self.blocks4 = []
        for i in range(n):
            self.blocks4.append(block(32))
        self.blocks4 = nn.Sequential(*self.blocks4)
        downsample = nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                               nn.BatchNorm2d(64))
        self.block5 = BasicBlock(32, 64, 2, downsample)
        self.blocks6 = []
        for i in range(n):
            self.blocks6.append(block(64))
        self.blocks6 = nn.Sequential(*self.blocks6)
        if self.growing:
            self.block5_add = BasicBlock(32, 64, 2, downsample)
            self.blocks6_add = []
            for i in range(n):
                self.blocks6_add.append(block(64))
            self.blocks6_add = nn.Sequential(*self.blocks6_add)
            self.gamma = nn.Parameter(torch.zeros(1).fill_(10))
        self.avgpool = nn.AvgPool2d(8)
        self.view = View(-1)

        self.fc = nn.Linear(64,num_classes)

        self.lwf = lwf
        if self.lwf:
            self.lwf_lyr = nn.Linear(64, num_source_cls)

        self.alphas = nn.ParameterList([nn.Parameter(torch.rand(3, 1, 1, 1)*0.1),
                nn.Parameter(torch.rand(3, 1, 1, 1)*0.1),
                nn.Parameter(torch.rand(3, 1, 1, 1)*0.1)])

        if self.growing:
            self.fc = nn.Linear(64*2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, i=None):
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        r1 = self.relu(b1)
        f0 = self.block1(r1)

        # for i in range(len(self.blocks2)):
        #     f = self.blocks2[i](f)
        f1 = self.blocks2(f0)
        f2 = self.block3(f1)
        # for i in range(len(self.blocks4)):
        #     if i == 0:
        #         f3 = self.blocks4[i](f2)
        #     else:
        #         f3 = self.blocks4[i](f3)
        f3 = self.blocks4(f2)
        f4 = self.block5(f3)
        # for i in range(len(self.blocks6)):
        #     if i == 0:
        #         f5 = self.blocks6[i](f4)
        #     else:
        #         f5 = self.blocks6[i](f5)
        f5 = self.blocks6(f4)
        
        f6 = self.avgpool(f5)
        f7 = self.view(f6)
        if self.lwf:
            old_out = self.lwf_lyr(f7)

        f7 = self.fc(f7)
        #f8 = self.fc(f7)
        
        if self.lwf:
            return x, [r1, f1, f3, f5], old_out
        else:
            return f7, [r1, f1, f3, f5]

    def forward_with_features(self, x):
        feat = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        feat = [x]
        for i in range(len(self.blocks2)):
            x = self.blocks2[i](x)
            feat.append(x)

        x = self.block3(x)
        feat.append(x)
        for i in range(len(self.blocks4)):
            x = self.blocks4[i](x)
            feat.append(x)

        x = self.block5(x)
        feat.append(x)
        for i in range(len(self.blocks6)):
            x = self.blocks6[i](x)
            feat.append(x)

        x = self.avgpool(x)
        x = self.view(x)
        x = self.fc(x)
        return x, feat

def resnet32(num_classes=10, growing=False):
    model = CResNet(4, block=lambda k: BasicBlock(k, k),num_classes=num_classes, growing=growing)

    return model

