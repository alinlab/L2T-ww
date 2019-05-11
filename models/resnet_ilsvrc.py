import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .modules import View, BasicBlock_meta
from utils.ops import linear, conv2d, batchnorm2d


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, meta=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.lwf = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        f1 = self.conv1(x)
        b1 = self.bn1(f1)
        r1 = self.relu(b1)
        p1 = self.maxpool(r1)

        f2 = self.layer1(p1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)

        f6 = self.avgpool(f5)
        f6 = f6.view(f6.size(0), -1)
        f7 = self.fc(f6)

        return f7, [r1, f2, f3, f4, f5]

    def forward_with_features(self, x):
        return self.forward(x)

class ResNet_branch(nn.Module):
    def __init__(self, feat_dim=512, num_classes=1000):
        super(ResNet_branch, self).__init__()
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.view = View(-1)
        self.fc = nn.Linear(feat_dim,num_classes)
        # self.num_classes = num_classes
        # if isinstance(num_classes, list):
        #     fcs = []
        #     for i in range(len(num_classes)):
        #         fcs.append(nn.Linear(64, num_classes[i]))
        #     self.fc = nn.ModuleList(fcs)
        # else:
        #     self.fc = nn.Linear(64, num_classes)
        # # if self.growing:
        # #     self.fc = nn.Linear(64*2, num_classes)

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False, idx=None):
        stop_gradient_ = True 
        x = self.avgpool(x)
        x = self.view(x)
        f7 = linear(inputs=x,
            weight=self.fc.weight,
            bias=self.fc.bias,
            meta_loss=meta_loss,
            meta_step_size=meta_step_size,
            stop_gradient=stop_gradient_)   

        return f7

class ResNet_meta(nn.Module):

    def __init__(self, block, layers, num_classes=1000, meta=None):
        self.inplanes = 64
        super(ResNet_meta, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )
            downsample = [
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)]
            

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):
        c1 = conv2d(inputs=x,
                    weight=self.conv1.weight,
                    bias=self.conv1.bias,
                    padding=self.conv1.padding,
                    stride=self.conv1.stride,
                    meta_loss=meta_loss,
                    meta_step_size=meta_step_size,
                    stop_gradient=stop_gradient)
        b1 = batchnorm2d(inputs=c1,
                        running_mean=self.bn1.running_mean,
                        running_var=self.bn1.running_var,
                        weight=self.bn1.weight,
                        bias=self.bn1.bias,
                        train=self.bn1.training,
                        track_running_stats=self.bn1.track_running_stats,
                        momentum=self.bn1.momentum,
                        eps=self.bn1.eps,
                        meta_step_size=meta_step_size,
                        meta_loss=meta_loss,
                        stop_gradient=stop_gradient)
        r1 = F.relu(b1, inplane=True)
        r1 = self.maxpool(r1)

        f1 = r1
        for i in range(len(self.layer1)):
            f1 = self.layer1[i](x=f1,
                                meta_loss=meta_loss,
                                meta_step_size=meta_step_size,
                                stop_gradient=stop_gradient)

        f2 = f1
        for i in range(len(self.layer2)):
            f2 = self.layer2[i](x=f2,
                                meta_loss=meta_loss,
                                meta_step_size=meta_step_size,
                                stop_gradient=stop_gradient)

        f3 = f2
        for i in range(len(self.layer3)):
            f3 = self.layer3[i](x=f3,
                                meta_loss=meta_loss,
                                meta_step_size=meta_step_size,
                                stop_gradient=stop_gradient)

        f4 = f3
        for i in range(len(self.layer4)):
            f4 = self.layer4[i](x=f4,
                                meta_loss=meta_loss,
                                meta_step_size=meta_step_size,
                                stop_gradient=stop_gradient)

        f5 = self.avgpool(f4)
        f6 = f5.view(f5.size(0), -1)
        f7 = self.fc(f6)

        return f7, [r1, f1, f2, f3, f4, f5]




def resnet18(pretrained=False, meta=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if meta:
        model = ResNet_meta(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, meta=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if meta:
        model = ResNet_meta(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
