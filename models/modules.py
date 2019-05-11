import torch.nn as nn
from utils.ops import linear, conv2d, batchnorm2d

class View(nn.Module):
    def __init__(self, *size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(x.size()[0], *self.size)

class BasicResBlockSELU(nn.Module):
    pass

class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, batchnorm_affine=True):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=batchnorm_affine)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=batchnorm_affine)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock_meta(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_meta, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    #def forward(self, x):
    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):
        stop_gradient_ = True
        residual = x

        #out = self.conv1(x)
        out = conv2d(inputs=x,
                    weight=self.conv1.weight,
                    bias=self.conv1.bias,
                    padding=self.conv1.padding,
                    stride=self.conv1.stride,
                    meta_loss=meta_loss,
                    meta_step_size=meta_step_size,
                    stop_gradient=stop_gradient)
        #out = self.bn1(out)
        out = batchnorm2d(inputs=out,
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
        out = self.relu(out)

        #out = self.conv2(out)
        out = conv2d(inputs=out,
                    weight=self.conv2.weight,
                    bias=self.conv2.bias,
                    padding=self.conv2.padding,
                    stride=self.conv2.stride,
                    meta_loss=meta_loss,
                    meta_step_size=meta_step_size,
                    stop_gradient=stop_gradient)
        #out = self.bn2(out)
        out = batchnorm2d(inputs=out,
                        running_mean=self.bn2.running_mean, 
                        running_var=self.bn2.running_var,
                        weight=self.bn2.weight,
                        bias=self.bn2.bias,
                        train=self.bn2.training,
                        track_running_stats=self.bn2.track_running_stats,
                        momentum=self.bn2.momentum,
                        eps=self.bn2.eps,
                        meta_step_size=meta_step_size,
                        meta_loss=meta_loss,
                        stop_gradient=stop_gradient)

        if self.downsample is not None:
            residual = conv2d(inputs=x,
                    weight=self.downsample[0].weight,
                    bias=self.downsample[0].bias,
                    padding=self.downsample[0].padding,
                    stride=self.downsample[0].stride,
                    meta_loss=meta_loss,
                    meta_step_size=meta_step_size,
                    stop_gradient=stop_gradient)
            residual = batchnorm2d(inputs=residual,
                        running_mean=self.downsample[1].running_mean, 
                        running_var=self.downsample[1].running_var,
                        weight=self.downsample[1].weight,
                        bias=self.downsample[1].bias,
                        train=self.downsample[1].training,
                        track_running_stats=self.downsample[1].track_running_stats,
                        momentum=self.downsample[1].momentum,
                        eps=self.downsample[1].eps,
                        meta_step_size=meta_step_size,
                        meta_loss=meta_loss,
                        stop_gradient=stop_gradient)

            #residual = self.downsample(x)

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
