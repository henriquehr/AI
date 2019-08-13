from collections import OrderedDict
import math
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['resnet34']


def init_weight(*args):
    return nn.Parameter(nn.init.kaiming_normal_(torch.zeros(*args), mode='fan_out', nonlinearity='relu'))


class ForwardSign(torch.autograd.Function):
    """Fake sign op for 1-bit weights.

    See eq. (1) in https://arxiv.org/abs/1802.08530

    Does He-init like forward, and nothing on backward.
    """

    @staticmethod
    def forward(ctx, x):
        return math.sqrt(2. / (x.shape[1] * x.shape[2] * x.shape[3])) * x.sign()

    @staticmethod
    def backward(ctx, g):
        return g


class ModuleBinarizable(nn.Module):

    def __init__(self, binarize=False):
        super().__init__()
        self.binarize = binarize

    def _get_weight(self, name):
        w = getattr(self, name)
        return ForwardSign.apply(w) if self.binarize else w

    def forward(self):
        pass


class BasicBlock(ModuleBinarizable):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.register_parameter('conv1', init_weight(planes, inplanes, 3, 3))
        # self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.register_parameter('conv2', init_weight(planes, planes, 3, 3))
        # self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.conv2d(x, self._get_weight('conv1'), padding=1, stride=self.stride)
        # out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = F.conv2d(out, self._get_weight('conv2'), padding=1)
        # out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            residual = torch.cat([residual, torch.zeros_like(residual)], dim=1)

        out += residual
        out = self.relu(out)

        return out


class WRN_McDonnell(ModuleBinarizable):
    """Implementation of modified Wide Residual Network.

    Differences with pre-activated ResNet and Wide ResNet:
        * BatchNorm has no affine weight and bias parameters
        * First layer has 16 * width channels
        * Last fc layer is removed in favor of 1x1 conv + F.avg_pool2d
        * Downsample is done by F.avg_pool2d + torch.cat instead of strided conv

    First and last convolutional layers are kept in float32.
    """

    def __init__(self, block, layers, binarize=False):
        super(WRN_McDonnell, self).__init__()
        self.inplanes = 64
        self.binarize = binarize

        self.register_parameter('conv1', init_weight(64, 3, 3, 3))

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.conv2d(x, self.conv1, padding=1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet34(**kwargs):
    model = WRN_McDonnell(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model