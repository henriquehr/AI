import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .resnet_cbam import *
from .resnet_cbam import Bottleneck, BottleneckRes, BasicBlockRes, BasicBlock

class RefinementModule(nn.Module):
    def __init__(self, in_ch):
        super(RefinementModule, self).__init__()

        self.conv0 = nn.Conv2d(in_ch , 32, 7, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.relu0 = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.P2_scale = ScaledL2Norm(64, initial_scale=10)

        self.P3_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.P3_conv = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.P3_scale = ScaledL2Norm(32, initial_scale=10)
        
        self.P4_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.P4_conv = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=1)
        self.P4_scale = ScaledL2Norm(32, initial_scale=10)

        self.P5_up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.P5_conv = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=3)
        self.P5_scale = ScaledL2Norm(32, initial_scale=10)

        self.conv_conc0 = nn.Conv2d(160, 32, kernel_size=1, padding=1)
        self.bn_conc0 = nn.BatchNorm2d(32)
        self.relu_conc0 = nn.ReLU(inplace=True)

        self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_conc2 = nn.Conv2d(32, 1, kernel_size=1, padding=0)


    def forward(self,x):
        hx1 = self.relu0(self.bn0(self.conv0(x)))
        hx1 = self.relu1(self.bn1(self.conv1(hx1)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))        
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        s2 = self.P2_scale(hx2)
        
        up3 = self.P3_up(hx3)
        s3 = self.P3_conv(up3)
        s3 = self.P3_scale(s3)

        up4 = self.P4_up(hx4)
        s4 = self.P4_conv(up4)
        s4 = self.P4_scale(s4)
        
        up5 = self.P5_up(hx5)
        s5 = self.P5_conv(up5)
        s5 = self.P5_scale(s5)

        conc = torch.cat([s2, s3, s4, s5], dim=1)

        c_c = self.conv_conc0(conc)
        c_c = self.bn_conc0(c_c)
        c_c = self.relu_conc0(c_c)

        c_c = self.upsamp(c_c)

        c_c = self.conv_conc2(c_c)
        
        out = c_c + x

        return out


class ScaledL2Norm(nn.Module):
    def __init__(self, in_channels, initial_scale):
        super(ScaledL2Norm, self).__init__()
        self.in_channels = in_channels
        self.scale = nn.Parameter(torch.Tensor(in_channels))
        self.initial_scale = initial_scale
        self.reset_parameters()

    def forward(self, x):
        return (F.normalize(x, p=2, dim=1)
             * self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3))

    def reset_parameters(self):
        self.scale.data.fill_(self.initial_scale)


# class SpatialAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SpatialAttention, self).__init__()
#         self.in_channels = in_channels
#         out_channels = 1
#         k = 9
#         self.conv_left_in = nn.Conv2d(in_channels, out_channels, kernel_size=[1, k], padding=2, bias=False)
#         self.bn_left_in = nn.BatchNorm2d(out_channels)
#         self.relu_left_in = nn.ReLU(inplace=True)
#         self.conv_left_out = nn.Conv2d(out_channels, 1, kernel_size=[k, 1], padding=2, bias=False)
#         self.bn_left_out = nn.BatchNorm2d(1)
#         self.relu_left_out = nn.ReLU(inplace=True)

#         self.conv_right_in = nn.Conv2d(in_channels, out_channels, kernel_size=[k, 1], padding=2, bias=False)
#         self.bn_right_in = nn.BatchNorm2d(out_channels)
#         self.relu_right_in = nn.ReLU(inplace=True)
#         self.conv_right_out = nn.Conv2d(out_channels, 1, kernel_size=[1, k], padding=2, bias=False)
#         self.bn_right_out = nn.BatchNorm2d(1)
#         self.relu_right_out = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()

#         self.conv_w = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
#         self.bn_w = nn.BatchNorm2d(in_channels)
#         self.relu_w = nn.ReLU(inplace=True)


#     def forward(self, x):
#         l = self.conv_left_in(x)
#         l = self.bn_left_in(l)
#         l = self.relu_left_in(l)
#         l = self.conv_left_out(l)
#         l = self.bn_left_out(l)
#         l = self.relu_left_out(l)

#         r = self.conv_right_in(x)
#         r = self.bn_right_in(r)
#         r = self.relu_right_in(r)
#         r = self.conv_right_out(r)
#         r = self.bn_right_out(r)
#         r = self.relu_right_out(r)

#         a = l + r
#         a = self.sigmoid(a)
#         a = a.repeat(1, self.in_channels, 1, 1)

#         w = self.conv_w(x)
#         w = self.bn_w(w)
#         w = self.relu_w(w)
#         m = a * w

#         out = torch.cat([x, m], dim=1)
#         return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MYNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MYNet, self).__init__()

        resnet = resnet34_mini_cbam(pretrained=True, cbam=False)
   
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.rl = resnet.relu
        self.mp = resnet.maxpool

        # block 1
        self.resnet_block1 = resnet.layer1
        # block 2
        self.resnet_block2 = resnet.layer2
        # block 3
        self.resnet_block3 = resnet.layer3
        # block 4
        self.resnet_block4 = resnet.layer4

        # self.P2_ca = ChannelAttention(128)
        # self.P2_sa = SpatialAttention()
        self.P2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.P2_conv0 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.P2_out = nn.Conv2d(64, 1, 3, padding=1)

        # self.P3_ca = ChannelAttention(256)
        # self.P3_sa = SpatialAttention()
        self.P3_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.P3_conv0 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.P3_out = nn.Conv2d(64, 1, 3, padding=1)

        # self.P4_ca = ChannelAttention(512)
        # self.P4_sa = SpatialAttention()
        self.P4_up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.P4_conv0 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.P4_out = nn.Conv2d(64, 1, 3, padding=1)

        self.P2_norm = ScaledL2Norm(64, initial_scale=10)
        self.P3_norm = ScaledL2Norm(64, initial_scale=10)
        self.P4_norm = ScaledL2Norm(64, initial_scale=10)
        # self.P5_norm = ScaledL2Norm(64, initial_scale=10)

        self.res_dec0_in = BasicBlockRes(192, 256, cbam=False)
        self._norm0 = ScaledL2Norm(256, initial_scale=10)

        self.res_dec1_in = BasicBlockRes(256, 128, cbam=False)
        self._norm1 = ScaledL2Norm(128, initial_scale=10)

        self.res_dec2_in = BasicBlockRes(128, 64, cbam=False)

        self.upscale2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upscale4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.upscale8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.outconv0 = nn.Conv2d(64, 1, 3, padding=1)

        # Refinement Module
        # self.ref_module = RefinementModule(1)


    def forward(self, x):
        hx = self.conv1(x)
        hx = self.bn1(hx)
        hx = self.rl(hx)
        hx = self.mp(hx)

        h1 = self.resnet_block1(hx)
        h2 = self.resnet_block2(h1)
        h3 = self.resnet_block3(h2)
        h4 = self.resnet_block4(h3)

        p2 = h2
        p2 = self.P2_conv0(self.P2_up(p2))

        p3 = h3
        p3 = self.P3_conv0(self.P3_up(p3))

        p4 = h4
        p4 = self.P4_conv0(self.P4_up(p4))

        p2_o = self.upscale4(p2)
        p2_o = self.P2_out(p2_o)
        p3_o = self.upscale4(p3)
        p3_o = self.P3_out(p3_o)
        p4_o = self.upscale4(p4)
        p4_o = self.P4_out(p4_o)

        norm2 = self.P2_norm(p2)
        norm3 = self.P3_norm(p3)
        norm4 = self.P4_norm(p4)

        conc = torch.cat([norm2, norm3, norm4], dim=1)

        up0 = self.res_dec0_in(conc)
        up0 = self.upscale2(up0)
        up0 = self._norm0(up0)

        up1 = self.res_dec1_in(up0)
        up1 = self.upscale2(up1)
        up1 = self._norm1(up1)

        dec2 = self.res_dec2_in(up1)

        out0 = self.outconv0(dec2)

        out1 = p2_o 
        out2 = p3_o 
        out3 = p4_o 
        
        # Refinement Module
        # out = self.ref_module(out0)

        # print(out0.shape)
        # print(out1.shape)
        # print(out2.shape)
        # print(out3.shape)
        # assert False

        return torch.sigmoid(out0), torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3)
