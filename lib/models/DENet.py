# -*- coding:utf-8 -*-
# @Project   :DENet
# @FileName  :DENet.py
# @Time      :2024/7/26 20:55
# @Author    :Zhiyue Lyu
# @Version   :1.0
# @Descript  :None

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

from lib.models.net_utils import BasicBlock, Bottleneck, MDSR_Block, PAPPM, segmenthead, PFAM, DE_Module_1, DE_Module_2, FFM
import matplotlib.cm as cm

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


class Net(nn.Module):

    def __init__(self, layers, num_classes=3, channels=32, ppm_channels=128, head_channels=64, augment=True):
        super(Net, self).__init__()

        self.augment = augment

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(channels, momentum=bn_mom),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(channels, momentum=bn_mom),
            nn.LeakyReLU(inplace=True),
        )

        self.relu = nn.LeakyReLU(inplace=False)

        self.layer1 = self._make_layer(BasicBlock, channels, channels, layers[0])
        self.layer2 = self._make_layer(BasicBlock, channels, channels * 2, layers[1], stride=2)       # out channels: 64
        self.layer3 = self._make_layer(MDSR_Block, channels * 2, channels * 4, layers[2], stride=2)   # out channels: 128
        self.layer4 = self._make_layer(MDSR_Block, channels * 4, channels * 8, layers[3], stride=2)   # out channels: 256

        # semantic features fuse to detail features

        self.compression1 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, kernel_size=1, bias=False),
            BatchNorm2d(channels * 2, momentum=bn_mom)
        )

        self.compression2 = nn.Sequential(
            nn.Conv2d(channels * 8, channels * 2, kernel_size=1, bias=False),
            BatchNorm2d(channels * 2, momentum=bn_mom)
        )

        self.pfam1 = PFAM(channels * 2)
        self.pfam2 = PFAM(channels * 2)

        # Detail Branch
        self.layer3_ = self._make_layer(BasicBlock, channels * 2, channels * 2, 2, enhance=True)
        self.layer4_ = self._make_layer(BasicBlock, channels * 2, channels * 2, 2, enhance=True)

        self.layer5_ = self._make_layer(Bottleneck, channels * 2, channels * 2, 1, enhance=True)
        self.layer5 = self._make_layer(Bottleneck, channels * 8, channels * 8, 1, stride=2)

        self.spp = PAPPM(channels * 16, ppm_channels, channels * 4)
        self.ffm = FFM(channels * 8, channels * 4)

        if self.augment:
            self.seghead_extra = segmenthead(channels*2, head_channels, num_classes)

        self.final_layer = segmenthead(channels * 4, head_channels, num_classes)


    def _make_layer(self, block, in_channels, channels, blocks, stride=1, enhance=False):
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(in_channels, channels, stride, downsample))
        in_channels = channels * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(in_channels, channels, stride=1, no_relu=True))
            else:
                layers.append(block(in_channels, channels, stride=1, no_relu=False))

        if enhance:
            if blocks==2:
                layers.append(DE_Module_1())
            if blocks==1:
                layers.append(DE_Module_1(input_channels=channels*2, output_channels=channels*2))

        return nn.Sequential(*layers)

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        layers = []

        x = self.conv1(x)   # 进行两次下采样，得到1/4的特征图

        x = self.layer1(x)  # 进行两次特征提取
        layers.append(x)    # layers[0]

        x = self.layer2(self.relu(x))   # 进行一次下采样，得到1/8的特征图
        layers.append(x)    # layers[1]

        x = self.layer3(self.relu(x))  # 语义分支，再次进行一次下采样，得到1/16的特征图
        layers.append(x)    # layers[2]
        f_d = self.layer3_(self.relu(layers[1]))          # 细节分支，1/8的特征图，经过特征增强

        # 无参数注意力机制特征提取  1/16特征图
        f_d = f_d + self.pfam1(
            F.interpolate(
                self.compression1(self.relu(layers[2])),
                size=[height_output, width_output],
                mode='bilinear')
        )

        if self.augment:
            temp = f_d

        x = self.layer4(self.relu(x))      # 语义分支， 进行一次下采样，得到1/32的特征图
        layers.append(x)    # layers[3]
        f_d = self.layer4_(self.relu(f_d))

        # 无参数注意力机制特征提取  1/32特征图
        f_d = f_d + self.pfam2(
            F.interpolate(
                self.compression2(self.relu(layers[3])),
                size=[height_output, width_output],
                mode='bilinear')
        )

        f_d = self.layer5_(self.relu(f_d))       # 经过瓶颈块 # 128channels

        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),  # 1/64  经过DAPPM
            size=[height_output, width_output],
            mode='bilinear')  # 1/8           # channels:128

        f_d = self.ffm(x, f_d)                # channels: 128

        f_d = self.final_layer(f_d)           # channels: 64

        if self.augment:
            x_extra = self.seghead_extra(temp)
            return [x_extra, f_d]
        else:
            return f_d

    def gray_to_heatmap(self, gray_tensor):
        # 将单通道的灰度图映射到热力图的RGB颜色
        heatmap = gray_tensor.detach().cpu().numpy().squeeze()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化到[0, 1]
        heatmap = cm.viridis(heatmap)[:, :, :3]  # 使用viridis颜色映射
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).unsqueeze(0).float()  # 转换为PyTorch张量
        return heatmap


def DeEnResNet_imagenet(cfg, pretrained=False):
    model = Net([2, 2, 2, 2], num_classes=cfg.DATASET.NUM_CLASSES, channels=32, ppm_channels=128, head_channels=64, augment=True)
    if pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)

        model.load_state_dict(model_dict, strict=False)
    return model


def get_seg_model(cfg, **kwargs):
    model = DeEnResNet_imagenet(cfg, pretrained=False)
    return model


if __name__ == '__main__':
    pass
