import os

import torch.nn as nn
import torch
import torch.nn.functional as F

import math
import numpy as np
from torchvision.utils import save_image
import matplotlib.cm as cm

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

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

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

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
        if self.no_relu:
            return out
        else:
            return self.relu(out)

# MDSR_Block
class MDSR_Block(nn.Module):   # mulit-scale depthwise separable convolution Residual block
    """
    添加通道注意力机制的MDSR_Block
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, no_relu=False):
        super(MDSR_Block, self).__init__()
        # self.DWconv1x1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, groups=in_planes, bias=False)  # 1x1depthwise卷积
        # self.DWconv3x3 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)  # 3x3depthwise卷积
        self.PWconv1x1 = nn.Conv2d(in_planes*3, out_planes, kernel_size=1, bias=False)  # 1x1pointwise卷积

        self.relu = nn.LeakyReLU(inplace=True)
        self.no_relu = no_relu
        self.downsample = downsample

        self.scale1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, groups=in_planes, bias=False),
            BatchNorm2d(in_planes, momentum=bn_mom),
            self.relu
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False),
            BatchNorm2d(in_planes, momentum=bn_mom),
            self.relu
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False),
            BatchNorm2d(in_planes, momentum=bn_mom),
            self.relu,
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes, bias=False),
            BatchNorm2d(in_planes, momentum=bn_mom),
            self.relu
        )
        self.pointwise = nn.Sequential(
            self.PWconv1x1,
            BatchNorm2d(out_planes, momentum=bn_mom),
        )

        self.ca = nn.Sequential(    # 加入通道注意力机制
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_planes*3, (in_planes*3) // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d((in_planes*3) // 16, in_planes*3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x

        x_list = []
        x_list.append(self.scale1(x))
        # print(f"x_list[0]:{x_list[0].shape}")
        x_list.append(self.scale2(x))
        # print(f"x_list[1]:{x_list[1].shape}")
        x_list.append(self.scale3(x))
        # print(f"x_list[2]:{x_list[2].shape}")
        out = torch.cat(x_list, dim=1)

        out = out * (self.ca(out))

        out = self.pointwise(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale_process = nn.Sequential(
            BatchNorm(branch_planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 4, branch_planes * 4, kernel_size=3, padding=1, groups=4, bias=False),
        )

        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                                        mode='bilinear', align_corners=algc) + x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                                        mode='bilinear', align_corners=algc) + x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                                        mode='bilinear', align_corners=algc) + x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                                        mode='bilinear', align_corners=algc) + x_)

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return out


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear')

        return out


class PFAM(nn.Module):     # Parameter-Free Attention Module
    """
    use the module to extract the feature of the low resolution branch
    """
    def __init__(self, Fd_channels=64):
        """
        :param Fd_channels: High resolution branch channel or Detail branch channel
        :param Fs_channels: Low resolution branch channel or Semantic branch channel
        :param BatchNorm:
        """
        super(PFAM, self).__init__()
        self.Fd_channels = Fd_channels
        # self.Fs_channels = Fs_channels
        # self.width_output = width_output
        # self.height_output = height_output

        # self.compression1 = nn.Sequential(
        #     nn.Conv2d(Fs_channels, Fd_channels, kernel_size=1, bias=False),
        #     BatchNorm2d(Fd_channels, momentum=bn_mom)
        # )

        self.conv3x3 = nn.Conv2d(Fd_channels, Fd_channels, kernel_size=3, padding=1, bias=False)
        self.bn = BatchNorm2d(Fd_channels, momentum=bn_mom)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, f_s):

        # f_s = F.interpolate(               # 经过1x1卷积，加强非线性化,然后进行上采样。
        #     self.compression1(self.relu(f_s)),
        #     size=[self.height_output, self.width_output],
        #     mode='bilinear')

        residual = f_s                     # 留出一个残差，用于后面的融合

        out = self.conv3x3(self.relu(f_s))   # 3x3卷积, 进一步提取特征
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv3x3(out)               # 3x3卷积, 进一步提取特征
        out = self.bn(out)

        # sim_att = torch.sigmoid(out)
        sim_att = torch.sigmoid(out) - 0.5    # 关于原点对称的注意力机制

        f_s_ = (out + residual) * sim_att     # 残差特征图 * 注意力特征图

        # # 获取 layers[1] 特征图
        # feature_map = f_s_
        #
        # # 获取通道数
        # num_channels = feature_map.size(1)
        #
        # # 随机选择10个通道索引
        # num_channels_to_save = 10
        # # random_channel_indices = np.random.choice(num_channels, num_channels_to_save, replace=False)
        # random_channel_indices = [1, 5, 9, 11, 23, 25, 27, 29, 33, 45]
        #
        # # 保存每个随机通道的特征图
        # for channel_index in random_channel_indices:
        #     channel_feature = feature_map[:, channel_index:channel_index + 1, :, :]
        #
        #     # 将单通道特征图映射到热力图的RGB颜色
        #     heatmap = self.gray_to_heatmap(channel_feature)
        #
        #     # 创建保存路径
        #     save_path = "output/coalflow/getfeatures/feature_map"
        #     os.makedirs(save_path, exist_ok=True)
        #
        #     # 保存特征图为图像文件
        #     save_image(heatmap, os.path.join(save_path, "feature_map_channel_{}.png".format(channel_index)))

        return f_s_        # 返回最后融合的结果。注意：最后融合后的结果没有进行激活函数。   # 高分辨率分支的结果，低分辨率分支的结果

    def gray_to_heatmap(self, gray_tensor):
        # 将单通道的灰度图映射到热力图的RGB颜色
        heatmap = gray_tensor.detach().cpu().numpy().squeeze()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化到[0, 1]
        heatmap = cm.viridis(heatmap)[:, :, :3]  # 使用viridis颜色映射
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).unsqueeze(0).float()  # 转换为PyTorch张量
        return heatmap

class DE_Module(nn.Module):       # Detail Enhance Module
    """
    use the module to enhance the detail branch feature map
    """

    def __init__(self):
        super(DE_Module, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.DWConv_3x3 = nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2, bias=False)
        self.PWConv_1x1 = nn.Conv2d(2, 1, kernel_size=1, bias=False)

    def forward(self, x):
        """
        再输入特征图之前，是否加入relu,看模型的具体效果，再决定加不加
        """

        residual = x

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.DWConv_3x3(x)
        x = self.PWConv_1x1(x)

        att = self.sigmoid(x)
        out = residual * att

        return out

class DE_Module_1(nn.Module):       # Detail Enhance Module   # spatial attention   + DWConv + PWConv
    """
    use the module to enhance the detail branch feature map
    并行
    """

    def __init__(self, input_channels=64, output_channels=64, ratio=2):
        super(DE_Module_1, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        init_channels = math.ceil(output_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(input_channels, init_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.DWConv_3x3 = nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2, bias=False)
        self.PWConv_1x1 = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        self.compression = nn.Conv2d(output_channels*2, output_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        在输入特征图之前，是否加入relu,看模型的具体效果，再决定加不加
        """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        x_g = torch.cat([x1, x2], dim=1)

        residual = x

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.DWConv_3x3(x)
        x = self.PWConv_1x1(x)

        att = self.sigmoid(x)
        x_s = residual * att

        out = torch.cat([x_s, x_g], dim=1)

        out = self.compression(out)

        return out

class DE_Module_2(nn.Module):       # Detail Enhance Module   # spatial attention   + DWConv + PWConv
    """
    use the module to enhance the detail branch feature map
    串行
    """

    def __init__(self, input_channels=64, output_channels=64, ratio=2):
        super(DE_Module_2, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        init_channels = math.ceil(output_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(input_channels, init_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.DWConv_3x3 = nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2, bias=False)
        self.PWConv_1x1 = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        self.compression = nn.Conv2d(output_channels*2, output_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        在输入特征图之前，是否加入relu,看模型的具体效果，再决定加不加
        """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        x = torch.cat([x1, x2], dim=1)

        residual = x

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.DWConv_3x3(x)
        x = self.PWConv_1x1(x)

        att = self.sigmoid(x)
        out = residual * att

        return out

class FFM(nn.Module):

    def __init__(self, in_channels=256, out_channels=128):
        super(FFM, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU(inplace=False)
        self.bn = BatchNorm2d(out_channels, momentum=bn_mom)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f_s, f_d):

        x = torch.cat([f_s, f_d], dim=1)
        x = self.conv1x1(x)
        x1 = self.bn(x)
        att = self.ca(x1)
        out = x * att
        return out


if __name__ == '__main__':
    pass


















