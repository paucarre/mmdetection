import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (VGG, xavier_init, constant_init, kaiming_init,
                      normal_init)
from mmcv.runner import load_checkpoint
from ..registry import BACKBONES
from ..utils import build_norm_layer
import math
'''
    MobileNet V2 backbone 
    
    Paper: 
        MobileNetV2: Inverted Residuals and Linear Bottlenecks
        https://arxiv.org/pdf/1801.04381.pdf

    
    The code is borrowed from https://github.com/tonylins/pytorch-mobilenet-v2

'''

def conv_3x3_stride_2_with_batch_norm(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)


@BACKBONES.register_module
class MobileNetV2(nn.Module):

    def __init__(self, 
                input_channel=32,
                interverted_residual_setting=[
                                                # t, c, n, s
                                                [1, 16, 1, 1],
                                                [6, 24, 2, 2],
                                                [6, 32, 3, 2],
                                                [6, 64, 4, 2],
                                                [6, 96, 3, 1],
                                                [6, 160, 3, 2],
                                                [6, 320, 1, 1],
                                            ],
                width_mult = 1.):
        super(MobileNetV2, self).__init__()
        # building first layer
        input_channel = int(input_channel * width_mult)
        self.features = [conv_3x3_stride_2_with_batch_norm(3, input_channel)]
        # building inverted residual blocks
        self.end_of_block_indices = []
        for expand_ratio, channels, repetitions, stride in interverted_residual_setting:
            output_channel = int(channels * width_mult)
            for i in range(repetitions):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=expand_ratio))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=expand_ratio))
                    if i == repetitions - 1 and stride > 1:
                        # end of block detected
                        self.end_of_block_indices.append(len(self.features) - 1)
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        outputs = []
        for feature in self.features:
            x = feature(x)
            outputs.append(x)
        outputs = [outputs[end_of_block] for end_of_block in self.end_of_block_indices]
        return outputs

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()