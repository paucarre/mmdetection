import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from .anchor_head import AnchorHead
from ..registry import HEADS
from ..utils import bias_init_with_prob
from ..utils import weight_init
import math

'''
    MobileNet V2 head 
    
    Paper: 
        MobileNetV2: Inverted Residuals and Linear Bottlenecks
        https://arxiv.org/pdf/1801.04381.pdf

    
    The code is borrowed from https://github.com/tonylins/pytorch-mobilenet-v2
    
    See mmdet/models/backbones/mobilenetv2.py for the backbone-side of the network

'''

@HEADS.register_module
class MobileNetV2Head(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 feat_channels=1280,
                 width_mult=1.,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        self.width_mult = width_mult
        super(MobileNetV2Head, self).__init__(
            num_classes,
            in_channels,
            anchor_scales=anchor_scales,
            use_sigmoid_cls=True,
            use_focal_loss=True,
            **kwargs)

        
    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        # building classifier
        self.last_channels = int(self.feat_channels * self.width_mult) if self.width_mult > 1.0 else self.feat_channels
        self.cls_convs = nn.ModuleList()
        self.bbox_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            in_channels = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.last_channels, 1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(self.last_channels),
                    self.relu
                ))
            self.bbox_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.last_channels, 1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(self.last_channels),
                    self.relu
                ))
        self.mobilenet_cls = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Conv2d( self.last_channels, self.num_anchors * self.cls_out_channels,
                                3, padding=1)
                )
        self.mobilenet_bbox = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Conv2d(self.last_channels, self.num_anchors * 4, 3, padding=1)
                )


    def init_weights(self):
        weight_init.xavier_sequential(self)

    def forward_single(self, x):
        cls_feat = x
        bbox_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for bbox_conv in self.bbox_convs:
            bbox_feat = bbox_conv(bbox_feat)
        cls_score = self.mobilenet_cls(cls_feat)
        bbox_pred = self.mobilenet_bbox(bbox_feat)
        return cls_score, bbox_pred
