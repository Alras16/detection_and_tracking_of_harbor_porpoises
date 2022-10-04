import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Sequence

from ..builder import BACKBONES
from mmcv.runner import BaseModule, Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from .resnet import Bottleneck, ResLayer 
from .cspnet import DarknetBottleneck, CSPStage

@BACKBONES.register_module()
class CRNetv2(BaseModule):
    def __init__(self,
            in_channels=(3, 32, 64, 128, 256, 512),
            out_channels=(32, 64, 128, 256, 512, 1024),
            num_blocks=(3, 4, 6, 4, 3),
            stride=(1, 2, 2, 2, 2),
            expand_ratio=2,
            bottle_ratio=1,
            has_downsampler=(False, True, True, True, True),
            down_growth=False,
            frozen_stages=-1,
            out_indices=-1,
            norm_eval=False,
        ):
        super().__init__()
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        
        mid_channels = out_channels[0]//2
        self.stem_layers = nn.Sequential(
            nn.Conv2d(in_channels[0], mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, eps=1.0e-5),
            nn.LeakyReLU(negative_slope=0.01, inplace=False), #Mish()
            nn.Conv2d(mid_channels, out_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[0], eps=1.0e-5),
            nn.LeakyReLU(negative_slope=0.01, inplace=False), #Mish()
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        stages = []
        num_stages = len(in_channels)
        for stage in range(1, num_stages):
            if stage % 2 == 0:
                csp_stage = CSPStage(
                    block_fn=DarknetBottleneck, #BasicBlock,
                    in_channels=in_channels[stage],
                    out_channels=out_channels[stage],
                    has_downsampler=has_downsampler[stage-1],
                    down_growth=down_growth,
                    expand_ratio=expand_ratio,
                    bottle_ratio=bottle_ratio,
                    num_blocks=num_blocks[stage-1],
                    
                )
                stages.append(csp_stage)
            else:
                res_stage = ResLayer(
                    block=Bottleneck,
                    num_blocks=num_blocks[stage-1],
                    in_channels=in_channels[stage],
                    out_channels=out_channels[stage],
                    stride=stride[stage-1]
                )
                stages.append(res_stage)
            self.stages = Sequential(*stages)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = len(self.stages) + index
            assert 0 <= out_indices[i] <= len(self.stages), \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices
    
    @staticmethod
    def expand_arch(arch):
        num_stages = len(arch['in_channels'])

        def to_tuple(x, name=''):
            if isinstance(x, (list, tuple)):
                assert len(x) == num_stages, \
                    f'The length of {name} ({len(x)}) does not ' \
                    f'equals to the number of stages ({num_stages})'
                return tuple(x)
            else:
                return (x, ) * num_stages

        full_arch = {k: to_tuple(v, k) for k, v in arch.items()}
        if 'block_args' not in full_arch:
            full_arch['block_args'] = to_tuple({})
        return full_arch

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem_layers.eval()
            for param in self.stem_layers.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(CRNetv2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        x = self.stem_layers(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)