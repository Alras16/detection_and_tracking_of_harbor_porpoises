import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Sequence

from ..builder import BACKBONES
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

# class Mish(nn.Module):
#     def __init__(self, inplace=False):
#         super(Mish, self).__init__()
#         nn.Module.__init__(self)
#         self.inplace = inplace

#     def forward(self, x):
#         return x *( torch.tanh(F.softplus(x)))

class BasicBlock(BaseModule):
    def __init__(self, 
            in_channels, 
            out_channels, 
            expansion=1, 
            stride=1, 
            dilation=1, 
            groups=1, 
            downsample=None,
        ):
        super(BasicBlock).__init__()
        self.mid_channels = out_channels // expansion
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.norm1 = nn.BatchNorm2d(self.mid_channels, eps=1.0e-5) 
        self.act1  = nn.LeakyReLU(negative_slope=0.1, inplace=True) #Mish()
        self.conv2 = nn.Conv2d(self.mid_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels, eps=1.0e-5)
        self.act2  = nn.LeakyReLU(negative_slope=0.1, inplace=True) #Mish()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.act2(out)
        return out

class BottleNeckBlock(BaseModule):
    def __init__(self, 
            in_channels, 
            out_channels, 
            expansion=1, 
            stride=1, 
            dilation=1, 
            groups=1, 
            downsample=None,
        ):
        super(BottleNeckBlock).__init__()
        self.mid_channels = out_channels // expansion
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=1, bias=False)
        self.norm1 = nn.BatchNorm2d(self.mid_channels, eps=1.0e-5)
        self.act1  = nn.LeakyReLU(negative_slope=0.1, inplace=True) #Mish()
        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.norm2 = nn.BatchNorm2d(self.mid_channels, eps=1.0e-5) 
        self.act2  = nn.LeakyReLU(negative_slope=0.1, inplace=True) #Mish()
        self.conv3 = nn.Conv2d(self.mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels, eps=1.0e-5) 
        self.act3  = nn.LeakyReLU(negative_slope=0.1, inplace=True) #Mish()
        self.downsample = downsample


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.act3(out)
        return out

class CSPStage(BaseModule):
    def __init__(self, 
            block_fn, 
            in_channels, 
            out_channels, 
            stride=1, 
            has_downsampler=True, 
            down_growth=True, 
            expand_ratio=0.5, 
            bottle_ratio=2, 
            num_blocks=1
        ):
        super.__init__()
        self.has_downsampler = has_downsampler
        # Downsample convolutional layer
        down_channels = out_channels if down_growth else in_channels
        self.downsample_conv = nn.Sequential(
            nn.Conv2d(in_channels, down_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(down_channels, eps=1.0e-5),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) #Mish()
        )
        # Expansion convolutional layer
        exp_channels = int(down_channels * expand_ratio)
        self.expand_conv = nn.Sequential(
            nn.Conv2d(down_channels, exp_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(down_channels, eps=1.0e-5)
        )
        assert exp_channels % 2 == 0, \
            'The channel number before blocks must be divisible by 2.'
        block_channels = exp_channels // 2
        blocks = []
        for block in range(num_blocks):
            block_cfg = dict(
                in_channels=block_channels,
                out_channels=block_channels,
                stride=stride,
                expansion=bottle_ratio,
            )
            blocks.append(block_fn(**block_cfg))
        self.blocks = nn.Sequential(*blocks)
        # Transition convolutional layer
        self.transit_conv = nn.Sequential(
            nn.Conv2d(block_channels, block_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(down_channels, eps=1.0e-5),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) #Mish()
        )
        # Final convolutional layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(2*block_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(down_channels, eps=1.0e-5),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) #Mish()
        )

    def forward(self, x):
        # Optional downsampling convolutional layer
        if self.has_downsampler:
            x = self.downsample_conv(x)

        # Expansion convolutional layer
        x = self.expand_conv(x)

        # Split feature map into two parts
        split = x.shape[1] // 2
        xa, xb = x[:, :split], x[:, split:]

        # Pass one part of the feature map through
        # the blocks and a transition layer
        xb = self.blocks(xb)
        xb = self.transit_conv(xb)

        # Concatenate the two feature map parts and then
        # pass through yet another transition layer
        xf = torch.cat((xa, xb), dim=1)
        xf = self.final_conv(xf)
        return xf

class ResStage(BaseModule):
    def __init__(self, 
            block_fn, 
            in_channels, 
            out_channels, 
            expansion, 
            stride, 
            has_downsampler=True, 
            num_blocks=1
        ):
        self.has_downsampler=has_downsampler
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(out_channels, eps=1.0e-5),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), #Mish()
        )
        
        blocks = []
        for block in range(num_blocks):
            if not blocks:
                block_cfg = dict(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion = expansion,
                    stride=stride,

                )
                blocks.append(block_fn(**block_cfg))
            else:
                in_channels = out_channels
                block_cfg = dict(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion = expansion,
                    stride=1,
                )
                blocks.append(block_fn(**block_cfg))            
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        if self.has_downsampler:
            x = self.downsample(x)
        out = self.blocks(x)
        return out

@BACKBONES.register_module()
class CRNet(BaseModule):
    def __init__(self,
            in_channels=(3, 32, 64, 128, 256, 512),
            out_channels=(32, 64, 128, 256, 512, 1024),
            num_blocks=(3, 4, 6, 4, 3),
            stride=(1, 2, 2, 2, 2),
            expand_ratio=4,
            bottle_ratio=2,
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
            nn.LeakyReLU(negative_slope=0.1, inplace=False), #Mish()
            nn.Conv2d(mid_channels, out_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[0], eps=1.0e-5),
            nn.LeakyReLU(negative_slope=0.1, inplace=False), #Mish()
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        stages = []
        num_stages = len(in_channels)
        for stage in range(1, num_stages):
            if stage % 2 == 0:
                csp_stage = CSPStage(
                    block_fn=BasicBlock,
                    in_channels=in_channels[stage],
                    out_channels=out_channels[stage],
                    stride=stride[stage-1],
                    has_downsampler=has_downsampler[stage-1],
                    down_growth=down_growth,
                    expand_ratio=expand_ratio,
                    bottle_ratio=bottle_ratio,
                    num_blocks=num_blocks[stage-1],
                    
                )
                stages.append(csp_stage)
            else:
                res_stage = ResStage(
                    block_fn=BottleNeckBlock,
                    in_channels=in_channels[stage],
                    out_channels=out_channels[stage],
                    expansion=bottle_ratio,
                    stride=stride[stage-1],
                    has_downsampler=has_downsampler[stage-1],
                    num_blocks=num_blocks[stage-1],
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
        super(CRNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        x = self.stem_layers(x)
        for i, stage in range(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)