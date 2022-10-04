import math
import torch
import warnings
import torch.nn as nn
import collections.abc
import torch.utils.checkpoint as cp
import torch.nn.functional as F 
from typing import Sequence
from itertools import repeat
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)

from mmcv.utils import digit_version
from ..builder import BACKBONES
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule, Sequential
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init, DepthwiseSeparableConvModule)

eps = 1.0e-5

#Class copied from mmclassification/mmcls/models/backbones/utils/helpers.py
def _ntuple(n):
    """A `to_tuple` function generator.

    It returns a function, this function will repeat the input to a tuple of
    length ``n`` if the input is not an Iterable object, otherwise, return the
    input directly.

    Args:
        n (int): The number of the target length.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

# Class copied from mmclassification/mmcls/models/backbones/resnet.py with few modifications
class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LeakyReLU', inplace=True), #dict(type='ReLU', inplace=True),
                 drop_path_rate=0.0,
                 init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg=init_cfg)
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

# Class copied from mmclassification/mmcls/models/backbones/resnet.py with few modifications
def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        #elif issubclass(block, BasicBlock):
        #    expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)

# Class copied from mmclassification/mmcls/models/backbones/cspnet.py with few modifications
class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet. Each DarknetBottleneck
    consists of two ConvModules and the input is added to the final output.
    Each ConvModule is composed of Conv, BN, and LeakyReLU. The first convLayer
    has filter size of 1x1 and the second one has the filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2.
            Defaults to 4.
        add_identity (bool): Whether to add identity to the out.
            Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d.
        drop_path_rate (float): The ratio of the drop path layer. Default: 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN', eps=1e-5)``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='Swish')``.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=2,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 drop_path_rate=0,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels / expansion)

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, hidden_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            hidden_channels,
            kernel_size=1)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            hidden_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.add_module(self.norm2_name, norm2)
        self.act = nn.LeakyReLU(inplace=True)
        # self.conv1 = ConvModule(
        #     in_channels,
        #     hidden_channels,
        #     1,
        #     conv_cfg=conv_cfg,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg)
        # self.conv2 = conv(
        #     hidden_channels,
        #     out_channels,
        #     3,
        #     stride=1,
        #     padding=1,
        #     conv_cfg=conv_cfg,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x
        # out = self.conv1(x)
        # out = self.conv2(out)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.drop_path(out)

        if self.add_identity:
            return out + identity
        else:
            return out

# Class copied from mmclassification/mmcls/models/backbones/cspnet.py with few modifications
class CSPStage(BaseModule):
    """Cross Stage Partial Stage.

    Args:
        block_fn (nn.module): The basic block function in the Stage.
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        has_downsampler (bool): Whether to add a downsampler in the stage.
            Default: False.
        down_growth (bool): Whether to expand the channels in the
            downsampler layer of the stage. Default: False.
        expand_ratio (float): The expand ratio to adjust the number of
             channels of the expand conv layer. Default: 0.5
        bottle_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        block_dpr (float): The ratio of the drop path layer in the
            blocks of the stage. Default: 0.
        num_blocks (int): Number of blocks. Default: 1
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', inplace=True)
    """

    def __init__(self,
                 block_fn,
                 in_channels,
                 out_channels,
                 has_downsampler=True,
                 down_growth=False,
                 expand_ratio=0.5,
                 bottle_ratio=2,
                 num_blocks=1,
                 block_dpr=0,
                 block_args={},
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        # grow downsample channels to output channels
        down_channels = out_channels if down_growth else in_channels
        block_dpr = to_ntuple(num_blocks)(block_dpr)

        if has_downsampler:
            self.downsample_conv = ConvModule(
                in_channels=in_channels,
                out_channels=down_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.downsample_conv = nn.Identity()

        exp_channels = int(down_channels * expand_ratio)
        self.expand_conv = ConvModule(
            in_channels=down_channels,
            out_channels=exp_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg if block_fn is DarknetBottleneck else None)

        assert exp_channels % 2 == 0, \
            'The channel number before blocks must be divisible by 2.'
        block_channels = exp_channels // 2
        blocks = []
        for i in range(num_blocks):
            block_cfg = dict(
                in_channels=block_channels,
                out_channels=block_channels,
                expansion=bottle_ratio,
                drop_path_rate=block_dpr[i],
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **block_args)
            blocks.append(block_fn(**block_cfg))
        self.blocks = Sequential(*blocks)
        self.atfer_blocks_conv = ConvModule(
            block_channels,
            block_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.final_conv = ConvModule(
            2 * block_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.downsample_conv(x)
        x = self.expand_conv(x)

        split = x.shape[1] // 2
        xa, xb = x[:, :split], x[:, split:]

        xb = self.blocks(xb)
        xb = self.atfer_blocks_conv(xb).contiguous()

        x_final = torch.cat((xa, xb), dim=1)
        return self.final_conv(x_final)

@BACKBONES.register_module()
class CRNet(BaseModule):
    def __init__(self,
            in_channels=(3, 32, 64, 128, 256, 512),
            out_channels=(32, 64, 128, 256, 512, 1024),
            num_blocks=(3, 4, 6, 4, 3),
            stride=(1, 1, 1, 1, 1),
            expand_ratio=2,
            bottle_ratio=1,
            has_downsampler=(False, True, True, True, True),
            down_growth=False,
            frozen_stages=-1, # 3 stages frozen instead of -1.
            out_indices=-1,
            norm_eval=False,
            zero_init_residual=True,
        ):
        super().__init__()
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.zero_init_residual = zero_init_residual
        
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        #super().init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, DarknetBottleneck):
                        constant_init(m.norm2, 0)

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
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)