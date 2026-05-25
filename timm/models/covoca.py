"""CVoCA-style complex-valued convolutional image classifiers.

This module provides software complex-valued convolution blocks for controlled
experiments with complex feature processing on RGB image classification tasks.
It is inspired by complex-valued optical convolution accelerators (CVoCA), but
does not model optical hardware effects.
"""
from typing import Any, Dict, Optional, Sequence, Set, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, SelectAdaptivePool2d, calculate_drop_path_rates, trunc_normal_
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model

__all__ = ['CVoCA']


class ComplexConv2d(nn.Module):
    """Complex-valued convolution on channel-concatenated real and imaginary features."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            bias: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.real_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, **dd)
        self.imag_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = x.chunk(2, dim=1)
        y_real = self.real_conv(x_real) - self.imag_conv(x_imag)
        y_imag = self.real_conv(x_imag) + self.imag_conv(x_real)
        return torch.cat((y_real, y_imag), dim=1)


class ComplexBatchNorm2d(nn.Module):
    """Split batch normalization for channel-concatenated complex features."""

    def __init__(
            self,
            channels: int,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.real_norm = nn.BatchNorm2d(channels, **dd)
        self.imag_norm = nn.BatchNorm2d(channels, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = x.chunk(2, dim=1)
        return torch.cat((self.real_norm(x_real), self.imag_norm(x_imag)), dim=1)


class SplitActivation(nn.Module):
    """Apply a real-valued activation independently to real and imaginary parts."""

    def __init__(self, act_layer: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.real_act = act_layer()
        self.imag_act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = x.chunk(2, dim=1)
        return torch.cat((self.real_act(x_real), self.imag_act(x_imag)), dim=1)


class ComplexMagnitude(nn.Module):
    """Convert channel-concatenated complex features to real magnitudes."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = x.chunk(2, dim=1)
        return torch.sqrt(x_real.square() + x_imag.square() + self.eps)


class ComplexInputStem(nn.Module):
    """Lift real RGB inputs into the complex domain and patchify them."""

    def __init__(
            self,
            in_chans: int,
            out_channels: int,
            patch_size: int = 4,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.proj = ComplexConv2d(
            in_chans, out_channels, kernel_size=patch_size, stride=patch_size, bias=False, **dd)
        self.norm = ComplexBatchNorm2d(out_channels, **dd)
        self.act = SplitActivation(act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, torch.zeros_like(x)), dim=1)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ComplexDownsample(nn.Module):
    """Stride-2 complex convolution used between stages."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, **dd)
        self.norm = ComplexBatchNorm2d(out_channels, **dd)
        self.act = SplitActivation(act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ComplexConvBlock(nn.Module):
    """ConvNeXt-style residual block built from complex-valued convolutions."""

    def __init__(
            self,
            dim: int,
            kernel_size: int = 7,
            mlp_ratio: float = 4.,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        hidden_dim = int(dim * mlp_ratio)
        padding = kernel_size // 2
        self.dwconv = ComplexConv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False, **dd)
        self.norm1 = ComplexBatchNorm2d(dim, **dd)
        self.act1 = SplitActivation(act_layer)
        self.pwconv1 = ComplexConv2d(dim, hidden_dim, kernel_size=1, bias=False, **dd)
        self.norm2 = ComplexBatchNorm2d(hidden_dim, **dd)
        self.act2 = SplitActivation(act_layer)
        self.pwconv2 = ComplexConv2d(hidden_dim, dim, kernel_size=1, bias=False, **dd)
        self.norm3 = ComplexBatchNorm2d(dim, **dd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.pwconv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        x = self.norm3(x)
        x = self.drop_path(x)
        return shortcut + x


class CVoCA(nn.Module):
    """Complex-valued convolutional classifier for image recognition experiments."""

    def __init__(
            self,
            depths: Sequence[int] = (2, 2, 6, 2),
            dims: Sequence[int] = (48, 96, 192, 384),
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            kernel_size: int = 7,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            output_stride: int = 32,
            device=None,
            dtype=None,
            **kwargs,
    ):
        super().__init__()
        assert output_stride == 32
        assert len(depths) == len(dims)
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_features = self.head_hidden_size = dims[-1]
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        self.stem = ComplexInputStem(in_chans, dims[0], patch_size=4, act_layer=act_layer, **dd)

        dpr = calculate_drop_path_rates(drop_path_rate, sum(depths))
        stages = []
        feature_info = []
        block_idx = 0
        prev_dim = dims[0]
        for stage_idx, (depth, dim) in enumerate(zip(depths, dims)):
            stage_layers = []
            if stage_idx > 0:
                stage_layers.append(ComplexDownsample(prev_dim, dim, act_layer=act_layer, **dd))
            stage_layers.extend([
                ComplexConvBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[block_idx + block_offset],
                    act_layer=act_layer,
                    **dd,
                )
                for block_offset in range(depth)
            ])
            block_idx += depth
            prev_dim = dim
            stages.append(nn.Sequential(*stage_layers))
            feature_info.append(dict(
                num_chs=dim * 2,
                reduction=4 * 2 ** stage_idx,
                module=f'stages.{stage_idx}',
            ))
        self.stages = nn.Sequential(*stages)
        self.feature_info = feature_info

        self.norm = ComplexBatchNorm2d(self.num_features, **dd)
        self.to_magnitude = ComplexMagnitude()
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes, **dd) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else r'^stages\.(\d+)\.(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head = nn.Linear(
            self.head_hidden_size,
            num_classes,
            device=self.head.weight.device if isinstance(self.head, nn.Linear) else None,
            dtype=self.head.weight.dtype if isinstance(self.head, nn.Linear) else None,
        ) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.to_magnitude(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': ('stem.proj.real_conv', 'stem.proj.imag_conv'), 'classifier': 'head',
        'origin_url': 'https://www.nature.com/articles/s41467-024-55321-8',
        'license': 'apache-2.0', **kwargs
    }


default_cfgs = generate_default_cfgs({
    'covoca_tiny.untrained': _cfg(),
    'covoca_small.untrained': _cfg(),
    'covoca_base.untrained': _cfg(),
    'cvoca_tiny.untrained': _cfg(),
    'cvoca_small.untrained': _cfg(),
    'cvoca_base.untrained': _cfg(),
})


def _create_covoca(variant: str, pretrained: bool = False, **kwargs: Any) -> CVoCA:
    model = build_model_with_cfg(
        CVoCA,
        variant,
        pretrained,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def covoca_tiny(pretrained: bool = False, **kwargs: Any) -> CVoCA:
    model_args = dict(depths=(1, 1, 3, 1), dims=(32, 64, 128, 256), mlp_ratio=3.)
    return _create_covoca('covoca_tiny', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def covoca_small(pretrained: bool = False, **kwargs: Any) -> CVoCA:
    model_args = dict(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), mlp_ratio=4.)
    return _create_covoca('covoca_small', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def covoca_base(pretrained: bool = False, **kwargs: Any) -> CVoCA:
    model_args = dict(depths=(3, 3, 9, 3), dims=(64, 128, 256, 512), mlp_ratio=4.)
    return _create_covoca('covoca_base', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cvoca_tiny(pretrained: bool = False, **kwargs: Any) -> CVoCA:
    model_args = dict(depths=(1, 1, 3, 1), dims=(32, 64, 128, 256), mlp_ratio=3.)
    return _create_covoca('cvoca_tiny', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cvoca_small(pretrained: bool = False, **kwargs: Any) -> CVoCA:
    model_args = dict(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), mlp_ratio=4.)
    return _create_covoca('cvoca_small', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def cvoca_base(pretrained: bool = False, **kwargs: Any) -> CVoCA:
    model_args = dict(depths=(3, 3, 9, 3), dims=(64, 128, 256, 512), mlp_ratio=4.)
    return _create_covoca('cvoca_base', pretrained=pretrained, **dict(model_args, **kwargs))
