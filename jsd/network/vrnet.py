import torch
import torch.nn as nn

from segmentation3d.network.module.weight_init import kaiming_weight_init, \
    gaussian_weight_init
from segmentation3d.network.module.vnet_inblock import InputBlock
from segmentation3d.network.module.vnet_downblock import DownBlock


def parameters_kaiming_init(net):
    """ model parameters initialization """
    net.apply(kaiming_weight_init)


def parameters_gaussian_init(net):
    """ model parameters initialization """
    net.apply(gaussian_weight_init)


class landmark_output_block(nn.Module):
    """ output block of vr-net """

    def __init__(self, in_shape, in_channels, out_channels):
        super(landmark_output_block, self).__init__()

        self.pool = nn.AvgPool3d(in_shape)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3,
                              padding=1)
        self.gn1 = nn.GroupNorm(1, num_channels=in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                              padding=1)
        self.gn2 = nn.GroupNorm(1, num_channels=out_channels)

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        
        out = self.pool(input)
        out = self.act1(self.gn1(self.conv1(out)))
        out = torch.squeeze(self.gn2(self.conv2(out)))
        return out


class RegressionNet(nn.Module):
    """ volumetric segmentation network """

    def __init__(self, in_shape, in_channels, out_channels):
        super(RegressionNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1)
        self.down_64 = DownBlock(32, 2)
        self.down_128 = DownBlock(64, 3)
        self.down_256 = DownBlock(128, 3)
        self.down_512 = DownBlock(256, 3)
        
        shape = [in_shape[idx] // self.max_stride() for idx in range(3)]
        self.out_block = landmark_output_block(shape, 512, out_channels)

    def forward(self, input):
        assert isinstance(input, torch.Tensor)

        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)
        out512 = self.down_512(out256)
        out = self.out_block(out512)
        return out

    def max_stride(self):
        return 32