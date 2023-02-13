import numpy as np
import h5py
import torch
import torch.nn as nn
import random

random.seed(1)


def activation_func(activation, inplace=False):
    '''
    Activation functions
    '''
    if activation is None: return None
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=inplace)],
        ['elu', nn.ELU(inplace=inplace)],
        ['leaky_relu', nn.LeakyReLU(inplace=inplace)],
        ['selu', nn.SELU(inplace=inplace)],
        ['none', nn.Identity()],
    ])[activation]


def normalization_func(input_size, normalization, n_dim):
    '''
    Normalization functions
    '''
    assert input_size in ['1D', '2D'], 'input_size: 1D or 2D.'
    if input_size == '1D':
        return nn.ModuleDict([
            ['batch', nn.BatchNorm1d(n_dim)],
            ['instance', nn.InstanceNorm1d(n_dim)],
            ['layer', nn.LayerNorm(n_dim)],
            ['none', nn.Identity()],
        ])[normalization]

    elif input_size == '2D':
        return nn.ModuleDict([
            ['batch', nn.BatchNorm2d(n_dim)],
            ['instance', nn.InstanceNorm2d(n_dim)],
            ['none', nn.Identity()]
        ])[normalization]


class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = (self.dilation[0] * (self.kernel_size[0] - 1) // 2,)


class ResBlock1d(nn.Module):
    def __init__(self, n_input=256, n_output=256, kernel_size=5, dilation=2,
                 dropout=0.0, activation='elu', normalization='batch', bias=False, *args, **kwargs):
        super().__init__()

        self.block = nn.Sequential(
            activation_func(activation),
            nn.Dropout(p=dropout),
            Conv1dAuto(n_input, n_output, kernel_size=kernel_size, dilation=dilation, bias=bias)
        )
        self.activate = activation_func(activation)
        self.conv1d = Conv1dAuto(n_input, n_output, kernel_size=kernel_size, dilation=dilation, bias=bias)
        self.norm = normalization_func('1D', normalization, 256)

    def forward(self, x1d):
        residual = x1d
        x1d = self.conv1d(x1d)
        x1d = x1d.permute(1, 0)
        x1d = self.norm(x1d)
        x1d = x1d.permute(1, 0)
        x1d = self.block(x1d)
        x1d = x1d.permute(1, 0)
        x1d = self.norm(x1d)
        x1d = x1d.permute(1, 0)
        x1d += residual
        x1d = self.activate(x1d)
        return x1d


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size and dilation
        self.padding = (self.dilation[0] * (self.kernel_size[0] - 1) // 2,
                        self.dilation[1] * (self.kernel_size[1] - 1) // 2)


class ResBlock2d(nn.Module):
    def __init__(self, n_input=553, kernel_size=5, dilation=2,
                 dropout=0.0, activation='elu', normalization='batch', bias=False, *args, **kwargs):
        super().__init__()
        self.block2d = nn.Sequential(
            Conv2dAuto(n_input, n_input, kernel_size=kernel_size, dilation=dilation, bias=bias),
            normalization_func('2D', normalization, n_input),
            activation_func(activation),
            nn.Dropout2d(p=dropout),
            Conv2dAuto(n_input, n_input, kernel_size=kernel_size, dilation=dilation, bias=bias),
            normalization_func('2D', normalization, n_input),
        )
        self.activate = activation_func(activation)

    def forward(self, x2d):
        residual = x2d
        x2d = self.block2d(x2d)
        x2d += residual
        x2d = self.activate(x2d)
        return x2d


class ResNet(nn.Module):

    def __init__(self, n_input1d=256,
                 n_input2d=553, kernel_size2d=5, dilation2d=2, n_ResNet1D_block=4, n_ResNet2D_block=4,
                 dropout=0.0, activation='elu', normalization='batch', bias=False, *args, **kwargs):
        super().__init__()

        self.linear = nn.Linear(2560, 256)
        self.proj_1D = nn.Conv1d(n_input1d, n_input1d, kernel_size=1, bias=True)
        self.ResNet1D_blocks = nn.ModuleList(
            [
                ResBlock1d(
                    n_input1d, n_input1d, kernel_size=3, dilation=1, dropout=0,
                    activation=activation, normalization=normalization,
                    bias=bias) for _ in range(n_ResNet1D_block)
            ]
        )

        self.ResNet2D_blocks = nn.ModuleList(
            [
                ResBlock2d(n_input2d, kernel_size=3, dilation=1, dropout=0.0,
                           activation='elu', normalization='batch', bias=False) for _ in range(n_ResNet2D_block)
            ]
        )

        self.activate = activation_func(activation)
        self.finalconv2d = nn.Sequential(
            nn.ReLU(),
            Conv2dAuto(n_input2d, 2, kernel_size=kernel_size2d, dilation=dilation2d, bias=bias)
        )

    def forward(self, x1d, x2d):

        x1d = self.linear(x1d)
        x1d = x1d.permute(1, 0)
        print(x1d.shape)
        x1d = self.proj_1D(x1d)
        for block in self.ResNet1D_blocks:
            x1d = block(x1d)
        x1d = self.activate(x1d)
        e = torch.ones(1, x1d.shape[1], 1)
        e = e.to('cuda')
        x1d = x1d.reshape(x1d.shape[0], 1, x1d.shape[1]) * e
        x1d = torch.cat((x1d, torch.transpose(x1d, 1, 2)), 0)
        x = torch.cat((x1d, x2d), dim=0)
        x = x.reshape(1, x.shape[0], x.shape[1], -1)
        for block in self.ResNet2D_blocks:
            x = block(x)
        x = self.finalconv2d(x)
        return x


import time

def ContactM(coor):
    x=torch.tensor(coor[:,3,0].reshape(-1,1))
    y=torch.tensor(coor[:,3,1].reshape(-1,1))
    z=torch.tensor(coor[:,3,2].reshape(-1,1))
    distance_m=torch.sqrt((x-x.T)**2+(y-y.T)**2+(z-z.T)**2)
    l=coor.shape[0]
    noncon_m=(distance_m>=8).long()
    return noncon_m.reshape(1,noncon_m.shape[0],-1)


