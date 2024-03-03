import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LeakyReLU as LReLU

import d2l
from valkyrie.ml import utils
from valkyrie.ml.utils import HyperParameters, weighted_l2_loss, weighted_l1_loss
from valkyrie.ml.modules import Module

import collections
import hashlib
import inspect


class Filter1D(nn.Module):  # 1D convolution kernel # this is old code
    def __init__(self, n_out_channel, kernel_size, stride=1, dilation=1, groups=1,
                 bias=False, padding_mode='zeros', device=None, dtype=None, **kwargs):
        super(Filter1D, self).__init__(**kwargs)
        n_out_channel = round(n_out_channel)
        padding = int((np.round(kernel_size) - 1) / 2)

        self.conv2d = nn.LazyConv2d(n_out_channel, kernel_size=[1, kernel_size], stride=[1, stride],
                                    padding=[0, padding], dilation=[1, dilation], groups=groups,
                                    bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)

    def forward(self, x):
        d1, d2, d3 = x.shape[0], x.shape[1], x.shape[2]
        y = self.conv2d(x)
        return y


class BatchNormAlongW(nn.Module):
    def __init__(self, num_channels, height):
        super(BatchNormAlongW, self).__init__()
        self.num_channels = num_channels
        self.height = height
        self.batch_norm = nn.BatchNorm1d(num_channels * height)
        # nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        # eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs

    def forward(self, x):
        N, C, H, W = x.shape
        # Check if the input dimensions match the expected dimensions
        assert C == self.num_channels and H == self.height, "Input dimensions do not match"

        # Permute and reshape tensor to shape (N, H, C*W)
        x = x.view(N, C * H, -1)

        # Apply BatchNorm1d
        x = self.batch_norm(x)

        # Reshape and permute back to original shape (N, C, H, W)
        return x.view(N, C, H, W)


class BatchNormAlongW(nn.Module):
    def __init__(self, num_channels, height):
        super(BatchNormAlongW, self).__init__()
        self.num_channels = num_channels
        self.height = height
        self.batch_norm = nn.BatchNorm1d(num_channels * height)
        # nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        # eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs

    def forward(self, x):
        N, C, H, W = x.shape
        # Check if the input dimensions match the expected dimensions
        assert C == self.num_channels and H == self.height, "Input dimensions do not match"

        # Permute and reshape tensor to shape (N, H, C*W)
        x = x.view(N, C * H, -1)

        # Apply BatchNorm1d
        x = self.batch_norm(x)

        # Reshape and permute back to original shape (N, C, H, W)
        return x.view(N, C, H, W)


class Regressor(Module):
    def __init__(self, func):
        if inspect.isfunction(func):
            self._loss_fun = func
        elif type(func) == str:
            kind2loss = {'l2': weighted_l2_loss,
                         'l1': weighted_l1_loss}
            self._loss_fun = kind2loss[func.lower()]
        super().__init__()
        self.epoch = 0

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    # def validation_step(self, batch):
    #    Y_hat = self(*batch[:-1])
    #    self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
    #    return self.loss(Y_hat, batch[-1])

    def loss(self, Y_hat, YW, averaged=True):
        Y, W = YW[:, 0], YW[:, 1]
        Y = utils.astype(utils.reshape(Y, Y_hat.shape), torch.float32)
        return self._loss_fun(Y_hat, Y, W)


class Classifier(Module):
    """Defined in :numref:`sec_classification`"""

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
        return self.loss(Y_hat, batch[-1])

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions.

        Defined in :numref:`sec_classification`"""
        Y_hat = utils.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = utils.astype(utils.argmax(Y_hat, axis=1), Y.dtype)
        compare = utils.astype(preds == utils.reshape(Y, -1), utils.float32)
        return utils.reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        # Y = utils.astype(utils.reshape(Y, Y_hat.shape), torch.float32)
        # return F.mse_loss(Y_hat, Y)
        Y_hat = utils.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = utils.reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')


class LrNet(Regressor):
    def __init__(self, lr):
        print("Lr Net")
        super().__init__('l1')
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1, bias=True)
        )


class AlexRegNet(Regressor):
    def __init__(self, lr, num_channels, height, down_mul=4):
        super().__init__('l1')
        self.save_hyperparameters()
        self.net = nn.Sequential(
            BatchNormAlongW(num_channels, height),
            nn.LazyConv2d(int(96 / down_mul), kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(int(384 / down_mul), kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.LazyConv2d(int(256 / down_mul), kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(int(4096 / (down_mul * down_mul))), nn.LeakyReLU(0.1), nn.Dropout(p=0.5),
            nn.LazyLinear(1))
        self.net.apply(d2l.torch.init_cnn)


class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(GoogleNet.Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.leaky_relu(self.b1_1(x), 0.1)
        b2 = F.leaky_relu(self.b2_2(F.leaky_relu(self.b2_1(x), 0.1)), 0.1)
        b3 = F.leaky_relu(self.b3_2(F.leaky_relu(self.b3_1(x), 0.1)), 0.1)
        b4 = F.leaky_relu(self.b4_2(self.b4_1(x)), 0.1)
        return torch.cat((b1, b2, b3, b4), dim=1)


class GoogleNet(Regressor):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b2(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.LeakyReLU(0.1),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b3(self):
        return nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b4(self):
        return nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b5(self):
        return nn.Sequential(
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    def b0(self):
        return BatchNormAlongW(self.n_channel, self.height)

    def __init__(self, n_channel, height, lr=0.1, num_classes=10):
        # super(GoogleNet, self).__init__()
        self.n_channel = n_channel
        self.height = height
        super(GoogleNet, self).__init__('l1')
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b0(), self.b1(), self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(1))
        self.net.apply(d2l.init_cnn)

