DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from valkyrie.tools import round

nn_Module = nn.Module

import collections
import hashlib
import inspect
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import datetime
import zipfile
from collections import defaultdict
import gym
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from scipy.spatial import distance_matrix

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch import tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from valkyrie.ml import utils
from valkyrie.ml.utils import Module, HyperParameters

class Classifier(Module):
  """Defined in :numref:`sec_classification`"""

  def validation_step(self, batch):
    Y_hat = self(*batch[:-1])
    self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
    self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

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

loss = nn.L1Loss(reduction='sum')

def weighted_l1_loss(y1, y2, w):
  return loss(y1, y2)
  #y1, y2, w = y1.view(-1), y2.view(-1), w.view(-1)
  #res = torch.sum(w * torch.abs(y1 - y2)) #/ torch.sum(w)
  #return res

def weighted_l2_loss(y1, y2, w):
  y1, y2, w = y1.view(-1), y2.view(-1), w.view(-1)
  z = y1 - y2
  z = z * z
  return torch.sum(w * z) #/ torch.sum(w)

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

  # def validation_step(self, batch):
  #   Y_hat = self(*batch[:-1])
  #   self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)

  def loss(self, Y_hat, YW, averaged=True):
    Y, W = YW[:,0], YW[:,1]
    Y = utils.astype(utils.reshape(Y, Y_hat.shape), torch.float32)
    return self._loss_fun(Y_hat, Y, W)

class LeNet(Classifier):
  """Defined in :numref:`sec_lenet`"""
  def __init__(self, lr=0.1, num_classes=10):
    print("LeNet Init")
    super().__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(
      nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
      nn.AvgPool2d(kernel_size=2, stride=2),
      nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
      nn.AvgPool2d(kernel_size=2, stride=2),
      nn.Flatten(),
      nn.LazyLinear(16), nn.Sigmoid(),
      nn.LazyLinear(84), nn.Sigmoid(),
      nn.LazyLinear(num_classes))

class Trainer(HyperParameters):
  def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    os.system(f'rm -f ./trainer_logger.txt')
    self.write_log(f'num_gpus = {num_gpus}')

    assert num_gpus == 0, 'No GPU support yet'

  def write_log(self, line):
    with open('./trainer_logger.txt', mode = 'a') as file:
      file.writelines([str(line),'\n'])

  def prepare_data(self, data):
    self.train_dataloader = data.train_dataloader()
    self.val_dataloader = data.val_dataloader()
    self.num_train_batches = len(self.train_dataloader)
    self.num_val_batches = (len(self.val_dataloader)
                            if self.val_dataloader is not None else 0)
    self.loss_total = []

  def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    self.model = model

  def fit(self, model, data):
    self.prepare_data(data)
    self.prepare_model(model)
    self.optim = model.configure_optimizers()
    self.scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.2, patience=5)
    self.train_batch_idx = 0
    self.val_batch_idx = 0

    for self.epoch in range(self.max_epochs):
      self.fit_epoch()
      self.model.epoch += 1

  def fit_epoch(self):
    self.model.train()
    now = datetime.datetime.now()
    self.write_log(f"====== {now} : fitting epoch {self.model.epoch} ")
    epoch_loss, n_samples = tensor(0.0, device=utils.gpu()), tensor(0.0, device=utils.gpu())

    for batch in self.train_dataloader:
      # print("=============== batch ==================")
      # print(batch[0].shape)
      # print(batch[1].shape)
      # print(batch[0])
      # print(batch[1])
      XYW = self.prepare_batch(batch)

      self.optim.zero_grad()
      loss = self.model.training_step(XYW)
      epoch_loss += loss
      n_samples += XYW[0].shape[0]

      with torch.no_grad():
        if self.gradient_clip_val > 0:  # To be discussed later
          self.clip_gradients(self.gradient_clip_val, self.model)
        loss.backward()
        self.optim.step()

      #self.train_batch_idx += 1

    now = datetime.datetime.now()


    self.model.plot('loss', epoch_loss, train=True)

    self.loss_total.append(epoch_loss)
    self.scheduler.step(epoch_loss)
    self.lr = self.optim.param_groups[0]['lr']
    self.model.lr = self.lr
    self.write_log(f"====== {now} : epoch loss={epoch_loss:.4g} lr = {self.lr:.3g}")

    #if not torch.isfinite(epoch_loss):
    #  raise Exception(f"epoch loss is not finite {epoch_loss}")

    # if self.val_dataloader is None:
    #   return
    # self.model.eval()
    # for batch in self.val_dataloader:
    #   with torch.no_grad():
    #     self.model.validation_step(self.prepare_batch(batch))
    #   self.val_batch_idx += 1

  def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    """Defined in :numref:`sec_use_gpu`"""
    self.save_hyperparameters()
    self.gpus = [utils.gpu(i) for i in range(min(num_gpus, utils.num_gpus()))]

  def prepare_batch(self, batch):
    """Defined in :numref:`sec_use_gpu`"""
    if self.gpus:
      batch = [utils.to(a, self.gpus[0]) for a in batch]
    return batch

  def prepare_model(self, model):
    """Defined in :numref:`sec_use_gpu`"""
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
      model.to(self.gpus[0])
    self.model = model

  def clip_gradients(self, grad_clip_val, model):
    """Defined in :numref:`sec_rnn-scratch`"""
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
      for param in params:
        param.grad[:] *= grad_clip_val / norm

class Filter1D(nn.Module): #1D convolution kernel
  def __init__(self, n_out_channel, kernel_size, stride=1, dilation=1, groups=1,
               bias=False, padding_mode='zeros', device=None, dtype=None, **kwargs):
    super(Filter1D, self).__init__(**kwargs)
    n_out_channel = round(n_out_channel)
    padding = int((np.round(kernel_size)-1) / 2)

    self.conv2d = nn.LazyConv2d(n_out_channel, kernel_size=[1, kernel_size], stride=[1, stride],
                            padding=[0, padding], dilation=[1, dilation], groups= groups,
                            bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)

  def forward(self, x):
    d1, d2, d3 = x.shape[0], x.shape[1], x.shape[2]
    y = self.conv2d(x)
    return y

class BN1D(nn.Module): #1D batch normalization
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        super(BN1D, self).__init__(**kwargs)
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        
    def forward(self, x):
        n, d1, d2, d3 = x.shape        
        x = x.view(n, d1*d2, d3)                        
        return self.bn(x).view(n,d1,d2,d3)
