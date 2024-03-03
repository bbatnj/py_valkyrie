from valkyrie.tools import format_df_nums

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
from valkyrie.ml.utils import HyperParameters, ProgressBoard, cpu
from valkyrie.ml.utils import to_numpy, to, weighted_l2_loss, weighted_l1_loss

class NetEvaluator:
    def __init__(self, model, dm, device='cuda:0'):
        self.model = model
        self.dm  = dm
        self.device = device

    def eval_performance(self, is_train = True):
        self.Y, self.Y_hat = [], []
        loss, init_loss = 0.0, 0.0
        dl = self.dm.get_dataloader(is_train)
        n = 0
        for i, batch in enumerate(dl):
            Y = batch[1][:,0].to(self.device)
            with torch.no_grad():
                Y_hat = self.model(batch[0].to(self.device))
                loss += (Y_hat.view(-1) - Y.view(-1)).abs().sum()
                init_loss += Y.abs().sum()
            self.Y.append(Y)
            self.Y_hat.append(Y_hat)
            n += batch[0].shape[0]
        score = 1.0 - loss/init_loss
        return score, loss / n, init_loss /n

class Trainer(HyperParameters):
  def __init__(self, max_epochs, log_fn, gpu_index = 0, plot = True, gradient_clip_val=0):
    self.save_hyperparameters()
    os.system('mkdir -p ./trainer_logs/')
    self.log_fn = f'./trainer_logs/{log_fn}'
    os.system(f'rm -f {self.log_fn}')
    self.gpus = [utils.gpu(i) for i in range(utils.num_gpus())]
    self.gpu_index = gpu_index
    num_gpus = {len(self.gpus)}
    self.write_log(f'num_gpus = {num_gpus}')
    self.shall_plot = plot

    assert num_gpus != 0, 'No GPU support yet'


  def write_log(self, line):
    with open(f'{self.log_fn}', mode = 'a') as file:
      file.writelines([str(line),'\n'])

  def prepare_data(self, data):
    self.train_dataloader  = data.train_dataloader()
    self.val_dataloader    = data.val_dataloader()
    self.num_train_batches = len(self.train_dataloader)
    self.num_val_batches   = (len(self.val_dataloader)
                            if self.val_dataloader is not None else 0)


    self.loss_total = []

  def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    self.model = model

  def fit(self, model, data):
    self.prepare_data(data)
    self.prepare_model(model)

    self.net_evaluator = NetEvaluator(self.model, data)

    _, _, self.init_train_ps = self.net_evaluator.eval_performance(True)
    _, _, self.init_val_ps = self.net_evaluator.eval_performance(False)

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
    n_train_samples, n_val_samples = 0, 0
    epoch_loss = torch.tensor([0.0], device='cuda:0')

    for batch in self.train_dataloader:
      XYW = self.prepare_batch(batch)

      self.optim.zero_grad()
      loss = self.model.training_step(XYW)
      epoch_loss += float(loss)
      n_train_samples += XYW[0].shape[0]

      with torch.no_grad():
        if self.gradient_clip_val > 0:  # To be discussed later
          self.clip_gradients(self.gradient_clip_val, self.model)
        loss.backward()
        self.optim.step()

      self.train_batch_idx += 1

    now = datetime.datetime.now()

    self.loss_total.append(epoch_loss)
    self.scheduler.step(epoch_loss)
    self.lr = self.optim.param_groups[0]['lr']
    self.model.lr = self.lr

    if not torch.isfinite(epoch_loss):
      raise Exception(f"epoch loss is not finite {epoch_loss}")

    if self.val_dataloader is None:
      return

    self.model.eval()
    val_loss = torch.tensor([0.0], device='cuda:0')

    for batch in self.val_dataloader:
      with torch.no_grad():
        XYW = self.prepare_batch(batch)
        val_loss += float(self.model.validation_step(XYW))
      self.val_batch_idx += 1
      n_val_samples += XYW[0].shape[0]

    init_train_loss_ps = self.init_train_ps
    init_val_loss_ps = self.init_val_ps
    train_loss_ps = epoch_loss / n_train_samples
    val_loss_ps = val_loss / n_val_samples
    if self.shall_plot:
        self.model.plot('loss', train_loss_ps, train=True)
        self.model.plot('loss', val_loss_ps, train=False)

    res  = pd.Series(
                  {f"train_score " : f'{float(1 - train_loss_ps/init_train_loss_ps):3g}',
                   f"init_train_loss" : f'{float(init_train_loss_ps):3g}',
                   f"train_loss" : f'{float(train_loss_ps):3g}',
                   f"val_score" : f'{float(1 - val_loss_ps/init_val_loss_ps):3g}',
                   f"init_val_loss" : f'{float(init_val_loss_ps):3g}',
                   f"val loss" : f"{float(val_loss_ps):3g}",
                   f"lr" : f"{self.lr:3g}"})

    res = format_df_nums(pd.DataFrame(res).T)

    self.write_log(f"====== {now} done for epoch:\n{res}")

  def prepare_batch(self, batch):
    """Defined in :numref:`sec_use_gpu`"""
    batch = [utils.to(a, self.gpus[self.gpu_index]) for a in batch]
    return batch

  def prepare_model(self, model):
    """Defined in :numref:`sec_use_gpu`"""
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    model.to(self.gpus[self.gpu_index])
    self.model = model

  def clip_gradients(self, grad_clip_val, model):
    """Defined in :numref:`sec_rnn-scratch`"""
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
      for param in params:
        param.grad[:] *= grad_clip_val / norm



class Module(nn_Module, HyperParameters):
    """Defined in :numref:`sec_oo-design`"""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        #if train:
        #    x = self.trainer.train_batch_idx / \
        #        self.trainer.num_train_batches
            #n = self.trainer.num_train_batches / \
            #    self.plot_train_per_epoch
        #else:
        #    x = self.trainer.epoch + 1
            #n = self.trainer.num_val_batches / \
            #    self.plot_valid_per_epoch
        self.board.draw(self.epoch, to_numpy(to(value, cpu())),
                        ('train_' if train else 'val_') + key,
                        f'epoch = {self.epoch}, lr = {self.lr}',
                        every_n=1)
                        #every_n = int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        #self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
      pass
        #l = self.loss(self(*batch[:-1]), batch[-1])
        #self.plot('loss', l, train=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        #return torch.optim.SGD(self.net.parameters(), lr=self.lr)
        #return torch.optim.LBFGS(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)


