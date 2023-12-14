DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

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
from torchvision import transforms


nn_Module = nn.Module

ones_like = torch.ones_like
ones = torch.ones
zeros_like = torch.zeros_like
zeros = torch.zeros
tensor = torch.tensor
arange = torch.arange
meshgrid = torch.meshgrid
sin = torch.sin
sinh = torch.sinh
cos = torch.cos
cosh = torch.cosh
tanh = torch.tanh
linspace = torch.linspace
exp = torch.exp
log = torch.log
normal = torch.normal
rand = torch.rand
randn = torch.randn
matmul = torch.matmul
int32 = torch.int32
int64 = torch.int64
float32 = torch.float32
concat = torch.cat
stack = torch.stack
abs = torch.abs
eye = torch.eye
sigmoid = torch.sigmoid
batch_matmul = torch.bmm
to_numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)
repeat = lambda x, *args, **kwargs: x.repeat(*args, **kwargs)

def parameter_summary(model):
  numel_list = [p.numel() for p in model.parameters()]
  total, param_list = sum(numel_list), numel_list
  print(f'total : {total}  param_list : {param_list}')
  return total, param_list

def layer_summary(model, X_shape):
  X = torch.randn(1, *X_shape)
  print(f'input shape {X_shape}')
  for layer in model.net:
      X = layer(X)
      name = layer.__class__.__name__
      print(f'name : {name} output shape:\t {X.shape}' )


# def init_zeros(module):
#   if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d:
#     module.weight = torch.zeros(module.weight.shape)


def init_cnn(module):
  if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d:
    with torch.no_grad():
      nn.init.xavier_uniform_(module.weight)

def init_zeros(module):
  if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d:
    with torch.no_grad():
      module.weight.zero_()

def use_svg_display():
  """Use the svg format to display a plot in Jupyter.

  Defined in :numref:`sec_calculus`"""
  backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
  """Set the figure size for matplotlib.

  Defined in :numref:`sec_calculus`"""
  use_svg_display()
  plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
  """Set the axes for matplotlib.

  Defined in :numref:`sec_calculus`"""
  axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
  axes.set_xscale(xscale), axes.set_yscale(yscale)
  axes.set_xlim(xlim), axes.set_ylim(ylim)
  if legend:
    axes.legend(legend)
  axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
  """Plot data points.

  Defined in :numref:`sec_calculus`"""

  def has_one_axis(X):  # True if `X` (tensor or list) has 1 axis
    return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
            and not hasattr(X[0], "__len__"))

  if has_one_axis(X): X = [X]
  if Y is None:
    X, Y = [[]] * len(X), X
  elif has_one_axis(Y):
    Y = [Y]
  if len(X) != len(Y):
    X = X * len(Y)

  set_figsize(figsize)
  if axes is None: axes = plt.gca()
  axes.cla()
  for x, y, fmt in zip(X, Y, fmts):
    axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
  set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def add_to_class(Class):
  """Defined in :numref:`sec_oo-design`"""

  def wrapper(obj):
    setattr(Class, obj.__name__, obj)

  return wrapper

def cpu():
  """Defined in :numref:`sec_use_gpu`"""
  return torch.device('cpu')


def gpu(i=0):
  """Defined in :numref:`sec_use_gpu`"""
  return torch.device(f'cuda:{i}')


def num_gpus():
  """Defined in :numref:`sec_use_gpu`"""
  return torch.cuda.device_count()


def try_gpu(i=0):
  """Return gpu(i) if exists, otherwise return cpu().

  Defined in :numref:`sec_use_gpu`"""

  if num_gpus() >= i + 1:
    return gpu(i)
  return cpu()


def try_all_gpus():
  """Return all available GPUs, or [cpu(),] if no GPU exists.

  Defined in :numref:`sec_use_gpu`"""
  return [gpu(i) for i in range(num_gpus())]

def to_gpu(x, dtype = torch.float32):
  return torch.tensor(x, dtype = dtype).to(gpu())

def corr2d(X, K):
  """Compute 2D cross-correlation.

  Defined in :numref:`sec_conv_layer`"""
  h, w = K.shape
  Y = zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
  for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
      Y[i, j] = reduce_sum((X[i: i + h, j: j + w] * K))
  return Y

class HyperParameters:
  def save_hyperparameters(self, ignore=[]): #this just set the input args as class attr.
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    self.hparams = {k: v for k, v in local_vars.items()
                    if k not in set(ignore + ['self']) and not k.startswith('_')}
    for k, v in self.hparams.items():
      setattr(self, k, v)

class ProgressBoard(HyperParameters):
    """Plot data points in animation.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='log',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(6.5, 5.5), display=True, counter = 0):
        self.save_hyperparameters()

    def draw(self, x, y, label, title_text, every_n=1):
        """Defined in :numref:`sec_utils`"""
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        use_svg_display()
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        axes.set_title(title_text)
        axes.grid(True, axis = 'y')
        display.display(self.fig)
        display.clear_output(wait=True)
        self.counter += 1


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
