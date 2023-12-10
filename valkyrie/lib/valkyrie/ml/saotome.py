from torch import nn
from torch.nn import LeakyReLU as LReLU

from valkyrie.ml import modules as ml_modules
from valkyrie.ml.modules import Filter1D, BN1D

n_field, n_instr = 1, 2


class SaotomeNetII(ml_modules.Regressor):
  def __init__(self, lr):
    super().__init__('l1')
    self.save_hyperparameters()

    c_mul = 2
    n_c0o = n_field
    n_c1o = c_mul * n_c0o
    n_c2o = c_mul * n_c1o
    n_c3o = c_mul * n_c2o
    n_c4o = c_mul * n_c3o
    n_c5o = c_mul * n_c4o

    self.net = nn.Sequential(
      Filter1D(groups=n_c0o, n_out_channel=n_c1o, kernel_size=5, stride=2), BN1D(n_c1o * n_instr), LReLU(0.01),
      Filter1D(groups=n_c1o, n_out_channel=n_c2o, kernel_size=5, stride=2), BN1D(n_c2o * n_instr), LReLU(0.01),
      Filter1D(groups=n_c2o, n_out_channel=n_c3o, kernel_size=5, stride=2), BN1D(n_c3o * n_instr), LReLU(0.01),
      Filter1D(groups=n_c3o, n_out_channel=n_c4o, kernel_size=5, stride=2), BN1D(n_c4o * n_instr), LReLU(0.01),
      # Filter1D(n_c4o, groups = n_c4o, n_out_channel = n_c5o, kernel_size = 5, stride = 2),BN1D(n_c5o * n_instr), LReLU(0.01),
      nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
      nn.LazyConv2d(128, kernel_size=3, padding=1), nn.ReLU(),
      nn.LazyConv2d(64, kernel_size=3, padding=1), nn.ReLU(),
      nn.LazyConv2d(32, kernel_size=3, padding=1), nn.ReLU(),
      nn.Flatten(),
      nn.LazyLinear(128), LReLU(0.01),
      nn.LazyLinear(32), LReLU(0.01),
      nn.LazyLinear(1)
    )
