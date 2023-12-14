import copy
from itertools import product as cartesian_product

from overrides import overrides
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from valkyrie.ml.utils import tensor, HyperParameters
from valkyrie.ml import utils as ml_utils
from numba import jit

from d2l.torch import FashionMNIST

class Df2DataSet(Dataset):
  def __init__(self, df: pd.DataFrame, ycol, wcol, dtype = torch.float32):
    super().__init__()
    cols = [c for c, t in df.dtypes.items() if np.issubdtype(t, np.number) and c != ycol and c != wcol]

    self.data = tensor(df[cols].values, dtype=dtype)

    self.columns = list(df[cols].columns)
    self.n_cols = len(self.columns)

    self.y = torch.zeros(df.shape[0])
    if ycol:
      self.y = df[ycol].values

    self.w = torch.zeros(df.shape[0])
    if wcol:
      self.w = df[wcol].values

    self.yw = torch.empty(2)

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    X = tensor(self.data[idx], dtype=torch.float32)
    self.yw[0] = self.y[idx]
    self.yw[1] = self.w[idx]
    return X, self.yw

class Df2Snapshot(Df2DataSet):
  def __init__(self, df: pd.DataFrame, ycol: str, wcol: str, t: int):
    super().__init__(df, ycol, wcol)
    self.t = t

  def __len__(self):
    return self.data.shape[0] - self.t + 1

  def __getitem__(self, idx):
    idx_e = idx + self.t
    # X = tensor(self.data[idx: idx_e].view(self.t, self.n_cols), dtype=torch.float32)
    X = self.data[idx: idx_e].W
    self.yw[0] = self.y[idx - 1]
    self.yw[1] = self.w[idx - 1]
    return (X, self.yw)


class MultiSnapshot(Df2Snapshot):
  def __init__(self, df: pd.DataFrame, ycol, t: int):
    super().__init__(df, ycol, t)
    sizes = df.groupby('ticker')['ticker'].count()

    if len(sizes.unique()) != 1:
      raise Exception(f'df must have equal size per ticker {sizes}')

    self.n_instr = len(sizes)
    self.n_sample_per_instr = sizes[0] - t + 1
    self.instr_stride = sizes[0]

  def __len__(self):
    return self.n_sample_per_instr * self.n_instr

  def __getitem__(self, idx):
    instr_i, offset = idx // self.n_sample_per_instr, idx % self.n_sample_per_instr
    idx = self.instr_stride * instr_i + offset
    idx_e = idx + self.t
    X = tensor(self.data[idx: idx_e], dtype=torch.float32)
    X = X.view(self.t, self.n_cols)
    X = torch.unsqueeze(X, 0)
    y = self.y[idx_e - 1]
    return (X, y)

class InsNB:
  def __init__(self, ins: str, nbs: list):
    self.ins = ins
    self.nbs = nbs.copy()

class MultiRecField(Dataset):
  def __init__(self, ins2df: dict, ins_nbs: list, xcols: list, ycol: str, wcol: str, t: int):
    super().__init__()
    self.inss = [ins_nb.ins for ins_nb in ins_nbs]

    # sanity check for all dfs must be same index
    ts_indices = [ins2df[ins].index for ins in self.inss]
    if len(ts_indices) != 1 and not all([ts_indices[0].equals(_) for _ in ts_indices]):
      raise Exception("time indices must be all equal")
    # end sanity check

    self.n_samples_per_ins = len(ts_indices[0]) - t + 1
    self.ins_nbs = copy.deepcopy(ins_nbs)
    self.n_instr = len(self.inss)
    self.total_n = self.n_samples_per_ins * self.n_instr
    self.xcols = copy.deepcopy(xcols)
    self.n_xcols = len(xcols)
    self.n_nb = len(self.ins_nbs[0].nbs)
    self.t = t
    # pre-generate indexed nbs
    for i, ins_nb in enumerate(self.ins_nbs):
      self.ins_nbs[i].nbs = list(enumerate(ins_nb.nbs))

    cols = xcols + [ycol, wcol]

    self.ins2mss = {ins: Df2Snapshot(ins2df[ins][cols], ycol, wcol, t) for ins in self.inss}
    additional_ins = set(nb for ins_nb in ins_nbs for nb in ins_nb.nbs)
    additional_ins = {ins for ins in additional_ins if ins not in self.inss}
    for ins in additional_ins:
      self.ins2mss[ins] = Df2Snapshot(ins2df[ins][xcols], None, None, t)

    self.X = torch.empty([len(xcols), (self.n_nb + 1) , t])
    # below is for 2D input
    # self.X = torch.empty([(self.n_nb + 1) * len(xcols), t])

  def __len__(self):
    return self.total_n

  def __getitem__(self, idx):
    ins_i, offset = idx // self.n_samples_per_ins, idx % self.n_samples_per_ins
    ins_nb = self.ins_nbs[ins_i]
    ins, nbs = ins_nb.ins, ins_nb.nbs
    R = self.ins2mss[ins][offset]

    X = torch.empty([len(self.xcols), (self.n_nb + 1), self.t])
    X[:,0, :] = R[0]
    for j, nb in nbs:
      X[:, j+1, :] = self.ins2mss[nb][offset][0]

    yw = R[1].clone() #### RH : Why we must clone here????
    return (X, yw)

  #below is for 2D input
  #def __getitem__(self, idx):
    # ins_i, offset = idx // self.n_samples_per_ins, idx % self.n_samples_per_ins
    # ins_nb = self.ins_nbs[ins_i]
    # ins, nbs = ins_nb.ins, ins_nb.nbs
    # R = self.ins2mss[ins][offset]

    # n_nb = self.n_nb
    # self.X[0::n_nb + 1, :] = R[0]
    # self.yw = R[1]
    # for j, nb in nbs:
    #   self.X[j + 1::n_nb + 1, :] = self.ins2mss[nb][offset][0]
    # return (self.X, self.yw)

def mrf_linear_fit(mrf, fit_intercept=True):
  n = len(mrf)
  n_xcols = np.prod(mrf[0][0].shape)
  X = np.empty([n, n_xcols])
  #Xnet = torch.empty([n] + list(mrf[0][0].shape))
  Y = np.empty(n)
  W = np.empty(n)
  for i in np.arange(len(mrf)):
    X[i, :] = mrf[i][0].view(1, -1)
    #Xnet[i,:] = mrf[i][0]
    Y[i] = mrf[i][1][0]
    W[i] = mrf[i][1][1]

  from sklearn.linear_model import LinearRegression as LR
  lr = LR(fit_intercept=fit_intercept)
  lr.fit(X, Y, W)
  Y_hat = lr.predict(X)
  abs_lr_loss = np.sum(np.abs(Y_hat.reshape(-1) - Y) * W) / np.sum(W)
  abs_zr_loss = np.sum(np.abs(Y) * W) / np.sum(W)
  return lr.score(X, Y, W), abs_lr_loss, abs_zr_loss #, X, Xnet, Y, W, lr


def mrf_linear_fit_2(mrf):
  n, p = len(mrf), np.prod(mrf[0][0].shape)

  XtX = torch.zeros(p, p)
  XtY = torch.zeros(p, 1)
  Y = torch.zeros(n, 1)
  Y_hat = torch.zeros(n, 1)

  for i in np.arange(len(mrf)):
    x, y, w = mrf[i][0][0].view(1, p), mrf[i][1][0], mrf[i][1][1]
    XtX += torch.matmul(x.W, x)  # * w
    XtY += x.W * y  # * w

  beta = torch.linalg.solve(XtX, XtY)

  abs_zr_loss, abs_lr_loss = 0.0, 0.0
  for i in np.arange(len(mrf)):
    x, y, w = mrf[i][0][0].view(1, p), mrf[i][1][0], mrf[i][1][1]
    Y[i], Y_hat[i] = y, torch.matmul(x, beta).view(-1)[0]
    abs_lr_loss += torch.abs(Y_hat[i] - y)  # * w
    abs_zr_loss += torch.abs(y)  # *w

  return abs_lr_loss, abs_zr_loss, Y, Y_hat

class DataModule(HyperParameters):
  @staticmethod
  def from_dataset(mss_train, mss_val, batch_size, num_workers=4):
    dm = DataModule(batch_size, num_workers)
    dm.train = mss_train
    dm.val = mss_val
    return dm

  def __init__(self, batch_size=32, root='../data', num_workers=4):
    self.save_hyperparameters()

  def get_dataloader(self, train):
    data = self.train if train else self.val
    if data is None:
      return None
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       pin_memory=True,
                                       num_workers=4)#self.num_workers)

  def train_dataloader(self):
    return self.get_dataloader(train=True)

  def val_dataloader(self):
    return self.get_dataloader(train=False)

  def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)


class Df2T2(Dataset):
    def __init__(self, df: pd.DataFrame, H, W, xcols, ycol, wcol=None, mul=2, dtype=torch.float32, device='cpu', yscaler = 1e4):
        super().__init__()

        self.H = H
        self.W = W
        self.n_features = len(xcols)

        final_x_cols = xcols.copy()  # we do not wanna change original xcols

        dfs = [df[xcols]]
        for freq_mul_index in np.arange(1, self.H):
            freq_mul = mul ** freq_mul_index
            df_temp = df[xcols].rolling(freq_mul).agg('mean')
            df_temp.columns = [f'{c}_x{freq_mul:04d}' for c in xcols]
            dfs.append(df_temp)
            final_x_cols += list(df_temp.columns)

        # xcols must be sorted by feature then freq_mul
        final_x_cols = sorted(final_x_cols)

        dfs.append(df[[ycol]])  # y
        self.df = pd.concat(dfs, axis=1)  # x's

        prev_n = self.df.shape[0]
        self.df.dropna(inplace=True)

        n = self.df.shape[0]
        print(f'after dropping na {prev_n} -> {n}')

        if wcol:  # w
            weights = df[wcol].values
        else:
            weights = np.ones(n)

        weights = weights / np.sqrt(np.mean(weights * weights))

        X = torch.tensor(self.df[final_x_cols].values, dtype=dtype, device=device)
        # xcols must be sorted by feature then freq_mul
        X = X.T.view(self.n_features, self.H, self.df.shape[0]).contiguous()
        self.X = X
        # X.shape is now (n_features, H, n_samples)

        self.offset = self.W - 1
        self.yw = torch.tensor(np.c_[yscaler * self.df[ycol].values, weights], dtype=dtype, device=device)
        self.n = self.X.shape[-1] - self.W + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[:,:,idx: idx+self.offset+1], self.yw[idx+self.offset]

    @jit
    def get_yw(self):
        n = self.n
        c = np.prod(self[0][0].shape)
        #X_sk = np.empty((n, c))
        y_sk = np.empty(n)
        w_sk = np.empty(n)
        for i in np.arange(n):
            z = self[i]
            #X_sk[i, :] = z[0].reshape(-1)
            y_sk[i] = z[1][0]
            w_sk[i] = z[1][1]
        return y_sk, w_sk

def create_random_sets_numpy(N, p):
    # Calculate the size of the first set (p% of N)
    size_of_first_set = int(p * N)

    # Create an array of numbers from 0 to N-1
    numbers = np.arange(N)

    # Shuffle the array randomly
    np.random.shuffle(numbers)

    # Split the shuffled array into the first set and the second set
    first_set = numbers[:size_of_first_set]
    second_set = numbers[size_of_first_set:]

    return first_set, second_set

class SubDataSet(Dataset):

    @staticmethod
    def create_train_valiation_pair(parent_data_set, train_percentage):
        n = len(parent_data_set)
        train_indices, validation_indices = create_random_sets_numpy(n, train_percentage)
        train_set = SubDataSet(parent_data_set, train_indices)
        validation_set = SubDataSet(parent_data_set, validation_indices)
        return train_set, validation_set

    def __init__(self, parent_dataset, indices):
        super().__init__()
        self.parent_dataset = parent_dataset
        self.indices = copy.deepcopy(indices)

    def __getitem__(self, index):
        return self.parent_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

class FashionMNIST(DataModule):
  def __init__(self, batch_size=64, resize=(28, 28)):
    super().__init__()
    self.save_hyperparameters()
    trans = transforms.Compose([transforms.Resize(resize),
                                transforms.ToTensor()])
    self.train = torchvision.datasets.FashionMNIST(
      root=self.root, train=True, transform=trans, download=True)
    self.val = torchvision.datasets.FashionMNIST(
      root=self.root, train=False, transform=trans, download=True)

  def text_labels(self, indices):
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]

  def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
      labels = self.text_labels(y)
    ml_utils.show_images(X.squeeze(1), nrows, ncols, titles=labels)
