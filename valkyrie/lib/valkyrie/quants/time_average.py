import copy
import gc
import pdb
from multiprocessing import Pool
from itertools import product as cartesianproduct
from functools import partial
from collections import defaultdict as dd
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from numba import jit
from sklearn.linear_model import LinearRegression as LR

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit

from valkyrie.securities import *
from valkyrie.tools import ymd, ymd_dir, calc_diff_days, cpum2
from valkyrie.quants.linear import WinsorizedLM

EPS = 1e-5

class Kalman1D:
  def __init__(self, mult : float, min_s2 : float, vol: float):
    self.mult, self.min_s2, = mult,  min_s2
    self.t, self.x_h, self.s2_h = T0, 0.0, 1e9 #s2_h -> very large so x_h does not matter
    self.diffusion_s2_per_sec = vol * vol / (252 * 6.5 * 3600)

  def onData(self, t : pd.Timestamp, x : float, s2 : float):
    dt = (t - self.t).total_seconds()
    if dt < 0:
      raise Exception(f"OnLineWEMA going backward in time {t} vs {self.t}")

    dt = min(max(dt, 0.1), 10800) # 3hr for overnight, at least 100 ms

    if not np.isfinite(x + s2):
      return self.x_h

    s2_h = self.s2_h + dt * self.diffusion_s2_per_sec * self.mult# / self.halflife_s
    x_h = self.x_h

    #eta  = np.exp(-dt / self.halflife_s)
    #ceta = 1.0 - eta
    eta = 1.0
    ceta = 1.0

    x_h = (x * s2_h * ceta + x_h * s2 * eta) / (s2_h * ceta + s2 * eta)
    s2_h = s2_h * s2 / (s2_h + s2)

    self.t = t
    self.x_h = x_h
    self.s2 = s2
    self.s2_h = max(s2_h, self.min_s2)
    return self.x_h

  # def set_time(self, t : pd.Timestamp):
  #   self.t = t

class EMS:
  def __init__(self, halflife, init_value = np.nan):
    self.halflife, self.x_h, self.t = halflife, init_value, T0

  def onTime(self, t):
    dt = (t - self.t).total_seconds()
    if dt < 0:
      raise Exception(f"Online EMS going backward in time {t} vs {self.t}")

    dt = min(max(dt, 0.1), 10800)  # 3hr for overnight, at least 100 ms
    eta = np.exp(-dt / self.halflife)
    self.x_h = self.x_h * eta
    self.t = t

  def onData(self, t: pd.Timestamp, x: float):
    self.onTime(t)

    if not np.isfinite(x):
      return self.x_h

    self.x_h += x
    return self.x_h

  def getData(self):
    return self.x_h

class EMA:
  def __init__(self, halflife, init_value = np.nan):
    self.halflife, self.x_h, self.t = halflife, init_value, T0
    self.eta = np.nan

  def onTime(self, t):
    dt = (t - self.t).total_seconds()
    if dt < 0:
      raise Exception(f"Online EMS going backward in time {t} vs {self.t}")

    self.t, self.eta = t, np.exp(-dt / self.halflife)

  def onData(self, t: pd.Timestamp, x: float):
    self.onTime(t)
    if not np.isfinite(x):
      return self.x_h

    self.x_h = self.x_h * self.eta + (1 - self.eta) * x
    return self.x_h

  def getData(self):
    return self.x_h

def add_wema(df, halflife_in_n, col, s2_col, min_s2=0.01 ** 2, y_col='y'):
  eta = np.exp(-1.0 / halflife_in_n)
  ceta = 1 - eta

  n = df.shape[0]
  x = df[col].values
  s2 = df[s2_col].clip(min_s2, np.inf).values
  y, sigma2_hat = np.zeros(n), np.zeros(n)
  x_h, s2_h, y[0], sigma2_hat[0] = x[0], s2[0], x[0], s2[0]

  for i in np.arange(1, n):
    if not np.isfinite(x[i] + s2[i]):
      y[i], sigma2_hat[i] = x[i], s2[i]
      continue

    if i >= halflife_in_n:
      x_h = eta * x_h / s2_h + (1 - eta) * x[i] / s2[i]
      x_h = x_h / (eta / s2_h + (1 - eta) / s2[i])
    else:
      x_h = (i * x_h / s2_h + x[i] / s2[i]) / (i / s2_h + 1 / s2[i])

    s2_h = s2_h * s2[i] / (ceta * s2_h + eta * s2[i])
    s2_h = max(s2_h, min_s2)

    y[i], sigma2_hat[i] = x_h, s2_h

  df['sigma2_hat'] = sigma2_hat
  df[y_col] = y

def calc_wemas(df, ema_mults, col, s2_col, min_s2=0.01 ** 2, ycol='y'):
  df = df.copy()
  for ema in ema_mults:
    add_wema(df, ema, col, s2_col, min_s2, y_col=f'{ycol}_{ema}')
  return df

class BuilderXY(ABC):
  def __init__(self, name):
    self.name = name

  @abstractmethod
  def get_xy(self):
    pass

  def __str__(self):
    return self.name


def fit_tscv(xy_builders, models,
             horizons=(100,), n_train_days=(80,), n_test_days=(5,)):
  # horizons=(100,200,400),
  # n_train_days=(20, 40, 60, 80, 100),
  # n_test_days=(5,)):

  param_grid = {'x_builders': xy_builders, 'models': models,
                'horizon': horizons, 'n_train_days': n_train_days, 'n_test_day': n_test_days}

  res = dd(lambda: [])

  param2res = {'timeseries': dd(lambda: []),
               'fitter': dd(lambda: []),
               'score': dd(lambda: []),
               'score_in': dd(lambda: []),
               }

  p2ts, p2fitter, p2score, p2score_in = param2res['timeseries'], param2res['fitter'], param2res['score'], param2res[
    'score_in']

  for xy_builder in xy_builders:
    df_xy, xcols, ycols, wcol = xy_builder.get_xy()
    df_xy['i'] = np.arange(df_xy.shape[0])
    all_dates = df_xy['ymd'].unique()
    n_all_dates = len(all_dates)

    for param in list(ParameterGrid(param_grid)):
      gc.collect()
      horizon, n_train_day, n_test_day, model = param['horizon'], param['n_train_days'], param['n_test_day'], param[
        'models']
      if model.__class__ not in {WinsorizedLM}:
        print(f'in fit_tcsv model type not supported! {model.__class__}')
        continue

      tscv = TimeSeriesSplit(gap=0, n_splits=int((n_all_dates - n_test_day) / n_test_day), max_train_size=n_train_day,
                             test_size=n_test_day)
      X = df_xy.iloc[:-horizon][xcols].values
      Y = np.log(df_xy.iloc[horizon:][ycols].values / df_xy.iloc[:-horizon][ycols].values)
      W = np.sqrt(df_xy.iloc[horizon:][wcol].values * df_xy.iloc[:-horizon][wcol].values)
      W = np.ones(Y.shape)
      df_ymd = df_xy.iloc[:-horizon][['ymd', 'i']].copy()

      param_str = f'{xy_builder}_h_{horizon}_m_{model}_nf_{n_train_day}_nt_{n_test_day}'
      # print(f'fitting {param_str}')

      df_test_result, scores = [], []
      for train_dates, test_dates in tscv.split(all_dates):
        if len(train_dates) < n_test_day:
          continue  # print('not enough training data skipping')
        train_dates, test_dates = all_dates[train_dates], all_dates[test_dates]

        # print(f'fitting for {train_dates[0]} {train_dates[-1]}')
        # training
        train_index = df_ymd.query('ymd in @train_dates')['i'].values
        current_model = copy.deepcopy(model)
        current_model.fit(X[train_index], Y[train_index], W[train_index])

        # predicting
        test_index = df_ymd.query('ymd in @test_dates')['i'].values
        y_hat_os = current_model.predict(X[test_index])
        y_os = Y[test_index].copy()

        df_y_hat = pd.DataFrame(y_hat_os, columns=[c + '_hat' for c in ycols], index=df_ymd.iloc[test_index]['i'])
        df_y = pd.DataFrame(y_hat_os, columns=ycols, index=df_ymd.iloc[test_index]['i'])

        score_in = current_model.score(X[train_index], Y[train_index], W[train_index])
        score = current_model.score(X[test_index], Y[test_index], W[test_index])

        p2ts[param_str].append(pd.concat([df_y_hat, df_y], axis=1))
        p2fitter[param_str].append(current_model)
        p2score[param_str].append(score)
        p2score_in[param_str].append(score_in)

      scores, scores_in = p2score[param_str], p2score_in[param_str]
      # print(f'{param_str} : score in : {scores_in} score out : {scores}')
      res['param'].append(param_str)
      res['n_cv'].append(len(scores_in))
      res['trains'].append(n_train_day)
      res['horizon'].append(horizon)
      res['score_in_median'].append(np.median(scores_in))
      res['score_in_mean'].append(np.median(scores_in))
      res['score_in_win'].append(np.sum(np.array(scores_in) >= 0) / len(scores_in))
      res['score_median'].append(np.median(scores))
      res['score_mean'].append(np.mean(scores))
      res['score_win'].append(np.sum(np.array(scores) >= 0) / len(scores))

  # for param in param2res['timeseries']:
  # param2res['timeseries'][param] = pd.concat(param2res['timeseries'][param])
  # path = f'{res_save_path}/{param}/'
  # os.makedirs(f'{path}', exist_ok = True)
  # param2res['timeseries'][param].to_hdf(f'{path}/y_hat.h5', key = 'data')

  df_res = pd.DataFrame(res)
  return df_res, param2res

########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

def create_stk_xy_builders(stocks, xy_builder_classes, xy_params, black_list={'PFF'}):
  df_res = []
  black_list = {'PFF'}
  xy_builders = []

  for stk in stocks:
    if stk in black_list:
      continue

    xy_builders += [c(**(xy_params | {'stock': stk})) for c in xy_builder_classes]

  return xy_builders


@jit
def calc_ts_ema(x, dt, halflife_in_s):
    eta = np.exp(-1.0 * dt / halflife_in_s)
    eta = np.clip(eta, 0, np.inf)
    if np.isscalar(eta):
      eta = np.full_like(x, eta)

    ome = 1.0 - eta
    n, y = x.shape[0], np.empty_like(x)
    y[0] = x[0]

    for i in np.arange(1, n):
        if not np.isfinite(x[i]):
            y[i] = y[i - 1]
            continue

        y[i] = y[i - 1] * eta[i] + ome[i] * x[i]
    return y

@jit
def calc_ts_ems(x, dt, halflife_in_s):
    eta = np.exp(-1.0 * dt / halflife_in_s)
    eta = np.clip(eta, 0, np.inf)
    if np.isscalar(eta):
      eta = np.full_like(x, eta)

    n, y = x.shape[0], np.empty_like(x)
    y[0] = x[0]

    for i in np.arange(1, n):
        if not np.isfinite(x[i]):
            y[i] = y[i - 1]
            continue

        y[i] = y[i - 1] * eta[i] + x[i]
    return y

def calc_ems_hl(df, fld, freq_sec, hl_sec_s =  [10, 60, 300, 1800]):
    if not np.issubdtype(type(freq_sec), np.number):
        freq_sec = str2sec(freq_sec)

    df = df[[fld]].ffill().fillna(0.0)
    for hl in hl_sec_s :
        df[f'net_qty_ems_{hl}'] = calc_ts_ems(df[fld].values, freq_sec, hl)
    return df[[c for c in df if 'net_qty_ems' in c]]


def calc_ema_hl(df, fld, freq_sec, hl_sec_s =  [10, 60, 300, 1800]):
    if not np.issubdtype(type(freq_sec), np.number):
        freq_sec = str2sec(freq_sec)

    df = df[[fld]].ffill().fillna(0.0)
    for hl in hl_sec_s :
        df[f'net_qty_ems_{hl}'] = calc_ts_ema(df[fld].values, freq_sec, hl)
    return df[[c for c in df if 'net_qty_ems' in c]]

def calc_ema_vol(df_mid, mid_fld, freq_sec, hl_sec_s = [10, 60, 300, 1800], max_vol = np.inf):
    if not np.issubdtype(type(freq_sec), np.number):
        freq_sec = str2sec(freq_sec)

    df_vol = df_mid[[mid_fld]].copy()
    n_samples_per_year = 365 * 24 * 3600 / freq_sec
    max_ret = max_vol / np.sqrt(n_samples_per_year) * 10.0
    df_vol['var_raw'] = (df_vol[mid_fld].shift(-1) / df_vol[mid_fld]).apply(np.log).clip(-max_ret, max_ret)
    df_vol['var_raw'] = df_vol['var_raw'].apply(np.square)
    df_vol['vol_raw'] = df_vol['var_raw'].apply(np.sqrt) * np.sqrt(n_samples_per_year)
    for hl in hl_sec_s:
        #df_vol[f'vol_{hl}']=df_vol[['var_raw']].rolling(hl).mean() #calc_ts_ema(df_vol['var_raw'].values, df_vol['dt'].values, hl)
        df_vol[f'vol_{hl}'] = calc_ts_ema(df_vol['var_raw'].values, freq_sec, hl)
        df_vol[f'vol_{hl}'] = df_vol[f'vol_{hl}'].apply(lambda var : np.sqrt(var * n_samples_per_year ) )
    return df_vol[[c for c in df_vol if 'vol_' in c or 'var_raw' in c or 'vol_raw' in c]]


# class DataMgrCYieldEma(DataMgr):
#     def __init__(self, sdate, edate, stocks,  name='good_dvd',
#                  data_path=f'd:/valkyrie/data/resampled/hist_ib_raw_md'):
#         super().__init__(sdate, edate, stocks,  name, data_path)
#
#     def build_xy(self, y_stock, ema_mults):
#         f_same_family = partial(isSameFamily, stk2=y_stock)
#         x_stocks = list(filter(f_same_family, self.stocks))
#
#         df_merged = pd.concat([self.stk2df[stk] for stk in x_stocks], keys=x_stocks, axis=1)
#
#         df_res = []
#         for stk in x_stocks:
#             df_res.append(calc_wemas(df_merged[stk], ema_mults, 'cy_upx', 's2', 0.01 ** 2, ycol='cy_upx_ema'))
#
#         df_res = pd.concat(df_res, keys=x_stocks, axis=1)
#         df_res.columns = ['_'.join(c) for c in df_res.columns]
#         df_res['ymd'] = df_res.index.map(lambda t: ymd(t))
#
#         df_res.columns = [bigpr2small(c) for c in df_res]
#         y_stock = bigpr2small(y_stock)
#         cols = list(df_res.columns)
#         cy_cols = [c for c in cols if 'cy_upx' in c and 'ema' not in c and y_stock in c]
#         ema_cols = [c for c in cols if 'cy_upx_ema' in c]
#
#         xcols = []
#         for c1, c2 in cartesianproduct(cy_cols, ema_cols):
#             c = c1 + '_minus_' + c2
#             df_res[c] = df_res.eval(f'{c1} - {c2}')
#             xcols.append(c)
#
#         ycols = cy_cols
#         wcol = 'weight'
#         df_res['weight'] = df_res.eval(f'1.0 / {y_stock}_s2').clip(1e-6,10000)
#         df_res.dropna(how='any', inplace=True)
#         return df_res, xcols, ycols, wcol
