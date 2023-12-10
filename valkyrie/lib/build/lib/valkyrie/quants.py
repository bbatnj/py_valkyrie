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
from sklearn.linear_model import LinearRegression as LR

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit

from valkyrie.securities import *
from valkyrie.tools import ymd, ymd_dir, calc_diff_days, cpum2

EPS = 1e-5

def add_fwd_ret_by_n(df: pd.DataFrame, cols: list, ns: list):
  for col, n in cartesianproduct(cols, ns):
    c, fwd_c = f'{col}_ret_{n}_n', f'{col}_fwd_{n}_n'
    df[fwd_c] = df[col].shift(-n)
    df[c] = df.eval(f'{fwd_c} / {col}').apply(np.log)

def calc_fwd_ret(df: pd.DataFrame, df_mkt: pd.DataFrame, hzs: list, col: str):
  df_ret = df[[col]].copy()
  for hz in hzs:
    fwd = df_mkt[col].copy()
    fwd.index -= pd.Timedelta(hz)
    fwd.name = f'{col}_{hz}'
    df_ret = pd.merge_asof(df_ret, fwd, left_index=True, right_index=True).ffill()
    df_ret[f'{col}_ret_{hz}'] = df_ret.eval(f'{col}_{hz} / {col}').apply(np.log)#.diff()

  df_ret = df_ret.drop([col], axis = 1)
  df = pd.concat([df, df_ret],axis = 1)
  return df

class WEMA:
  def __init__(self, halflife_s : float, min_s2 : float, vol: float):
    self.halflife_s, self.min_s2, = halflife_s,  min_s2
    self.t, self.x_h, self.s2_h = T0, 0.0, 1e9 #s2_h -> very large so x_h does not matter
    self.diffusion_s2_per_sec = vol * vol / (252 * 6.5 * 3600)

  def onData(self, t : pd.Timestamp, x : float, s2 : float):
    dt = (t - self.t).total_seconds()
    if dt < 0:
      raise Exception(f"OnLineWEMA going backward in time {t} vs {self.t}")

    dt = min(max(dt, 0.1), 10800) # 3hr for overnight, at least 100 ms

    if not np.isfinite(x + s2):
      return self.x_h

    s2_h = self.s2_h + dt * self.diffusion_s2_per_sec / self.halflife_s
    x_h = self.x_h

    eta  = 1.0#max(0.7, np.exp(-dt / self.halflife_s))
    ceta = 1.0# - eta

    x_h = (x * s2_h * ceta + x_h * s2 * eta) / (s2_h * ceta + s2 * eta)
    s2_h = s2_h * s2 / (s2_h + s2)

    self.t = t
    self.x_h = x_h
    self.s2 = s2
    self.s2_h = max(s2_h, self.min_s2)
    return self.x_h

  # def set_time(self, t : pd.Timestamp):
  #   self.t = t


def add_clip2col(x, lows, highs):
  for i in np.arange(x.shape[1]):
    x[:, i] = x[:, i].clip(lows[i], highs[i])


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


def win_ratio(s):
  return np.sum(s > 0) / len(s)


def sharpe(s):
  n = len(s)
  if n <= 1:
    return np.nan

  std = max(0.01, s.std())
  return (s.mean() / std) * np.sqrt(252)

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


class WinsorizedLM:
  def __init__(self, quantile, linear_model, **args):
    self.lm = linear_model(**args)
    self.quantile = quantile
    self.args = args

  def fit(self, X, Y, W = None):
    if not W:
      W = np.ones(Y.shape)
    X, Y, W = X.copy(), Y.reshape(X.shape[0], -1).copy(), W.copy()
    self.xlows, self.xhighs = np.quantile(X, self.quantile, axis=0), np.quantile(X, 1 - self.quantile, axis=0)
    add_clip2col(X, self.xlows, self.xhighs)
    self.ylows, self.yhighs = np.quantile(Y, self.quantile, axis=0), np.quantile(Y, 1 - self.quantile, axis=0)
    add_clip2col(Y, self.ylows, self.yhighs)
    self.lm.fit(X, Y)  # , W)

  def predict(self, X):
    X = X.copy()
    add_clip2col(X, self.xlows, self.xhighs)
    Y = self.lm.predict(X)
    add_clip2col(Y, self.ylows, self.yhighs)
    return Y

  def score(self, X, Y, W = None):
    if not W:
      W = np.ones(Y.shape)
    X, Y, W = X.copy(), Y.reshape(X.shape[0], -1).copy(), W.copy()
    add_clip2col(X, self.xlows, self.xhighs)
    add_clip2col(Y, self.ylows, self.yhighs)
    r2 = self.lm.score(X, Y)  # , W)
    #Y_hat = self.predict(X)
    #Y_diff2 = (Y - Y_hat) * (Y - Y_hat)
    # Y_m = np.mean(Y, axis=0)
    # Y2 = (Y - Y_m) * (Y - Y_m)
    #r2m = 1 - np.sum(Y_diff2) / np.sum(Y * Y)
    # print(f'r2 : {r2}, r2m : {r2m}')
    return r2

  def __str__(self):
    s = '_'.join([f'{k[0:3]}_{v}' for k, v in self.args.items()])
    return str(self.lm).strip('()')[0:4] + s + f'_q={self.quantile}'


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


def calc_current_yield(df, dvd_freq, dvd_path='d:/valkyrie/data/universe/dvd/research_20220205', ):
  freq2param = {
    'Q': {'n_per_year': 4, 'n_days': 90},
    'M': {'n_per_year': 12, 'n_days': 30}
  }
  # copy md df
  df_md = df.copy()
  df_md['ymd'] = df_md.index.map(lambda x: ymd(x))
  stocks = df_md['ticker'].unique()

  # create dvd df, last dvd on/before date
  df_dvd = []
  for stk in stocks:
    try:
      df = pd.read_csv(f'{dvd_path}/{stk}.csv').sort_values('date')
      df['ymd'] = df['date'].apply(ymd)
      if df.shape[0] <= 1:
        raise Exception(f'Not enough dvd history for {stk} number of records = {df.shape[0]} ')

      df['ticker'] = stk
      df = df.iloc[1:].copy()  # remove the first one
      df_dvd.append(df)
    except Exception as e:
      print(e)

  df_dvd = pd.concat(df_dvd)
  df_dvd = df_dvd.rename({'date': 'exDate'}, axis=1).drop(['note', 'ymd'], axis=1)
  df_dvd.index = df_dvd['exDate'].apply(lambda t: pd.Timestamp(t))
  df_dvd.index.name = ''
  df_dvd = df_dvd.sort_values(['ticker', 'exDate'])

  # filter tickers witout dvd info
  stocks_with_dvd = set(df_dvd['ticker'])
  removed_stocks = {s for s in stocks if s not in stocks_with_dvd}
  stocks = stocks_with_dvd.intersection(stocks)
  df_md = df_md.query('ticker in @stocks').copy()
  if len(removed_stocks) > 0:
    print(f'removed stocks {removed_stocks} due to no dvd info')

  # merge with dvd and calc yield
  df_merged = pd.merge_asof(df_md, df_dvd.sort_index(), left_index=True, right_index=True, by='ticker')
  df_merged['dExDate'] = calc_diff_days(df_merged, 'ymd', 'exDate')
  n_days = freq2param[dvd_freq]['n_days']
  df_merged['accrued'] = df_merged.eval('dExDate/@n_days * amount')
  df_merged['ymd'] = df_merged.index.map(ymd)
  # df_merged['accrued'] = df_merged[['accrued', 'amount']].min(axis=1)

  # add extra dvd adj
  df_merged['adj'] = 0.0
  df_extra = pd.read_csv(f'{dvd_path}/extra.csv')
  df_extra['sdate'] = df_extra['sdate'].apply(ymd)
  df_extra['edate'] = df_extra['edate'].apply(ymd)
  extra_adj_records = df_extra.to_dict('records')
  for rec in extra_adj_records:
    t, s, e, m = rec['ticker'], rec['sdate'], rec['edate'], rec['amount']
    df_merged['active'] = df_merged.eval('@s <= ymd <= @e and ticker == @t').astype(int)
    df_merged['adj'] += df_merged.eval('active * @m')

  df_merged['upx'] = df_merged.eval('(bid * az + ask * bz) / (bz + az)')
  n_per_year = freq2param[dvd_freq]['n_per_year']
  df_merged['cy_upx'] = df_merged.eval(' @n_per_year * amount / (upx - accrued + adj)')
  df_merged['cy_bid'] = df_merged.eval(' @n_per_year * amount / (bid - accrued + adj)')
  df_merged['cy_ask'] = df_merged.eval(' @n_per_year * amount / (ask - accrued + adj)')

  df_dvd_err = df_merged.groupby(['ymd', 'ticker'])[['dExDate']].first().query('dExDate > 105')
  if df_dvd_err.shape[0] > 0:
    print(f'possible dvd error :')
    print(df_dvd_err)

  return df_merged



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
