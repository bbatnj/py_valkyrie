import numpy as np
import pandas as pd

from valkyrie.securities import *

EPS = 1e-5


def feature_filter(df, xcols, threshold_pct=0.2, kind='any'):
  tp1 = threshold_pct / 2
  x_lower = dict(df[xcols].quantile(tp1))
  x_higher = dict(df[xcols].quantile(1 - tp1))

  fmt = []
  for x in xcols:
    l, h = x_lower[x], x_higher[x]
    fmt.append(f'({x} <= {l} or {x} >= {h})')
  kind2op = {
    'any': ' or ',
    'all': ' and '
  }

  fmt = kind2op[kind].join(fmt)
  return df.query(fmt).copy()

def big_rel_diff(x, y, threshold):
  if x * y == 0:
    return x != y
  return not( (np.abs(x / y - 1) < threshold) and (np.abs(y / x  - 1) < threshold) )

def add_fwd_ret_by_n_(df: pd.DataFrame, kind: str, col, ns: list, s2col = None):
  if kind not in {"abs", "log"}:
    raise Exception(f"unknown kind for using add fwd_ret_by_n")

  for n in ns:
    c, fwd_c = f'{col}_ret_{n}_n', f'{col}_fwd_{n}_n'
    df[fwd_c] = df[col].shift(-n)

    if kind == "abs":
      df[c] = df.eval(f'{fwd_c} - {col}')
    elif kind == "log":
      df[c] = df.eval(f'{fwd_c}/{col}').apply(np.log)

    if s2col:
      df[f'wgt_{c}'] = df[s2col] + df[s2col].shift(-n)
      df[f'wgt_{c}'] = 1.0 / df[f'wgt_{c}']

# RH: 2023/12/01: pd.concat(..., axis=1) seems to have a bug
# RH: the below func assume df and df_mkt have the same index?
# RH: remove this function and use add_fwd_ret_by_n instead
# def calc_fwd_ret(df: pd.DataFrame, df_mkt: pd.DataFrame, hzs: list, col: str):
#   df_ret = df[[col]].copy()
#   for hz in hzs:
#     fwd = df_mkt[col].copy()
#     fwd.index -= pd.Timedelta(hz)
#     fwd.name = f'{col}_{hz}'
#     df_ret = pd.merge_asof(df_ret, fwd, left_index=True, right_index=True).ffill()
#     df_ret[f'{col}_ret_{hz}'] = df_ret.eval(f'{col}_{hz} / {col}').apply(np.log)#.diff()

def calc_expect_ret_bps(t_sec, vol = 0.6):
    return 1e4  * np.sqrt(t_sec / 3600 / 24 / 365 ) * vol

def weigted_mean(x : np.array, w : np.array):
  w = w.reshape(-1)
  sum_w = np.sum(w)
  return (x * w).sum(axis = 1) / sum_w


#   df_ret = df_ret.drop([col], axis = 1)
#   df = pd.concat([df, df_ret],axis = 1)
#   return df

def add_clip2col(x, lows, highs):
  for i in np.arange(x.shape[1]):
    x[:, i] = x[:, i].clip(lows[i], highs[i])

def win_ratio(s):
  return (s > 0).sum() / len(s)

def sharpe(s, days_per_year = 252):
  n = len(s)
  if n <= 1:
    return np.nan

  std = max(0.01, s.std())
  return (s.mean() / std) * np.sqrt(days_per_year)
