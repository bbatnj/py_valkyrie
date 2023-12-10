import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


class EMA:
  def __init__(self, halfLife, overnight_w=0.9):
    self.mu = -1.0 / halfLife
    self.overnight_w = overnight_w

  def calc(self, df, t_col, x_col):
    t = df[t_col].values
    x = df[x_col].values
    n = t.shape[0]
    y = np.zeros(n)
    y[0] = x[0]
    for i in np.arange(1, n):
      if (t[i] - t[i - 1]) < 3600 * 12:
        w = np.exp((t[i] - t[i - 1]) * self.mu)
      else:
        w = self.overnight_w
      y[i] = w * y[i - 1] + (1 - w) * x[i - 1]
    return y


class EMA2:
  def __init__(self, ticker, halfLife, overnight_w=0.9):
    print("init EMA2")
    self.ticker = ticker
    self.mu = -1.0 / halfLife
    self.overnight_w = overnight_w

  def calc(self, df, t_col, x_col, dt_col, ticker_col):
    t = df[t_col].values
    x = df[x_col].values
    dt = df[dt_col].values
    tickers = df[ticker_col].values
    n = t.shape[0]
    y = np.zeros(n)
    y[0] = x[0]
    current_dt = dt[0]
    for i in np.arange(1, n):
      if current_dt != dt[i]:  # starting a new day
        if self.ticker == tickers[i]:
          y[i] = self.overnight_w * y[i - 1] + (1 - self.overnight_w) * x[i]
          current_dt = dt[i]
        else:
          y[i] = y[i - 1]
      else:
        w = np.exp((t[i] - t[i - 1]) * self.mu)
        y[i] = w * y[i - 1] + (1 - w) * x[i - 1]
    return y


def drop_duplicate_index(df, which='last'):
  return df[~df.index.duplicated(keep=which)]


def df_times_series(df, s):
  for col in df.columns:
    df[col] = df[col] * s
  return df


def concat_df(sym2df):
  return pd.concat(item[1] for item in sym2df.items())


def calc_w_mean(df, cv, cw):
  if df[cw].shape[0] == 0:
    return np.nan

  df["__mask"] = np.isfinite(df[cv] * df[cw])
  d = df[df["__mask"]]
  df.drop("__mask", axis=1, inplace=True)
  return np.average(d[cv], weights=d[cw])


# a class that automatically init according to args in init
class X(object):
  def __init__(self, a, b, c, d):
    self.__dict__.update(locals())
    self.__dict__.pop("self")


def flatTwoLeveLColumn(df):
  df.columns = [c[0] + "_" + c[1]
                for c in zip(df.columns.get_level_values(0), df.columns.get_level_values(1))]
  df.columns = [c.strip("_") for c in df.columns]
  return df


def printdf(df):
  """
  Print dataframe without row/column limits
  :param df:
  :return:
  """
  from pandas import set_option
  set_option('display.max_rows', None)
  set_option('display.max_columns', None)
  set_option('display.width', 800)
  print(df)


def print_ex():
  exc_type, exc_obj, exc_tb = sys.exc_info()
  fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
  print(" %s : in FILE %s : LINE  %d" % (exc_type, fname, exc_tb.tb_lineno))


def compute_forward_values(top_of_book, at_times, horizon,
                           columns={'time': 'ts', 'mid': 'mid_price'}):
  c_time, c_mid = columns['time'], columns['mid']
  if at_times.values[0] < top_of_book[c_time].values[0]:
    raise ValueError("compute_forward_values can NOT extroplate backwards")
  x = top_of_book[c_time].values
  y = np.arange(x.shape[0])
  f = interp1d(x, y, kind='zero', bounds_error=False, fill_value="extrapolate", assume_sorted=True)
  idx = f(at_times.values + horizon)

  idx = np.int64(np.floor(idx))

  mid_price = top_of_book[c_mid].values

  return np.array(mid_price[idx]).reshape(-1)


def compute_fwd_return(top_of_book, at_times, horizon,
                       columns={'time': 'ts', 'mid': 'mid_price'}):
  c_mid = columns['mid']
  y_current = top_of_book[c_mid].values
  y_fwd = compute_forward_values(top_of_book, at_times, horizon, columns)
  return np.log(y_fwd / y_current)
