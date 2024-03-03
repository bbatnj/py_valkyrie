import multiprocessing
from collections import defaultdict
from itertools import product as cartesian_product
import copy
import os
from timeit import default_timer as timer

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import logging
import socket


import cProfile
import pstats
from functools import wraps



HOSTNAME = socket.gethostname()
HOSTTYPE= "LINUX" if 'VirtualBox' in HOSTNAME or 'aorus' in HOSTNAME else "WIN"

ROOT_DIR = "D:/valkyrie_shared/" if HOSTTYPE == "WIN" else '/mnt/win/valkyrie_shared/'
SIM_ROOT_DIR = "D:/valkyrie_shared/sim_dir" if HOSTTYPE == "WIN" else '/mnt/win/valkyrie_shared/sim_dir'
CFG_DIR = "D:/valkyrie_shared/configs" if HOSTTYPE == "WIN" else '/mnt/win/valkyrie_shared/configs'

DAY = pd.Timedelta('1d')
WEEK = pd.Timedelta('7d')
TRADING_SEC_PER_DAY = 3600 * 6.5

str2loglevel = {
  'DEBUG': logging.DEBUG,
  'INFO': logging.INFO,
  'WARNING': logging.WARNING,
  'CRITICAL': logging.CRITICAL}

class Timer:
  def __init__(self, des):
    self.des = des

  def __enter__(self):
    self.start = timer()

  def __exit__(self, exc_type, exc_value, traceback):
    end = timer()
    self.dur = pd.Timedelta(end - self.start, unit="second")
    print(f'{self.des} finished in : {self.dur.total_seconds()} secs')

def cpu_at_80():
  return int(multiprocessing.cpu_count() * 0.8)

def cpum2():
  return max(multiprocessing.cpu_count() - 2, 1)

def to_cents(x):
  return np.round(x * 100)

def round(x):
  return int(np.round(x))

# alias
px2cents = to_cents

def substr(s, i):
  re = s.split()
  if i < len(re):
    return re[i]
  else:
    return None


def str2ts(s):
  try:
    return pd.Timestamp(s)
  except:
    return None

def plot2y(df,ycols1, ycols2,title='' ,styles = ['-o', '-d']):
  ax = df[ycols1].plot(ax=ax, title='trades', style=styles[0], alpha=0.7)
  ax2 = ax.twinx()
  df[ycols2].plot(ax=ax2, style=styles[1], alpha=0.7)

def ymd(d):
  return str(pd.Timestamp(d).date()).replace('-', '')


def ymd_dir(d, sep='/'):
  d = str(pd.Timestamp(d).date()).replace('-', '')
  return sep + sep.join([d[0:4], d[4:6], d[6:8]])


def index2dt(df):
  return df.index.map(lambda x: str(pd.Timestamp(x).date()))



def calc_diff_days(df, col1, col2):
  s = df[col1].apply(lambda x: pd.Timestamp(x)) - df[col2].apply(lambda x: pd.Timestamp(x))
  return s.apply(lambda x: x.days)


def df_col2str(df):
  for c, t in df.dtypes.items():
    if t == object:
      df[c] = df[c].astype(str)
  return df

def drop_dup_col_(df):
  duplicate_cols = df.columns[df.columns.duplicated()]
  df.drop(columns=duplicate_cols, inplace=True)

def printMkt(obj):
  return f"{obj.bz:.0f}x{obj.bp}-{obj.ap}x{obj.az:.0f}"

T0 = pd.Timestamp(0, tz='US/Eastern')

def format_df_nums(df, cols_with_comma = []):
    df = df.copy()
    for c, t in df.dtypes.items():
        if np.issubdtype(t, np.number):
            if c in cols_with_comma:
                df[c] = df[c].apply(lambda x : f'${x:,.2f}')
            else:
                df[c] = df[c].apply(lambda x : f'{x:.4g}')
    return df

def df_des(df):
    return df.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.995])

def str2sec(s):
    try:
        return pd.Timedelta(s).total_seconds()
    except:
        raise Exception(f'invalid time format {s}')

def format2kmgt(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f'{num:.1f}{" " + " kMGTPEZY"[magnitude]}'

def print_df_mem_usage(df : pd.DataFrame):
      x = df.memory_usage().sum()
      print(f'{format2kmgt(x)}B')

def toDf32_(df):
  for c, kind in df.dtypes.items():
    if np.issubdtype(kind, np.float64):
      df[c] = df[c].astype(np.float32)
    elif np.issubdtype(kind, np.int64):
      df[c] = df[c].astype(np.int32)

class ConfigGenerator:
  def __init__(self, src: str, p2p: dict):
    with open(src, 'r') as file:
      lines = file.readlines()
    self.text = ''.join(lines)

    self.src = src
    self.p2p = copy.deepcopy(p2p)

  def addParam(self, pattern: str, params: dict):
    self.p2p[pattern] = copy.deepcopy(params)

  @staticmethod
  def multi_replace(s: str, p2p: dict):
    for k, v in p2p.items():
      s = s.replace(k, str(v))
    return s

  def generate(self, dst_dir: str):
    os.system(f'rm -rf {dst_dir}')
    os.system(f'mkdir -p {dst_dir}')
    patterns = [k for k in self.p2p.keys()]
    params = [v for _, v in self.p2p.items()]

    tpl_name = '.'.join(self.src.split('/')[-1].split('.')[0:-1])
    for i, param in enumerate(cartesian_product(*params)):
      cur_p2p = dict(zip(patterns, param))
      with open(f'{dst_dir}/{tpl_name}_{i + 1 :04d}'.replace('.tpl','') + '.json', 'w') as file:
        file.writelines([self.multi_replace(self.text, cur_p2p)])

def plot_multiple_lines(df, y_axis, t1=None, t2=None):
    if len(y_axis) > 4:
        raise Exception(f'Max 4 y_axis supported, given {len(y_axis)}')

    y1_label, y2_label, y3_label, y4_label = '', '', '', ''
    y1_cols, y2_cols, y3_cols, y4_cols = [], [], [], []

    for i, e in enumerate(list(y_axis.items())):
        label, cols = e[0], e[1]
        if i == 0:
            y1_label = label
            y1_cols = cols.copy()
        elif i == 1:
            y2_label = label
            y2_cols = cols.copy()
        elif i == 2:
            y3_label = label
            y3_cols = cols.copy()
        elif i == 3:
            y4_label = label
            y4_cols = cols.copy()

    if t1 or t2:
        df = df.between_time(t1, t2).copy()

    transparent = 0.5
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.8)

    lines = []

    for i, col in enumerate(y1_cols):
        line_i1 = ax1.plot(df[col], c=cmap[i], label=col, alpha=transparent)
        lines += line_i1

    ax1.set_xlabel("Time")
    ax1.set_ylabel(y1_label, fontsize=20)
    ax1.yaxis.label.set_color(lines[-1].get_color())
    ax1.tick_params(axis="x")
    ax1.tick_params(axis="y", colors=lines[-1].get_color())
    ax1.grid(True, axis="both", which="both")

    if y2_cols:
        ax2 = ax1.twinx()
        for i2, col in enumerate(y2_cols, len(y1_cols)):
            line_i2 = ax2.plot(df[col], c=cmap[i2], label=col, alpha=transparent)
            lines += line_i2
        ax2.set_ylabel(y2_label)
        ax2.yaxis.label.set_color(lines[-1].get_color())
        ax2.tick_params(axis="both", colors=lines[-1].get_color())

    if y3_cols:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        for i3, col in enumerate(y3_cols, len(y1_cols) + len(y2_cols)):
            line_i3 = ax3.plot(df[col], c=cmap[i3], label=col, alpha=transparent)
            lines += line_i3
        ax3.set_ylabel(y3_label)
        ax3.yaxis.label.set_color(lines[-1].get_color())
        ax3.tick_params(axis="both", colors=lines[-1].get_color())

    if y4_cols:
        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 120))
        for i4, col in enumerate(y4_cols, len(y1_cols) + len(y2_cols) + len(y3_cols)):
            line_i4 = ax4.plot(df[col], '-', c=cmap[i4], label=col)
            lines += line_i4
        ax4.set_ylabel(y4_label)
        ax4.yaxis.label.set_color(lines[-1].get_color())
        ax4.tick_params(axis="both", colors=lines[-1].get_color())

    labs = [l.get_label() for l in lines]
    plt.legend(lines, labs, loc="lower left")
    plt.close(fig)
    return fig



def profile(output_file=None, sort_by='cumulative', lines_to_print=None, strip_dirs=False):
  """A time profiler decorator.
  Inspired by and modified the profile decorator of Giampaolo Rodola:
  http://code.activestate.com/recipes/577817-profile-decorator/
  Args:
      output_file: str or None. Default is None
          Path of the output file. If only name of the file is given, it's
          saved in the current directory.
          If it's None, the name of the decorated function is used.
      sort_by: str or SortKey enum or tuple/list of str/SortKey enum
          Sorting criteria for the Stats object.
          For a list of valid string and SortKey refer to:
          https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
      lines_to_print: int or None
          Number of lines to print. Default (None) is for all the lines.
          This is useful in reducing the size of the printout, especially
          that sorting by 'cumulative', the time consuming operations
          are printed toward the top of the file.
      strip_dirs: bool
          Whether to remove the leading path info from file names.
          This is also useful in reducing the size of the printout
  Returns:
      Profile of the decorated function
  """

  def inner(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      _output_file = output_file or func.__name__ + '.prof'
      pr = cProfile.Profile()
      pr.enable()
      retval = func(*args, **kwargs)
      pr.disable()
      pr.dump_stats(_output_file)

      with open(_output_file, 'w') as f:
        ps = pstats.Stats(pr, stream=f)
        if strip_dirs:
          ps.strip_dirs()
        if isinstance(sort_by, (tuple, list)):
          ps.sort_stats(*sort_by)
        else:
          ps.sort_stats(sort_by)
        ps.print_stats(lines_to_print)
      return retval

    return wrapper

  return inner
