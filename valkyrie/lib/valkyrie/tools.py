import multiprocessing
from collections import defaultdict
from itertools import product as cartesian_product
import copy
import os
from timeit import default_timer as timer

import cProfile
import pstats
from functools import wraps

import numpy as np
import pandas as pd
import logging
import socket


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

def printMkt(obj):
  return f"{obj.bz:.0f}x{obj.bp}-{obj.ap}x{obj.az:.0f}"

T0 = pd.Timestamp(0, tz='US/Eastern')

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f'{num:.1f}{" " + " kMGTPEZY"[magnitude]}'

def print_df_mem_usage(df : pd.DataFrame):
      x = df.memory_usage().sum()
      print(f'{human_format(x)}B')

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
