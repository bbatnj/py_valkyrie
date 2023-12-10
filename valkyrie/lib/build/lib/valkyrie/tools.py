import multiprocessing
from collections import defaultdict
from timeit import default_timer as timer
import cProfile
import pstats
from functools import wraps

import numpy as np
import pandas as pd
import logging

DAY = pd.Timedelta('1d')
WEEK = pd.Timedelta('7d')

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

def ymd(d):
  return str(pd.Timestamp(d).date()).replace('-', '')


def ymd_dir(d, sep='/'):
  d = str(pd.Timestamp(d).date()).replace('-', '')
  return sep + sep.join([d[0:4], d[4:6], d[6:8]])


def index2dt(df):
  return df.index.map(lambda x: str(pd.Timestamp(x).date()))


def df_column_types(df, float_cols, str_cols):
  for c in float_cols:
    df[c] = df[c].astype(float)

  for c in str_cols:
    df[c] = df[c].astype(str)

  return df

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

T0 = pd.Timestamp(0, tz = 'US/Eastern')

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