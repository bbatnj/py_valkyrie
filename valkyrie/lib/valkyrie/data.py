import os

import pandas as pd

import shutil
import glob
from pathlib import Path

from valkyrie.tools import ymd_dir, ymd
from valkyrie.securities import stocks_good_dvd, ROOT_PATH, nyse

def resample_bid_ask(date, stocks, data_path, freq='30s',
                     stime='09:30:00', etime="16:00:00",
                     verbose=False):
  path = data_path + '/' + ymd_dir(date)
  df_re = []
  for stock in stocks:
    try:
      df = pd.read_hdf(f'{path}/{stock}.h5')
      df = df.resample(freq, closed='right', label='right').last().ffill()
      df_re.append(df)
    except Exception as e:
      if verbose:
        print(f'error resampling for {date} {stock} due to {e}')
  df_re = pd.concat(df_re).sort_index().between_time(stime, etime).copy()
  df_re.rename({'priceBid': 'bid',
                'priceAsk': 'ask',
                'sizeBid': 'bz',
                'sizeAsk': 'az'}, axis=1, inplace=True)
  df_re['spread'] = df_re.eval('ask - bid')
  df_re = df_re.ffill()
  return df_re


def copy_data():
  hist_raw_root_dir = f'{ROOT_PATH}/hist_ib_raw_md'
  dst_dir = f'{ROOT_PATH}/raw/hist_ib'

  for filename in glob.iglob(hist_raw_root_dir + '**/**', recursive=True):
    if os.path.isfile(filename):
      p = Path(filename).parts
      y, m, d, kind, fn = p[4], p[5], p[6], p[7], p[8]
      dst = f'{dst_dir}/{kind}/{y}/{m}/{d}'
      os.makedirs(dst, exist_ok=True)
      shutil.copy(filename, f'{dst}/{fn}')


def check_hist_ib_data_integrity(date, stocks, path, kind):
  stock2err = {}

  for stk in stocks:
    try:
      err_msg = ''
      fn = f'{path}/{kind}/{ymd_dir(date)}/{stk}.h5'
      df = pd.read_hdf(fn)
      df['ymd'] = df.index.map(lambda t: ymd(t))

      if len(df['ymd'].unique()) > 1:
        err_msg += ' not unique ymd'

      if df.shape[0] == 0:
        err_msg += f' no data for {kind}'

      if not df.index.is_monotonic_increasing:
        err_msg += ' not sorted'
    except Exception as e:
      err_msg += " " + str(e)

    if err_msg:
      stock2err[stk] = err_msg
      print(f'removing {stk} for {date}')
      try:
        os.remove(fn)
      except Exception as e:
        pass

  return stock2err


MONTHLY_DVD_STOCKS = {'PFF'}
