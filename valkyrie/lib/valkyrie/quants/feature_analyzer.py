from itertools import product as cartesian_product
import gc

from sklearn.linear_model import LinearRegression as LR
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from valkyrie.quants.linear_model import WinsorizedLM
from valkyrie.securities import stocks_good_dvd, nyse, ymd_dir
from valkyrie.tools import ymd, SIM_ROOT_DIR, toDf32_

class FeatureMgr:
  def __init__(self, sdate, edate, stocks, sim,
               root_dir=SIM_ROOT_DIR):
    self.sdate, self.edate, self.stocks = sdate, edate, stocks
    self.root_dir, self.sim = root_dir, sim
    self._loaded = False
    self.ret_calc_black_list = {'PFF'}

  def load(self):
    def _load_per_stock(stk):
      gc.collect()
      df_per_stk = []
      for date in nyse.schedule(start_date=self.sdate, end_date=self.edate).index:
        try:
          fn = f'{self.root_dir}/{self.sim}/{ymd_dir(date)}/{stk}/features.h5'
          df = pd.read_hdf(fn)
          df_per_stk.append(df)
        except Exception as e:
          print(f'error loading stocks {e} on {date}')

      df_per_stk = pd.concat(df_per_stk)
      df_per_stk['date'] = df_per_stk['t'].apply(ymd)

      return df_per_stk.set_index('t')

    self.stk2df = {stk: _load_per_stock(stk) for stk in self.stocks}
    self._loaded = True

  def calc_rets_(self, kind: str, col: str, ns: list, s2col=None, float32=False, wmul = 1.0):
    if not self._loaded:
      self.load()

    df_ret = []
    for stk in self.stk2df:
      if stk in self.ret_calc_black_list:
        continue

      df = self.stk2df[stk]
      df['ticker'] = stk
      df['mid'] = df.eval('0.5 * (bid + ask)')
      add_fwd_ret_by_n(df, kind, col, ns, s2col, wmul)
      df.dropna(inplace=True)
      df_ret.append(df)

    df_ret = pd.concat(df_ret)
    if float32:
      toDf32_(df_ret)
    return df_ret

  def calc_per_instr_(self, func, *args):
    for instr in self.instr2df:
      df = self.instr2df[instr]
      func(df, *args)

