from functools import partial
from itertools import product as cartesianproduct
import pandas as pd
import numpy as np

from valkyrie.data import *
from valkyrie.quants import *
from valkyrie.tools import *
from valkyrie.securities import *


from valkyrie.components.common import *
from valkyrie.tools import WEEK, DAY

from valkyrie.avalon.base_pricer import BasePricer
def current_vs_auto_es(data_mgr, y_stock, ema_mults):
  # current cy spread vs self ema cy spread (es)
  f_same_family = partial(isSameFamily, stk2=y_stock)
  x_stocks = [y_stock]

  df_res = pd.concat([data_mgr.stk2df[stk] for stk in x_stocks], keys=x_stocks, axis=1)
  df_res.columns = ['_'.join(c) for c in df_res.columns]
  df_res['ymd'] = df_res.index.map(lambda t: ymd(t))
  df_res.columns = [bigpr2small(c) for c in df_res]

  ema1, ema2 = ema_mults[0], ema_mults[1]
  y_stock = bigpr2small(y_stock)
  cy_col = f'{y_stock}_cy_upx'

  xc1 = f'{y_stock}_cy_upx_ema_{ema2}' + '_minus_' + f'{cy_col}'
  xc2 = f'{y_stock}_cy_upx_ema_{ema1}' + '_minus_' + f'{y_stock}_cy_upx_ema_{ema2}'
  xcols = [xc1, xc2]

  for c in xcols:
    df_res[c] = df_res.eval(c.replace('_minus_', ' - '))

  ycols = [cy_col]
  wcol = 'weight'
  df_res['weight'] = df_res.eval(f'1.0 / {y_stock}_s2').clip(1e-6, 10000)
  df_res.dropna(how='any', inplace=True)
  print(xcols)
  return df_res, xcols, ycols, wcol


def current_vs_pff_es(data_mgr, y_stock, ema_mults):
  # current cy spread vs currnet-pff ema cy spread (es)
  x_stocks = [y_stock, 'PFF']

  df_res = pd.concat([data_mgr.stk2df[stk] for stk in x_stocks], keys=x_stocks, axis=1)
  df_res.columns = ['_'.join(c) for c in df_res.columns]
  df_res['ymd'] = df_res.index.map(lambda t: ymd(t))
  df_res.columns = [bigpr2small(c) for c in df_res]

  ema1, ema2 = ema_mults[0], ema_mults[1]
  y_stock = bigpr2small(y_stock)

  ycol = f'{y_stock}_cy_upx'
  # xc1 = f'PFF_cy_upx_ema_{ema1}' + '_minus_' + f'{y_stock}_cy_upx_ema_{ema1}'
  xc1 = f'PFF_cy_upx' + '_minus_' + ycol
  xc2 = f'PFF_cy_upx_ema_{ema1}' + '_minus_' + f'{y_stock}_cy_upx_ema_{ema1}'
  xc3 = f'PFF_cy_upx_ema_{ema2}' + '_minus_' + f'{y_stock}_cy_upx_ema_{ema2}'
  xcols = [xc1, xc2, xc3]

  for c in xcols:
    df_res[c] = df_res.eval(c.replace('_minus_', ' - '))

  df_res[xc1] = -df_res[xc1]

  wcol = 'weight'
  df_res['weight'] = df_res.eval(f'1.0 / {y_stock}_s2').clip(1e-6, 10000)
  df_res.dropna(how='any', inplace=True)
  print(xcols)
  return df_res, xcols, [ycol], wcol


def current_vs_family_es(data_mgr, y_stock, x_stocks, ema_mults):
  # current cy spread vs family ema cy spread (es)
  f_same_family = partial(isSameFamily, stk2=y_stock)
  x_stocks = list(filter(f_same_family, x_stocks))

  df_res = pd.concat([data_mgr.stk2df[stk] for stk in x_stocks], keys=x_stocks, axis=1)
  df_res.columns = ['_'.join(c) for c in df_res.columns]
  df_res['ymd'] = df_res.index.map(lambda t: ymd(t))
  df_res.columns = [bigpr2small(c) for c in df_res]

  y_stock = bigpr2small(y_stock)
  x_stocks = [bigpr2small(stk) for stk in x_stocks]
  cy_col = f'{y_stock}_cy_upx'

  xcols = []
  for ema, stk in cartesianproduct(ema_mults, x_stocks):
    if stk == y_stock:
      continue
    xc = f'{y_stock}_cy_upx_ema_{ema}' + '_minus_' + f'{stk}_cy_upx_ema_{ema}'
    df_res[xc] = df_res.eval(xc.replace('_minus_', ' - '))
    xcols.append(xc)

  for stk in x_stocks:
    if stk == y_stock:
      continue
    xc = f'{y_stock}_cy_upx' + '_minus_' + f'{stk}_cy_upx'
    df_res[xc] = df_res.eval(xc.replace('_minus_', ' - '))
    xcols.append(xc)

  wcol = 'weight'
  df_res['weight'] = df_res.eval(f'1.0 / {y_stock}_s2').clip(1e-6, 10000)
  df_res.dropna(how='any', inplace=True)
  print(xcols)
  return df_res, xcols, [cy_col], wcol


class EodTradePricer(BasePricer):
  def __init__(self, group_mgr, msg_dispatcher, stk, config):
    super().__init__(group_mgr, msg_dispatcher, stk, config)