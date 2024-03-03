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


class EodTradePricer(BasePricer):
  def __init__(self, group_mgr, msg_dispatcher, stk, config):
    super().__init__(group_mgr, msg_dispatcher, stk, config)
