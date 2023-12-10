import os
from .tools import *
from collections import defaultdict as DefaultDict
import pandas_market_calendars as mcal
import json

ROOT_PATH = '/home/bb/projects/valkyrie/data'  # 'd:/valkyrie/data'
nyse = mcal.get_calendar('NYSE')

def secfromjson(fn):
  with open(f'{ROOT_PATH}/universe/trading_config_test.json') as f:
    config = json.load(f)

  return list(config['universe'].keys())

def parent(stk):
  if ' PR' in stk:
    return stk.split(' ')[0]
  elif stk == 'total': #special case
    return 'total'
  else:
    return stk[0:4]

def isSameFamily(stk1, stk2):
  return parent(stk1) == parent(stk2)

def pr2dash(stock):
  if ' PR' in stock:
    stock = stock.replace(' PR', '-')
  return stock

def bigpr2small(stock):
  if ' PR' in stock:
    stock = stock.replace(' PR', 'pr')
  return stock

def build_universe_df(url):
  from html_table_parser.parser import HTMLTableParser
  with open(url, 'r') as f:
    xhtml = f.read()
  p = HTMLTableParser()
  p.feed(xhtml)
  table = p.tables[-2]

  orig_len = len(table)

  table = [r for r in table if len(r) == 12]
  new_len = len(table)
  print(f'remaing table rows : {new_len}/{orig_len}')

  cols = ['SymbolCUSIP', 'des', 'exch', 'IPO', 'cpn', 'LiqPref', 'CallMat', 'rating', 'Tax15', 'Conv', 'Prosp',
          'Distribution Dates']
  df = pd.DataFrame(table[1:], columns=['orig_' + c for c in cols])
  df['symbol'] = df['orig_SymbolCUSIP'].apply(lambda s: substr(s, 0).replace('-', ' PR'))
  df['cusip'] = df['orig_SymbolCUSIP'].apply(lambda s: substr(s, 1))
  df['des'] = df['orig_des']
  df['exch'] = df['orig_exch'].apply(lambda s: substr(s, 0))
  df['ipo'] = df['orig_IPO'].apply(str2ts)
  for c in ['symbol', 'cusip', 'des', 'exch']:
    df[c] = df[c].astype(str)

  df['float'] = df['des'].apply(lambda x: 'FLOAT' in x.upper())
  df['parent'] = df['symbol'].apply(lambda x: parent(x))
  return df[[c for c in df if 'orig' not in c]].set_index('symbol')

class EnabledByParent:
  def __init__(self, parents):
    self.parents_stks = parents

  def __call__(self, sym):
    if ' PR' in sym:
      return sym.split()[0] in self.parents_stks
    elif len(sym) >= 5:
      return sym[0:4] in self.parents_stks
    else:
      return False


def bank_plus_reit():
  df_univ = pd.read_hdf(f'{ROOT_PATH}/universe/universe_20220112.h5')
  reit_fn = f'{ROOT_PATH}/universe/ListofMortgageREITs.csv'
  df_reit = pd.read_csv(reit_fn)
  parent_stks = ['JPM', 'BAC', 'WFC', 'C']
  parent_stks += list(df_reit['Symbol'])
  parent_stks = set(parent_stks)

  ebp = EnabledByParent(parent_stks)
  stock_list = list(filter(ebp, list(df_univ['symbol'])))
  return stock_list


def missing_date_for_hist_data(sdate, edate):
  stock_list = bank_plus_reit()
  date2stk = DefaultDict(lambda: set())
  for date in pd.date_range(sdate, edate):
    ymd_path = ymd_dir(date)
    for stk in stock_list:
      if not os.path.exists(f'{ROOT_PATH}/hist_ib_raw_md/{ymd_path}/bid_ask/{stk}.h5'):
        date2stk[ymd(date)].add(stk)
      if not os.path.exists(f'{ROOT_PATH}/hist_ib_raw_md/{ymd_path}/trades/{stk}.h5'):
        date2stk[ymd(date)].add(stk)
  return date2stk


def dvd_from_website(stock):
  import urllib
  from html_table_parser.parser import HTMLTableParser

  stock = pr2dash(stock)
  url = f"https://www.dividendinformation.com/search_ticker/?identifier={stock}"
  """ Opens a website and read its binary contents (HTTP Response Body) """
  req = urllib.request.Request(url=url)
  xhtml = urllib.request.urlopen(req).read().decode('utf-8')

  p = HTMLTableParser()
  p.feed(xhtml)
  df = pd.DataFrame(p.tables[-1])

  df.columns = ['date', 'amount', 'note']
  df = df.iloc[1:].copy()
  df['date'] = df['date'].apply(lambda t: pd.Timestamp(t))
  df['amount'] = df['amount'].apply(lambda x: float(x.strip('$')))

  df_check = df.copy()
  df_check['ddate'] = df_check['date'].diff().apply(lambda t: t.days)
  df_check['damount'] = df_check['amount'].diff()
  df_check = df_check.iloc[1:].copy()

  error_msg = ''
  if df_check.query('abs(damount) >= 0.01').shape[0] > 1:
    error_msg += f'{stock} dvd amount changed'

  if df_check.query('ddate < -98 or  ddate > -82').shape[0] > 1:
    error_msg += f' {stock} dvd date error'
  return df, error_msg


def gen_dvd_files(stocks, outpath='d:/valkyrie/data/universe/dvd/research_20220205'):
  os.makedirs(outpath, exist_ok=True)
  for stk in stocks:
    print(f'generating dvd for {stk}')
    df_dvd, error_msg = dvd_from_website(stk)
    df_dvd = df_dvd.query('amount > 0').copy()
    if error_msg:
      print(f'error for {stk} due to {error_msg}')
      # continue
    df_dvd.to_csv(f'{outpath}/{stk}.csv', index=False)


def stocks_good_dvd():
  return ['AGNCM',
          'AGNCN',
          'AGNCO',
          'AGNCP',
          'BAC PRB',
          'BAC PRE',
          'BAC PRK',
          # 'BAC PRL',
          'BAC PRM',
          'BAC PRN',
          'C PRJ',
          'C PRK',
          'CIM PRA',
          'CIM PRB',
          'CIM PRC',
          'CIM PRD',
          'JPM PRC',
          'JPM PRD',
          'JPM PRJ',
          'NLY PRF',
          'NLY PRG',
          'NLY PRI',
          'NRZ PRA',
          'NRZ PRB',
          'NRZ PRC',
          'PMT PRA',
          'PMT PRB',
          'STAR PRD',
          'STAR PRG',
          'STAR PRI',
          'TWO PRA',
          'TWO PRB',
          'TWO PRC',
          # 'WFC PRL',
          'WFC PRQ',
          'WFC PRR',
          'WFC PRY',
          'WFC PRZ']
