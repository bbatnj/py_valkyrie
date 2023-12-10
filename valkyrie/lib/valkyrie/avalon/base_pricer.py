import gc
import os
import json
import copy
from collections import defaultdict as DD

import numpy as np
import pandas as pd
from valkyrie.components.common import SecurityGroup, PerSecurityHandler, BaseEnvironment
from valkyrie.tools import ymd_dir, T0, ROOT_DIR
from overrides import overrides

class BasePricer(PerSecurityHandler):
  dvd_freq2days = {
    'M' : 30,
    'Q' : 90
  }

  def calc_annual_dvd(self):
    mult = 360 / BasePricer.dvd_freq2days[self.dvd_freq]
    return self.get_avg_dvd() * mult

  def cy2px(self, cy):
    return self.get_accured_dvd() + (self.calc_annual_dvd() / cy)

  def __init__(self, group_mgr, stk, config):
    super().__init__(group_mgr, stk, config)
    self.feature_arrays = DD(lambda : [])
    self.record_fields = {}

  def load_dvd_fn(self, dvd_fn):
    stk = self.stk
    df = pd.read_csv(dvd_fn).sort_values('date')
    df['ymd'] = df['date'].apply(lambda s : pd.Timestamp(s, tz = 'US/Eastern'))
    if df.shape[0] <= 1:
      raise Exception(f'Not enough dvd history for {stk} number of records = {df.shape[0]}')

    df['ticker'] = stk
    self.df_dvd = df.iloc[1:].copy()  # remove the first one dvd as it may be inaccurate
    self.dvd_freq = self.group_mgr.dvd_freq[stk]
    self._dvd_today = np.nan

  #orverride me
  def _pricerSod(self, date: pd.Timestamp):
    pass

  @staticmethod
  def calc_avg_dvd(df_dvd):
    #if df_dvd.shape[0] < 3 :
    #  raise Exception('Requires at least 3 dvds to calc dvd')
    return df_dvd['amount'].mean()

  def get_today_dvd(self):
    return self._dvd_today

  def get_accured_dvd(self):
    return self._accured_amt

  def get_avg_dvd(self):
    return self._avg_dvd_amt

  def save_fields(self):
    if self.config['record_feature']:
      self.feature_arrays['t'].append(self.now())
      self.feature_arrays['bid'].append(self.bp)
      self.feature_arrays['bz'].append(self.bz)
      self.feature_arrays['ask'].append(self.ap)
      self.feature_arrays['az'].append(self.az)

      for fld in self.record_fields:
        self.feature_arrays[fld].append(self.record_fields[fld])
         #self.feature_arrays[f'cywa_{hl}_s2'].append(cywa.s2_h)
    return

  @overrides
  def onSoD(self, date):
    #calc dvd accrued
    df_dvd = self.df_dvd.query("ymd <= @date").copy()
    last_xdvd_date = df_dvd['ymd'].iloc[-1]

    self._avg_dvd_amt = BasePricer.calc_avg_dvd(df_dvd)

    if last_xdvd_date == date:
      self._accured_amt  = 0
      self._dvd_today = df_dvd['amount'].iloc[-1]
    else:
      n_accrued = (date - last_xdvd_date).days
      accured_amt = self._avg_dvd_amt * n_accrued / BasePricer.dvd_freq2days[self.dvd_freq]
      self._accured_amt = np.clip(accured_amt, 0.0, self._avg_dvd_amt)
      self._dvd_today = 0.0

    #for derived class
    self._pricerSod(date)

  def onEoD(self, date):
    if self.config['record_feature']:
      sim_dir = self.config['sim_dir'] + '/' + ymd_dir(date) + f'/{self.stk}'
      os.system(f'mkdir -p "{sim_dir}"')

      df = pd.DataFrame(self.feature_arrays)
      df.to_hdf(f'{sim_dir}/features.h5', key='data')

    self.feature_arrays = DD(lambda : [])
    gc.collect()

  def __str__(self):
    return f'BasePricer_{self.stk}'

class BasePricerGroup(SecurityGroup):
  def __init__(self, name : str, pricer_class: BasePricer, env : BaseEnvironment, config : dict):
    config = copy.deepcopy(config)
    config['default']['sim_dir'] = env.config['sim_dir']
    super().__init__(name, pricer_class, env, config)

    self.last_update = T0
    self.sample_interval_s = config["sample_interval_s"]


    dvd_dir = f"{ROOT_DIR}/{config['default']['dvd_dir']}"
    with open(f"{dvd_dir}/dvd_freq.json") as f:
      dvd_freq = json.load(f)

    self.dvd_freq = DD(lambda :dvd_freq['default'])
    for k, v in dvd_freq.items():
      self.dvd_freq[k] = v

    for stk in self.stk2sec:
      self.stk2sec[stk].load_dvd_fn(f"{dvd_dir}/{stk}.csv")

  @overrides
  def onSoD(self, date):
    for stk in self.stk2sec:
      self.stk2sec[stk].onSoD(date)

