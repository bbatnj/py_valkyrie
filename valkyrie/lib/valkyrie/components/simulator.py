import datetime
import json
import copy
import os
import logging
import shutil
import sys
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import hashlib

from valkyrie.avalon.eve import EveGroup
from valkyrie.avalon.base_pricer import BasePricerGroup
from valkyrie.avalon.fwd import EodTradePricer

from valkyrie.components.common import SecurityGroup, MessageDispatcher, BaseEnvironment
from valkyrie.components.quoter import Quoter
from valkyrie.components.comet import Comet
from valkyrie.components.sim_order_matcher import SimHiddenOrderMatcher
from valkyrie.components.risk import SecurityRiskGroup
from valkyrie.components.trade_summary import EodSummaryGroup

from valkyrie.securities import nyse
from valkyrie.tools import ymd_dir, Timer, str2loglevel, T0, profile, ROOT_DIR, SIM_ROOT_DIR, CFG_DIR

pd.set_option('display.max_columns', 500)

class LocalDataMgr:
  def __init__(self, data_path, fn):
    self.data_path = f'{ROOT_DIR}/{data_path}'
    self.fn = fn

  def at_date(self, date):
    fn = f'{self.data_path}/{ymd_dir(date)}/{self.fn}'
    # with Timer(f"reading data for {str(date)[0:10]}") as t:
    df = pd.read_hdf(fn)

    return df
class Simulator(BaseEnvironment):
  def getLogger(self):
    return self.logger

  def now(self):
    return self._now

  def __init__(self, sim_name, group_name, data_mgr, config_json : str, log_level: str):
    super().__init__(config_json)

    self.sim_name = sim_name
    self.group_name = group_name
    self.data_mgr  = data_mgr
    self._now = T0

    self.timer_interval = pd.Timedelta(self.config['env']['sim_timer_interval'])
    self.sim_root_dir = SIM_ROOT_DIR
    self.fee = self.config['env']['fee']

    if self.sim_name == "":
      self.sim_name = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()[0:8]

    if self.group_name:
      self.config['sim_dir'] = f'{self.sim_root_dir}/{self.group_name}/{self.sim_name}'
    else:
      self.config['sim_dir'] = f'{self.sim_root_dir}/{self.sim_name}'
    sim_dir = self.config['sim_dir']

    shutil.rmtree(f'{sim_dir}', ignore_errors=True)
    os.makedirs(f'{sim_dir}', exist_ok=True)
    shutil.rmtree(f'{self.sim_root_dir}/latest', ignore_errors=True)
    #os.system(f'rm -rf {sim_dir}')
    #os.system(f'mkdir -p {sim_dir}')
    #os.system(f'rm -rf {self.sim_root_dir}/latest')
    os.system(f'ln -s {sim_dir} {self.sim_root_dir}/latest')

    with open(f'{sim_dir}/sim_config.json','w') as f:
      json.dump(self.config, f)

    logger = logging.getLogger()
    fileHandler = logging.FileHandler(f'{sim_dir}/sim_out.txt')
    streamHandler = logging.StreamHandler(sys.stdout)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    #level = str2loglevel[self.config['log']['level'].upper()]
    logger.setLevel(log_level)
    self.logger = logger

    self.msg_dispatcher = MessageDispatcher()
    self._setup()

  def createPricer(self, config):
    pricer_type = config['pricer']['type']
    if pricer_type == "Eve":
      return EveGroup('pricer', self, config['pricer'])
    elif pricer_type == "Fwd":
      return BasePricerGroup('pricer', EodTradePricer, self, config['pricer'])

  def _setup(self):
    config = defaultdict(dict, self.config)
    self.components = {}

    if config['sim_matcher']:
      self.components['order_engine'] = SecurityGroup('order_engine', SimHiddenOrderMatcher, self, config['sim_matcher'])

    if config['comet']:
      self.components['comet'] = SecurityGroup('comet', Comet, self, config['comet'])
      self.components['comet'].order_id_step = 10

    if config['quoter']:
      self.components['quoter'] = SecurityGroup('quoter', Quoter, self, config['quoter'])
      self.components['quoter'].order_id_step = 10

    if config['pricer']:
      self.components['pricer'] = self.createPricer(config)

    if config['risk']:
      self.components['risk_mgr'] = SecurityRiskGroup('risk_mgr', self, config['risk'])

    self.components['eod_summarizer'] = EodSummaryGroup('eod_summarizer', self, self.components['pricer'])

    def com_from_names(names):
      if type(names) is str:
        names = names.split()
      return [self.components[n] for n in names if n in self.components]

    self.msg_dispatcher.add_tv_listeners(com_from_names('comet quoter risk_mgr order_engine eod_summarizer'))
    self.msg_dispatcher.add_risk_listeners(com_from_names('comet quoter order_engine eod_summarizer'))
    self.msg_dispatcher.add_order_instr_listeners(com_from_names('order_engine eod_summarizer'))
    self.msg_dispatcher.add_order_update_listeners(com_from_names('comet quoter risk_mgr eod_summarizer'))

  def _run_a_day(self, date):
    df = self.data_mgr.at_date(date).between_time("09:30:00", '16:00:00')
    n = df.shape[0]
    ts = df.index
    bid, ask, bz, az = df['bid'].values, df['ask'].values, df['bz'].values, df['az'].values
    ticker, trd_px, trd_sz, exch = df['ticker'].values, df['trd_px'].values, df['trd_sz'].values, df['exch'].values

    # tracemalloc.start()
    for _, c in self.components.items():
      c.onSoD(date)

    start_time = date + pd.Timedelta("09:30:00")
    self._now, self._last_timer_event = start_time, start_time

    for i in range(n):
      #mimic timer event
      while ts[i] >= (self.timer_interval + self._last_timer_event):
        self._last_timer_event += self.timer_interval
        self._now = self._last_timer_event
        for _, c in self.components.items():
          c.onTimer(self._now)

      self._now = ts[i]

      if not np.isfinite(trd_px[i]):
        for _, c in self.components.items():
          c.onBidAsk(ts[i], ticker[i], bid[i], bz[i], ask[i], az[i])
      else:
        for _, c in self.components.items():
          c.onMktTrade(ts[i], exch[i], ticker[i], trd_px[i], trd_sz[i], "")

    for _, c in self.components.items():
      c.onEoD(date)

  def run(self, sdate, edate):
    for date in nyse.schedule(start_date=sdate, end_date=edate).index:
      date = pd.Timestamp(date, tz='US/Eastern')
      with Timer(f'simulation for {str(date)[0:10]}') as t:
        self.logger.critical(f'starting sim {self.sim_name} for {date}')
        self._run_a_day(date)

        self.components['eod_summarizer'].summarize()

    summary_fn = self.config['log']['summary']
    #os.system(f'mkdir -p {"/".join(summary_fn.split("/")[0:-1])}')
    #df_res.to_hdf(summary_fn, key='data')

#@profile(output_file="simulation.prf")
def main(args = None):
  print(f'trading simulation started at {datetime.datetime.now()}')

  if not args:
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', help='config file')
    parser.add_argument('--sdate', help='sim start date')
    parser.add_argument('--xdate',  help='trading start date')
    parser.add_argument('--edate', help='trading end date')
    parser.add_argument('--log', help='logging') #    'CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'
    parser.add_argument('--name', help='sim name', default="")
    parser.add_argument('--group', help='sim group', default="")

    args = vars(parser.parse_args())

  sim_name = args['name']
  with Timer(f'simulation {sim_name} finished ') as t:
    local_data_mgr = LocalDataMgr('data/merged/hist_ib', 'stocks_good_dvd.h5')
    config_fn, sdate, edate, log_level, group \
      = args['config'], args['sdate'], args['edate'], args['log'].upper(), args['group']
    simulator = Simulator(sim_name, group, local_data_mgr, f'{CFG_DIR}/{config_fn}', log_level)
    simulator.run(sdate, edate)

if __name__ == "__main__":
  main()