import os
from collections import namedtuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from valkyrie.securities import parent
from valkyrie.tools import ymd, SIM_ROOT_DIR
from valkyrie.components.common import *
from valkyrie.quants.utils import sharpe, win_ratio

class Summary:
  summary_funs = ('mean', 'median', 'max', 'min', 'std')
  def __init__(self, summary_dir):
    h5files = sorted(os.listdir(summary_dir))
    self.df = pd.concat([pd.read_hdf(summary_dir + '/' + fn) for fn in h5files])

  @staticmethod
  def eod_by_issue(df):
    return df.groupby('parent')['pnl pos_pnl volume'.split(' ')] \
      .agg(['sum', 'count', 'std', 'min', 'median', 'max'])

  @classmethod
  def calc_pnl(cls, df, summary_funs = None):
    if not summary_funs:
      summary_funs = (sharpe, win_ratio) + cls.summary_funs

    df = df[['edge', 'pnl', 'pos_pnl', 'fee_pnl', 'dvd_rcvd', 'dvd_paid', 'total_pnl']].agg(summary_funs)
    return df

  @classmethod
  def avg_pnl(cls, df):
    df = df.copy()
    if 'total_pnl' not in df:
      df['total_pnl'] = df.eval('pnl + pos_pnl + dvd_rcvd + dvd_paid + fee_pnl')
    return cls.calc_pnl(df)

  @classmethod
  def avg_trades(cls, df):
    trade_cols = ['$pos', '$volume', '$bs', 'volume', 'bs', '#buy_order', '#sell_order', '#fill']
    df = df[trade_cols].agg(cls.summary_funs)
    return df

class GroupSimViewer:
  def __init__(self, group_name, sim_root_dir=SIM_ROOT_DIR):
    self.group_name = group_name
    self.sim_root_dir = sim_root_dir
    sim_names = os.listdir(f'{SIM_ROOT_DIR}/{group_name}')
    self.sim_viewers = {sim_name : SimViewer(f'{group_name}/{sim_name}', sim_root_dir) for sim_name in sim_names}

  def view(self, col = 'total_pnl' ):
    return pd.DataFrame({n : self.sim_viewers[n].view(plot = False)[col] for n in self.sim_viewers}).T.sort_index()

class SimViewer:
  def __init__(self, sim_name, sim_root_dir=SIM_ROOT_DIR):
    sim_dir = f'{sim_root_dir}/{sim_name}/summary'
    h5files = sorted(os.listdir(sim_dir))
    self.df = pd.concat([pd.read_hdf(sim_dir + '/' + fn) for fn in h5files if fn.endswith('.h5')])
    self.df['total_pnl'] = self.df.eval('pnl + pos_pnl + dvd_rcvd + dvd_paid + fee_pnl')

  def view(self, plot = True):
    if plot:
      fig, axes = plt.subplots(nrows=1, ncols=3)
      fig.set_figwidth(15)
      self.plot_cum_pnl(axes[0])
      self.plot_daily_pos(axes[1])
      self.plot_cum_trade(axes[2])
    df_pnl = Summary.avg_pnl(self.df.query('ticker == "total"'))
    df_trades = Summary.avg_trades(self.df.query('ticker == "total"'))
    return pd.concat([df_pnl, df_trades], axis = 1)

  def plot_cum_pnl(self, ax=None):
    df_total = self.df.query('ticker == "total"').copy()
    df_total['dvd'] = df_total.eval('dvd_rcvd + dvd_paid')
    df_total[['pnl', 'pos_pnl', 'fee_pnl', 'total_pnl', 'dvd_rcvd', 'dvd_paid']]\
      .cumsum().plot(ax=ax, title='cum_pnl')
  def plot_daily_pos(self, ax=None):
    df_total = self.df.query('ticker == "total"').copy()
    df_total['$pos_mean'] = df_total['$pos'].mean()
    df_total['$pos_abs_mean'] = df_total['$pos_abs'].mean()
    df_total[['$pos', '$pos_abs', '$pos_mean', '$pos_abs_mean']]\
      .plot(ax=ax, title='position', style=['-b','-r','-.b','-.r'])

  def plot_cum_trade(self, ax=None):
    df = self.df.query('ticker == "total"').copy()
    df = df[['$volume', '$bs']].cumsum()
    ax = df[['$volume']].plot(ax=ax, title='trades', color='b', alpha=0.7)
    ax2 = ax.twinx()
    df[['$bs']].plot(ax=ax2, color='g', alpha=0.7)

  def by_issuer(self, cols=['pnl', 'pos_pnl', 'total_pnl'], summary_funcs=['sum', win_ratio]):
    return Summary.calc_pnl(self.df.groupby('parent'), summary_funcs)[cols]
class EodTradeSummary(PerSecurityHandler):
  FillRecord = namedtuple('Fill', ['ts', 'stk', 'side', 'px', 'size', 'tv', 'fee'])

  def __init__(self, group_mgr, stk, config):
    super().__init__(group_mgr, stk, config)
    self.open_pos, self.open_risk_tv = 0.0, np.nan
    self.pricer = None
    self.order_count, self.buy_order_count, self.sell_order_count, self.fill_records = None, None, None, None
    self.dvd_rcvd, self.dvd_paid = None, None

  def onSoD(self, date):
    self.order_count = {
      '#buy_order' : 0,
      '#sell_order': 0
    }
    self.buy_order_count, self.sell_order_count  = 0, 0
    self.fill_records = []  # clear for new day
    self.df_fill = None

    if self.pricer:
      net_dvd = self.open_pos * self.pricer.get_today_dvd()
      self.dvd_rcvd = max(0.0, net_dvd)
      self.dvd_paid = min(0.0, net_dvd)

  def onOrderInstr(self, order_instr : OrderInstr):
    if not np.isfinite(order_instr.px):
      return

    if order_instr.side == Side.Buy:
      self.order_count['#buy_order'] += 1
    elif order_instr.side == Side.Sell:
      self.order_count['#sell_order'] += 1


  def onBidAsk(self, ts, stk, bp, bz, ap, az):
    self.bp, self.bz, self.ap, self.az = bp, bz, ap, az
    self.curr_sprd = self.ap - self.bp

  def onRiskAdj(self, risk_update: RiskUpdate):
    self.logger.info(f'onRiskAdj stk : {self.stk} ')
    self.risk_update = risk_update

  def _onFill(self, order_update: OrderUpdate):
    ts, tv = self.group_mgr.now(), self.tv_update.tv

    px, side, size = order_update.px, order_update.side, order_update.sz

    edge = side * (tv - order_update.px)
    #edge_adj = SRC.calc_adj_edge(side, edge_adj, self.fee, self.risk_update)
    if edge < 0:
      self.logger.warning(f'Negative edge {self.stk} tv : {tv} order {order_update} edge_adj {edge}')

    self.logger.info(f'summarizer : {self.stk} {edge:.4f} {int(size)}')

    fee = self.group_mgr.fee_calc.calc_fee(order_update.tactic, order_update.sz)

    record = \
      self.FillRecord(ts=ts, stk=self.stk, side=side, size=size, px=px, tv=tv, fee=fee)
    self.fill_records.append(record)

  def onTV(self, tv_update: TVUpdate):
    self.tv_update = tv_update
    self.logger.debug(f'{self.now()} Summary onTV: {self.stk} '                      
                      f'r_adj: {self.tv_update.tv:.3f} ')

  def onEoD(self, date : pd.Timestamp):
    open_pos, open_risk_tv = self.open_pos, self.open_risk_tv
    risk_tv = self.tv_update.risk_tv
    self.open_risk_tv = risk_tv  # prepare for next day #TODO: this should be adjusted for dvd

    if len(self.fill_records) == 0:  # create dummy fill record
      ts = self.group_mgr.now()
      record = self.FillRecord(ts=ts, stk=self.stk, side=Side.Buy, size=0, px=risk_tv, tv=risk_tv, fee=0.0)
      self.fill_records.append(record)

    df_fill = pd.DataFrame(self.fill_records)

    # sum cols
    df_fill['cash'] = df_fill.eval('-px * size * side')

    df_fill['edge'] = df_fill.eval('side * size * (tv - px)')
    df_fill['bs'] = df_fill.eval('side * size')
    df_fill['volume'] = df_fill['size'].abs()
    df_fill['$volume'] = df_fill.eval('volume * @risk_tv')
    df_fill['$bs'] = df_fill.eval('bs * @risk_tv')
    df_fill['#bs'] = 1
    eod_tv = self.tv_update.risk_tv
    df_fill['pnl'] = df_fill.eval('side * size * (@eod_tv - px)')
    df_fill['fee_pnl'] = df_fill.eval('-1.0 * fee')
    sum_cols = ['fee_pnl', 'pnl', 'cash', '#bs', 'bs', 'volume', '$bs', '$volume', 'edge']

    # last cols
    df_fill['pos'] = df_fill['bs'].cumsum() + open_pos
    df_fill['$pos'] = df_fill.eval('pos * @risk_tv')  # note : $pos using eod tv
    df_fill['$pos_abs'] = df_fill['$pos'].abs()
    df_fill['pos_pnl'] = df_fill.eval('@open_pos * (@risk_tv - @open_risk_tv)')
    last_cols = ['pos_pnl', 'pos', '$pos', '$pos_abs']
    res = df_fill[sum_cols + last_cols].agg(
      {c: 'sum' for c in sum_cols} | {c: lambda x: x.iloc[-1] for c in last_cols})

    res['sod_tv'], res['eod_tv'] = open_risk_tv, self.tv_update.risk_tv
    res['sod_share'], res['eod_share'] = open_pos, df_fill['pos'].iloc[-1]

    res['#buy_order'] = self.order_count['#buy_order']
    res['#sell_order'] = self.order_count['#sell_order']
    res['#fill'] = df_fill.query('size > 0 ').shape[0]

    res['dvd_rcvd'] = self.dvd_rcvd
    res['dvd_paid'] = self.dvd_paid
    #res['total_pnl'] = res['pnl'] + res['pos_pnl'] + res['dvd_rcvd'] + res['dvd_paid']
    self.eod, self.df_fill = res, df_fill

    # prepare for next day
    self.open_pos = df_fill['pos'].iloc[-1]

    #save today's fills
    today = ymd(self.group_mgr.env.now())
    save_dir = f'{self.group_mgr.save_dir}/fills/{today}/'
    #os.system(f'mkdir -p {save_dir}')
    os.makedirs(f'{save_dir}', exist_ok=True)

    df_fill.to_hdf(f'{save_dir}/{self.stk}.h5', key = 'data')

class EodSummaryGroup(SecurityGroup):

  def __init__(self, name, env, pricer_group):
    config = {}
    super().__init__(name, EodTradeSummary, env, config)
    self.dfs = []
    self.pricer_group = pricer_group
    for stk in self.stk2sec:
      self.stk2sec[stk].pricer = pricer_group.stk2sec[stk]

    sim_dir = env.config['sim_dir']
    save_dir = f'{sim_dir}/summary'
    #os.system(f'mkdir -p {save_dir}')
    os.makedirs(f'{save_dir}', exist_ok=True)
    self.save_dir = save_dir

  def onEoD(self, date):
    super().onEoD(date)

    stk2eod_stats = {stk: self.stk2sec[stk].eod for stk in self.stk2sec}

    df = pd.DataFrame(stk2eod_stats).T.fillna(0.0)
    df.loc['total'] = df.sum()
    df['date'] = date
    df.index.name = 'ticker'
    df['parent'] = df.index.map(lambda x : parent(x))
    df = df.reset_index().set_index('date')
    self.dfs.append(df.copy())

    self.logger.critical(f"============Simulation Done for {date}====================")
    #cols = 'ticker pnl pos_pnl bs sod_tv eod_tv dvd_rcvd dvd_paid sod_share eod_share #buy_order #sell_order #fill'
    #print(df.groupby('parent')['pnl pos_pnl sod_share eod_share #fill'.split(' ')].sum())

  def summarize(self):
    pd.set_option('display.float_format', '{:3g}'.format)

    df_lastday = self.dfs[-1]

    self.logger.warning('============= EoD By issuer =======================')
    self.logger.warning(Summary.eod_by_issue(df_lastday))

    today = ymd(self.env.now())
    df_lastday.to_hdf(f'{self.save_dir}/eod_summary_{today}.h5', key = 'data')

    df_res = pd.concat(self.dfs)

    df_trades = Summary.avg_trades(df_res.query('ticker == "total"'))
    self.logger.warning('==============avg trades================\n' + str(df_trades))

    df_pnl = Summary.avg_pnl(df_res.query('ticker == "total"'))
    self.logger.warning('===========avg PnL===================\n' + str(df_pnl))
    return df_res




