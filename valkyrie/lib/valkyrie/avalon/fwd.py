from valkyrie.components.common import *
from valkyrie.tools import WEEK, DAY

from valkyrie.avalon.base_pricer import BasePricer

class EodTradePricer(BasePricer):
  def __init__(self, group_mgr, stk, config):
    super().__init__(group_mgr, stk, config)

  def __str__(self):
    return f'EodTradePricer_{self.stk}'

  def _pricerSod(self, date): #override
    horizon = pd.Timedelta('0d')
    sdate = date + horizon
    fwd_date = nyse.schedule(sdate, sdate + WEEK).index[0]

    df_mkt = self.group_mgr.data_mgr.at_date(fwd_date)

    stk = self.stk
    df_mkt = df_mkt.query(f'ticker == "{stk}"').copy()
    df_trade = df_mkt.query('trd_sz > 0')[['bid', 'ask', 'trd_px']].copy()
    if df_trade.shape[0] > 0:
      self.eod_px = df_trade['trd_px'].iloc[-1]
    else:
      self.eod_px = df_mkt.eval('0.5*(bid+ask)').iloc[-1]

  def onBidAsk(self, ts, stk, bp, bz, ap, az):
    eod_px = self.eod_px
    w, mid = 1.0, 0.5 * (bp + ap)
    tv = w * eod_px + (1 - w) * mid
    risk_tv = eod_px
    tv_update = TVUpdate(tv=tv, risk_tv=risk_tv, stk=stk)
    self.msg_dispatcher.onTV(tv_update)

  def onTimer(self, now: pd.Timestamp):
    pass

# class EodTradePricerGroup(SecurityGroup):
#     def __init__(self, name, parent, message_dispatcher):
#         super().__init__(name, parent, EodTradePricer, message_dispatcher)
#
#     def onSoD(self, date, df_mkt, horizon):
#         for stk in self.stk2sec:
#             self.stk2sec[stk].onSoD(df_mkt, horizon)

# class FwdPriceGroup(SecurityGroup):
#     def __init__(self, name, parent, message_dispatcher):
#         super().__init__(name, parent, FwdPricer, message_dispatcher)
#
#     def onSoD(self, date, df_mkt, horizon):
#         for stk in self.stk2sec:
#             self.stk2sec[stk].onSoD(df_mkt, horizon)
#
