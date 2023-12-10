from itertools import product as cartesianproduct
import copy

from overrides import overrides
import pandas as pd
import numpy as np

from valkyrie.components.common import RiskUpdate, BaseEnvironment
from valkyrie.components.definition import TVUpdate
from valkyrie.avalon.base_pricer import BasePricer, BasePricerGroup
from valkyrie.securities import parent
from valkyrie.quants.time_average import Kalman1D, EMS
from valkyrie.quants.utils import big_rel_diff
from valkyrie.tools import to_cents

class Eve(BasePricer):
  def __init__(self, group_mgr, stk, config):
    super().__init__(group_mgr, stk, config)
    self.volatility = config['volatility']
    self.nm2cywa = {nm : Kalman1D(mult, 0.01 ** 2 / (25 * 25), self.volatility) for nm, mult in config['wema_mults'].items()} #current yield weighted avg
    self.cywa_mkt = self.nm2cywa['mkt']
    self.trade_hls = copy.deepcopy(config['trade_hls'])
    self.nm2buytrade = {nm : EMS(hl, 0.0) for nm, hl in self.trade_hls.items()}
    self.nm2selltrade = {nm: EMS(hl, 0.0) for nm, hl in self.trade_hls.items()}
    self.nm2midtrade = {nm: EMS(hl, 0.0) for nm, hl in self.trade_hls.items()}
    self.features = config['features']
    self.pred_weight = config['pred_weight']
    self.bp, self.bz, self.ap, self.az = np.nan, np.nan, np.nan, np.nan
    self.last_sent_tv = None

    self.bs2trades = {
      1  : self.nm2buytrade,
      0  : self.nm2midtrade,
      -1 : self.nm2selltrade,
    }

  def calc_per_symbol(self):
    bp, bz, ap, az = self.bp, self.bz, self.ap, self.az
    rf = self.record_fields

    accrued_dvd = self.get_accured_dvd()
    annual_dvd  = self.calc_annual_dvd()

    sprd = (ap - bp)
    clean_px = (bp * az + ap * bz) / (az + bz) - accrued_dvd
    # when bid = 0, ask = 25, fair = 24.9, dvd = 14; clean px is negative
    cy = max(annual_dvd / clean_px, 0.0)
    s2 = (sprd * sprd) / max((clean_px * clean_px), (sprd * sprd))

    rf['clean_px'] = clean_px
    rf['accrued_dvd'] = accrued_dvd
    rf['annual_dvd'] = annual_dvd
    rf['s2'], rf['cy'] = s2, cy

    t = self.now()
    for nm in self.nm2cywa:
      self.nm2cywa[nm].onData(t, cy, s2)
      self._record_cywa(nm)

    for nm, hl in self.trade_hls.items():
      rf[f'buytrade_{nm}'] = self.nm2buytrade[nm].onTime(t)
      rf[f'selltrade_{nm}'] = self.nm2selltrade[nm].onTime(t)
      rf[f'midtrade_{nm}'] = self.nm2midtrade[nm].onTime(t)

      rf[f'buytrade_{nm}'] = self.nm2buytrade[nm].getData()
      rf[f'selltrade_{nm}'] = self.nm2selltrade[nm].getData()
      rf[f'midtrade_{nm}'] = self.nm2midtrade[nm].getData()

  def _record_cywa(self, nm):
    xs = self.record_fields
    xs[f'cywa_{nm}_xh'] = self.nm2cywa[nm].x_h
    xs[f'cywa_{nm}_s2h'] = self.nm2cywa[nm].s2_h
    #xs[f'cywa_{nm}_s2'] = self.nm2cywa[nm].s2
    #xs[f'cywa_{nm}_tv'] = self.cy2px(self.nm2cywa[nm].x_h)

    try:
      pff = self.pff
      xs[f'pff_cywa_{nm}_xh'] = pff.nm2cywa[nm].x_h
      xs[f'pff_cywa_{nm}_s2h'] = pff.nm2cywa[nm].s2_h
      #xs[f'pff_cywa_{nm}_s2'] = pff.nm2cywa[nm].s2
      #xs[f'pff_cywa_{nm}_tv'] = pff.cy2px(pff.nm2cywa[nm].x_h)
    except:
      pass

  def calc_combined(self):
    xs = self.record_fields
 
    xs['intercept'] = 1.0

    #Cluster
    for nm in self.nm2cywa:
      if self.sibs:
        s2 = [(sib.nm2cywa[nm].s2 + sib.nm2cywa['mkt'].s2) for sib in self.sibs]
        fn = f'cluster_{nm}_avg_s2'
        xs[fn] = np.sum(s2)
        weights = [1.0 / s for s in s2]
        fn = f'cluster_{nm}_avg_xh'
        xs[fn] = np.average([sib.nm2cywa[nm].x_h - sib.nm2cywa['mkt'].x_h for sib in self.sibs], weights = weights)
      else:
        xs[f'cluster_{nm}_avg_xh'] = 0.0
        xs[f'cluster_{nm}_avg_s2'] = np.nan

    for nm in self.trade_hls:
      if self.sibs:
        xs[f'cluster_buy_{nm}'] = np.sum([sib.nm2buytrade[nm].x_h for sib in self.sibs])
        xs[f'cluster_sell_{nm}']= np.sum([sib.nm2selltrade[nm].x_h for sib in self.sibs])
        xs[f'cluster_mid_{nm}'] = np.sum([sib.nm2midtrade[nm].x_h for sib in self.sibs])

    #Feature calculation
    xs['feature_cluster'] = xs[f'cluster_long_avg_xh'] - (xs['cywa_long_xh'] - xs['cywa_mkt_xh'])
    xs['feature_long_dpff'] = xs[f'pff_cywa_long_xh'] - xs[f'cywa_long_xh']
    xs['feature_mkt_dpff'] = xs[f'pff_cywa_mkt_xh'] - xs[f'cywa_mkt_xh']

  @overrides
  def _pricerSod(self, date: pd.Timestamp):
    gm, family = self.group_mgr, parent(self.stk)
    self.sibs = [ gm.stk2sec[stk] for stk in gm.stk2sec if parent(stk) == family and stk != self.stk]
    self.pff = self.group_mgr.stk2sec['PFF']

    #for hl in self.cywas:
    #  self.cywas[hl].set_time(date + pd.Timedelta("09:30:00")) #fast forward time directly, may have to add decay

  @overrides
  def onBidAsk(self, ts, stk, bp, bz, ap, az):
    self.bp, self.bz, self.ap, self.az = bp, bz, ap, az
    #self._calc_features()

  def publish_tv(self):
    #coefs = self.config['coeffs']

    rf = self.record_fields

    cy_pred = 0.0
    for k in self.features:
         cy_pred += self.features[k] * rf[k]

    self.cy_pred = cy_pred * self.pred_weight

    risk_tv = self.cy2px(self.cywa_mkt.x_h)
    #tv = risk_tv
    tv = self.cy2px(self.cywa_mkt.x_h + self.cy_pred)
    #tv = self.cy2px(np.exp(self.cy_pred) * self.cywa_mkt.x_h) #X * b = log(y1/y) -> exp(X*b) & y = y1

    dtv = tv - risk_tv
    self.record_fields['cy_pred'] = cy_pred
    self.record_fields['cy_theo'] = self.cywa_mkt.x_h + self.cy_pred
    self.record_fields['risk_tv'] = risk_tv
    self.record_fields['risk_tv_clean'] = risk_tv - self.get_accured_dvd()
    self.record_fields['tv'] = tv
    self.record_fields['dtv'] = dtv

    self.tv_update = TVUpdate(tv, risk_tv, self.stk)

    prev, th = self.last_sent_tv, self.config['min_tv_diff']
    if (not prev) or big_rel_diff(prev.tv, tv, th) or big_rel_diff(prev.risk_tv, risk_tv, th):
      self.msg_dispatcher.onTV(self.tv_update)
      self.last_sent_tv = self.tv_update

  def onMktTrade(self, ts, exch, stk, px, sz, cond):
    t = self.now()
    mid = (self.bp + self.ap) / 2
    bs = np.sign(to_cents(px) - to_cents(mid))
    nm2trade = self.bs2trades[int(np.round(bs))]

    for nm, hl in self.trade_hls.items():
      nm2trade[nm].onData(t, sz)

class EveGroup(BasePricerGroup):
  def __init__(self, name: str, env: BaseEnvironment, config: dict):
    super().__init__(name, Eve, env, config)
    self.issuers = {parent(stk) for stk in self.stk2sec}

  @overrides
  def onTimer(self, now : pd.Timestamp):
    dt = (self.now() - self.last_update).total_seconds()

    if dt < self.sample_interval_s:
      return

    #self.last_update = self.now()

    for stk in self.stk2sec:
      self.stk2sec[stk].calc_per_symbol()

    for stk in self.stk2sec:
      self.stk2sec[stk].calc_combined()

    for stk in self.stk2sec:
      self.stk2sec[stk].publish_tv()

    for stk in self.stk2sec:
      self.stk2sec[stk].save_fields()

    #super().onTimer(now)

    # benchmark = 'PFF'
    # pff = self.stk2sec[benchmark]
    # pff_ema_hls = [hl for hl in pff.cywas]
    # for stk, hl in cartesianproduct(self.stk2sec, pff_ema_hls):
    #   eve = self.stk2sec[stk]
    #   eve.features[f'{benchmark}_cywa_{hl}'].append(pff.cywas[hl].x_h)