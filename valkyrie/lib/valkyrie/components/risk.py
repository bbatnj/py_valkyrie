import numpy as np

from valkyrie.components.definition import *
from valkyrie.components.common import *

from collections import namedtuple


class RiskFactor:
  def __init__(self, name, tol, cost):
    self.dpos = 0
    self.name, self.tol, self.cost = name, tol, cost

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return f'RiskFactor [{self.name}, tol : {self.tol}, cost : {self.cost}]'

  def set_param(self, tol, cost):
    self.tol, self.cost = tol, cost

  def onDPos(self, dpos):
    self.dpos += dpos

  def calc_adj(self):
    adj = -self.dpos * self.cost / self.tol
    # print(f'risk factor : adj : {adj:g} {self.name} dpos : {self.dpos:.2f} cost : {self.cost:.2f} tol : {self.tol:g}')
    return adj

  def calc_u(self):
    u = self.cost / self.tol
    return u


LoadedFactor = namedtuple('LoadedFactor', ['factor_ref', 'loading'])


class SecurityRiskCalc(PerSecurityHandler):
  @staticmethod
  def calc_adj_edge(side: Side, edge, fee, risk_update):
    risk_adj_edge = min(edge - fee, edge - fee + side * risk_update.adj)
    return risk_adj_edge
#   edge -= fee
#   if edge > 0:
#     return edge + side * risk_update.adj
#   else:
#     return min(edge, edge + side * risk_update.adj)

  @staticmethod
  def calc_opt_size(side: Side, tv, edge, fee, risk_update):
    risk_adj_edge = SecurityRiskCalc.calc_adj_edge(side, edge, fee, risk_update)
    if not risk_adj_edge > 0.0:
      return 0.0

    return risk_adj_edge / tv / risk_update.u

  @staticmethod
  def calc_delta_utility(side: Side, tv, edge, size, fee, risk_update: RiskUpdate):
    risk_adj_edge = SecurityRiskCalc.calc_adj_edge(side, edge, fee, risk_update)
    return risk_adj_edge * size * tv - 0.5 * risk_update.u * size * size * tv * tv

  def __init__(self, group_mgr, stk, config):
    super().__init__(group_mgr, stk, config)
    self.msg_dispatcher = group_mgr.msg_dispatcher
    self.stk = stk
    self.loaded_factors = []
    self.risk_tv = 0.0
    self.sz = 0.0
    self.counter = 0
    self.prev_risk_adj = np.nan

  def set_factors(self, factor, load):
    self.loaded_factors.append(LoadedFactor(factor, load))

  def onTV(self, tv_update: TVUpdate):
    dtv = tv_update.risk_tv - self.risk_tv
    self.risk_tv = tv_update.risk_tv
    for loaded_factor in self.loaded_factors:
      factor_ref, loading = loaded_factor.factor_ref, loaded_factor.loading
      factor_ref.onDPos(dtv * self.sz * loading)

  def onOrderUpdate(self, order_update):
    if self.risk_tv == 0.0 or not np.isfinite(self.risk_tv):
      raise f"{self} must have risk_tv before position update"

    sz = order_update.sz * order_update.side
    self.sz += sz
    self.counter += 1

    if self.counter == 1:
      x = 4

    # print(f'Fill : {self.stk} {self.counter} sz:{self.sz} fill:{sz}')
    for loaded_factor in self.loaded_factors:
      factor_ref, loading = loaded_factor.factor_ref, loaded_factor.loading
      factor_ref.onDPos(self.risk_tv * order_update.sz * order_update.side * loading)

  def calc_risk_params(self):
    adj, u = 0.0, 0.0
    # print(f'calc risks for stk {self.stk} ')
    for loaded_factor in self.loaded_factors:
      factor_ref, loading = loaded_factor.factor_ref, loaded_factor.loading
      adj += factor_ref.calc_adj()
      u += factor_ref.calc_u() * loading * loading
    return adj, u

  @staticmethod
  def opt_sz(edge, side, px, adj, u):
    if edge < 0.0:
      return 0.0

    adj = max(edge * adj, 0)
    dpos = (edge + adj) / u
    return dpos / px

class SecurityRiskGroup(SecurityGroup):

  @staticmethod
  def _rpl_sym_issuer(input, instr, issuer):
    return {k.replace('@SYMBOL@',instr).replace('@ISSUER@', issuer)
            : v
            for k, v in input.items()}

  def _load_risk_factors(self, config):
    risk_factors = {}

    if 'default_factors' in config:
      default_config = config['default_factors']

      default_idio_factor = default_config['IDIO_@SYMBOL@'].copy()
      for instr in self.stk2sec:
        risk_factors[f"IDIO_{instr}"] = self._rpl_sym_issuer(default_idio_factor, instr, '')

      default_issuer_factor = default_config["RES_@ISSUER@"].copy()
      for iss in self.issuers:
        risk_factors[f"RES_{iss}"] = self._rpl_sym_issuer(default_issuer_factor, '', iss)

    risk_factors |= config['factors']
    self.risk_factors = risk_factors

  def _load_risk_loadings(self, config):
    loadings = {}

    dl = 'default_loadings'
    if dl in config:
      default_loadings = config[dl].copy()

      for instr in self.stk2sec:
        iss = parent(instr)
        loadings[f"{instr}"] = self._rpl_sym_issuer(default_loadings, instr, iss)

    loadings |= config['loadings']
    self.risk_loadings = loadings

  def __init__(self, name : str, env : BaseEnvironment, config : dict):
    super().__init__(name, SecurityRiskCalc, env, config)

    self._load_risk_factors(config)
    risk_factors = self.risk_factors
    self.name2factors = {k: RiskFactor(k, v['tol'], v['cost']) for k, v in risk_factors.items()}

    self._load_risk_loadings(config)
    risk_loadings = self.risk_loadings

    # link risk factors to secs
    for stk, loadings in risk_loadings.items():
      if stk not in self.stk2sec:
        raise Exception(f'{stk} found in risk loading but not in sec fn ?')

      for factor_name, load in loadings.items():
        factor = self.name2factors[factor_name]
        self.stk2sec[stk].set_factors(factor, load)

    self.ready_stk = set()

  def _updateRiskParam(self):
    for stk in self.stk2sec:
      sec = self.stk2sec[stk]
      adj, u = sec.calc_risk_params()
      ru = RiskUpdate(adj, u, stk)

      prev_adj = sec.prev_risk_adj

      if np.abs(prev_adj - adj) < 0.005:
        continue

      sec.prev_risk_adj = adj
      self.msg_dispatcher.onRiskAdj(ru)

  def onOrderUpdate(self, order_update: OrderUpdate):
    if order_update.response == OrderResponse.Fill:
      stk = order_update.stk
      self.stk2sec[stk].onOrderUpdate(order_update)
      if len(self.ready_stk) == len(self.stk2sec):
        self._updateRiskParam()

  def onTV(self, tv_update :TVUpdate):
    if not np.isfinite(tv_update.risk_tv):
      raise Exception(f'{tv_update.stk} having non valid risk tv update')

    stk = self.stk2sec[tv_update.stk]
    dtv = tv_update.risk_tv - stk.risk_tv

    if dtv / stk.risk_tv < 0.005:
      return

    self.ready_stk.add(tv_update.stk)
    self.stk2sec[tv_update.stk].onTV(tv_update)
    if len(self.ready_stk) == len(self.stk2sec):
      self._updateRiskParam()

