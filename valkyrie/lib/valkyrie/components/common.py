import datetime
import copy

from overrides import overrides
from abc import ABC, abstractmethod

import json

from valkyrie.components.definition import *
from valkyrie.components.fee_calculator import FeeCalculator
from valkyrie.securities import parent


class BaseEnvironment(ABC):

  def __init__(self, config_json):
    if type(config_json) == str:
      with open(config_json) as f:
        self.config = json.load(f)
    else:
      self.config = copy.copy(config_json)

    self.fee_calc = FeeCalculator(self.config['env']['fee'])

  @abstractmethod
  def getLogger(self):
    pass

  @abstractmethod
  def now(self):
    pass

  @abstractmethod
  def _setup(self):
    pass

  def get_fee_calculator(self):
    return self.fee_calc


class GenericComponent(ABC):
  def __init__(self):
    pass

  def onTimer(self, now : pd.Timestamp):
    pass

  def now(self):
    pass

  def onSoD(self, date):
    pass

  def onEoD(self, date):
    pass

  def onBidAsk(self, ts, stk, bp, bz, ap, az):
    pass

  def onMktTrade(self, ts, exch, stk, px, sz, cond):
    pass

  def onOrderInstr(self, order_instr):
    pass

  def onTV(self, tv_update):
    pass

  def onRiskAdj(self, risk_update):
    pass

  def onOrderUpdate(self, order_update : OrderUpdate):
    pass


class MessageLogger(GenericComponent):
  def __init__(self, verbose=True):
    super().__init__()
    self.verbose = verbose
    self.stk2tv, self.stk2risk, self.stk2order = {}, {}, {}

  @overrides
  def onBidAsk(self, ts, stk, bp, bz, ap, az):
    pass

  @overrides
  def onMktTrade(self, ts, exch, stk, px, sz, cond):
    pass

  @overrides
  def onOrderInstr(self, order_instr):
    self.stk2order[order_instr.stk] = order_instr
    # print(order_instr)

  @overrides
  def onTV(self, tv_update):
    self.stk2tv['tv_update'] = tv_update
    # print(tv_update)

  @overrides
  def onRiskAdj(self, risk_update):
    self.stk2risk[risk_update.stk] = risk_update
    # print(risk_update)

  def onOrderUpdate(self, fill_update):
    if self.verbose:
      print(fill_update)


class MessageDispatcher(GenericComponent):
  def __init__(self):
    #self.mkt_listeners = []
    self.tv_listeners = []
    self.risk_listeners = []
    self.order_listeners = []
    self.order_update_listeners = []

    self.order_update_counter = 0

  def add_tv_listeners(self, listeners):
    self.tv_listeners += listeners

  def add_risk_listeners(self, listeners):
    self.risk_listeners += listeners

  def add_order_instr_listeners(self, listeners):
    self.order_listeners += listeners

  def add_order_update_listeners(self, listeners):
    self.order_update_listeners += listeners



  @overrides
  def onMktTrade(self, ts, exch, stk, px, sz, cond):
    for c in self.mkt_listeners:
      c.onBidAsk(exch, stk, px, sz, cond)

  def onOrderInstr(self, order_instr):
    for e in self.order_listeners:
      e.onOrderInstr(order_instr)

  @overrides
  def onTV(self, tv_update):
    for e in self.tv_listeners:
      e.onTV(tv_update)

  @overrides
  def onRiskAdj(self, risk_update):
    for e in self.risk_listeners:
      e.onRiskAdj(risk_update)

  @overrides
  def onOrderUpdate(self, order_update: OrderUpdate):
    self.order_update_counter += 1

    for e in self.order_update_listeners:
      # print(id(order_update))
      e.onOrderUpdate(order_update)

class SecurityGroup(GenericComponent):
  mkt_open = datetime.time(9, 30, 0)
  mkt_close = datetime.time(16, 0, 0)

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    return f"SecirtyGroup [{self.name}]"

  def __init__(self, name: str,
               per_stk_handler: 'PerSecurityHandler',  # fwd type declaration
               env: BaseEnvironment,
               config : dict
               ):
    self.name, self.env, self.logger = name, env, env.getLogger()
    self.msg_dispatcher = env.msg_dispatcher
    self.data_mgr = env.data_mgr
    self.config = copy.deepcopy(config)
    self.fee_calc = env.get_fee_calculator()

    # with open(self.env.config['trading_config_fn']) as file:

    self.stk2sec = {}
    for i, instr in enumerate(self.env.config['universe']):
      params = config['default'].copy() if 'default' in config else {}
      params['stk_index'] = i
      if instr in config: #overrides
        params |= config[instr]

      self.stk2sec[instr] = per_stk_handler(self, instr, params)

    self.issuers = {parent(instr) for instr in self.stk2sec}
      #self.stk2params[ticker]['order_id_seed'] = int(i * 1e6)

  def now(self):
    return self.env.now()

  @overrides
  def onBidAsk(self, ts, stk, bp, bz, ap, az):
    if stk not in self.stk2sec:
      return

    if ts.time() < self.mkt_open or ts.time() > self.mkt_close:
      return

    if 0 < bp < ap and bz > 0 and az > 0:
      self.stk2sec[stk].onBidAsk(ts, stk, bp, bz, ap, az)
    else:
      pass
      # print(f'{self.now()} warning invalid market data {bz} {bp}-{ap} {az}')

  @overrides
  def onMktTrade(self, ts, exch, stk, px, sz, cond):
    if stk not in self.stk2sec:
      return

    if ts.time() < self.mkt_open or ts.time() > self.mkt_close:
      return

    if np.isfinite(px):
      self.stk2sec[stk].onMktTrade(ts, exch, stk, px, sz, cond)

  @overrides
  def onOrderInstr(self, order_instr):
    order_instr = order_instr
    self.stk2sec[order_instr.stk].onOrderInstr(order_instr)

  @overrides
  def onTV(self, tv_update):
    self.stk2sec[tv_update.stk].onTV(tv_update)

  @overrides
  def onRiskAdj(self, risk_update):
    risk_update = risk_update
    self.stk2sec[risk_update.stk].onRiskAdj(risk_update)

  @overrides
  def onOrderUpdate(self, order_update):
    stk = order_update.stk
    if order_update.response == OrderResponse.Fill:
      self.stk2sec[stk]._onFill(order_update)
    elif order_update.response == OrderResponse.Ack:
      self.stk2sec[stk]._onAck(order_update)
    elif order_update.response == OrderResponse.Cancel:
      self.stk2sec[stk]._onCancel(order_update)

    self.stk2sec[order_update.stk].onOrderUpdate(order_update)

  @overrides
  def onSoD(self, date):
    for stk in self.stk2sec:
      self.stk2sec[stk].onSoD(date)

  @overrides
  def onEoD(self, date):
    for stk in self.stk2sec:
      self.stk2sec[stk].onEoD(date)

  @overrides
  def onTimer(self, now : pd.Timestamp):
    for stk in self.stk2sec:
      self.stk2sec[stk].onTimer(now)


class PerSecurityHandler(GenericComponent):
  def __init__(self, group_mgr: SecurityGroup, stk: str, config: dict):
    super().__init__()
    self.group_mgr = group_mgr
    self.logger = group_mgr.logger
    self.msg_dispatcher = group_mgr.msg_dispatcher
    self.stk = stk

    self.config = config.copy()

  @overrides
  def now(self):
    return self.group_mgr.now()

  def _onFill(self, order_update):
    pass

  def _onAck(self, order_update):
    pass

  def _onCancel(self, order_update):
    pass

# DUMMY_SECURITY = PerSecurityHandler(None, None, "DUMMY", None)
