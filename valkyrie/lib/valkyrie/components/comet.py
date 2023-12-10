import gc

from valkyrie.components.common import *
from valkyrie.tools import T0
from valkyrie.quants.utils import to_cents
from valkyrie.components.risk import SecurityRiskCalc as SRC

class Comet(PerSecurityHandler):
  def __init__(self, group_mgr, stk, config):
    super().__init__(group_mgr, stk, config)
    self.parent = group_mgr
    self.enable = self.config['enable']
    self.min_size = self.config['min_order_sz']
    self.max_size = self.config['max_order_sz']
    self.min_edge = self.config['min_edge']
    self.max_spread_allowed = self.config['max_spread_allowed']
    self.order_id = int(self.config['stk_index'] * 1e6) + int(self.config['order_id_seed'])
    self.min_order_interval_sec = int(self.config['min_order_interval_sec'])
    self.side2order = {Side.Buy : None, Side.Sell : None}
    self.side2last_ordersubmit_time = {Side.Buy : T0, Side.Sell : T0}
    self.bid_ask_update_counter = 0
    self.sod_cooling = self.config['sod_cooling']
    self.tactic = config['tactic']
    self.fee = self.group_mgr.fee_calc.calc_fee(self.tactic, 1) #fee for 1 share

    self.tv_update, self.risk_update = None, None
    self.prev_mkt_bid, self.prev_mkt_ask = np.nan, np.nan
    self.counter = 0

  def getNextOrderId(self):
      self.order_id += self.parent.order_id_step
      return self.order_id

  def __str__(self):
    return f'Comet_{self.stk}'

  def cancelOrders(self):
    for side, order_instr in self.side2order.items():
      if not order_instr:
        continue

      order_cancel = OrderInstr.create(self.tactic, np.nan, 0, order_instr.id, side, self.stk, 'valkyrie')

      self.msg_dispatcher.onOrderInstr(order_cancel)
      self.side2order[side] = None

  def _updateOrder(self):
    self.counter += 1
    if not self.enable or self.now() < self.trading_start_time:
      return

    if not self.tv_update or not self.risk_update:
      return

    if not np.isfinite(self.tv_update.tv) or not np.isfinite(self.risk_update.adj):
      return

    if to_cents((self.ap - self.bp) - self.max_spread_allowed) > 0.0:
      return

    #which side?
    bs = (self.tv_update.tv + self.risk_update.adj) > self.ap
    side = Side.Buy if bs else Side.Sell

    px = self.ap if bs else self.bp
    mkt_sz = self.az if bs else self.bz

    #opt size
    tv = self.tv_update.tv
    edge = side * (tv - px)
    opt_sz = SRC.calc_opt_size(side, tv, edge, self.fee, self.risk_update)
    opt_sz = min(int(np.round(opt_sz)), mkt_sz, self.max_size)
    edge_adj = SRC.calc_adj_edge(side, edge, self.fee, self.risk_update)

    if opt_sz < self.min_size or edge_adj < self.min_edge or opt_sz < mkt_sz :
      return

    last_ts = self.side2last_ordersubmit_time[side]
    if (self.now() - last_ts).total_seconds() < self.min_order_interval_sec:
      return

    order_id = self.getNextOrderId()
    order = OrderInstr.create(self.tactic, px, opt_sz, order_id, side, self.stk, 'valkyrie')

    self.side2order[side] = order
    self.side2last_ordersubmit_time[side] = self.now()
    self.logger.debug(f'{self.parent.now()} Comet: {self.stk} '
                      f'side_counter {self.counter} mkt: {self.bp:.2f}x{self.ap:.2f} '
                      f'order: {order} tv: {self.tv_update.tv:.3f} '
                      f'r_adj: {self.risk_update.adj:.3f} ')

    self.msg_dispatcher.onOrderInstr(order)

  @overrides
  def onTimer(self, now : pd.Timestamp):
    self.cancelOrders()

  def onBidAsk(self, ts, stk, bp, bz, ap, az):
    self.bid_ask_update_counter += 1
    px_update = self.bp != bp or self.ap != ap
    self.bp, self.bz, self.ap, self.az = bp, bz, ap, az

    self._updateOrder()

  def onRiskAdj(self, risk_update: RiskUpdate):
    self.risk_update = risk_update
    self._updateOrder()

  def onTV(self, tv_update: TVUpdate):
    self.tv_update = tv_update
    self.logger.debug(f'{self.now()} Comet onTV: {self.stk} '
                      f'r_adj: {self.tv_update.tv:.3f} ')
    self._updateOrder()

  def onSoD(self, date):
    self.bp, self.bz, self.ap, self.az = np.nan, np.nan, np.nan, np.nan
    self.tv_update = None
    self.side2order = {Side.Buy: None, Side.Sell: None}
    self.side2last_ordersubmit_time = {Side.Buy: T0, Side.Sell: T0}
    self.trading_start_time \
      = pd.Timestamp(date) + pd.Timedelta("09:30:00") + pd.Timedelta(self.sod_cooling)

  def onEoD(self, date):
    gc.collect(0)
    self.cancelOrders()

  def _onFill(self, order_update: OrderUpdate):
    if order_update.tactic != self.tactic:
      return

    cur_order = self.side2order[order_update.side]

    if not cur_order or cur_order.id != order_update.id:
      raise Exception(f'Comet received fill order id {order_update.id} for unknown order')

    if cur_order.sz < order_update.sz:
      raise Exception(f'{self} overfill for {self.stk} : {cur_order}')

    self.logger.debug(f'{self.group_mgr.now()} : {self.stk} '
                      f'filled {cur_order.side} {order_update.sz} @ {cur_order.px:.2f}'
                      f'mkt: {self.bp}x{self.ap} tv: {self.tv_update.tv:.3f} '
                      f'r_adj: {self.risk_update.adj:.3f}'
                      f'order:{cur_order} for sz={order_update.sz} ')


    order, _ = cur_order.onFill(order_update.sz)
    self.side2order[order_update.side] = order

  def _onAck(self, order_update: OrderUpdate):
    return

  def _onCancel(self, order_update):
    if order_update.tactic != self.tactic:
      return
    if order_update.id == self.side2order[Side.Buy].id:
      self.side2order[Side.Buy] = None
    elif order_update.id == self.side2order[Side.Sell].id:
      self.side2order[Side.Sell] = None
    else:
      raise Exception(f'Comet received cancel order id {order_update.id} for unknown order')
