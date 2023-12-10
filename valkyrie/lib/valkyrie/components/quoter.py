import gc

from valkyrie.components.common import *
from valkyrie.components.risk import SecurityRiskCalc as SRC
from valkyrie.tools import *

class Quoter(PerSecurityHandler):
  def __init__(self, group_mgr, stk, config):
    super().__init__(group_mgr, stk, config)

    self.parent = group_mgr
    self.enable = self.config['enable']
    self.order_id = int(self.config['stk_index'] * 1e6) + int(self.config['order_id_seed'])
    self.order_update = OrderState.Confirmed
    self.exchs = self.config['exchs'].copy()
    self.config['fill_prob_k'] = self.config['fill_prob_coef'] / self.config['avg_sprd']
    self.tactic = config['tactic']
    self.fee = self.group_mgr.fee_calc.calc_fee(self.tactic, 1)  # fee for 1 share

    self.buy_side, self.sell_side = QuoteSide(self, Side.Buy), QuoteSide(self, Side.Sell)
    self.fill2side = {Side.Buy: self.buy_side, Side.Sell: self.sell_side}

    self.bid_ask_update_counter = 0
    self.max_offset_in_cent = int(np.round(self.config['max_offset'] * 100))

    self.tv_update, self.risk_update = None, None
    self.prev_mkt_bid, self.prev_mkt_ask = np.nan, np.nan

  def __str__(self):
    return f'Quoter_{self.stk}'

  def getNextOrderId(self):
    self.order_id += self.parent.order_id_step
    return self.order_id

  def updateOrders(self):
    if not self.tv_update or not self.risk_update:
      return

    if not np.isfinite(self.tv_update.tv) or not np.isfinite(self.risk_update.adj):
      return

    bp, bz, ap, az = self.bp, self.bz, self.ap, self.az
    max_off = self.config['max_offset']

    max_bp = min(ap - 0.01, bp + max_off)
    min_ap = max(bp + 0.01, ap - max_off)

    buy_prices = 1e-2 * np.arange(to_cents(bp), to_cents(max_bp), 1)
    sell_prices = 1e-2 * np.arange(to_cents(ap), to_cents(min_ap), -1)

    if self.enable:
      self.buy_side.updateOrderByPrices(buy_prices)
      self.sell_side.updateOrderByPrices(sell_prices)

  def onBidAsk(self, ts, stk, bp, bz, ap, az):
    #if bp == self.prev_mkt_bid and ap == self.prev_mkt_ask:
    #  return

    self.prev_mkt_bid, self.prev_mkt_ask = bp, ap

    self.logger.debug(
      f'=== {self.group_mgr.now()} quoter {self.stk} : {self.bid_ask_update_counter} : {bz}x{bp}--{ap}x{az}')

    self.bid_ask_update_counter += 1
    self.bp, self.bz, self.ap, self.az = bp, bz, ap, az
    self.curr_sprd = self.ap - self.bp
    self.fill_prob_base = self.config['fill_prob_base'] * self.config['avg_sprd'] / self.curr_sprd
    self.updateOrders()

  def onRiskAdj(self, risk_update: RiskUpdate):
    self.risk_update = risk_update
    self.updateOrders()

  def onTV(self, tv_update: TVUpdate):
    self.logger.debug(f'{self.now()} {self.stk} tv_update : tv = {tv_update.tv}')
    self.tv_update = tv_update
    self.buy_side.onTv(tv_update)
    self.sell_side.onTv(tv_update)
    self.updateOrders()

  def onEoD(self, date):
    gc.collect(0)

  def _onFill(self, order_update: OrderUpdate):
    if order_update.tactic != self.tactic:
      return

    quote_side = self.fill2side[order_update.side]
    if order_update.exch not in quote_side.exch2order_instr:
      return

    exch = order_update.exch
    cur_order = quote_side.exch2order_instr[exch]

    if order_update.id != cur_order.id:
      raise Exception(f'Warning fill received for non-cur order {self.stk} : {order_update} -- {cur_order}')

    if cur_order.sz < order_update.sz:
      raise Exception(f'{self} overfill for {self.stk} : {cur_order}')

    self.logger.debug(f'{self.group_mgr.now()} : quoter {self.stk}'
                       f'filled {cur_order.side} {order_update.sz} @ {cur_order.px:.2f}'
                       f'mkt: {self.bp}x{self.ap} tv: {self.tv_update.tv:.3f} r_adj: {self.risk_update.adj:.3f}'
                       f'order:{cur_order} for sz={order_update.sz} ')

    quote_side.exch2order_instr[exch], _ = cur_order.onFill(order_update.sz)
    quote_side.exch2order_state[exch] = OrderState.Confirmed

  def _onAck(self, order_update: OrderUpdate):
    if order_update.tactic != self.tactic:
      return

    quote_side = self.fill2side[order_update.side]
    exch = order_update.exch
    if exch not in quote_side.exch2order_state:
      raise Exception(f'Receiving order ack for exch not known {order_update.exch}')

    if order_update.id != quote_side.exch2order_instr[exch].id:
      raise Exception(
        f'{self} received order ACK for not current order : {order_update} -- {quote_side.exch2order_instr[exch]}')

    if quote_side.exch2order_state[exch] != OrderState.Submitting:
      raise Exception(f'{self} received order ACK, but current order state is not submitted')

    quote_side.exch2order_state[exch] = OrderState.Confirmed

  def _onCancel(self, order_update):
    if order_update.tactic != self.tactic:
      return
    raise NotImplemented('Quote on Cacnel not implemented')
    # quote_side = self.fill2side[order_update.side]
    # if order_update.id != quote_side.cur_order.id:
    #     self.logger.debug(f'Warning cancel received for non-cur order {self.stk} : {order_update} -- {quote_side.cur_order}')
    #     return
    #
    # quote_side = self.fill2side[order_update.side]
    # if order_update.id != quote_side.cur_order.id:
    #     raise Exception(
    #         f'{self} received order Cancel for not current order : {order_update} -- {quote_side.cur_order}')
    #
    # quote_side.cur_order.px = np.nan
    # quote_side.cur_order.sz = 0
    # quote_side.order_state = OrderState.Confirmed

class QuoteSide:
  def __init__(self, parent: Quoter, side):
    self.parent = parent
    self.cfg = parent.config
    self.logger = parent.logger
    self.min_delta_order_sz = parent.config['min_delta_sz']
    self.msg_dispatcher = parent.msg_dispatcher
    self.order_interval_in_sec = pd.Timedelta(parent.config['order_interval_in_sec'], unit="sec")


    self.side = side
    self.delta_util = -np.inf
    self.counter = 0
    self.tactic = parent.tactic
    self.stk = parent.stk

    self.fee = parent.fee

    self.last_order_send_time = T0
    self.opt_px, self.opt_sz = np.nan, 0
    self.cur_px, self.cur_sz  = np.nan, 0
    self.exch2order_instr = {exch: None for exch in self.cfg['exchs']}
    self.exch2order_state = {exch: OrderState.Confirmed for exch in self.cfg['exchs']}
    self.tv, self.itv = np.nan, np.nan

  def onTv(self, tv_update):
    self.tv = tv_update.tv
    self.itv = 1.0 / self.tv

  def _calc_util(self, px, sz=None):
    edge = self.side * (self.tv - px)
    if not sz:
      sz = SRC.calc_opt_size(self.side, self.tv, edge, self.fee, self.parent.risk_update)
      sz = int(np.round(np.clip(sz, self.cfg['min_order_sz'], self.cfg['max_order_sz'])))

    nbbo_px = self.parent.bp if self.side == Side.Buy else self.parent.ap
    offset = self.side * (px - nbbo_px)

    fill_prob = self.parent.fill_prob_base + offset * self.cfg['fill_prob_k']
    fill_prob = np.clip(fill_prob, 0.0, 1.0)

    du = fill_prob * SRC.calc_delta_utility(self.side, self.tv, edge, sz, self.fee, self.parent.risk_update)
    # self.logger.debug(f'{self.parent.group_mgr.now()} calc util stk:{self.parent.stk} du:{du:3g} fill_prob:{fill_prob:2g} '
    #      f'{self.side}@{px:.2f} tv:{self.tv:.2f} '
    #      f'risk:{self.parent.risk.adj:.2f} sz:{sz} mkt:{self.parent.bp:.2f}x{self.parent.ap:.2f}')
    du = du * fill_prob
    return du, sz

  def _calc_optimal_orders(self, prices):
    # nulling order
    self.delta_util = 0.0
    self.opt_px, self.opt_sz = np.nan, 0

    # iterate for best via allowed px
    for px in prices:
      edge = self.side * (self.tv - px)
      edge_adj = SRC.calc_adj_edge(self.side, edge, self.fee, self.parent.risk_update)
      if edge_adj < self.cfg['min_edge']:
        continue

      du, sz = self._calc_util(px)
      if du > self.delta_util:
        self.opt_px, self.opt_sz = px, sz
        self.delta_util = du

  def _is_current_order_good(self, cur_px, prices):
    if not np.isfinite(cur_px):
      return True

    if len(prices) == 0:
      return False

    px_allowed = max(prices) if self.side == Side.Buy else min(prices)
    trade_sign = 1 if self.side == Side.Buy else -1
    if (trade_sign * to_cents(cur_px)) > trade_sign * to_cents(px_allowed) + 1: #current px too aggresive
      return False

    current_du, _ = self._calc_util(cur_px, self.opt_sz)
    return current_du > 0 #neg util

  def calc_order_update(self, prices):
    cur_px, cur_sz = self.cur_px, self.cur_sz
    is_cur_good = self._is_current_order_good(cur_px, prices)
    self._calc_optimal_orders(prices)

    def _is_big_change():
      return (np.isfinite(cur_px) ^ np.isfinite(self.opt_px)) \
        or to_cents(self.opt_px - cur_px) >= 1.0 \
        or np.abs(np.abs(cur_sz - self.opt_sz) > self.min_delta_order_sz)

    def _time_allowed():
      now, last_time, internval = self.parent.now(), self.last_order_send_time, self.order_interval_in_sec
      time_allowed = (now - last_time) > internval
      return time_allowed

    shall_update_order = (not is_cur_good) or (_time_allowed() and _is_big_change())
    if shall_update_order:
      xxxx = 4
    return shall_update_order

  def updateOrderByPrices(self, prices: list):
    self.counter += 1
    #if self.counter == 1407 and self.side == Side.Sell:
    #  print("a")

    test_stk_name = ''
    shall_update_order = self.calc_order_update(prices)
    parent = self.parent

    self.logger.debug(f'{parent.now()} {test_stk_name} update orders: '
                      f'mkt: {parent.bp}x{parent.ap} tv: {parent.tv_update.tv:.3f} r_adj: {parent.risk_update.adj:.3g} '
                      f'order: {self.opt_px} {self.opt_sz} {parent.order_id}')

    if not shall_update_order:
      return None

    self.cur_px, self.cur_sz = self.opt_px, self.opt_sz

    # quote_side.cur_order = order
    for exch in self.exch2order_instr:
      if self.exch2order_instr[exch] and self.exch2order_instr[exch].id == 32:
        x =4343

      if self.exch2order_instr[exch] and self.exch2order_state[exch] != OrderState.Confirmed:
        self.logger.critical(
          f'{self.parent.now()} Quoter: {self.stk} exch : {exch} order not confirm '
          f'for {self.stk} : {self.exch2order_instr[exch]}')
        continue

      order_id = parent.getNextOrderId()

      orderInstr = OrderInstr.create(self.tactic, self.opt_px, self.opt_sz, order_id, self.side, parent.stk, exch)

      self.exch2order_instr[exch] = orderInstr
      self.exch2order_state[exch] = OrderState.Submitting
      self.last_order_send_time = self.parent.now()

      self.msg_dispatcher.onOrderInstr(orderInstr)
      self.logger.debug(f'{self.parent.now()} Quoter: {parent.stk} exch: {exch} '
                       f'side_counter {self.counter} mkt: {parent.bp:.2f}x{parent.ap:.2f} '
                       f'order: {orderInstr} tv: {parent.tv_update.tv:.3f} '
                       f'r_adj: {parent.risk_update.adj:.3f} ')

