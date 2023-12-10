from .common import *
from collections import defaultdict as DD
from valkyrie.tools import printMkt, T0, to_cents

class SimHiddenOrderMatcher(PerSecurityHandler):
  def __init__(self, group_mgr, stk, config):
    super().__init__(group_mgr, stk, config)
    self.exch2buy_order, self.exch2sell_order = DD(lambda: None), DD(lambda: None)
    back_of_line = 1.0
    self.back_line_adj = 1e-4 * back_of_line
    self.onSoD(None)

  def onSoD(self, date):
    self.last_update = T0
    self.bp, self.ap, self.tv, self.risk_adj = np.nan, np.nan, np.nan, np.nan
    self.az, self.bz, self.sim_az, self.sim_bz = np.nan, np.nan, np.nan, np.nan

  def onBidAsk(self, ts, stk, bp, bz, ap, az):
    if self.bp != bp:
      self.sim_bz = bz
    else: #delta_sz + last_sim_sz
      self.sim_bz = max(bz - self.bz + self.sim_bz, 0)

    if self.ap != ap:
      self.sim_az = az
    else: #delta_sz + last_sim_sz
      self.sim_az = max(az - self.az + self.sim_az, 0)

    self.bp, self.ap, self.bz, self.az = bp, ap, bz, az

    for exch in self.exch2buy_order:
      buy_order = self.exch2buy_order[exch]
      if buy_order and to_cents(buy_order.px) >= to_cents(self.ap):
        self._fillOrder(buy_order.px, self.sim_az, buy_order)

    for exch in self.exch2sell_order:
      sell_order = self.exch2sell_order[exch]
      if sell_order and to_cents(sell_order.px) <= to_cents(self.bp):
        self._fillOrder(sell_order.px, self.sim_bz, sell_order)

  def onTV(self, tv_update: TVUpdate):
    self.tv = tv_update.tv

  def onRiskAdj(self, risk_update: RiskUpdate):
    self.risk_adj = risk_update.adj

  def _fillOrder(self, px, sz, order):
    fill_sz = min(sz, order.sz)
    if not fill_sz > 0:
      return 0

    self.logger.info(f"{self.now()} mkt: {printMkt(self)} filling {fill_sz} for {order} tv = {self.tv:.3f} mkt_trade@{px}x{sz}")
    new_order, fill_response = order.onFill(fill_sz)

    e2o = self.exch2buy_order if order.side == Side.Buy else self.exch2sell_order
    e2o[order.exch] = new_order
    self.msg_dispatcher.onOrderUpdate(fill_response)
    return fill_sz

  def onMktTrade(self, ts, exch, stk, px, sz, cond):
    if exch not in self.exch2buy_order and exch not in self.exch2sell_order:
      return

    buy_order, sell_order = self.exch2buy_order[exch], self.exch2sell_order[exch]

    self.logger.info(f'MktTrade {self.now()} {stk} {exch} {px}@{sz} buy:{buy_order} sell_order:{sell_order}')

    if buy_order and buy_order.px >= (px + self.back_line_adj):
      self._fillOrder(px, sz, self.exch2buy_order[exch])
    elif sell_order and sell_order.px <= (px - self.back_line_adj):
      self._fillOrder(px, sz, self.exch2sell_order[exch])

  def onOrderInstr(self, order_instr: OrderInstr):
    self.last_update = self.group_mgr.now()
    ack_response = order_instr.onAck()
    self.msg_dispatcher.onOrderUpdate(ack_response)

    e2o = self.exch2buy_order if order_instr.side == Side.Buy else self.exch2sell_order

    # if order_instr.sz == 0 or not np.isfinite(order_instr.px):# cancel
    #   if e2o[order_instr.exch]
    #     e2o[order_instr.exch] = None
    # else:#replace
    e2o[order_instr.exch] = copy.copy(order_instr)

    #crossing if possible
    if order_instr.side == Side.Buy and to_cents(order_instr.px - self.ap) >= 0.0:
      px, sz = self.ap, self.sim_az
      fill_sz = self._fillOrder(px, sz, order_instr)
      self.sim_az -= fill_sz
    elif order_instr.side == Side.Sell and to_cents(order_instr.px - self.bp) <= 0.0:
      px, sz = self.bp, self.sim_bz
      fill_sz = self._fillOrder(px, sz, order_instr)
      self.sim_bz -= fill_sz
