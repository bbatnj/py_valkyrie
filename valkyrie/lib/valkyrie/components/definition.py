from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class SlotsFreezer:
  '''Freeze any class such that instantiated
  objects become immutable. Also use __slots__ for speed.
  '''
  # __slots__ = []
  _frozen = False

  def __init__(self):
    # if self._frozen:
    #    return
    # generate __slots__ list dynamically
    # for attr_name in dir(self):
    # self.__slots__.append(attr_name)
    self._frozen = True

  def __delattr__(self, *args, **kwargs):
    if self._frozen:
      raise AttributeError('This object is frozen!')
    object.__delattr__(self, *args, **kwargs)

  def __setattr__(self, *args, **kwargs):
    if self._frozen:
      raise AttributeError('This object is frozen!')
    object.__setattr__(self, *args, **kwargs)



class Side(IntEnum):
  Sell = -1
  Invalid = 0
  Buy = 1

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    if self.value == self.Sell:
      return "Sell"
    elif self.value == self.Invalid:
      return "Invalid"
    elif self.value == self.Buy:
      return "Buy"
    raise Exception(f'Unknown Side {self.value}')


class OrderResponse(IntEnum):
  Ack = 0
  Fill = 1
  Cancel = 2
  Invalid = 3


class OrderState(IntEnum):
  Confirmed = 0
  Submitting = 1

@dataclass
class TimerEvent(SlotsFreezer):
  def __init__(self, now : pd.Timestamp):
    self.now = now
    super().__init__()

@dataclass
class RiskUpdate(SlotsFreezer):
  adj: float
  u: float
  stk: str

  def __init__(self, adj, u, stk):
    self.adj, self.u, self.stk = adj, u, stk
    super().__init__()


@dataclass
class TVUpdate(SlotsFreezer):
  tv: float
  risk_tv: float
  stk: str

  def __init__(self, tv, risk_tv, stk):
    self.tv, self.risk_tv, self.stk = tv, risk_tv, stk
    super().__init__()


@dataclass
class OrderInstr(SlotsFreezer):
  tactic: str
  px: float
  sz: int
  id: int
  side: int
  stk: str
  exch: str

  @staticmethod
  def create(tactic, px, sz, id, side, stk, exch):
    return OrderInstr(tactic, px, sz, id, side, stk, exch)
  def __init__(self, tactic, px, sz, id, side, stk, exch):
    self.tactic, self.px, self.sz, self.id, self.side, self.stk, self.exch \
      = tactic, px, sz, id, side, stk, exch
    super().__init__()

  def onFill(self, fill_sz, px = np.nan):
    px = self.px if not np.isfinite(px) else px
    order = OrderInstr.create(self.tactic, px, self.sz - fill_sz, self.id, self.side, self.stk, self.exch)
    fill_response = OrderUpdate(self.tactic, self.id, OrderResponse.Fill, self.stk, self.side, self.px, fill_sz, self.exch)
    return order, fill_response

  def onAck(self):
    ack_response = OrderUpdate(self.tactic, self.id, OrderResponse.Ack, self.stk, self.side, self.px, self.sz, self.exch)
    return ack_response

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    return f'OrderInstr[id:{self.id} stk:{self.stk} side:{self.side} px:{self.px:.2f} sz:{self.sz} ]'


@dataclass
class OrderUpdate(SlotsFreezer):
  tactic: str
  id: int
  response: OrderResponse
  stk: str
  side: Side
  px: float
  sz: int
  exch: str

  def __init__(self, tactic, id, response, stk, side, px, sz, exch):
    self.tactic, self.id, self.response, self.stk, self.side, self.px, self.sz, self.exch \
      = tactic, id, response, stk, side, px, sz, exch
    super().__init__()

  @staticmethod
  def from_order_instr(self, response, px, oi: OrderInstr):
    return OrderUpdate(oi.tactic, oi.id, response, oi.stk, oi.side, px, oi.sz, oi.exch)
