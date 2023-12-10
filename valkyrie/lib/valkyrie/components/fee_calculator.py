import copy
class FeeCalculator:
  def __init__(self, config):
    self.base = config['fee_base']
    self.fee_adj = copy.deepcopy(config['fee_adj'])

  def calc_fee(self, tactic : str, sz : int):
    return (self.base + self.fee_adj[tactic]) * sz
