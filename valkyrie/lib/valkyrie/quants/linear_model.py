import sklearn.linear_model
from sklearn import linear_model as LM
from sklearn.linear_model import LinearRegression as LR

from valkyrie.securities import *
from valkyrie.quants.utils import add_clip2col

def wcorr(df, xcols: list, ycols: list, wcols: list = None):
  wcoeff = {}
  if wcols and len(ycols) != len(wcols):
    raise Exception("len of y and w must be equal")

  if not wcols:
    W = np.ones((df.shape[0], len(ycols)))
  else:
    W = df[wcols].values

  from statsmodels.stats.weightstats import DescrStatsW
  for i, y in enumerate(ycols):
    X = df[xcols + [y]].values
    dsw = DescrStatsW(X, weights=W[:, i])
    wcoeff[y] = dsw.corrcoef[:-1, -1]
  return pd.DataFrame(wcoeff, index=xcols)


class WinsorizedLM:
  def __init__(self, quantile, linear_model, **args):
    self.lm = linear_model(**args)
    self.quantile = quantile
    self.args = args

  def fit(self, X, Y, W = None):
    if W is None:
      W = np.ones(Y.shape)
    X, Y, W = X.copy(), Y.reshape(X.shape[0], 1).copy(), W.copy()
    self.xlows, self.xhighs = np.quantile(X, self.quantile, axis=0), np.quantile(X, 1 - self.quantile, axis=0)
    add_clip2col(X, self.xlows, self.xhighs)
    self.ylows, self.yhighs = np.quantile(Y, self.quantile, axis=0), np.quantile(Y, 1 - self.quantile, axis=0)
    add_clip2col(Y, self.ylows, self.yhighs)
    self.lm.fit(X, Y, W)

  def predict(self, X):
    X = X.copy()
    add_clip2col(X, self.xlows, self.xhighs)
    Y = self.lm.predict(X)
    add_clip2col(Y, self.ylows, self.yhighs)
    return Y.reshape(-1.1)

  def score(self, X, Y, W = None):
    if W is None:
      W = np.ones(Y.shape)
    X, Y, W = X.copy(), Y.reshape(X.shape[0], -1).copy(), W.copy()
    add_clip2col(X, self.xlows, self.xhighs)
    add_clip2col(Y, self.ylows, self.yhighs)
    r2 = self.lm.score(X, Y, W)
    #Y_hat = self.predict(X)
    #Y_diff2 = (Y - Y_hat) * (Y - Y_hat)
    # Y_m = np.mean(Y, axis=0)
    # Y2 = (Y - Y_m) * (Y - Y_m)
    #r2m = 1 - np.sum(Y_diff2) / np.sum(Y * Y)
    # print(f'r2 : {r2}, r2m : {r2m}')
    return r2

  def __str__(self):
    s = '_'.join([f'{k[0:3]}_{v}' for k, v in self.args.items()])
    return str(self.lm).strip('()')[0:4] + s + f'_q={self.quantile}'



def lm_fit(df, xcols : list, ycols : list, wcols : list, quantile=0.03, fit_intercept=False):
  y2coeffs, r2s, wlms = {}, {}, {}

  for ycol, wcol in zip(ycols, wcols):
    X = df[xcols].values.copy()
    y = df[ycol].values.copy()
    w = df[wcol].values.copy()

    ym = np.sum(y * w) / np.sum(w)
    y = y - ym

    wlm = WinsorizedLM(quantile=quantile, linear_model=LR, fit_intercept=fit_intercept)
    wlm.fit(X, y, w)
    r2s[ycol] = wlm.score(X, y, w)
    wlms[ycol] = wlm
    coeffs = {c: wlm.lm.coef_[0][i] for i, c in
              enumerate(df[xcols])}  # | {'intercept' : wlm.lm.intercept_[0]}
    if fit_intercept:
      coeffs = coeffs | {'intercept': wlm.lm.intercept_[0]}
    y2coeffs[ycol] = coeffs

  wlms = pd.DataFrame(wlms, index=['model'])
  r2s = pd.DataFrame(r2s, index=['r2'])
  y2coeffs = pd.DataFrame(y2coeffs)
  return pd.concat([r2s, y2coeffs, wlms])

def params_from_pls(pls):
  # X -= self._x_mean
  # X /= self._x_std
  # Ypred = np.dot(X, self.coef_)
  x_coef = (1 / pls._x_std) * pls.coef_.reshape(-1)
  x_bias = (pls._x_mean / pls._x_std).dot(pls.coef_.reshape(-1))
  offset = -x_bias + pls._y_mean
  return {'coeff' : x_coef, 'offset' : offset}


def params_from_lm(lm):
  if type(lm) in (LM.LinearRegression, LM.Ridge, LM.Ridge, LM.ElasticNet):
    x_coef = lm.coef_.reshape(-1)
    offset = lm.intercept_
    return {'coeff' : x_coef, 'offset' : offset}


def linear_pred(X, x_coef, offset):
  return X.dot(x_coef.W) + offset

def analyze_features(df, exclude_features, xcols, ycols, wcols):
  res = {}
  res[f'all'] = lm_fit(df, xcols, ycols, wcols).loc['r2']
  for ef in exclude_features:
      xm_cols = [c for c in xcols if not c.endswith(ef)]
      c = f'frm {ef}'
      res[c] = lm_fit(df, xm_cols, ycols, wcols).loc['r2']
      res[c] = res[f'all'] - res[c] #r2 from removing ef
      res[ef] = lm_fit(df, [ef], ycols, wcols).loc['r2'] #r2 from ef alone
  res = pd.DataFrame(res)
  return res
