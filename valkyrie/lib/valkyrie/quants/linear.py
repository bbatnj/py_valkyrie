import sklearn.linear_model
from sklearn import linear_model as LM
from sklearn.linear_model import LinearRegression as LR

from valkyrie.securities import *
from valkyrie.quants.utils import add_clip2col

from sklearn.linear_model import SGDRegressor as SGDReg
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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


from sklearn.cross_decomposition import PLSRegression as PLS

class PCR:
  def __init__(self, k_size):
      self.pls = PLS(k_size)

  def fit(self, X, y, w = None):
      return self.pls.fit(X, y)

  def predict(self, X):
      return self.pls.predict(X).reshape(-1)

  def score(self, X, y, w = None):
      return self.pls.score(X, y)

  def get_params(self):
      # X -= self._x_mean
      # X /= self._x_std
      # Ypred = np.dot(X, self.coef_)
      pls = self.pls
      x_coef = (1 / pls._x_std) * pls.coef_.reshape(-1)
      x_bias = (pls._x_mean / pls._x_std).dot(pls.coef_.reshape(-1))
      offset = -x_bias + pls._y_mean
      return {'coeff': x_coef, 'offset': offset}


  class SGD:
    def __init__(self, alpha, l1_ratio):
        self.scaler = StandardScaler(with_mean=False)
        a, l1_ratio = float(alpha), float(l1_ratio)
        self.sgd_reg = SGDReg(loss='epsilon_insensitive',  # 'squared_error',
                              # epsilon = 1e-6,
                              penalty='elasticnet',  # None,
                              fit_intercept=False,
                              alpha=a,
                              l1_ratio=l1_ratio,
                              # tol = 1e-9,
                              # eta0 = 1e-11,
                              learning_rate='adaptive')
        self.pipeline = make_pipeline(self.scaler, self.sgd_reg)

    def fit(self, X, y, w=None):
        X_scaled = self.scaler.fit_transform(X, y)
        self.sgd_reg.fit(X_scaled, y, sample_weight = w)

    def predict(self, X):
        y = self.pipeline.predict(X)
        return y

    def score(self, X, y, w=None):
        res = self.pipeline.score(X, y, sample_weight = w)
        return res

    def get_params(self):
        return self.sgd_reg.coef_ / self.scaler.scale_

    def get_reg_params(self):
        return self.sgd_reg.coef_

    def get_scalers(self):
        return self.scaler.scale_

def calc_auto_corr(df, cols, lags):
    if type(cols) is str:
        cols = [cols]

    df_res = []

    for col in cols:
        for lag in lags:
            df[f'{col}_{lag}'] = df[col].shift(lag)
        df_corr = df[[c for c in df if col in c]].corr()
        df_corr.index = np.array([0.0] + lags)
        df_corr = df_corr.iloc[1:, :].copy()
        df_res.append(df_corr)
    return pd.concat(df_res, axis = 1)

def weighted_correlation(series1, series2, weights):
    if len(series1) != len(series2) or len(series1) != len(weights):
        raise ValueError("Input Series and weights must have the same length")

    weighted_mean1 = np.average(series1, weights=weights)
    weighted_mean2 = np.average(series2, weights=weights)

    weighted_cov = np.average((series1 - weighted_mean1) * (series2 - weighted_mean2), weights=weights)
    weighted_var1 = np.average((series1 - weighted_mean1)**2, weights=weights)
    weighted_var2 = np.average((series2 - weighted_mean2)**2, weights=weights)

    weighted_corr = weighted_cov / np.sqrt(weighted_var1 * weighted_var2)

    return weighted_corr


def calc_corr(df, first_cols, second_cols):
    correlations = []

    for col1 in first_cols:
        row = []
        for col2 in second_cols:
            correlation = df[col1].corr(df[col2])
            row.append(correlation)
        correlations.append(row)

    correlation_df = pd.DataFrame(correlations, index=first_cols, columns=second_cols)

    return correlation_df


def calculate_weighted_correlations(df, first_cols, second_cols, weight_col):
    correlations = []

    for col1 in first_cols:
        row = []
        for col2 in second_cols:
            correlation = np.corrcoef(df[col1], df[col2], aweights=df[weight_col])[0, 1]
            row.append(correlation)
        correlations.append(row)

    correlation_df = pd.DataFrame(correlations, index=first_cols, columns=second_cols)

    return correlation_df

#exp weighted beta of col_x vs col_y
def calc_exp_weighted_beta(df, col_x, col_y, beta_hl):
    xtx = df.eval(f'{col_x} * {col_x}').ewm(halflife=beta_hl).mean()
    xty = df.eval(f'{col_x} * {col_y}').ewm(halflife=beta_hl).mean()
    return xty / xtx
