import gc
import datetime
from datetime import datetime
import copy
from itertools import product as cartesian_product

import numpy as np
import pandas as pd

from overrides import overrides
from d2l import torch as d2l

from sklearn.linear_model import LinearRegression as LR
from sklearn.cross_decomposition import PLSRegression as PLSR
from statsmodels.stats.weightstats import DescrStatsW
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.max_rows', 500)

#################################
# Torch
#################################
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.nn import LeakyReLU as LReLU

#############################
#valkyrie common
#############################
from valkyrie.securities import stocks_good_dvd, parent
from valkyrie.quants.linear import lm_fit, wcorr, analyze_features
#from valkyrie.quants.feature_analyzer import FeatureMgr
from valkyrie.tools import *
from valkyrie.components.trade_summary import GroupSimViewer

#############################
#valkyrie Machine learning
#############################
import valkyrie.ml.data as ml_data
import valkyrie.ml.modules as ml_modules
import valkyrie.ml.utils as ml_utils
from valkyrie.ml.data import mrf_linear_fit
from valkyrie.ml.modules import NetEvaluator
from valkyrie.ml.utils import tensor, HyperParameters
from valkyrie.ml.data import Df2T2, TrainValDataSet
from valkyrie.ml.net import LrNet, AlexRegNet
from valkyrie.ml.data import DataModule

#############################
#valkyrie Nibelungen
#############################
from valkyrie.nibelungen.data import DataMgr
