### 导入模块
import datetime
from datetime import datetime
from math import sqrt
import numpy as np
import pandas as pd
from numpy import concatenate
import folium
import lightgbm as lgb
from folium.plugins import HeatMap
from keras.models import Sequential
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import norm, skew 
import plotly_express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
sns.set(rc={'figure.figsize':(25,15)})

import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为50
pd.set_option('max_colwidth',50)
