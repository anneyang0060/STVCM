import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import product
import multiprocessing as mp
import os
import warnings
warnings.filterwarnings("ignore")
import sys
# path='/data1/Prophet/sluo/projects/manual/order_dispatch/validation/temporal/src'
# path='/nfs/project/liuqi/yangying/validation/temporal/src'
path='C:/Users/annie/OneDrive - pku.edu.cn/projects/3. Finished/stvcm/Code+Data20210825/temporal/src'  
if path not in sys.path:
    sys.path.append(path)
from sklearn.model_selection import KFold
from model_new import VCM

### simulation settings ###
file = 'V2_hangzhou_serial_order_dispatch_AA.csv'
ycol = 'gmv'
xcols = ['cnt_call','sum_online_time']
scols = ['cnt_call_1','sum_online_time_1']
acol = 'is_exp'
regcols = ['const'] + xcols

df = pd.read_csv('C:/Users/annie/OneDrive - pku.edu.cn/projects/3. Finished/stvcm/Code+Data20210825/temporal/data/'+file)
df['const'] = 1
xycols = [ycol] + regcols +['date', 'time']
df = df[xycols]  

NN = 40
idx = [i+1 for i in range(NN)]

kf = KFold(n_splits=5, shuffle=True)

param_grid = [0.05*i for i in range(20)] * NN ** (-1/3)

K = 3; M =48

res = [] 

for train_index, test_index in kf.split(idx):
    df_train = df.loc[df['date'].isin(train_index)].set_index(['date','time'])
    df_test = df.loc[df['date'].isin(test_index)].set_index(['date','time'])
    for hc in param_grid:
        Amat = df_train.groupby('date').apply(lambda dt: np.dot(dt[regcols].T.values, dt[regcols].values)).sum()
        bvec = df_train.groupby('date').apply(lambda dt: np.dot(dt[regcols].T, dt[ycol])).sum()
        eps_diag = np.eye(Amat.shape[0])*1e-3
        theta = np.linalg.solve(Amat+eps_diag, bvec)
        theta = pd.DataFrame(theta.reshape((M, K)), columns=regcols)
        tmat = np.mat(np.reshape(np.repeat(np.arange(M)/(M-1), M), (M,M)))
        theta = smooth(theta.T, ker_mat((tmat.T-tmat),hc)).T
        df_test['fitted'] = df_test[regcols].dot(theta_DE.values.flatten())
        df_test['resid'] = df_test[ycol] - df_test['fitted']
        res.append(sum((df_test['resid'])**2))

res = np.array(res).reshape(5,20)
res = res.sum(axis=0)

np.array(param_grid)[np.where(np.min(res))]