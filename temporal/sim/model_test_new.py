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
path='C:/Users/annie/temporal/src'  
if path not in sys.path:
    sys.path.append(path)
from model_new import VCM


### simulation settings ###
file = 'V2_cityA_serial_order_dispatch_AA.csv'
# file = 'V2_cityB_serial_order_dispatch_AA.csv'
ycol = 'gmv'
xcols = ['cnt_call','sum_online_time']
scols = ['cnt_call_lag','sum_online_time_lag']
acol = 'is_exp'
main_regcols = ['const'] + xcols
state_regcols = ['const'] + scols

nsim = 400

two_sided = False
wild_bootstrap = False

DDs = [0.0, 0.0025, 0.005, 0.0075, 0.01]
NNs = [8, 14, 20]
TIs = [1, 3, 6]

wbi = 1 if wild_bootstrap else 0
tsi = 1 if two_sided else 0
ini = 0
hc = 0.01


df = pd.read_csv('C:/Users/annie/temporal/data/'+file, index_col=['date','time'])
df['const'] = 1
df[acol] = -1
M = len(df.index.get_level_values(1).unique())
xyscols = [ycol] + main_regcols + scols + [acol]
df = df[xyscols]

model0 = VCM(df, ycol, xcols, acol, scols,
                     two_sided=two_sided, 
                     wild_bootstrap=wild_bootstrap, 
                    center_x=True, scale_x=True,hc=hc)
model0.estimate(null = True)

df['fitted_DE'] = model0.holder['fitted_DE'].values
df['eta_DE'] = model0.holder['eta_DE'].values; df['eps_DE'] = model0.holder['eps_DE'].values 
df['fitted_IE1'] = model0.holder['fitted_IE1'].values
df['eta_IE1'] = model0.holder['eta_IE1'].values; df['eps_IE1'] = model0.holder['eps_IE1'].values
df['fitted_IE2'] = model0.holder['fitted_IE2'].values
df['eta_IE2'] = model0.holder['eta_IE2'].values; df['eps_IE2'] = model0.holder['eps_IE2'].values 

def generate(df, ndays, ycol, main_regcols, state_regcols, acol, ti=1, delta=0):
    dates = (df.index.get_level_values(0).unique())
    number_of_days, nsamples_per_day = df[ycol].unstack().shape
    dates_ = np.random.choice(dates, size=(ndays,), replace=True)
    df_ = df.loc[[(x,y) for x in dates_ for y in range(1, nsamples_per_day+1)],:]
    df_ = df_.reset_index()
    df_['date'] = np.repeat(np.arange(ndays), nsamples_per_day)
    df_.set_index(['date','time'], inplace=True)

    mt = int(24/ti)
    if ti < 24: # intra-day time interval
        abv = np.tile(np.repeat([-1,1], nsamples_per_day//mt), mt//2)
        bav = np.tile(np.repeat([1,-1], nsamples_per_day//mt), mt//2)
        vec = np.hstack([abv, bav])
    elif ti == 24: # inter-day time interval
        av = -np.ones(nsamples_per_day)
        bv =  np.ones(nsamples_per_day)
        vec = np.hstack([av, bv])
    df_[acol] = np.tile(vec, ndays//2)
    df_[['eps_IE1', 'eps_IE2']] = df_[['eps_IE1', 'eps_IE2']]* np.repeat(np.random.randn(ndays), 2*M).reshape(ndays*M,2)
    df_[['eta_IE1', 'eta_IE2']] = df_[['eta_IE1', 'eta_IE2']]* np.repeat(np.random.randn(ndays), 2*M).reshape(ndays*M,2)
    for i in range(ndays):
        for j in range(2,M+1):
            df_.loc[(i,j),xcols] = df_.loc[(i,j-1),main_regcols].dot(model0.holder['theta_IE_0'][np.arange((j-1)*3, j*3),:]) + \
            df_.loc[(i,j),['eta_IE1', 'eta_IE2']].values + df_.loc[(i,j),['eps_IE1', 'eps_IE2']].values
    df_.loc[df_[acol]==1, xcols] *= (1+delta)
    # generate outcome via params learned under null       
    for i in range(ndays):
        for j in range(1,M+1):
            df_.loc[(i,j),'fitted_DE'] = sum(df_.loc[(i,j),main_regcols] * model0.holder['theta_DE_0'].loc[j-1,main_regcols])
    df_[ycol] = (df_['fitted_DE'] + df_['eps_DE'] * np.repeat(np.random.randn(ndays), M) + \
                            df_['eta_DE'] * np.repeat(np.random.randn(ndays), M)).values
    df_.loc[df_[acol]==1, ycol] *= (1+delta+delta)
    for i in range(len(scols)):
        df_[scols[i]] = np.append(0,np.delete(df_[xcols[i]].values, ndays*M-1))

    return df_

def one_step(seed):

    np.random.seed(seed)

    ret = []

    df_ = generate(df, NN, ycol, main_regcols, state_regcols, acol, TI, DD)

    model = VCM(df_, ycol, xcols, acol, scols,
                 two_sided=two_sided, 
                 wild_bootstrap=wild_bootstrap, 
                center_x=True, scale_x=True,hc=hc)
    model.inference()
    ret.append([model.holder['pvalue_DE'], model.holder['pvalue_IE']])

    return ret

df_ = generate(df, 20, ycol, main_regcols, state_regcols, acol, 1,0.05)

for (TI, NN, DD) in product(TIs, NNs, DDs): 
    
    rets = list(map(one_step, range(nsim)))

    name = 'C:/Users/annie/temporal/res/cityA_{}_{}_{}_{}.npy'.format( ycol, TI, NN, DD)

    np.save(name, rets)