import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import product
import multiprocessing as mp
from numpy import kron
import os
import warnings
warnings.filterwarnings("ignore")
import sys
path='C:/Users/annie/Spatio-temporal/src'  
if path not in sys.path:
    sys.path.append(path)
from model_st_new import VCM

### simulation settings ###
ycol = 'gmv'#,'cnt_grab', 'cnt_finish']
xcol = 'cnt_call'#,'sum_online_time']
scol = 'cnt_call_1'#the lag term
acol = 'is_exp'
acol_n = 'is_exp_n'
regcols = ['const'] + [xcol]
adj_mat = np.array([[0,1,1,1,0,0,0,0,0,0],
          [1,0,0,1,1,0,0,0,0,0],
          [1,0,0,1,0,1,0,0,0,0],
          [1,1,1,0,1,1,1,0,0,0],
          [0,1,0,1,0,0,1,1,0,0],
          [0,0,1,1,0,0,1,0,1,0],
          [0,0,0,1,1,1,0,1,1,1],
          [0,0,0,0,1,0,1,0,0,1],
          [0,0,0,0,0,1,1,0,0,1],
          [0,0,0,0,0,0,1,1,1,0]])
G = 10
adj_mat = adj_mat/np.repeat(adj_mat.sum(axis=0),G).reshape(G,G)
nsim = 400

two_sided = False
wild_bootstrap = False
interaction = False

DDS = [0.00, 0.005, 0.01]
IIS = [0.00, 0.005, 0.01]
IIS_n = [0.00, 0.005, 0.01]
NNs = [8,14,20]
TIs = [1,3,6]
designs = ['st','t']

wbi = 1 if wild_bootstrap else 0
tsi = 1 if two_sided else 0
ini = 1 if interaction else 0
hc = 0.01
hc_b = 0.01

for (design, TI, NN, DD, II, II_n) in product(designs, TIs, NNs, DDS, IIS, IIS_n):
    
    if II + II_n != DD:
        continue
    
    file = 'V1_CityF_pool.csv'
    df = pd.read_csv('C:/Users/annie/Spatio-temporal/data/'+file, index_col=['grid_id','date','time'])
    path = 'C:/Users/annie/Spatio-temporal/res/{}_{}_{}_{}_{}_{}_{}.npy'.format(design, file, NN, TI, DD, II, II_n)
    if os.path.exists(path):
        continue

    df['const'] = 1
    M = len(df.index.get_level_values(2).unique())
    N = len(df.index.get_level_values(1).unique())
    NM = M*N
    if IE:
        df[scol] = np.append(np.delete(df[xcol].values *(df.index.get_level_values(2)>0),0),0)
        xyscols = [ycol] + regcols + [scol]
        df = df[xyscols]
    else:
        xycols = [ycol] + regcols
        df = df[xycols]
    df[acol] = -1

    model0 = VCM(df, ycol, xcol, acol, scol,IE,
                         interaction=interaction,
                         two_sided=two_sided, 
                         wild_bootstrap=wild_bootstrap, 
                        center_x=True, scale_x=True,hc=hc)
    model0.estimate(null = True)
    df['fitted_DE'] = model0.holder['fitted_DE'].values
    df['eta_DE'] = model0.holder['eta_DE'].values; df['eps_DE'] = model0.holder['eps_DE'].values 
    df['fitted_IE'] = model0.holder['fitted_IE'].values
    df['eta_IE'] = model0.holder['eta_IE'].values; df['eps_IE'] = model0.holder['eps_IE'].values

    def generate(df, N, ycol, regcols, acol, ti=1, delta=0, delta_s=0, delta_s_n=0):
        grids = (df.index.get_level_values(0).unique())
        G = len(grids)
        dates = (df.index.get_level_values(1).unique())
        number_of_days = len(dates)
        M = len(df)// G // number_of_days

        dates_ = np.random.choice(dates, size=(N,), replace=True)
        df_ = df.loc[[(x,y,z) for x in grids for y in dates_ for z in range(M)],:]
        df_ = df_.reset_index()
        df_['date'] = np.tile(np.repeat(np.arange(N),M), G)
        df_.set_index(['grid_id','date','time'], inplace=True)

        mt = int(24/ti)
        if ti < 24: # intra-day time interval
            abv = np.tile(np.repeat([-1,1], M//mt), mt//2)
            bav = np.tile(np.repeat([1,-1], M//mt), mt//2)
            vec = np.hstack([abv, bav])
        elif ti == 24: # inter-day time interval
            av = -np.ones(M)
            bv = np.ones(M)
            vec = np.hstack([av, bv])
        gvs = np.array([])
        gv = np.tile(vec, N//2)
        if design == 'st':
            for i in range(G):
                gvs = np.append(gvs, np.random.choice([-1,1])*gv)
        else:
            for i in range(G):
                gvs = np.append(gvs, gv)
        df_[acol] = gvs
        df[acol_n] = np.dot(adj_mat, ((df[acol].values+1)/2).reshape(G,M*N)).ravel()

        if IE:
            idx1 = np.arange(df_.shape[0])[df_.index.get_level_values(2)>0]
            a=(df_['fitted_IE'] + df_['eps_IE'] * np.repeat(np.random.randn(N*G), M) + \
                                df_['eta_IE'] * np.repeat(np.random.randn(N*G), M)).values
            df_[xcol].iloc[idx1]=a[~np.isnan(a)]
            df_[xcol] *= (1+delta_s_n)
            df_.loc[df_[acol]==1, xcol] *= (1+delta_s)
            df_[scol] = np.append(np.delete(df_[xcol].values *(df_.index.get_level_values(2)>0),0),0)
            df_[scol][df_[scol]==0] = np.nan
        df_[ycol] = (df_['fitted_DE'] + df_['eps_DE'] * np.repeat(np.random.randn(N*G), M) + \
                                df_['eta_DE'] * np.repeat(np.random.randn(N*G), M)).values
        df_[ycol] *= (1+delta_s_n)
        df_.loc[df_[acol]==1, ycol] *= (1+delta+delta_s)

        return df_    

    def one_step(seed):
        
        np.random.seed(seed)
        ret = []
        
        df_ = generate(df, NN, ycol, regcols, acol, TI, DD, II, II_n) 
        model = VCM(df_, ycol, xcol, acol, acol_n, scol,IE,
                     interaction=interaction,
                     two_sided=two_sided, 
                     wild_bootstrap=wild_bootstrap, 
                    center_x=True, scale_x=True,hc=hc)
        model.inference()
        ret.append(model.holder['pvalue'])
            
        return ret
    
    IE = False
    pool = mp.Pool(20)
    rets0 = pool.map(one_step, range(nsim))
    pool.close()
    IE = True
    pool = mp.Pool(20)
    rets1 = pool.map(one_step, range(nsim))
    pool.close()
    rets=np.array([rets0,rets1])
    
    np.save(path, rets)