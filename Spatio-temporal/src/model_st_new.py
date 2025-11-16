import numpy as np
from numpy import kron
import pandas as pd
import statsmodels.api as sm

def ker(t, h):
    return np.exp(-t**2/h**2)

def ker_mat(mat,h):
    n, m = mat.shape
    res = np.zeros((n, m))    
    for i in range(n):
        for j in range(m):
            res[i, j] = ker(mat[i, j],h)            
    return res

def smooth(df, kmat):
    index, columns = df.index, df.columns
    dfs = np.dot(df.fillna(0.), kmat) / kmat.sum(axis=0)
    dfs = pd.DataFrame(dfs, index=index, columns=columns)
    isnan = df.isnull()
    dfs[isnan] = np.nan
    return dfs

def compute_tilde_cov(wts1,wts2,V,G,M,pr):
    wts1 = np.repeat(wts1,G*M*pr)
    wts2 = np.tile(np.repeat(wts2,M*pr),G)
    wts = wts1*wts2
    V['wts'] = wts
    res = V.groupby(['grid1','grid2']).apply(lambda dt:np.array(dt.iloc[:,0:(M*pr)])*dt['wts'].unique()).sum()
    return res

class VCM(object):
    def __init__(self, df, ycol, xcol, acol,scol,IE, interaction=False, two_sided=True, wild_bootstrap=False, copy=True,
                 center_x=True, scale_x=True, center_y=False, scale_y=False, # not used this time
                 keep_ratio=0.95, he=4, smooth_coef=False, hc=0.01, hc_b=0.01): # extra args for future model update
        '''
        y_i(t) = alpha(t) + (x_i(t)-x_bar(t))^T beta(t) + a_i(t)[alpha'(t) + (x_i(t)-x_bar(t))^T beta'(t)]
                 + eta_i(t) + varepsilon_i(t) if interaction = True
               = alpha(t) + (x_i(t)-x_bar(t))^T beta(t) + a_i(t)gamma(t)
                 + eta_i(t) + varepsilon_i(t) if interaction = False
        
        df: pandas DataFrame, multiindex: ('date','time'), 
                              columns: [ycol='gmv', xcols=['demand','supply'], acol='A']
                              
        ycol: str, column name for response of interest
        xcols: list, can be None or [], column names for covariates
        acol: str, column name for treatment indicator
        
        interaction: boolean, whether to include treatment and covariates interactions
        two_sided:   boolean, one sided or two sided test
        wild_bootstrap: boolean, wild_bootstrap based statistical inference
        copy:        whether to copy df, df will be modified
        
        keep_ratio:  functional PCA of cov_eta, keep ratio
        he:          smoothing bandwidth of eta_i(t), relative to 1
        smooth_coef: boolean, whether to smooth varying coeffs
        hc:          smoothing bandwidth of varying coeffs, relative to 1
        '''
        self.df = df.copy() if copy else df
        self.df['const'] = 1
        
        if xcol is None: xcol = ''
        # grids
        self.grids = self.df.index.get_level_values(0).unique()
        # dates
        self.dates = self.df.index.get_level_values(1).unique()
        # times
        self.times = self.df.index.get_level_values(2).unique()
        # number of grids
        self.G = len(self.grids)
        # number of days
        self.N = len(self.dates)
        # number of obs for each day
        self.M = len(self.times)
        # number of obs for each grid
        self.NM = self.N * self.M
        # total number of obs
        self.GNM = self.df.shape[0]
        # num of features
        self.px = len([xcol])

        # Agnostic notes on Regression Adjustments to Experimental Data: Reexamining Freedman's Critique
#         self.mx = np.repeat(self.df[xcol].groupby('grid_id').mean(), self.NM)
#         self.df[xcol] = self.df[xcol].values - self.mx.values
#         self.sx = np.repeat(self.df[xcol].groupby('grid_id').std(), self.NM)
#         self.df[xcol] = self.df[xcol].values / self.sx.values
#         self.sy = np.repeat(self.df[ycol].groupby('grid_id').std(), self.NM).values
#         self.sy1 = self.df[ycol].groupby('grid_id').std().values
#         self.df[ycol] /= self.sy
#         self.ss = np.repeat(self.df[scol].groupby('grid_id').std(), self.NM).values
#         self.ss1 = self.df[scol].groupby('grid_id').std().values
#         self.df[scol] /= self.ss
        
        # regression column names
        self.regcols = ['const'] + [xcol] + [acol]
        self.regcols0 = ['const'] + [xcol] 
        
        # extend feature columns, TODO: make it more efficient
        self.regcols_  = [col + str(time) for time in self.times for col in self.regcols]
        self.regcols0_ = [col + str(time) for time in self.times for col in self.regcols0]
        self.xcols_  = [col + str(time) for time in self.times for col in [xcol]]
        self.df = self.df.reset_index()
        self.df = self.df.set_index(['date','time'])
        self.df_ = pd.DataFrame(0, index=self.df.index, columns=self.regcols_)
        for time in self.times:
            regcols_time = [col+str(time) for col in self.regcols]
            self.df_.loc[(slice(None),time), regcols_time] = self.df.loc[(slice(None),time), self.regcols].values
        self.df_[ycol] = self.df[ycol].values
        self.df_['grid_id'] = self.df['grid_id'].values
        self.df_[scol] = self.df[scol].values
        self.IE = IE
        if self.IE:
            self.df_b = self.df_.copy()
            
        # main args
        self.ycol = ycol
        self.xcol = xcol
        self.acol = acol
        self.scol = scol
        self.interaction = interaction
        self.two_sided = two_sided
        self.wild_bootstrap = wild_bootstrap
        self.copy = copy
        # not used args
        self.keep_ratio = keep_ratio
        self.he = he
        self.hc = hc
        self.hc_b=hc_b
        self.smooth_coef = smooth_coef
        # number of regressors
        self.pr = len(self.regcols)
        self.null, self.alternative = False, False # estimation done under H0 or H1
        self.holder = {} # result holder
        self.wts = np.ones(self.M)
        
    def estimate(self,df_=None, ycol=None, xcol=None, scol=None, IE=True, null=False, suffix='', eps=0.001, hc=None):
        if null: suffix = '_0'
        if ycol is None: ycol = self.ycol
        if scol is None: scol = self.scol
        if xcol is None: xcol = self.xcol
        if IE is True: IE = self.IE
        regcols_ = self.regcols0_ if null else self.regcols_ # extended columns
        regcols  = self.regcols0  if null else self.regcols
        xcols_, xcol, acol = self.xcols_, self.xcol, self.acol
        interaction = self.interaction
        df_ = self.df_ if df_ is None else self.df_b 
        wts = self.wts
        px, pr = self.px, self.pr
        N, M, K = self.N, self.M, len(regcols)
        times = self.times
        holder = self.holder
        grids = self.grids
        G = self.G
        M = self.M
        GNM = self.GNM

        hrz = [int(grids[i][7:10]) for i in range(len(grids))]
        vtc = [int(grids[i][11:14]) for i in range(len(grids))]
        hmat = np.mat( np.reshape( np.repeat( hrz, G), (G,G) ) )
        vmat = np.mat( np.reshape( np.repeat( vtc, G), (G,G) ) )

        if hc is None: hc = self.hc
        holder['hc'] = hc
        Kmat_grid = ker_mat(hmat - hmat.T, h=hc)*ker_mat(vmat - vmat.T, h=hc)
        Kmat_grid = np.array([Kmat_grid[:,i]/np.sum(Kmat_grid,axis=1)[i] for i in range(G)])
        tmat = np.mat(np.reshape(np.repeat(np.arange(M)/(M-1), M), (M,M)))
        
        # estimate theta
        theta_DE_pool = np.zeros(G*M*K).reshape(G,M*K)
        for g in range(G):
            df_sub = df_[df_['grid_id'] == grids[g]]
            df_sub = df_sub.fillna(0)
            Amat = df_sub.groupby('date').apply(lambda dt: np.dot(dt[regcols_].T.values, dt[regcols_].values)).sum()
            bvec = df_sub.groupby('date').apply(lambda dt: np.dot(dt[regcols_].T, dt[ycol])).sum()
            eps_diag = np.eye(Amat.shape[0])*1e-3
            theta_DE = np.linalg.solve(Amat+eps_diag, bvec).reshape(M,K)
            theta_DE = (np.dot(theta_DE.T, ker_mat((tmat.T-tmat),hc))/ker_mat((tmat.T-tmat),hc).sum(axis=0)).T
            theta_DE_pool[g,:] = theta_DE.flatten()
        theta_DE_pool0 = theta_DE_pool.copy()
        for g in range(G):
            theta_DE_pool[g,:] = (theta_DE_pool0.T * Kmat_grid[g,:]).T.sum(axis=0)
        holder['theta_DE'+suffix] = theta_DE_pool.copy()
            
        if suffix!='_b':
            df_['fitted_DE'] = 0 ;df_['resid_DE'] = 0 ; 
            df_['eta_DE'] = 0 ; df_['eps_DE'] = 0
            for g in range(G):
                idx = np.arange((g*N*M),((g+1)*N*M))          
                df_['fitted_DE'].iloc[idx] = df_[regcols_].iloc[idx].dot(theta_DE_pool[g,:])
            df_['resid_DE'] = df_[ycol] - df_['fitted_DE']
            df_['eta_DE'] = pd.DataFrame(df_[['resid_DE','grid_id']]).reset_index().set_index(['grid_id','date','time']).groupby(['grid_id','date']).apply(lambda dt:
                            smooth(pd.DataFrame(dt).T, ker_mat((tmat.T-tmat),hc)).T).values.ravel()
            df_['eps_DE'] = df_['resid_DE'] - df_['eta_DE'] 
            holder['fitted_DE'] = df_['fitted_DE']
            holder['eta_DE'] = df_['eta_DE']
            holder['eps_DE'] = df_['eps_DE']
        
            if not null and not IE:    
                Amat = df_.loc[df_['grid_id'] == grids[g]].groupby('date').apply(lambda dt: np.dot(dt[regcols_].T.values, dt[regcols_].values)).sum()
                eps_diag = np.eye(Amat.shape[0])*1e-3
                Ai = np.linalg.pinv(Amat+eps_diag)
                W = df_['eta_DE'+suffix].unstack().fillna(0).groupby('date').apply(lambda dt: np.outer(dt, dt)).sum()/(N-pr)
                V = df_['resid_DE'+suffix].unstack().var(axis=0)
                W = W + np.diag(V)
                Wi = 1/W
                Wi = pd.DataFrame(Wi, index=times, columns=times)
                Cmat = df_.groupby('date').apply(lambda dt: np.dot(np.dot(dt[regcols_].T.values, Wi[dt.index.get_level_values(1)].values), dt[regcols_].values)).sum()
                if g>0:
                    Ai_pool = np.hstack((Ai_pool, Ai))
                    Cmat_pool = np.hstack((Cmat_pool,Cmat))
                else:
                    Ai_pool = Ai
                    Cmat_pool = Cmat
                    
        if not null and not IE:   
            theta_hat_cov = np.zeros((G*M*pr)**2).reshape(G*M*pr,G*M*pr)
            for i in range(G):
                idx_left = np.arange(i*M,(i+1)*M)
                idx_left_regcols = np.arange(i*M*N,(i+1)*M*N)
                idx_left_Ai = np.arange(i*M*pr,(i+1)*M*pr)
                for j in range(G):
                    idx_right = np.arange(j*M,(j+1)*M)
                    idx_right_regcols = np.arange(j*M*N,(j+1)*M*N)
                    idx_right_Ai = np.arange(j*M*pr,(j+1)*M*pr)
                    theta_hat_cov[min(idx_left_Ai):max(idx_left_Ai+1),:][:,min(idx_right_Ai):max(idx_right_Ai+1)] = \
                               np.dot(np.dot(Ai_pool[:,idx_left_Ai],Cmat_pool[idx_left_Ai,idx_right_Ai]), Ai_pool[:,idx_right_Ai].T)
            theta_hat_cov_df = np.array([theta_hat_cov[(i*M*pr):((i+1)*M*pr),(j*M*pr):((j+1)*M*pr)]\
                                         for i in range(G) for j in range(G)]).reshape(G*G*M*pr,M*pr)

            index1 = np.repeat(np.arange(G),G*M*pr)
            index2 = np.tile(np.repeat(np.arange(G),M*pr),G)
            theta_hat_cov_df = pd.DataFrame(theta_hat_cov_df)
            theta_hat_cov_df['grid1'] = index1
            theta_hat_cov_df['grid2'] = index2
            theta_hat_cov_df = theta_hat_cov_df.set_index(['grid1','grid2'])

            theta_tilde_cov = np.zeros((G*M*pr)**2).reshape(G*M*pr,G*M*pr)
            theta_tilde_cov0 = np.zeros((G*M*pr)**2).reshape(G*M*pr,G*M*pr)
            for i in range(G):
                wts1 = Kmat_grid[:,i]
                for j in range(G):
                    wts2 = Kmat_grid[:,j]
                    tmp = compute_tilde_cov(wts1,wts2,theta_hat_cov_df,G,M,pr)
                    theta_tilde_cov[(i*M*pr):((i+1)*M*pr),:][:,(j*M*pr):((j+1)*M*pr)] = tmp
                    theta_tilde_cov0[(i*M*pr):((i+1)*M*pr),:][:,(j*M*pr):((j+1)*M*pr)] = tmp
                
            idx1 = np.array([px + 1 + pr * np.arange(M*g,M*(g+1)) for g in range(G)]).ravel()
            idx2 = np.array([px + 2 + pr * np.arange(M*g,M*(g+1)) for g in range(G)]).ravel()
            idx = np.hastack([idx1,idx2])
            gamma = theta_DE_pool.ravel()[idx]
            gamma_tilde_cov = theta_tilde_cov0[idx,:][:,idx]
            wts = np.ones(len(idx))
            test_stats = gamma.sum()/np.dot(np.dot(wts,gamma_tilde_cov),wts)**0.5

            holder['test_stats'+suffix] = test_stats
            holder['theta_tilde'+suffix] = theta_DE_pool.reshape(G,M*pr)
            holder['theta_tilde_cov'+suffix] = theta_tilde_cov         
            holder[ycol] = df_[ycol]
            holder['gamma'] = gamma
            holder['gamma_tilde_cov'] = gamma_tilde_cov
        
        if IE:
            theta_IE_pool = np.zeros(G*M*K).reshape(G,M*K)
            for g in range(G):
                df_sub = df_[df_['grid_id'] == grids[g]]
                df_sub = df_sub[df_sub.index.get_level_values(1)<47]
                df_sub = df_sub.fillna(0)
                Amat = df_sub.groupby('date').apply(lambda dt: np.dot(dt[regcols_].T.values, dt[regcols_].values)).sum()
                bmat = df_sub.groupby('date').apply(lambda dt: np.dot(dt[regcols_].T, dt[scol])).sum()
                theta_IE = np.linalg.solve(Amat+eps_diag, bmat).reshape(M,K)
                theta_IE = (np.dot(theta_IE.T, ker_mat((tmat.T-tmat),hc))/ker_mat((tmat.T-tmat),hc).sum(axis=0)).T
                theta_IE_pool[g,:] = theta_IE.flatten()
            theta_IE_pool0 = theta_IE_pool.copy()
            for g in range(G):
                theta_IE_pool[g,:] = (theta_IE_pool0.T * Kmat_grid[g,:]).T.sum(axis=0)
            holder['theta_IE'+suffix] = theta_IE_pool.copy()
            if suffix!='_b':             
                df_['fitted_IE'] = 0 ;df_['resid_IE'] = 0 ; 
                df_['eta_IE'] = 0 ; df_['eps_IE'] = 0
                for g in range(G):
                    idx = np.arange((g*N*M),((g+1)*N*M))          
                    df_['fitted_IE'].iloc[idx] = df_[regcols_].iloc[idx].dot(theta_IE_pool[g,:])
                df_['resid_IE'] = df_[scol] - df_['fitted_IE']
                df_['eta_IE'] = pd.DataFrame(df_[['resid_IE','grid_id']]).reset_index().set_index(['grid_id','date','time']).groupby(['grid_id','date']).apply(lambda dt:
                                smooth(pd.DataFrame(dt).T, ker_mat((tmat.T-tmat),hc)).T).values.ravel()
                df_['eps_IE'] = df_['resid_IE'] - df_['eta_IE'] 
                holder['fitted_IE'] = df_['fitted_IE']
                holder['eta_IE'] = df_['eta_IE']
                holder['eps_IE'] = df_['eps_IE'] 
            
            if not null:
#                 idx1 = 1+pr*np.arange(M)
#                 test_stat = (theta_IE_pool[:,idx1+1]).sum()
                IE = np.zeros(G)
                DE = np.zeros(G)
                for g in range(G):
                    idx1 = 1+pr*np.arange(M)
                    beta = theta_DE_pool[g,idx1]
                    gamma = theta_DE_pool[g,idx1+1]
                    DE[g] = gamma.sum()
                    alpha = np.zeros(M)
                    sigma_IE = theta_IE_pool[g,idx1]
                    v_IE = theta_IE_pool[g,idx1+1]
                    for t in np.arange(1,M):
                        for j in np.arange(t-1):
                            s = 1
                            for i in np.arange(t-1-j,t-1):
                                s = s * sigma_IE[i]
                            alpha[t] = alpha[t] + np.dot(v_IE[t-1-j,],s)
                    IE[g] = (beta*alpha).sum()
                test_stat = IE.sum()   
                DE_est = DE.sum()
                holder['test_stat_IE'+suffix] = test_stat
                holder['DE_est'+suffix] = DE_est
            
#         holder['resid'+suffix] = df_['resid'+suffix] * self.sy
#         holder['pre'+suffix] = df_['fittedvalues'+suffix] * self.sy
        
        if null: self.null = True
        else: self.alternative = True

        
    def inference(self, nb=100, eps=0.001):
        if not self.alternative: # solve under alternative
            self.estimate(null=False)
        
        holder = self.holder
        df_ = self.df_
        G, N, M, NM, GNM = self.G, self.N, self.M, self.NM, self.GNM        
        two_sided = self.two_sided
        wild_bootstrap = self.wild_bootstrap
        scol, ycol, xcol, xcols_ = self.scol, self.ycol, self.xcol, self.xcols_
        regcols, regcols_ = self.regcols, self.regcols_
        IE = self.IE
        
        if IE:
            test_stat = holder['test_stat_IE']
            df = self.df
            test_stats_wb = np.zeros(nb)
            for idx in range(nb):
                # generate data via params learned under null
                df1 = df.copy()
                df1 = df1.reset_index().set_index(['grid_id','date','time'])
                idx1 = np.arange(df1.shape[0])[df1.index.get_level_values(2)>0]
                a=(df['fitted_IE'] + df['eps_IE'] * np.repeat(np.random.randn(N*G), M) + \
                                    df['eta_IE'] * np.repeat(np.random.randn(N*G), M)).values
                df1[xcol].iloc[idx1]=a[~np.isnan(a)]
                df1[scol] = np.append(np.delete(df1[xcol].values *(df1.index.get_level_values(2)>0),0),0)
                df1[scol][df1[scol]==0] = np.nan
                df1[ycol] = (df['fitted_DE'] + df['eps_DE'] * np.repeat(np.random.randn(N*G), M) + \
                                        df['eta_DE'] * np.repeat(np.random.randn(N*G), M)).values
                # Agnostic notes on Regression Adjustments to Experimental Data: Reexamining Freedman's Critique
#                 mx1 = np.repeat(df1[xcol].groupby('grid_id').mean(), NM)
#                 df1[xcol] = df1[xcol].values - mx1.values
#                 sx1 = np.repeat(df1[xcol].groupby('grid_id').std(), NM)
#                 df1[xcol] = df1[xcol].values / sx1.values
#                 sy1 = np.repeat(df1[ycol].groupby('grid_id').std(), NM).values
#                 sy11 = df1[ycol].groupby('grid_id').std().values
#                 df1[ycol] /= sy1
#                 ss1 = np.repeat(df1[scol].groupby('grid_id').std(), NM).values
#                 ss11 = df1[scol].groupby('grid_id').std().values
#                 df1[scol] /= ss1
                df1 = df1.reset_index().set_index(['date','time'])
                self.df_b = pd.DataFrame(0, index=df1.index, columns=regcols_)
                for time in self.times:
                    regcols_time = [col+str(time) for col in regcols]
                    self.df_b.loc[(slice(None),time), regcols_time] = df1.loc[(slice(None),time), regcols].values
                self.df_b[ycol] = df1[ycol].values
                self.df_b[scol] = df1[scol].values
                self.df_b['grid_id'] = df1['grid_id'].values
                # estimate under alternative
                self.estimate(df_=self.df_b,null=False, suffix='_b',eps=eps, hc = self.hc_b)
                test_stats_wb[idx] = holder['test_stat_IE_b'] - holder['test_stat_IE']

#             pvalue1 = (test_stats_wb > test_stat).mean()
#             pvalue2 = min(1.0, 2.0*(test_stats_wb > abs(test_stat)).mean())
            holder['test_stat'] = test_stat
            holder['test_stats_wb'] = test_stats_wb
        else:
            test_stat = holder['test_stat_DE']
            from scipy.stats import norm
            pvalue1 = 1 - norm.cdf(test_stats)
            pvalue2 = 2 - 2*norm.cdf(abs(test_stats))
            holder['test_stat'] = test_stat

        pvalue = pvalue2 if two_sided else pvalue1
        
        holder['pvalue1'] = pvalue1
        holder['pvalue2'] = pvalue2
