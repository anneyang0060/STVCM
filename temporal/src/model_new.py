import numpy as np
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


class VCM(object):
    def __init__(self, df, ycol, xcols, acol, scols, two_sided=True, wild_bootstrap=False, copy=True,
                 center_x=True, scale_x=True, center_y=False, scale_y=False, # not used this time
                 keep_ratio=0.95, he=0.05, smooth_coef=False, hc=0.01): # extra args for future model update
        '''
        y_i(t) = alpha(t) + (x_i(t)-x_bar(t))^T beta(t) + a_i(t)gamma(t)
                 + eta_i(t) + varepsilon_i(t)
        
        df: pandas DataFrame, multiindex: ('date','time'), 
                              columns: [ycol='gmv', xcols=['demand','supply'], acol='A']
                              
        ycol: str, column name for response of interest
        xcols: list, can be None or [], column names for covariates
        acol: str, column name for treatment indicator
        
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
        
        # dates
        self.dates = self.df.index.get_level_values(0).unique()
        # times
        self.times = self.df.index.get_level_values(1).unique()
        # number of days
        self.N = len(self.dates)
        # number of obs for each day
        self.Ms = self.df[acol].groupby('date').count()
        # maximum number of obs for a given day
        self.M = self.Ms.max()
        # total number of obs
        self.NM = self.df.shape[0]    
        # num of features
        self.px = len(xcols)

        # regression column names
        self.main_regcols = ['const'] + xcols + [acol]
        self.main_regcols0 = ['const'] + xcols
        self.state_regcols = ['const'] + scols + [acol]
        self.state_regcols0 = ['const'] + scols
        
        # extend feature columns, TODO: make it more efficient
        self.main_regcols_  = [col + str(time) for time in self.times for col in self.main_regcols]
        self.main_regcols0_ = [col + str(time) for time in self.times for col in self.main_regcols0]
        self.df_main = pd.DataFrame(0, index=self.df.index, columns=self.main_regcols_)
        for time in self.times:
            regcols_time = [col+str(time) for col in self.main_regcols]
            self.df_main.loc[(slice(None),time), regcols_time] = self.df.loc[(slice(None),time), self.main_regcols].values
        self.df_main[ycol] = self.df[ycol].values
        self.state_regcols_  = [col + str(time) for time in self.times for col in self.state_regcols]
        self.state_regcols0_ = [col + str(time) for time in self.times for col in self.state_regcols0]
        self.df_state = pd.DataFrame(0, index=self.df.index, columns=self.state_regcols_)
        for time in self.times:
            regcols_time = [col+str(time) for col in self.state_regcols]
            self.df_state.loc[(slice(None),time), regcols_time] = self.df.loc[(slice(None),time), self.state_regcols].values
        self.df_state[xcols] = self.df[xcols]
        self.df_main_b, self.df_state_b = None, None
    
        # main args
        self.ycol = ycol
        self.scols = scols
        self.xcols = xcols
        self.acol = acol
        self.two_sided = two_sided
        self.wild_bootstrap = wild_bootstrap
        self.copy = copy
        # not used args
        self.keep_ratio = keep_ratio
        self.he = he
        self.smooth_coef = smooth_coef
        self.hc = hc
        # number of regressors
        self.pr = len(self.main_regcols)
        self.null, self.alternative = False, False # estimation done under H0 or H1
        self.holder = {} # result holder
        self.wts = np.ones(self.M)
        
    def estimate(self, df_main=None, df_state=None, ycol=None, xcols=None, scols=None, IE=True, null=False, suffix='', max_iter=20, eps=0.001, hc=0.01):
        if null: suffix = '_0'
        if ycol is None: ycol = self.ycol
        if scols is None: scols = self.scols
        if xcols is None: xcols = self.xcols
        acol = self.acol
        df_ = self.df.copy()
        main_regcols_ = self.main_regcols0_ if null else self.main_regcols_
        main_regcols = self.main_regcols0 if null else self.main_regcols
        df_main = self.df_main if df_main is None else self.df_main_b
        state_regcols_ = self.state_regcols0_ if null else self.state_regcols_
        state_regcols = self.state_regcols0 if null else self.state_regcols
        df_state = self.df_state if df_state is None else self.df_state_b
        wts = self.wts
        px, pr = self.px, self.pr
        N, M, K = self.N, self.M, len(main_regcols)
        times = self.times
        holder = self.holder

        # DE: estimate theta
        Amat = df_main.groupby('date').apply(lambda dt: np.dot(dt[main_regcols_].T.values, dt[main_regcols_].values)).sum()
        bvec = df_main.groupby('date').apply(lambda dt: np.dot(dt[main_regcols_].T, dt[ycol])).sum()
        eps_diag = np.eye(Amat.shape[0])*1e-3
        theta_DE = np.linalg.solve(Amat+eps_diag, bvec)
        theta_DE = pd.DataFrame(theta_DE.reshape((M, K)), columns=main_regcols)
        tmat = np.mat(np.reshape(np.repeat(np.arange(M)/(M-1), M), (M,M)))
        theta_DE = smooth(theta_DE.T, ker_mat((tmat.T-tmat),hc)).T
        holder['theta_DE'+suffix] = theta_DE.copy()
        df_['fitted_DE'] = df_main[main_regcols_].dot(theta_DE.values.flatten())
        df_['resid_DE'] = df_main[ycol] - df_['fitted_DE']
        df_['eta_DE'] = pd.DataFrame(df_['resid_DE']).groupby('date').apply(lambda dt:smooth(dt.T, ker_mat((tmat.T-tmat),hc)).T).values
        df_['eps_DE'] = df_['resid_DE'] - df_['eta_DE'] 
        if suffix == '_0':
            holder['fitted_DE'] = df_['fitted_DE']; holder['eta_DE'] = df_['eta_DE']; holder['eps_DE'] = df_['eps_DE']
        if suffix == '':
            holder['fitted_DE_1'] = df_['fitted_DE']; holder['resid_DE_1'] = df_['resid_DE']
        
        # IE: estimte coefficient
        Amat = df_state.groupby('date').apply(lambda dt: np.dot(dt[state_regcols_].T.values, dt[state_regcols_].values)).sum()
        bmat = df_state.groupby('date').apply(lambda dt: np.dot(dt[state_regcols_].T, dt[xcols])).sum()
        theta_IE = np.linalg.solve(Amat+eps_diag, bmat)
        theta_IE[:,0] = (np.dot(theta_IE[:,0].reshape(M,K).T, ker_mat((tmat.T-tmat),hc))/ker_mat((tmat.T-tmat),hc).sum(axis=0)).T.flatten()
        theta_IE[:,1] = (np.dot(theta_IE[:,1].reshape(M,K).T, ker_mat((tmat.T-tmat),hc))/ker_mat((tmat.T-tmat),hc).sum(axis=0)).T.flatten()
        holder['theta_IE'+suffix] = theta_IE.copy()
        df_['fitted_IE1'] = df_state[state_regcols_].dot(theta_IE[:,0])
        df_['resid_IE1'] = df_state[xcols[0]] - df_state[state_regcols_].dot(theta_IE[:,0])
        df_['eta_IE1'] = pd.DataFrame(df_['resid_IE1']).groupby('date').apply(lambda dt:smooth(dt.T, ker_mat((tmat.T-tmat),hc)).T).values
        df_['eps_IE1'] = df_['resid_IE1'] - df_['eta_IE1']
        df_['fitted_IE2'] = df_state[state_regcols_].dot(theta_IE[:,1])
        df_['resid_IE2'] = df_state[xcols[1]] - df_state[state_regcols_].dot(theta_IE[:,1])
        df_['eta_IE2'] = pd.DataFrame(df_['resid_IE2']).groupby('date').apply(lambda dt:smooth(dt.T, ker_mat((tmat.T-tmat),hc)).T).values
        df_['eps_IE2'] = df_['resid_IE2'] - df_['eta_IE2']
        if suffix=='_0':   
            holder['fitted_IE1'] = df_['fitted_IE1']
            holder['eta_IE1'] = df_['eta_IE1']
            holder['eps_IE1'] = df_['eps_IE1']
            holder['fitted_IE2'] = df_['fitted_IE2']
            holder['eta_IE2'] = df_['eta_IE2']
            holder['eps_IE2'] = df_['eps_IE2']
        if suffix=='':   
            holder['fitted_IE1_1'] = df_['fitted_IE1']
            holder['resid_IE1_1'] = df_['resid_IE1']
            holder['fitted_IE2_1'] = df_['fitted_IE2']
            holder['resid_IE2_1'] = df_['resid_IE2']
            
        if not null: # compute standard error
            W = df_['eta_DE'].unstack().fillna(0).groupby('date').apply(lambda dt: np.outer(dt, dt)).sum()/(N-K)
            V = df_['eps_DE'].unstack().var(axis=0)
            W = W + np.diag(V)
            Wi = 1/W
            Wi = pd.DataFrame(Wi, index=times, columns=times)

            eps_diag = np.eye(Amat.shape[0])*1e-3
            Ai = np.linalg.pinv(Amat+eps_diag)

            Cmat = df_main.groupby('date').apply(lambda dt: np.dot(np.dot(dt[main_regcols_].T.values, Wi[dt.index.get_level_values(1)].values), dt[main_regcols_].values)).sum()
            theta_cov_hat = np.dot(np.dot(Ai, Cmat), Ai)
            idx = px + 1 + K * np.arange(M)
            gamma_hat = theta_DE.values.flatten()[idx]
            gamma_cov_hat = theta_cov_hat[idx,:][:,idx]
            gamma = (wts * gamma_hat).sum()
            gamma_se = abs(np.dot(np.dot(wts, gamma_cov_hat), wts))**0.5
            test_stats = gamma / gamma_se
            holder['gamma'+suffix] = gamma
            holder['gamma_se'+suffix] = gamma_se
            holder['test_stat_DE'+suffix] = test_stats
            holder['theta_cov_hat'+suffix] = theta_cov_hat

        beta = theta_DE[xcols].values
        alpha = np.zeros(M*px).reshape(M,px)
        sigma_IE = np.zeros(px*px*M).reshape(M,px,px)
        idx1 = K*np.arange(M)
        sigma_IE[:,0,:] = theta_IE[idx1-1,:]
        sigma_IE[:,1,:] = theta_IE[idx1,:]
        v_IE = theta_IE[idx1,:]
        for t in np.arange(1,M):
            for j in np.arange(t-1):
                s = 1
                for i in np.arange(t-1-j,t-1):
                    s = s * sigma_IE[i,:,:]
                alpha[t,:] = alpha[t,:] + np.dot(v_IE[t-1-j,],s)
        IE = 0
        for t in np.arange(1,M):
            IE = IE + np.dot(beta[t,:],alpha[t,:].T)
        test_stat = IE                
        holder['test_stat_IE'+suffix] = test_stat

        if null: self.null = True
        else: self.alternative = True
        
    def inference(self, IE=True, nb=400, max_iter=20, eps=0.001):
        self.estimate(null=False)
        
        holder = self.holder
#         df, df_ = self.df, self.df_
        N, M, Ms, NM = self.N, self.M, self.Ms, self.NM
        two_sided = self.two_sided
        wild_bootstrap = self.wild_bootstrap
        xcols, scols, ycol = self.xcols, self.scols, self.ycol
        main_regcols, main_regcols_ = self.main_regcols, self.main_regcols_
        state_regcols, state_regcols_ = self.state_regcols, self.state_regcols_
        pr = len(main_regcols)
        df = self.df
        test_stats_wb = np.zeros(nb)
        # DE
        test_stat = holder['test_stat_DE']
        from scipy.stats import norm
        pvalue = 1 - norm.cdf(test_stat)
        holder['pvalue_DE'] = pvalue
        test_stat = holder['test_stat_IE']
   
        for idx in range(nb):
            df_c = df.copy()
            # generate state via params learned under null
            df_c[['eps_IE1', 'eps_IE2']] = df_c[['eps_IE1', 'eps_IE2']]* np.repeat(np.random.randn(N), 2*M).reshape(N*M,2)
            df_c[['eta_IE1', 'eta_IE2']] = df_c[['eta_IE1', 'eta_IE2']]* np.repeat(np.random.randn(N), 2*M).reshape(N*M,2)
            for i in range(N):
                for j in range(2,M+1):
                    df_c.loc[(i,j),xcols] = df_c.loc[(i,j-1),main_regcols].dot(holder['theta_IE'][np.arange((j-1)*pr, j*pr),:]) + \
                    df_c.loc[(i,j),['eta_IE1', 'eta_IE2']].values + df_c.loc[(i,j),['eps_IE1', 'eps_IE2']].values
            # generate outcome via params learned under null       
            for i in range(N):
                for j in range(1,M+1):
                    df_c.loc[(i,j),'fitted_DE'] = sum(df_c.loc[(i,j),main_regcols] * holder['theta_DE'].loc[j-1,main_regcols])
            df_c[ycol] = (df_c['fitted_DE'] + df['eps_DE'] * np.repeat(np.random.randn(N), M) + \
                                    df['eta_DE'] * np.repeat(np.random.randn(N), M)).values
            for i in range(len(scols)):
                df_c[scols[i]] = np.append(0,np.delete(df_c[xcols[i]].values, N*M-1))
            self.df_main_b = pd.DataFrame(0, index=df_c.index, columns=main_regcols_)
            self.df_state_b = pd.DataFrame(0, index=df_c.index, columns=state_regcols_)
            for time in self.times:
                regcols_time = [col+str(time) for col in main_regcols]
                self.df_main_b.loc[(slice(None),time), regcols_time] = df_c.loc[(slice(None),time), main_regcols].values
                regcols_time = [col+str(time) for col in state_regcols]
                self.df_state_b.loc[(slice(None),time), regcols_time] = df_c.loc[(slice(None),time), state_regcols].values
            self.df_main_b[ycol] = df_c[ycol]
            self.df_state_b[xcols] = df_c[xcols]
            # estimate under alternative
            self.estimate(df_main=self.df_main_b, df_state=self.df_state_b, null=False, suffix='_b', max_iter=max_iter, eps=eps, hc=self.hc)
            test_stats_wb[idx] = holder['test_stat_IE_b']-holder['test_stat_IE']

        # IE
        pvalue = 1-(test_stats_wb < test_stat).mean()
        holder['pvalue_IE'] = pvalue
