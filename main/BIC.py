import numpy as np
import pandas as pd
import tqdm as tqdm
from main.BCD_LS import *
from main.BCD_MLE import *
def BIC(y,p_m=2,r_m =None, s_m = None):
    N,T = y.shape
    
    P=int(np.floor(np.log(T)))
    if r_m is None:
        r_m = N
    if s_m is None:
        s_m = int(np.floor((N)/2))
    order = [[h for h in range(p_m+1)]]+[[i for i in range(r_m+1)]]+[[j for j in range(s_m+1)]]
    BIC_table_index =  pd.MultiIndex.from_product(order, names=['p','r','s'])
    LS_BIC_table = pd.DataFrame(columns=['BIC','initial'],index=BIC_table_index) # record BIC value and intial parameter corresponding to min loss
    #  p:0-r_m; r: 0-N; s:0-N//2
    ML_BIC_table = pd.DataFrame(columns=['BIC','initial'],index=BIC_table_index)
    for p in range(p_m+1): 
        for r in range(r_m+1): # max r_m
            for s in range(s_m+1): # max s_m
                if r+2*s > N: # r+2s must le N
                    continue
                if (p,r,s)==(0,0,0):
                    LS_log_loss = np.var(y,axis= 1).sum()
                    ML_log_loss = np.var(y,axis= 1).sum()
                else:
                    if (r,s)!=(0,0):
                        # grid_iterative in lmbd,gamma,phi
                        lmbd_range = [np.linspace(-0.5,0.5,3).round(3).tolist() for i in np.arange(r)] 
                        gamma_range = [np.linspace(0.2,0.8,3).round(3).tolist() for i in np.arange(s)]
                        phi_range = [np.linspace(0.5,np.pi-0.4*np.pi,3).round(3).tolist() for i in np.arange(s)]
                        alpha_range = lmbd_range+gamma_range+phi_range
                        index = pd.MultiIndex.from_product(alpha_range)
                        LS_result = pd.Series(index=index)
                        ML_result = pd.Series(index=index)
                        for alpha in index:
                            lmbd = alpha[:r]
                            gamma = alpha[r:r+s]
                            phi = alpha[r+s:r+2*s]
                            if ((len(lmbd) > len(set(lmbd))) or 
                            (len(gamma) > len(set(gamma))) or
                            (len(phi) > len(set(phi)))):
                                continue
                            
                            LS_result.loc[alpha] = np.log(np.linalg.det(BCD_LS(y,p,r,s,P,n_iter=20,lmbd=np.array(lmbd), gamma = np.array([gamma]), phi = np.array([phi]),stop_thres=1e-3,result_show=False)['Sigma']))
                            ML_result.loc[alpha] = np.log(np.linalg.det(BCD_MLE(y,p,r,s,P,n_iter=20,lmbd=np.array(lmbd), gamma = np.array([gamma]), phi = np.array([phi]),stop_thres=1e-3,result_show=False)['Sigma']))
                            
                        LS_log_loss = np.min(LS_result)
                        ML_log_loss = np.min(ML_result)
                        LS_initial_value = LS_result.idxmin()
                        ML_initial_value = ML_result.idxmin()
                        LS_BIC_table.loc[p,r,s]['initial'] = LS_initial_value
                        ML_BIC_table.loc[p,r,s]['initial'] = ML_initial_value
                    else:
                        LS_log_loss=np.log(np.linalg.det(BCD_LS(y,p,r,s,P,n_iter=20,lmbd=np.array([lmbd]), gamma = np.array([gamma]), phi = np.array([phi]),stop_thres=1e-2,result_show=False)['Sigma']))
                        ML_log_loss=np.log(np.linalg.det(BCD_MLE(y,p,r,s,P,n_iter=20,lmbd=np.array([lmbd]), gamma = np.array([gamma]), phi = np.array([phi]),stop_thres=1e-2,result_show=False)['Sigma']))
                d_m = r+2*s+(N**2)*(p+r+2*s) # No. of all params
                LS_BIC_table.loc[p,r,s]['BIC']= LS_log_loss + d_m*np.log(T)/T
                ML_BIC_table.loc[p,r,s]['BIC']= ML_log_loss + d_m*np.log(T)/T
    LS_min_index = LS_BIC_table['BIC'].dropna().astype(float).idxmin()           
    LS_initial_value = LS_BIC_table.loc[LS_min_index]
    
    ML_min_index = ML_BIC_table['BIC'].dropna().astype(float).idxmin()           
    ML_initial_value = ML_BIC_table.loc[ML_min_index]
    
    table_result = dict(zip(['LS_BIC_table', 'LS_min_index', 'LS_initial_value','ML_BIC_table', 'ML_min_index', 'ML_initial_value'],
                            [LS_BIC_table, LS_min_index, LS_initial_value,ML_BIC_table, ML_min_index, ML_initial_value]))
    return table_result
