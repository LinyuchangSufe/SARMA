import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time 
from tqdm import tqdm
import math
from multiprocessing import Pool
import matplotlib.pyplot as plt
from main.BCD_LS import *
from main.BCD_MLE import *
import main.help_function_for_LS as lsfunc
import main.help_function_for_MLE as mlfunc
from statsmodels.tsa.seasonal import seasonal_decompose
from main.tensorOp import * 
from main.IOLS_VARMA import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
'''
FEDFUNDS:  Effective Federal Funds Rate---d1
CPIAUCSL:  CPI : All Items (Consumer Price Index)--d2log
PPICMM:    PPI: Metals and metal products--d2log
M2SL:      M2 Money Stock--d2log
PCEPI:     Personal Cons. Expend.: Chain Index--d2log
GS1:       1-Year Treasury Rate---d1
'''


df = pd.read_csv("Application//current.csv",header=0, index_col=0)
df=df[['FEDFUNDS','CPIAUCSL','PPICMM','M2SL','PCEPI','GS1']]
df = df.drop(df.index[0])
df.index = pd.to_datetime(df.index)
df.head(5)
null_columns=df.isnull().any()

df['FEDFUNDS'] = df['FEDFUNDS'].diff()
df['CPIAUCSL'] = np.log(df['CPIAUCSL']).diff().diff()
df['PPICMM'] = np.log(df['PPICMM']).diff().diff()
df['M2SL'] = np.log(df['M2SL']).diff().diff()
df['PCEPI'] = np.log(df['PCEPI']).diff().diff()
df['GS1'] = df['GS1'].diff()
df.dropna(inplace=True)


"================================================================"

'''
CPIAUCSL:  CPI : All Items (Consumer Price Index)--d2log
UNRATE:    Civilian Unemployment Rate--d1
M2SL:      M2 Money Stock--d2log
'''
os.chdir(r'C:\Users\Administrator\LYC\SARMA\SARMA_code')
df = pd.read_csv("Application//current.csv",header=0, index_col=0)
df=df[['FEDFUNDS','CPIAUCSL','UNRATE','M2SL','S&P 500']]
df = df.drop(df.index[0])
df.index = pd.to_datetime(df.index)
df.head(5)
null_columns=df.isnull().any()

df['FEDFUNDS'] = df['FEDFUNDS'].diff()
df['CPIAUCSL'] = np.log(df['CPIAUCSL']).diff().diff()
df['UNRATE'] = np.log(df['UNRATE']).diff().diff()
df['M2SL'] = np.log(df['M2SL']).diff().diff()
df['S&P 500'] = np.log(df['S&P 500']).diff()
df.dropna(inplace=True)

"==================================== "
'''
'RPI' (Retail Price Index) 5
'INDPRO' (Industrial Production Index) 5
'UNRATE' (Civilian Unemployment Rate) 5 
'W875RX1' Real personal income ex transfer receipts
'CPIAUCSL' Consumer Price Index 6
'DPCERA3M086SBEA' Real personal consumption expenditures 5
'''

os.chdir(r'C:\Users\Administrator\LYC\SARMA\SARMA_code')
df = pd.read_csv("Application//current.csv",header=0, index_col=0)
df=df[['RPI','INDPRO','UNRATE','M2SL','CPIAUCSL','DPCERA3M086SBEA']]
df = df.drop(df.index[0])
df.index = pd.to_datetime(df.index)
df.head(5)
null_columns=df.isnull().any()

df = df[:-4]


df['RPI'] = df['RPI'].diff()
df['INDPRO'] = np.log(df['INDPRO']).diff()
df['UNRATE'] = np.log(df['UNRATE']).diff()
df['M2SL'] = np.log(df['M2SL']).diff().diff()
df['CPIAUCSL'] = np.log(df['CPIAUCSL']).diff().diff()
df['DPCERA3M086SBEA'] = np.log(df['DPCERA3M086SBEA']).diff()
df.dropna(inplace=True)


T,N = df.shape
df
# plot
fig, axes = plt.subplots(nrows=N, ncols=1, sharex=True, figsize=(20, 15))
for idx, col in enumerate(df.columns):
    axes[idx].plot(df[col],color='black',lw=0.6)
    axes[idx].set_ylabel(col,fontsize=14)
plt.show()

T,N = df.shape
df
# plot
fig, axes = plt.subplots(nrows=N, ncols=1, sharex=True, figsize=(20, 15))
for idx, col in enumerate(df.columns):
    axes[idx].plot(df[col],color='black',lw=0.6)
    axes[idx].set_ylabel(col,fontsize=14)
plt.show()

df_norm = (df-df.mean())/df.std()

ADF_value = np.zeros(N)
for i,col in enumerate(df_norm.columns):
    ADF_value[i] = adfuller(df_norm[col])[1] 

    
T,N = df_norm.shape
fig, axes = plt.subplots(nrows=N, ncols=1, sharex=True, figsize=(20, 15))
for idx, col in enumerate(df_norm.columns):
    axes[idx].plot(df_norm[col],color='black',lw=0.6)
    axes[idx].set_ylabel(col,fontsize=14)
plt.show()

## ACF PACF
fig , axes = plt.subplots(nrows=N, ncols=2, figsize= (12,8))
for i,col in enumerate(df_norm.columns):
    df_column = df_norm[col]
    plot_acf(df_column, ax=axes[i,0])
    plot_pacf(df_column, ax=axes[i,1])
plt.show()

df_norm.to_csv('turn_R_data.csv')
df = df_norm.values


MACM_y=MACM(df_norm.values,15)
lag=16
err_bound = 2 / np.sqrt(T)
xlabel_lags=np.arange(1,lag)

fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(20, 15), sharex='all', sharey='all')
for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        ax.vlines(xlabel_lags, [0], MACM_y[:,i,j] ,color='k',linewidth=0.7)
        ax.axhline(0, color='k',linewidth=0.7)
        ax.axhline(err_bound, color='r', linestyle='--',linewidth=0.7)
        ax.axhline(-err_bound, color='r', linestyle='--',linewidth=0.7)
        if i == 5:  # 如果是最下方的子图，设置x轴的标签和刻度
            ax.set_xlabel('x-label')
            # axs[i, j].set_xticks([-0.5, 0, 0.5])
        if j == 0:  # 如果是最左侧的子图，设置y轴的标签和刻度
            ax.set_ylabel('y-label')
            ax.set_yticks(np.linspace(-0.5,0.5,3))
        # for ax in axs.flat:
        #     ax.set(xlabel='x-label', ylabel='y-label')
        #     ax.label_outer()  # 仅显示最外围的标签
plt.tight_layout()
plt.savefig("MACM_realdate.png", bbox_inches='tight')
plt.show()


# BIC
df=df_norm.values.T
N,T = df.shape
p_m =2; r_m = 2; s_m = 1
BIC_reuslt = BIC(df,p_m,r_m ,s_m )

# estimation
p=0; r=1; s=0; P=20; n_iter=50
lmbd_list_1 = np.linspace(-0.95,0.95,20)
lmbd_list_2 = np.linspace(-0.95,0.95,20)
# pd.MultiIndex.from_product(lmbd_list_1,lmbd_list_2)
# record = pd.Series( index=lmbd_list)
record = np.zeros(3)
record[0] = np.inf
for lmbd_1 in lmbd_list_1:
    for lmbd_2 in lmbd_list_2:
        if lmbd_1 == lmbd_2:
            continue
        result =  BCD_LS(df,p,r,s,P,n_iter=30,lmbd=np.array([lmbd_1,lmbd_2]), gamma = np.array([]), phi = np.array([]),result_show=False)
        if result['Loss']< record[0]:
            record[0] = result['Loss']
            record[1:] = [lmbd_1,lmbd_2]
   
lmbd_list = np.linspace(-0.95,0.95,10)
record = np.zeros(2)
record[0] = np.inf
for lmbd in lmbd_list:
    result =  BCD_LS(df,p,r,s,P,n_iter=30,lmbd=np.array([lmbd]), gamma = np.array([]), phi = np.array([]),result_show=False)
    if result['Loss']< record[0]:
        record[0] = result['Loss']
        record[1:] = [lmbd]
        print(record)

lmbd=[0.95];gamma=[];phi=[]
result_LS = BCD_LS(df,p,r,s,P,n_iter,lmbd=np.array(lmbd), gamma = np.array(gamma), phi = np.array(phi),stop_thres=1e-5,result_show=True)
LS_lmbd,LS_eta,LS_G,LS_A,LS_Sigma=result_LS[['lmbd','eta','G','A','Sigma']]
LS_gamma,LS_phi = LS_eta
y=df
Y = y[:,1:]
x = (np.flip(y,axis=1)).ravel(order='F') # N(T-1) array: vectorized from y_T to y_1
X1 = np.zeros((N*(T-1),T-1))
for i in range(T-1):
    X1[:(i+1)*N,i] = x[-(i+1)*N:]
X2 = np.zeros((T-1,N,T-p-1)) #  shape: (T-1)*N*(T-p-1)  
for i in range(p,T-1):
    X2[i,:,:(i+1-p)] = np.flip(y[:,:i+1-p],axis=1)
L = get_L(LS_lmbd,LS_eta,r,s,T,p) # L: T*d matrix
z = kron([L[:T-1,:].T,np.identity(N)]) @ X1
epsilon = Y-tensor_op.unfold(LS_A[:,:,:T-1],0).numpy() @ X1
Sigma_inv = np.linalg.inv(LS_Sigma)
ASD_LS = np.sqrt(np.diag(lsfunc.asymptotic(LS_lmbd,LS_eta,LS_G,LS_Sigma,z,X2,epsilon,p,r,s,N,T))).round(3)
ASD_LS_G = ASD_LS[1:].reshape((6,6),order='F')
LS_G[:,:,0].round(3)

LS_Sigma.round(3)
vech_sigma_deva=0
for i in range(T-1):
    de_cen = (np.outer(epsilon[:,i], epsilon[:,i])-LS_Sigma)[np.triu_indices(N)]
    vech_sigma_deva += np.outer(de_cen,de_cen)/(T-1)
a= np.zeros((6,6))
N=6
a[np.triu_indices(N)]=np.sqrt(np.diag(vech_sigma_deva)/T).round(3)


LS_G[:,:,0].round(2)
LS_G[:,:,1].round(2)

lmbd_list = np.linspace(-0.95,0.95,20)
record = np.zeros(2)
record[0] = np.inf
for lmbd in lmbd_list:
    result =  BCD_MLE(df,p,r,s,P,n_iter=30,lmbd=np.array([lmbd]), gamma = np.array([]), phi = np.array([]),result_show=False)
    if result['Loss']< record[0]:
        record[0] = result['Loss']
        record[1:] = [lmbd]


lmbd=[0.95];gamma=[];phi=[]
result_ML = BCD_MLE(df,p,r,s,P,n_iter,lmbd=np.array(lmbd), gamma = np.array(gamma), phi = np.array(phi),stop_thres=1e-5,result_show=True)
ML_lmbd,ML_eta,ML_G,ML_A,ML_Sigma=result_ML[['lmbd','eta','G','A','Sigma']]
ML_gamma,ML_phi = ML_eta
ML_G[:,:,0].round(3)
y=df
Y = y[:,1:]
x = (np.flip(y,axis=1)).ravel(order='F') # N(T-1) array: vectorized from y_T to y_1
X1 = np.zeros((N*(T-1),T-1))
for i in range(T-1):
    X1[:(i+1)*N,i] = x[-(i+1)*N:]
X2 = np.zeros((T-1,N,T-p-1)) #  shape: (T-1)*N*(T-p-1)  
for i in range(p,T-1):
    X2[i,:,:(i+1-p)] = np.flip(y[:,:i+1-p],axis=1)
L = get_L(ML_lmbd,ML_eta,r,s,T,p) # L: T*d matrix
z = kron([L[:T-1,:].T,np.identity(N)]) @ X1
epsilon = Y-tensor_op.unfold(ML_A[:,:,:T-1],0).numpy() @ X1
Sigma_inv = np.linalg.inv(ML_Sigma)
ASD_ML = np.sqrt(np.diag(mlfunc.asymptotic(ML_lmbd,ML_eta,ML_G,Sigma_inv,z,X2,epsilon,p,r,s,N,T))).round(3)
ASD_ML_G = ASD_ML[1:-21].reshape((6,6),order='F')
ASD_ML_G.round(3)
ML_G[:,:,0].round(2)

a= np.zeros((6,6))
N=6
ML_Sigma.round(3)
a[np.triu_indices(N)]=ASD_ML[-21:].round(3)


import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # 调整为3列，最后一列用于颜色条

index = ['RPI','IPI','CUR','RPIR','CPI','RPCE']  # 横纵坐标标签

# 绘制第一个热力图和标记
ax0 = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[2])  # 创建一个新的颜色条子图
sns.heatmap(LS_G[:, :, 0].round(3), fmt=".3f", cmap='coolwarm', ax=ax0, vmin=-1, vmax=1, cbar_ax=cax)
for i in range(LS_G.shape[0]):
    for j in range(LS_G.shape[1]):
        text = f"{LS_G[i, j, 0]:.3f}({ASD_LS_G[i, j]:.3f})"
        ax0.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=12)
ax0.set_xticklabels(index, rotation=0)  # 设置x轴标签，并让标签竖直显示
ax0.set_yticklabels(index, rotation=0)  # 设置y轴标签

# 绘制第二个热力图和标记
ax1 = fig.add_subplot(gs[1])
sns.heatmap(ML_G[:, :, 0].round(3), fmt=".3f", cmap='coolwarm', ax=ax1, vmin=-1, vmax=1, cbar=False)
for i in range(ML_G.shape[0]):
    for j in range(ML_G.shape[1]):
        text = f"{ML_G[i, j, 0]:.3f}({ASD_ML_G[i, j]:.3f})"
        ax1.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=12)
ax1.set_xticklabels(index, rotation=0)  # 设置x轴标签，并让标签竖直显示
ax1.set_yticklabels(index, rotation=0)  # 设置y轴标签

plt.tight_layout()
plt.savefig("realdata_coef.png", bbox_inches='tight')
plt.show()



y=df
Y = y[:,1:] # N*(T-1) matrix
x = (np.flip(y,axis=1)).ravel(order='F') # NT array: vectorized from y_T to y_1
X1 = np.zeros((N*(T-1),T-1)) # y_t = [A_1,...,A_{T-1}] * vec(y_{t-1},...,y_{t-T+1})
for i in range(T-1):
    X1[:(i+1)*N,i] = x[-(i+1)*N:]

epsilon_LS = Y-tensor_op.unfold(LS_A[:,:,:T-1],0).numpy() @X1
epsilon_ML = Y-tensor_op.unfold(ML_A[:,:,:T-1],0).numpy() @X1

plt.plot(-(tensor_op.unfold(LS_A[:,:,:T-1],1).numpy() @X1)[2],color='red')
plt.plot(y[2],color='black')
plt.show()

plt.plot(epsilon_LS[0])
plt.show()


epsilon_LS_MACM=MACM(epsilon_LS.T,15)

fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(20, 15))
lag=16
err_bound = 2 / np.sqrt(T)
xlabel_lags=np.arange(1,lag)

for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        ax.vlines(xlabel_lags, [0], epsilon_LS_MACM[:,i,j] ,color='k',linewidth=0.7)
        ax.axhline(0, color='k',linewidth=0.7)
        ax.axhline(err_bound, color='r', linestyle='--',linewidth=0.7)
        ax.axhline(-err_bound, color='r', linestyle='--',linewidth=0.7)
        ax.set_yticks(np.linspace(-0.5,0.5,3))
# plt.savefig('MACM_y_5.pdf')
plt.show()


# VARMA ordering 1，3
max_p = 10
max_q = 10

best_aic = np.inf
best_order = None
best_model = None

for p in range(max_p + 1):
    for q in range(max_q + 1):
        try:
            # 拟合VARMA模型
            model = VARMAX(df_norm.values, order=(p, q))
            results = model.fit(maxiter=10, disp=False)

            # 获取当前模型的AIC
            aic = results.aic

            # 检查是否找到了更好的模型
            if aic < best_aic:
                best_aic = aic
                best_order = (p, q)
                best_model = results

        except:
            continue

# VAR ordering ##9
max_p = 10

best_aic = np.inf
best_order = None
best_model = None

for p in range(max_p + 1):
    try:
        # 拟合VARMA模型
        model = VARMAX(df_norm.values, order=(p,0))
        results = model.fit(maxiter=10, disp=False)

        # 获取当前模型的AIC
        aic = results.aic

        # 检查是否找到了更好的模型
        if aic < best_aic:
            best_aic = aic
            best_order = (p)
            best_model = results

    except:
        continue
    

# VMA ordering 2
max_q = 10

best_aic = np.inf
best_order = None
best_model = None


for q in range(max_q + 1):
    try:
        # 拟合VARMA模型
        model = VARMAX(df_norm.values, order=(0,q))
        results = model.fit(maxiter=10, disp=False)

        # 获取当前模型的AIC
        aic = results.aic

        # 检查是否找到了更好的模型
        if aic < best_aic:
            best_aic = aic
            best_order = (q)
            best_model = results
    except:
        continue
      


y_train =df[:,:600]
y_test=df[:,600:]
h=1
p=0
def for_SARMA(y_train, y_test, lmbd,gamma,phi, G, h=1,estimation = "LS", method = 'rolling'):
    '''
    h: h-step forecast;
    method: rolling--fix window;
            recursive--fix start point
    -----------------------------------
    output:
    N*[length(y_test)-h+1]
    '''
    N,T_train = y_train.shape
    T_test = y_test.shape[1]
    y_oos = np.zeros((N,T_test-h+1)) ## save forecasts

    for t in range(T_test-h+1):
        if method == "rolling":
            y = np.hstack((y_train[:,t:], y_test[:,:t])) 
        elif method  == "recursive":
            y = np.hstack((y_train[:,:], y_test[:,:t])) 
        if estimation == "LSE":
            fit = BCD_LS(y,p,r,s,P,n_iter,lmbd=np.array(lmbd), gamma = np.array([gamma]), phi = np.array([phi]),G = G,result_show=False)
        if estimation == "MLE":
            fit = BCD_MLE(y,p,r,s,P,n_iter,lmbd=np.array(lmbd), gamma = np.array([gamma]), phi = np.array([phi]),G = G,result_show=False)
        lmbd = fit['lmbd']
        gamma,phi = fit['eta']
        G = fit['G']
        y_oos[:,t] = forecast_SARMA(y,p,r,s,lmbd,gamma,phi, G, h=h)[:,-1]
        
    return y_oos

LSE_roll = for_SARMA(y_train, y_test, LS_lmbd,LS_gamma,LS_phi, LS_G, h=1,estimation = "LSE", method = 'rolling')
MLE_roll = for_SARMA(y_train, y_test, ML_lmbd,ML_gamma,ML_phi, ML_G, h=1,estimation = "MLE", method = 'rolling')

def for_VARMA(y_train, y_test ,h=1,order= (1,1), method = 'rolling'):
    N,T_train = y_train.shape
    T_test = y_test.shape[1]
    y_oos = np.zeros((N,T_test-h+1)) ## save forecasts

    for t in range(T_test-h+1):
        if method == "rolling":
            y = np.hstack((y_train[:,t:], y_test[:,:t])) 
        elif method  == "recursive":
            y = np.hstack((y_train[:,:], y_test[:,:t])) 
        model = VARMAX(y.T, order=(1,1))
        results = model.fit(maxiter= 20, disp=True)
        y_oos[:,t]= results.forecast(steps=1)
    return y_oos

forcast_VARMA = for_VARMA(y_train, y_test, h=1,order = (1,1) ,method = 'rolling')



def for_VAR(y_train, y_test ,h=1,order= (1,1), method = 'rolling'):
    N,T_train = y_train.shape
    T_test = y_test.shape[1]
    y_oos = np.zeros((N,T_test-h+1)) ## save forecasts

    for t in range(T_test-h+1):
        if method == "rolling":
            y = np.hstack((y_train[:,t:], y_test[:,:t])) 
        elif method  == "recursive":
            y = np.hstack((y_train[:,:], y_test[:,:t])) 
        model = VAR(y.T)
        results = model.fit(2)
        y_oos[:,t]= results.forecast(y.T[-2:],steps=1)
    return y_oos

forcast_VAR = for_VAR(y_train, y_test, h=1,order = (1,1) ,method = 'rolling')





def for_IOLS_VARMA(y_train, y_test ,h=1, method = 'rolling'):
    N,T_train = y_train.shape
    T_test = y_test.shape[1]
    y_oos = np.zeros((N,T_test-h+1)) ## save forecasts

    for t in range(T_test-h+1):
        if method == "rolling":
            y = np.hstack((y_train[:,t:], y_test[:,:t])) 
        elif method  == "recursive":
            y = np.hstack((y_train[:,:], y_test[:,:t])) 
        y_oos[:,t]= IOLS_pre(y).ravel('F')
    return y_oos
forcast_IOLS_VARMA = for_IOLS_VARMA(y_train, y_test, h=1 ,method = 'rolling')
# plot
fig, axes = plt.subplots(nrows=N, ncols=1, sharex=True, figsize=(20, 15))
for idx in range(N):
    axes[idx].plot(y_test[idx],color='black',lw=0.6)
    axes[idx].plot(LSE_roll[idx],color='red',lw=0.6)
    axes[idx].plot(MLE_roll[idx],color='blue',lw=0.6)
    axes[idx].plot(forcast_IOLS_VARMA[idx],color='g',lw=0.6)
    axes[idx].plot(forcast_VARMA[idx],color='y',lw=0.6)
    axes[idx].plot(forcast_VAR[idx],color='orange',lw=0.6)
    # axes[idx].plot(forcast_VAR[idx],color='green',lw=0.6)
    # axes[idx].yaxis.set_label_position('right')
    # axes[idx].set_ylabel(col,fontsize=14)
# fig.autofmt_xdate()
# plt.savefig('timeplot_logreturn_5.pdf')
plt.show()

SFE_table = pd.DataFrame(columns = ['QMLE', 'LSE', 'IOLS_VARMA','QMLE_VARMA','VAR'])
SAE_table = pd.DataFrame(columns = ['QMLE', 'LSE', 'IOLS_VARMA','QMLE_VARMA','VAR'])

SFE_table['IOLS_VARMA'] = np.linalg.norm(forcast_IOLS_VARMA- y_test,ord=2,axis=0)**2
SFE_table['LSE']= np.linalg.norm(LSE_roll-y_test,ord= 2,axis=0)**2
SFE_table['QMLE']= np.linalg.norm(MLE_roll-y_test,ord= 2,axis=0)**2
SFE_table['QMLE_VARMA'] = np.linalg.norm(forcast_VARMA-y_test,ord= 2,axis=0)**2
SFE_table['VAR'] = np.linalg.norm(forcast_VAR-y_test,ord= 2,axis=0)**2


SAE_table['IOLS_VARMA'] = np.linalg.norm(forcast_IOLS_VARMA- y_test,ord=1,axis=0)
SAE_table['LSE']= np.linalg.norm(LSE_roll-y_test,ord= 1,axis=0)
SAE_table['QMLE']= np.linalg.norm(MLE_roll-y_test,ord= 1,axis=0)
SAE_table['QMLE_VARMA'] = np.linalg.norm(forcast_VARMA-y_test,ord= 1,axis=0)
SAE_table['VAR'] = np.linalg.norm(forcast_VAR-y_test,ord= 1,axis=0)


np.sqrt(SFE_table.mean())
SAE_table.mean()
SFE_table
min_in_row = SFE_table.min(axis=1)
mask = SFE_table.eq(min_in_row, axis=0)
counts = mask.sum()

min_in_row = SAE_table.min(axis=1)
mask = SAE_table.eq(min_in_row, axis=0)
counts = mask.sum()

LSE_err_MSFE = 0
LSE_err_MAFE = 0
MLE_err_MSFE = 0
MLE_err_MAFE = 0
IOLS_err_MSFE = 0
IOLS_err_MAFE = 0
# VARMA_err_MSFE = 0
# VARMA_err_MAFE = 0
T_test = y_test.shape[1]
for t in range(T_test):
    # LSE_err_MSFE += np.linalg.norm(LSE_roll[:,t]-y_test[:,t],ord= 2)**2/T_test
    # LSE_err_MAFE += np.linalg.norm(LSE_roll[:,t]-y_test[:,t],ord= 1)/T_test
    # MLE_err_MSFE += np.linalg.norm(MLE_roll[:,t]-y_test[:,t],ord= 2)**2/T_test
    # MLE_err_MAFE += np.linalg.norm(MLE_roll[:,t]-y_test[:,t],ord= 1)/T_tes
    
    IOLS_err_MSFE += np.linalg.norm(forcast_IOLS_VARMA[:,t]-y_test[:,t],ord= 2)**2/T_test
    IOLS_err_MAFE += np.linalg.norm(forcast_IOLS_VARMA[:,t]-y_test[:,t],ord= 1)/T_test
    
    # VARMA_err_MSFE += np.linalg.norm(forcast_VARMA[:,t]-y_test[:,t],ord= 2)/T_test
    # VARMA_err_MAFE += np.linalg.norm(forcast_VARMA[:,t]-y_test[:,t],ord= 1)/T_test
np.sqrt(LSE_err_MSFE)
LSE_err_MAFE
np.sqrt(MLE_err_MSFE)
MLE_err_MAFE
VARMA_err_MSFE
VARMA_err_MAFE
np.sqrt(IOLS_err_MSFE)
IOLS_err_MAFE