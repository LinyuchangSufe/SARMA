import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


def var_one_step_forecast(Y, p):
    """
    单变量用 AutoReg
    多变量用 VAR
    """
    T, N = Y.shape
    
    # 单变量 case
    if N == 1:
        y = Y[:, 0]
        model = AutoReg(y, lags=p).fit()
        pred = model.predict(start=len(y), end=len(y))
        return np.array([pred[0]])
    
    # 多变量 case
    df = pd.DataFrame(Y, columns=[f"y{i}" for i in range(N)])
    model = VAR(df)
    results = model.fit(maxlags=p)
    lag = results.k_ar
    pred = results.forecast(df.values[-lag:], steps=1)
    return pred[0]

def varma_one_step_forecast(Y, p, q):
    """
    N>1 使用 VARMA(p,q)
    N=1 使用 ARMA(p,q)
    返回统一为 (N,) 数组
    """
    T, N = Y.shape
    
    # 单变量 case → 自动走 ARMA
    if N == 1:
        y = Y[:, 0]
        # ARMA = ARIMA(order=(p,0,q))
        model = ARIMA(y, order=(p, 0, q)).fit()
        pred = model.forecast(steps=1)
        return np.array([pred[0]])

    # 多变量 → VARMA
    df = pd.DataFrame(Y, columns=[f"y{i}" for i in range(N)])
    model = VARMAX(df, order=(p, q), trend='c')
    results = model.fit(disp=False)
    y_hat = results.forecast(steps=1).values[0]

    return y_hat

from statsmodels.tsa.ar_model import AutoReg

def ar_one_step_forecast(Y, p):
    """
    对 Y 的每一个变量分别构建 AR(p)，并进行一步预测。
    参数:
        Y: numpy array of shape (T, N)
        p: AR 阶数
    返回:
        pred: 长度 N 的一步预测向量
    """
    T, N = Y.shape
    pred = np.zeros(N)

    for j in range(N):
        y = Y[:, j]

        # AutoReg 会自动处理 y[t] = c + sum φ_i y[t-i]
        model = AutoReg(y, lags=p, old_names=False)
        res = model.fit()

        # 一步预测
        pred[j] = res.predict(start=T, end=T)[0]

    return pred