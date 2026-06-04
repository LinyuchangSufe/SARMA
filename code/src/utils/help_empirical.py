
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss

# ================================
# 0) 你的 MACM 函数（原样保留）
# ================================
def MACM(df, lag):
    df = np.asarray(df)
    T, N = np.shape(df)
    mean_df = np.mean(df, 0)
    cov = np.zeros((lag + 1, N, N))
    for i in range(lag + 1):
        for t in range(i, T):
            cov[i] += np.outer(df[t] - mean_df, df[t - i] - mean_df) / (T - i)

    MACM_out = np.zeros((lag, N, N))
    cov_0 = np.outer(1 / np.sqrt(np.diag(cov[0])), 1 / np.sqrt(np.diag(cov[0])))
    for k in range(lag):
        MACM_out[k] = cov[k + 1] * cov_0
    return MACM_out
# ================================
# 1) 平稳性检验工具：ADF + KPSS
# ================================
def stationarity_tests(x: pd.Series):
    x = x.dropna().astype(float)

    # ADF: H0 = unit root (non-stationary)
    adf_stat, adf_p, _, _, _, _ = adfuller(x, autolag="AIC")

    # KPSS: H0 = stationary
    # regression='c' => level stationarity; 若你想检验趋势平稳，可用 'ct'
    try:
        kpss_stat, kpss_p, _, _ = kpss(x, regression="c", nlags="auto")
    except Exception:
        # 少数情况下 KPSS 可能因为数值问题报错，这里兜底
        kpss_stat, kpss_p = np.nan, np.nan

    return {
        "ADF stat": adf_stat,
        "ADF p": adf_p,
        "KPSS stat": kpss_stat,
        "KPSS p": kpss_p
    }


# =========================================
# 2) 每个指标：时序图 + ACF + 检验结果输出
# =========================================
def plot_series_and_acf(df_std: pd.DataFrame, acf_lag: int = 36):
    # 尝试把 index 转成 datetime（如果你的 index 是 YYYY-MM 或 YYYY-MM-DD）
    try:
        df_plot = df_std.copy()
        df_plot.index = pd.to_datetime(df_plot.index)
    except Exception:
        df_plot = df_std

    results = {}

    for col in df_plot.columns:
        x = df_plot[col].dropna()

        # ---- 平稳性检验 ----
        results[col] = stationarity_tests(x)

        # ---- 画图：时序 + ACF ----
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)

        axes[0].plot(x.index, x.values)
        axes[0].set_title(f"{col} | time series (standardized after FRED-MD transform)")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel(col)

        plot_acf(x.values, lags=acf_lag, ax=axes[1], zero=False)
        axes[1].set_title(f"{col} | ACF (lags<= {acf_lag})")

        plt.tight_layout()
        plt.show()

    # ---- 打印检验汇总表（论文/记录用）----
    test_df = pd.DataFrame(results).T
    # 只保留常用列顺序
    test_df = test_df[["ADF stat", "ADF p", "KPSS stat", "KPSS p"]]
    print("\nStationarity tests (ADF & KPSS):")
    print(test_df)

    return test_df


# =========================================
# 3) MACM：计算 + 画热力图（每个lag一张）
# =========================================
def plot_macm(df_std: pd.DataFrame, macm_lag: int = 12, lags_to_plot=None):
    X = df_std.to_numpy(dtype=float)
    cols = list(df_std.columns)

    macm = MACM(X, macm_lag)  # shape: (macm_lag, N, N)

    if lags_to_plot is None:
        # 默认把 1..macm_lag 都画出来（注意图会比较多）
        lags_to_plot = list(range(1, macm_lag + 1))

    for L in lags_to_plot:
        if not (1 <= L <= macm_lag):
            continue

        M = macm[L - 1]  # lag L 的 N×N matrix

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, aspect="auto", vmin=-1, vmax=1)
        ax.set_title(f"MACM heatmap | lag={L}")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticklabels(cols)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Autocorrelation")

        plt.tight_layout()
        plt.show()

    return macm