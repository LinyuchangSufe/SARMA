import pandas as pd
import numpy as np

# -----------------------------
# Step 0. Load raw CSV
# -----------------------------
df_raw = pd.read_csv("data/FRED-MD.csv")

# 第 0 行是 TCODE
tcode = df_raw.iloc[0].drop("sasdate")
df = df_raw.iloc[1:].copy()

# 转换日期
df["sasdate"] = pd.to_datetime(df["sasdate"])
df = df.set_index("sasdate")

# 数值列转成 float
df = df.astype(float)

# -----------------------------
# Step 1. Remove 1950
# -----------------------------
# df = df[df.index.year > 1959]
# print("After removing 1950:", df.shape)

# -----------------------------
# Step 2. Apply TCODE transformations
# -----------------------------
def apply_tcode(series, code):
    if code == 1:
        return series
    elif code == 2:
        return series.diff()
    elif code == 3:
        return series.diff().diff()
    elif code == 4:
        return np.log(series)
    elif code == 5:
        return np.log(series).diff()
    elif code == 6:
        return np.log(series).diff().diff()
    else:
        # unknown TCODE: keep as is
        return series

df_t_list = {}  # use dict to store columns

for col in df.columns:
    code = int(tcode[col])
    df_t_list[col] = apply_tcode(df[col], code)

# one-shot construction, no fragmentation
df_t = pd.DataFrame(df_t_list, index=df.index)

print("After TCODE transform:", df_t.shape)

df_t = df_t[df_t.index.year > 1959]
print("After removing 1950:", df_t.shape)
# -----------------------------
# Step 3. Remove outliers (winsorize at 1%–99%)
# -----------------------------
def winsorize(s):
    q1 = s.quantile(0.01)
    q99 = s.quantile(0.99)
    return s.clip(q1, q99)

df_wo = df_t.apply(winsorize)
print("After outlier removal:", df_wo.shape)

# -----------------------------
# Step 4. Missing value filtering
#   (a) Remove variables with consecutive NA > 3 months
#   (b) Remove variables with missing rate >= 0.005
# -----------------------------

def max_consecutive_na(s):
    is_na = s.isna().astype(int)
    # count consecutive blocks
    return (is_na.groupby((is_na != is_na.shift()).cumsum()).cumsum().max())

valid_cols = []
for col in df_wo.columns:
    if max_consecutive_na(df_wo[col]) <= 3:
        valid_cols.append(col)

df_mv = df_wo[valid_cols]
print("After removing variables with >3 consecutive NA:", df_mv.shape)

# Missing rate filtering
missing_rate = df_mv.isna().mean()
df_mv = df_mv[missing_rate[missing_rate < 0.005].index]
print("After filtering missing rate < 0.005:", df_mv.shape)

# -----------------------------
# Step 5. Prepare missing (initial fill)
#   Simple mean fill (or median fill) for initial EM
# -----------------------------
df_init = df_mv.copy()
for col in df_init.columns:
    df_init[col] = df_init[col].fillna(df_init[col].mean())

print("After initial missing fill:", df_init.shape)

# -----------------------------
# Step 6. Standardization (mean=0, std=1)
# -----------------------------
df_std = (df_init - df_init.mean()) / df_init.std()

print("Final standardized dataset:", df_std.shape)

# df_std 即为 Step 0–6 完成后的数据，可直接用于 PCA 或 EM

df_std

output_path = "data/FRED_MD_processed.csv"
df_std.to_csv(output_path, index=False)


# ======================
# 1. 读取文件（你上传的路径）
# ======================
df_raw = pd.read_csv("data/FRED-QD.csv")


# 第 1 行是 TCODE
tcode = df_raw.iloc[1].drop("sasdate")
df = df_raw.iloc[2:].copy()

# 转换日期
df["sasdate"] = pd.to_datetime(df["sasdate"])
df = df.set_index("sasdate")


# 数值列转成 float
df = df.astype(float)

# -----------------------------
# Step 1. Remove 1950
# -----------------------------
# df = df[df.index.year > 1959]
# print("After removing 1950:", df.shape)

# -----------------------------
# Step 2. Apply TCODE transformations
# -----------------------------
def apply_tcode(series, code):
    code = int(code)
    if code == 1:
        return series
    elif code == 2:
        return series.diff()
    elif code == 3:
        return series.diff().diff()
    elif code == 4:
        return np.log(series)
    elif code == 5:
        return np.log(series).diff()
    elif code == 6:
        return np.log(series).diff().diff()


df_t_list = {}  # use dict to store columns

for col in df.columns:
    code = int(tcode[col])
    df_t_list[col] = apply_tcode(df[col], code)

# one-shot construction, no fragmentation
df_t = pd.DataFrame(df_t_list, index=df.index)

print("After TCODE transform:", df_t.shape)


# -----------------------------
# Step 3. Remove outliers (winsorize at 1%–99%)
# -----------------------------
def winsorize(s):
    q1 = s.quantile(0.01)
    q99 = s.quantile(0.99)
    return s.clip(q1, q99)

df_wo = df_t.apply(winsorize)
print("After outlier removal:", df_wo.shape)

# -----------------------------
# Step 4. Missing value filtering
#   (a) Remove variables with consecutive NA > 3 months
#   (b) Remove variables with missing rate >= 0.005
# -----------------------------


# Missing rate filtering
missing_rate = df_wo.isna().mean()
df_rate = df_wo[missing_rate[missing_rate < 0.05].index]
print("After filtering missing rate < 0.005:", df_rate.shape)

def max_consecutive_na(s):
    is_na = s.isna().astype(int)
    # count consecutive blocks
    return (is_na.groupby((is_na != is_na.shift()).cumsum()).cumsum().max())

valid_cols = []
for col in df_rate.columns:
    if max_consecutive_na(df_rate[col]) <= 3:
        valid_cols.append(col)

df_mv = df_rate[valid_cols]
print("After removing variables with >3 consecutive NA:", df_mv.shape)


# -----------------------------
# Step 5. Prepare missing (initial fill)
#   Simple mean fill (or median fill) for initial EM
# -----------------------------
df_init = df_mv.copy()
for col in df_init.columns:
    df_init[col] = df_init[col].fillna(df_init[col].mean())

print("After initial missing fill:", df_init.shape)

# -----------------------------
# Step 6. Standardization (mean=0, std=1)
# -----------------------------
df_std = (df_init - df_init.mean()) / df_init.std()

print("Final standardized dataset:", df_std.shape)

# df_std 即为 Step 0–6 完成后的数据，可直接用于 PCA 或 EM

df_std

output_path = "data/FRED_QD_processed.csv"
df_std.to_csv(output_path, index=False)

# df_two = df_std[["GDPC1", "PCECC96"]].copy()
df_two = df_std[["GDPC1"]].copy()
df_two.to_csv("data/2index.csv", index=False)
df_std = df_std.drop(columns=["GDPC1"])
df_std.to_csv("data/FRED_QD_processed.csv", index=False)
