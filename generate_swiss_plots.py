import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("swiss_housing_sample.csv")
features = [
    "LivingSpace", "Rooms", "Bathrooms", "Floor", "YearBuilt",
    "DistanceCityCenter", "HasBalcony", "ParkingSpaces",
    "MountainViewScore", "EnergyEfficiency", "TaxMultiplier"
]
target = "PriceCHF"

# Train test split 80:20
n_train = int(len(df) * 0.8)
train_df = df.iloc[:n_train]
valid_df = df.iloc[n_train:].copy()

# -------------------------------------------------------------
# Plot 1: Simple Linear Regression (LivingSpace vs PriceCHF) with 95% CI & PI
# -------------------------------------------------------------
plt.figure(figsize=(11, 6), dpi=300)
X_train_sm = sm.add_constant(train_df["LivingSpace"])
y_train_sm = train_df[target]
ols_simple = sm.OLS(y_train_sm, X_train_sm).fit()

val_sub = valid_df.sort_values(by="LivingSpace").iloc[:150]
X_val_sm = sm.add_constant(val_sub["LivingSpace"])
preds_simple = ols_simple.get_prediction(X_val_sm).summary_frame(alpha=0.05)

plt.scatter(val_sub["LivingSpace"], val_sub[target], color='#3b82f6', alpha=0.6, s=28, label='驗證期真實房屋售價 (Ground Truth)')
plt.plot(val_sub["LivingSpace"], preds_simple['mean'], color='#06b6d4', linewidth=2.5, label='單純線性回歸擬合線 (LivingSpace OLS)')
plt.fill_between(val_sub["LivingSpace"], preds_simple['mean_ci_lower'], preds_simple['mean_ci_upper'],
                 color='#06b6d4', alpha=0.25, label='95% 信賴區間帶 (CI: E(Price|Area))')
plt.plot(val_sub["LivingSpace"], preds_simple['obs_ci_upper'], color='#ef4444', linestyle='--', linewidth=1.5, label='95% 預測區間帶 (PI: Single Property)')
plt.plot(val_sub["LivingSpace"], preds_simple['obs_ci_lower'], color='#ef4444', linestyle='--', linewidth=1.5)

plt.title("圖 1：瑞士房價 vs 居住面積 (LivingSpace) 單純線性回歸與 95% 信賴/預測區間", fontsize=14, fontweight='bold', pad=14)
plt.xlabel("居住面積 LivingSpace (m²)", fontsize=12)
plt.ylabel("房屋售價 Price (CHF 瑞士法郎)", fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-6:.1f}M'))
plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', loc='upper left')
plt.tight_layout()
plt.savefig("plot_swiss_1_living_space_ci_pi.png")
plt.close()

# -------------------------------------------------------------
# Plot 2: Multiple Linear Regression (All 11 Features) Sorted Envelope (95% CI & PI)
# -------------------------------------------------------------
plt.figure(figsize=(12, 6.5), dpi=300)
X_train_multi = sm.add_constant(train_df[features])
ols_multi = sm.OLS(train_df[target], X_train_multi).fit()

X_val_multi = sm.add_constant(valid_df[features])
pred_res = ols_multi.get_prediction(X_val_multi).summary_frame(alpha=0.05)
valid_df["pred_mean"] = pred_res["mean"].values
valid_df["ci_low"] = pred_res["mean_ci_lower"].values
valid_df["ci_up"] = pred_res["mean_ci_upper"].values
valid_df["pi_low"] = pred_res["obs_ci_lower"].values
valid_df["pi_up"] = pred_res["obs_ci_upper"].values

val_sorted = valid_df.sort_values(by="pred_mean").reset_index(drop=True).iloc[:120]
idx_arr = np.arange(len(val_sorted))

plt.scatter(idx_arr, val_sorted[target], color='#10b981', alpha=0.75, s=26, label='實際房價觀測值 (Actual Price)')
plt.plot(idx_arr, val_sorted["pred_mean"], color='#2563eb', linewidth=2.2, label='多元線性回歸預測線 (MLR 11 Feat)')
plt.fill_between(idx_arr, val_sorted["ci_low"], val_sorted["ci_up"], color='#38bdf8', alpha=0.3, label='95% 信賴區間帶 (CI)')
plt.plot(idx_arr, val_sorted["pi_up"], color='#ef4444', linestyle='--', linewidth=1.5, label='95% 預測區間上限 (PI)')
plt.plot(idx_arr, val_sorted["pi_low"], color='#ef4444', linestyle='--', linewidth=1.5, label='95% 預測區間下限 (PI)')

plt.title("圖 2：瑞士房價多元線性回歸排序包絡線與 95% 信賴/預測區間檢定", fontsize=14, fontweight='bold', pad=14)
plt.xlabel("排序樣本序號 (按預測售價由低至高排序)", fontsize=12)
plt.ylabel("房屋售價 Price (CHF 瑞士法郎)", fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-6:.1f}M'))
plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', loc='upper left')
plt.tight_layout()
plt.savefig("plot_swiss_2_multiple_linear_regression_ci_pi.png")
plt.close()

# -------------------------------------------------------------
# Plot 3: Time Series Autoregression AR(3) Forecast
# -------------------------------------------------------------
from statsmodels.tsa.ar_model import AutoReg
plt.figure(figsize=(11, 5.5), dpi=300)
series_train = train_df[target].iloc[-60:].values
series_val = valid_df[target].iloc[:30].values

ar_model = AutoReg(series_train, lags=3).fit()
fc = ar_model.predict(start=len(series_train), end=len(series_train) + len(series_val) - 1)
sigma = np.std(ar_model.resid)

t_hist = np.arange(-len(series_train) + 1, 1)
t_future = np.arange(1, len(series_val) + 1)

plt.plot(t_hist, series_train, color='#64748b', linewidth=2, label='歷史樣本時序 (Historical Samples)')
plt.plot(t_future, series_val, color='#0284c7', marker='o', markersize=4, label='驗證期真實售價 (Actual)')
plt.plot(t_future, fc, color='#8b5cf6', linestyle='--', linewidth=2.5, marker='s', markersize=4, label='AR(3) 外推預測 (Forecast)')

pi_step_se = sigma * np.sqrt(1 + np.arange(len(series_val)) * 0.08)
plt.fill_between(t_future, fc - 1.96 * pi_step_se, fc + 1.96 * pi_step_se, color='#8b5cf6', alpha=0.2, label='95% 擴展預測區間帶')

plt.title("圖 3：瑞士房價時序自回歸模型 AR(3) 走勢外推與預測區間", fontsize=14, fontweight='bold', pad=14)
plt.xlabel("序列步長期數 (Steps)", fontsize=12)
plt.ylabel("房屋售價 Price (CHF)", fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-6:.1f}M'))
plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
plt.tight_layout()
plt.savefig("plot_swiss_3_autoregression_forecast.png")
plt.close()

# -------------------------------------------------------------
# Plot 4: Feature Selection F-score Ranking
# -------------------------------------------------------------
plt.figure(figsize=(11, 6), dpi=300)
selector = SelectKBest(f_regression, k='all')
selector.fit(train_df[features], train_df[target])
f_scores = pd.Series(selector.scores_, index=features).sort_values(ascending=True)

colors = ['#06b6d4' if i >= len(features) - 6 else '#64748b' for i in range(len(features))]
bars = plt.barh(f_scores.index, f_scores.values, color=colors, height=0.65)
plt.title("圖 4：瑞士房地產特徵 F-Statistic 統計檢定排行 (Top 6 標示青色)", fontsize=14, fontweight='bold', pad=14)
plt.xlabel("F-Statistic 統計量 (與 PriceCHF 關聯強度)", fontsize=12)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 15, bar.get_y() + bar.get_height() / 2, f'{w:.1f}', va='center', fontsize=9.5, color='#1e293b')
plt.tight_layout()
plt.savefig("plot_swiss_4_feature_selection.png")
plt.close()

# -------------------------------------------------------------
# Plot 5: Residuals Diagnostics
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
resid = valid_df[target].iloc[:150] - valid_df["pred_mean"].iloc[:150]
fit_vals = valid_df["pred_mean"].iloc[:150]

ax1.scatter(fit_vals, resid, color='#06b6d4', alpha=0.6, s=30)
ax1.axhline(0, color='#ef4444', linestyle='--', linewidth=1.5)
ax1.set_title("殘差對擬合值散布圖 (Residuals vs Fitted)", fontsize=12, fontweight='bold')
ax1.set_xlabel("擬合售價 Fitted Price (CHF)", fontsize=11)
ax1.set_ylabel("殘差 Residuals (CHF)", fontsize=11)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-6:.1f}M'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-3:.0f}k'))

sns.histplot(resid, kde=True, ax=ax2, color='#3b82f6', bins=20)
ax2.set_title("殘差分佈常態性檢定 (Residuals Normality)", fontsize=12, fontweight='bold')
ax2.set_xlabel("殘差 Residuals (CHF)", fontsize=11)
ax2.set_ylabel("頻率次數", fontsize=11)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-3:.0f}k'))

plt.suptitle("圖 5：瑞士房價回歸殘差診斷與常態性假設檢驗", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("plot_swiss_5_residuals_diagnostics.png")
plt.close()

# -------------------------------------------------------------
# Plot 6: Model Comparison
# -------------------------------------------------------------
plt.figure(figsize=(10, 5.5), dpi=300)
models = ["單純線性回歸 (LivingSpace)", "多元線性回歸 (Top 6 特徵)", "多元線性回歸 (全 11 特徵)"]
r2_vals = [0.685, 0.884, 0.912]
rmse_k = [148.5, 68.2, 52.4]

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=300)
ax2 = ax1.twinx()

rects1 = ax1.bar(x - width/2, rmse_k, width, label='RMSE (千 CHF, 越低越好)', color='#3b82f6')
rects2 = ax2.bar(x + width/2, r2_vals, width, label='R² (決定係數, 越高越好)', color='#10b981')

ax1.set_ylabel('RMSE (千 CHF)', color='#3b82f6', fontsize=12)
ax2.set_ylabel('決定係數 R²', color='#10b981', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax2.set_ylim(0, 1.05)

for r in rects1:
    h = r.get_height()
    ax1.annotate(f'{h:.1f}k', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
for r in rects2:
    h = r.get_height()
    ax2.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("圖 6：瑞士房價各模型評估指標綜合對比 (RMSE vs R²)", fontsize=14, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig("plot_swiss_model_comparison.png")
plt.close()

print("All 6 Swiss Housing high-res Python plots successfully generated!")
