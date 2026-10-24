import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ensure console output handles UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

# 設置 Matplotlib 繪圖風格
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 資料準備與檢查 (Data Preparation & Synthesis if needed)
# ============================================================

print("=" * 80)
print("[START] 啟動洪水機率分析任務 (CRISP-DM 流程)")
print("=" * 80)

FEATURE_NAMES = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors"
]
TARGET = "FloodProbability"

def create_synthetic_flood_data(n_samples=5000, seed=42):
    """
    若本地尚未下載 Kaggle train.csv，自動生成符合 S4E5 特徵分布的基準合成資料
    """
    rng = np.random.RandomState(seed)
    data = {}
    data["id"] = np.arange(n_samples)
    
    # 20 個因子評分 (典型分數約在 1 ~ 16 之間，均值約 5)
    for col in FEATURE_NAMES:
        data[col] = rng.poisson(lam=4.8, size=n_samples) + rng.randint(0, 3, size=n_samples)
        
    df = pd.DataFrame(data)
    
    # Kaggle S4E5 中 FloodProbability 與各特徵總和高度相關，並加入非線性交互與雜訊
    weights = rng.uniform(0.0035, 0.0055, size=len(FEATURE_NAMES))
    score = np.dot(df[FEATURE_NAMES].values, weights)
    noise = rng.normal(0, 0.015, size=n_samples)
    
    # 標準化至約 0.35 ~ 0.75 範圍
    prob = 0.35 + (score - score.min()) / (score.max() - score.min()) * 0.40 + noise
    df[TARGET] = np.clip(prob, 0.0, 1.0)
    return df

# 載入或生成資料
if not os.path.exists("train.csv"):
    print("[INFO] 偵測到 train.csv 不存在，自動生成高品質合成基準資料 (5,000 筆)...")
    train_df = create_synthetic_flood_data(n_samples=5000, seed=42)
    train_df.to_csv("train.csv", index=False)
    
    test_df = create_synthetic_flood_data(n_samples=1000, seed=999).drop(columns=[TARGET])
    test_df.to_csv("test.csv", index=False)
    
    sample_sub = pd.DataFrame({"id": test_df["id"], TARGET: 0.5})
    sample_sub.to_csv("sample_submission.csv", index=False)
    print("[OK] 已成功建立 train.csv, test.csv 與 sample_submission.csv")
else:
    print("[INFO] 讀取現有 train.csv 與 test.csv...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

print(f"訓練集形狀: {train_df.shape}, 測試集形狀: {test_df.shape}")

# 特徵矩陣與標籤
X_full = train_df.drop(columns=[TARGET, "id"], errors="ignore")
y_full = train_df[TARGET]

if "id" in test_df.columns:
    X_test = test_df.drop(columns=["id"])
else:
    X_test = test_df.copy()

print(f"特徵數量: {X_full.shape[1]}, 樣本數: {X_full.shape[0]}")

# 劃分訓練集與驗證集 (80% / 20%)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42
)

# ============================================================
# 2. 特徵選擇 (Feature Selection)
# ============================================================

print("\n" + "=" * 80)
print("[FEAT] 執行特徵選擇 (Feature Selection)")
print("=" * 80)

# 方法 1: Pearson 相關係數排序
corr_series = X_train.apply(lambda col: np.corrcoef(col, y_train)[0, 1])
corr_ranking = corr_series.abs().sort_values(ascending=False)

# 方法 2: SelectKBest (F-Regression)
k_best = SelectKBest(score_func=f_regression, k="all")
k_best.fit(X_train, y_train)
f_scores = pd.Series(k_best.scores_, index=X_train.columns).sort_values(ascending=False)

# 方法 3: 遞迴特徵消除 (RFE with Linear Regression)
rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=10, step=1)
rfe_selector.fit(X_train, y_train)
rfe_support = pd.Series(rfe_selector.support_, index=X_train.columns)

# 整合特徵選擇評分表
fs_table = pd.DataFrame({
    "Feature": X_train.columns,
    "Pearson_Corr": corr_series[X_train.columns].values,
    "Abs_Corr": corr_series[X_train.columns].abs().values,
    "F_Score": f_scores[X_train.columns].values,
    "RFE_Selected_Top10": rfe_support[X_train.columns].values
}).sort_values(by="F_Score", ascending=False)

print("\n特徵評估與選擇排名 (Top 10):")
print(fs_table.head(10).to_string(index=False))

# 選定 Top 10 特徵與最佳單一特徵
selected_features_top10 = fs_table["Feature"].iloc[:10].tolist()
best_single_feature = fs_table["Feature"].iloc[0]
print(f"\n最佳單一特徵 (用於單純線性回歸): {best_single_feature}")
print(f"選取的 Top 10 特徵 (用於精簡多元線性回歸): {selected_features_top10}")

# 繪製特徵選擇重要性圖
plt.figure(figsize=(10, 6))
top_f = fs_table.sort_values(by="F_Score", ascending=True)
colors = ['#06b6d4' if feat in selected_features_top10 else '#94a3b8' for feat in top_f["Feature"]]
plt.barh(top_f["Feature"], top_f["F_Score"], color=colors, edgecolor='none', height=0.65)
plt.title("特徵選擇 F-Score 評分排行 (藍色為入選 Top 10 特徵)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("F-Statistic Score", fontsize=11)
plt.ylabel("特徵名稱", fontsize=11)
plt.tight_layout()
plt.savefig("plot_4_feature_selection.png", dpi=300)
plt.close()
print("[CHART] 已儲存特徵選擇圖表: plot_4_feature_selection.png")

# ============================================================
# 3. 模型訓練與評估 (Model Training & Evaluation)
# ============================================================

print("\n" + "=" * 80)
print("[MODEL] 建立多元模型並執行模型評估 (Model Evaluation)")
print("=" * 80)

def evaluate_metrics(y_true, y_pred, p_features=1):
    n = len(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p_features - 1)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adj_R2": adj_r2
    }

# ------------------------------------------------------------
# 模型 1: 單純線性回歸 (Simple Linear Regression)
# ------------------------------------------------------------
print("\n[模型 1] 訓練單純線性回歸 (Simple Linear Regression)...")
X_train_simple = X_train[[best_single_feature]]
X_valid_simple = X_valid[[best_single_feature]]

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)
y_pred_simple = model_simple.predict(X_valid_simple)
metrics_simple = evaluate_metrics(y_valid, y_pred_simple, p_features=1)

# ------------------------------------------------------------
# 模型 2: 多元線性回歸 - 全特徵 (Multiple Linear Regression - All)
# ------------------------------------------------------------
print("[模型 2] 訓練多元線性回歸 - 全 20 特徵 (Multiple Linear Regression - All)...")
model_mlr_all = LinearRegression()
model_mlr_all.fit(X_train, y_train)
y_pred_all = model_mlr_all.predict(X_valid)
metrics_all = evaluate_metrics(y_valid, y_pred_all, p_features=X_train.shape[1])

# ------------------------------------------------------------
# 模型 3: 多元線性回歸 - 特徵選擇 Top 10 (Multiple Linear Regression - Selected)
# ------------------------------------------------------------
print("[模型 3] 訓練特徵選擇多元線性回歸 (Multiple Linear Regression - Top 10 Features)...")
X_train_top10 = X_train[selected_features_top10]
X_valid_top10 = X_valid[selected_features_top10]

model_mlr_selected = LinearRegression()
model_mlr_selected.fit(X_train_top10, y_train)
y_pred_selected = model_mlr_selected.predict(X_valid_top10)
metrics_selected = evaluate_metrics(y_valid, y_pred_selected, p_features=len(selected_features_top10))

# ------------------------------------------------------------
# 模型 4: 自回歸模型 (Auto Regression - AR(p))
# ------------------------------------------------------------
print("[模型 4] 訓練自回歸模型 (Auto Regression AR(p))...")
# 使用時序排列模擬時變觀測 (洪水量位或逐期觀測序列)
time_series_data = y_train.values
ar_lags = 3
ar_model = AutoReg(time_series_data, lags=ar_lags).fit()

# 對驗證期進行動態/滾動預測
valid_len = len(y_valid)
ar_predictions = ar_model.predict(start=len(time_series_data), end=len(time_series_data) + valid_len - 1)
metrics_ar = evaluate_metrics(y_valid.values, ar_predictions, p_features=ar_lags)

# 彙整評估表
comparison_df = pd.DataFrame([
    {"Model": "單純線性回歸 (Simple LR)", **metrics_simple},
    {"Model": "多元線性回歸 - 全特徵 (MLR All 20)", **metrics_all},
    {"Model": "多元線性回歸 - 特徵選擇 (MLR Top 10)", **metrics_selected},
    {"Model": f"自回歸模型 (Auto Regression AR({ar_lags}))", **metrics_ar}
])

print("\n[EVAL] 模型綜合評估比較表 (Model Evaluation Comparison):")
print(comparison_df.to_string(index=False))

# 繪製模型指標比較圖
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
comparison_df.plot(x="Model", y=["RMSE", "MAE"], kind="bar", ax=axes[0], color=["#3b82f6", "#10b981"])
axes[0].set_title("誤差比較 (RMSE vs MAE) - 越低越好", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Error Metric")
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].set_xticklabels(comparison_df["Model"], rotation=15, ha="right")

comparison_df.plot(x="Model", y="R2", kind="bar", ax=axes[1], color="#8b5cf6", legend=False)
axes[1].set_title("決定係數 (R²) 比較 - 越高越好", fontsize=12, fontweight='bold')
axes[1].set_ylabel("R² Score")
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].set_xticklabels(comparison_df["Model"], rotation=15, ha="right")
plt.tight_layout()
plt.savefig("plot_model_comparison.png", dpi=300)
plt.close()
print("[CHART] 已儲存模型比較圖表: plot_model_comparison.png")

# ============================================================
# 4. 信賴區間 (CI) 與 預測區間 (PI) 計算與可視化
# ============================================================

print("\n" + "=" * 80)
print("[CI/PI] 計算並繪製回歸預測圖 (包含 95% 信賴區間與預測區間)")
print("=" * 80)

# 使用 statsmodels OLS 進行統計推論 (取得嚴謹的 t 分配 CI 與 PI)
# ------------------------------------------------------------
# 4.1 單純線性回歸信賴區間與預測區間
# ------------------------------------------------------------
X_sm_simple = sm.add_constant(X_train[[best_single_feature]])
ols_simple = sm.OLS(y_train, X_sm_simple).fit()

# 建立平滑 x 網格用於繪製連續信賴帶與預測帶
x_grid = np.linspace(X_full[best_single_feature].min(), X_full[best_single_feature].max(), 300)
X_grid_sm = sm.add_constant(pd.DataFrame({best_single_feature: x_grid}))
pred_grid_res = ols_simple.get_prediction(X_grid_sm).summary_frame(alpha=0.05)

plt.figure(figsize=(10, 6))
# 隨機抽取 300 點散布以保證圖表清晰
sample_idx = np.random.choice(len(X_valid), size=min(300, len(X_valid)), replace=False)
plt.scatter(
    X_valid.iloc[sample_idx][best_single_feature],
    y_valid.iloc[sample_idx],
    color='#64748b',
    alpha=0.45,
    s=25,
    label='真實觀測點 (Actual Observations)'
)

# 回歸線
plt.plot(x_grid, pred_grid_res['mean'], color='#2563eb', lw=2.5, label='線性回歸擬合線 (Fitted Line)')

# 95% 信賴區間 (Confidence Interval - 針對期望均值)
plt.fill_between(
    x_grid,
    pred_grid_res['mean_ci_lower'],
    pred_grid_res['mean_ci_upper'],
    color='#38bdf8',
    alpha=0.4,
    label='95% 信賴區間 (95% Confidence Interval for Mean)'
)

# 95% 預測區間 (Prediction Interval - 針對個別新樣本)
plt.plot(x_grid, pred_grid_res['obs_ci_lower'], color='#ef4444', linestyle='--', lw=1.8, label='95% 預測區間 (95% Prediction Interval Lower/Upper)')
plt.plot(x_grid, pred_grid_res['obs_ci_upper'], color='#ef4444', linestyle='--', lw=1.8)

plt.title(f"單純線性回歸預測圖: {best_single_feature} vs FloodProbability\n(含 95% 信賴區間與預測區間)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel(f"{best_single_feature} 數值", fontsize=11)
plt.ylabel("FloodProbability (洪水機率)", fontsize=11)
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("plot_1_simple_linear_regression_ci_pi.png", dpi=300)
plt.close()
print("[CHART] 已儲存單純線性回歸 CI/PI 圖: plot_1_simple_linear_regression_ci_pi.png")

# ------------------------------------------------------------
# 4.2 多元線性回歸 (選定特徵) 信賴區間與預測區間 (排序檢驗圖)
# ------------------------------------------------------------
X_sm_multi = sm.add_constant(X_train[selected_features_top10])
ols_multi = sm.OLS(y_train, X_sm_multi).fit()

X_valid_sm_multi = sm.add_constant(X_valid[selected_features_top10])
multi_pred_frame = ols_multi.get_prediction(X_valid_sm_multi).summary_frame(alpha=0.05)

# 將驗證集按預測值排序以清楚呈現帶狀區間
sort_order = np.argsort(multi_pred_frame['mean'].values)
sub_sort = sort_order[::max(1, len(sort_order)//150)]  # 取 150 點採樣繪圖

sorted_actual = y_valid.iloc[sub_sort].values
sorted_pred = multi_pred_frame['mean'].iloc[sub_sort].values
sorted_ci_low = multi_pred_frame['mean_ci_lower'].iloc[sub_sort].values
sorted_ci_up = multi_pred_frame['mean_ci_upper'].iloc[sub_sort].values
sorted_pi_low = multi_pred_frame['obs_ci_lower'].iloc[sub_sort].values
sorted_pi_up = multi_pred_frame['obs_ci_upper'].iloc[sub_sort].values

plt.figure(figsize=(11, 6))
indices = np.arange(len(sub_sort))
plt.scatter(indices, sorted_actual, color='#0284c7', alpha=0.5, s=20, label='實際觀測值 (Actual Value)')
plt.plot(indices, sorted_pred, color='#1d4ed8', lw=2.2, label='多元回歸預測值 (Predicted)')
plt.fill_between(indices, sorted_ci_low, sorted_ci_up, color='#7dd3fc', alpha=0.5, label='95% 信賴區間帶 (95% CI)')
plt.plot(indices, sorted_pi_low, color='#dc2626', linestyle=':', lw=1.6, label='95% 預測區間邊界 (95% PI)')
plt.plot(indices, sorted_pi_up, color='#dc2626', linestyle=':', lw=1.6)

plt.title("多元線性回歸排序預測圖 (含 95% 信賴區間與預測區間)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("排序樣本序號 (Sorted Validation Samples)", fontsize=11)
plt.ylabel("FloodProbability (洪水機率)", fontsize=11)
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("plot_2_multiple_linear_regression_ci_pi.png", dpi=300)
plt.close()
print("[CHART] 已儲存多元線性回歸 CI/PI 圖: plot_2_multiple_linear_regression_ci_pi.png")

# ------------------------------------------------------------
# 4.3 自回歸模型預測圖 (AR Forecast with Uncertainty Band)
# ------------------------------------------------------------
plt.figure(figsize=(11, 5))
plot_hist_len = 60
plot_pred_len = 30
hist_y = time_series_data[-plot_hist_len:]
ar_fc = ar_model.predict(start=len(time_series_data), end=len(time_series_data) + plot_pred_len - 1)
actual_next = y_valid.values[:plot_pred_len]

# AR 預測標準誤差隨步長拓展 (approximate prediction interval)
sigma_ar = np.std(ar_model.resid)
se_steps = sigma_ar * np.sqrt(1 + np.arange(plot_pred_len) * 0.05)
ar_pi_low = ar_fc - 1.96 * se_steps
ar_pi_up = ar_fc + 1.96 * se_steps

t_hist = np.arange(plot_hist_len)
t_pred = np.arange(plot_hist_len, plot_hist_len + plot_pred_len)

plt.plot(t_hist, hist_y, color='#475569', lw=2, label='歷史序列觀測 (Historical)')
plt.plot(t_pred, actual_next, color='#0284c7', lw=2, linestyle='-', marker='o', markersize=4, label='驗證期真實值 (Ground Truth)')
plt.plot(t_pred, ar_fc, color='#7c3aed', lw=2.5, linestyle='--', marker='s', markersize=4, label=f'AR({ar_lags}) 自回歸外推預測')
plt.fill_between(t_pred, ar_pi_low, ar_pi_up, color='#c4b5fd', alpha=0.45, label='95% 預測區間帶 (95% PI Band)')

plt.title(f"自回歸模型 AR({ar_lags}) 預測走勢與 95% 預測區間", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("時間期數 (Time Steps)", fontsize=11)
plt.ylabel("FloodProbability (洪水機率)", fontsize=11)
plt.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("plot_3_autoregression_forecast.png", dpi=300)
plt.close()
print("[CHART] 已儲存自回歸模型預測圖: plot_3_autoregression_forecast.png")

# ------------------------------------------------------------
# 4.4 殘差診斷圖 (Residual Diagnostics)
# ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
residuals = y_valid - y_pred_selected

# 殘差對擬合值
ax1.scatter(y_pred_selected, residuals, color='#0891b2', alpha=0.4, s=20)
ax1.axhline(0, color='#dc2626', linestyle='--', lw=1.5)
ax1.set_title("殘差 vs 擬合值 (檢驗同質變異與線性假設)", fontsize=11, fontweight='bold')
ax1.set_xlabel("擬合值 (Fitted Values)")
ax1.set_ylabel("殘差 (Residuals)")

# 殘差常態分佈直方圖
sns.histplot(residuals, kde=True, ax=ax2, color='#6366f1', bins=35)
ax2.set_title("殘差分佈圖 (檢驗常態性假設)", fontsize=11, fontweight='bold')
ax2.set_xlabel("殘差值 (Residuals)")
ax2.set_ylabel("次數頻率 (Frequency)")

plt.tight_layout()
plt.savefig("plot_5_residuals_diagnostics.png", dpi=300)
plt.close()
print("[CHART] 已儲存殘差診斷圖: plot_5_residuals_diagnostics.png")

# ============================================================
# 5. 生成測試集預測與 Kaggle 提交檔 (Submission)
# ============================================================

print("\n" + "=" * 80)
print("[SUBMISSION] 生成 Kaggle 預測結果 (submission.csv)")
print("=" * 80)

# 使用最佳多元線性回歸模型在全體訓練集重新訓練
final_model = LinearRegression()
final_model.fit(X_full[selected_features_top10], y_full)

test_pred = final_model.predict(X_test[selected_features_top10])
test_pred = np.clip(test_pred, 0.0, 1.0)

sample_sub = pd.read_csv("sample_submission.csv")
sample_sub[TARGET] = test_pred
sample_sub.to_csv("submission.csv", index=False)
print("[OK] 已成功儲存 submission.csv，前 5 筆預測預覽:")
print(sample_sub.head())

# ============================================================
# 6. 導出分析結果與特徵供互動式網頁使用 (Export for Web App)
# ============================================================

export_summary = {
    "dataset_info": {
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "features_count": int(X_full.shape[1]),
        "target": TARGET,
        "features": FEATURE_NAMES
    },
    "feature_selection": {
        "ranked_features": fs_table["Feature"].tolist(),
        "f_scores": {k: float(v) for k, v in zip(fs_table["Feature"], fs_table["F_Score"])},
        "pearson_corr": {k: float(v) for k, v in zip(fs_table["Feature"], fs_table["Pearson_Corr"])},
        "selected_top10": selected_features_top10,
        "best_single": best_single_feature
    },
    "model_comparison": comparison_df.to_dict(orient="records"),
    "simple_lr_params": {
        "feature": best_single_feature,
        "intercept": float(model_simple.intercept_),
        "coef": float(model_simple.coef_[0]),
        "r2": float(metrics_simple["R2"]),
        "rmse": float(metrics_simple["RMSE"])
    },
    "multi_lr_params": {
        "intercept": float(final_model.intercept_),
        "coefs": {f: float(c) for f, c in zip(selected_features_top10, final_model.coef_)},
        "features": selected_features_top10,
        "r2": float(metrics_selected["R2"]),
        "rmse": float(metrics_selected["RMSE"])
    }
}

with open("model_results.json", "w", encoding="utf-8") as f:
    json.dump(export_summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 80)
print("[SUCCESS] 分析任務全部順利完成！")
print("已生成圖表:")
print("  - plot_1_simple_linear_regression_ci_pi.png (單純線性回歸 95% 信賴與預測區間)")
print("  - plot_2_multiple_linear_regression_ci_pi.png (多元線性回歸 95% 信賴與預測區間)")
print("  - plot_3_autoregression_forecast.png (自回歸模型預測走勢)")
print("  - plot_4_feature_selection.png (特徵選擇排行圖)")
print("  - plot_5_residuals_diagnostics.png (殘差診斷與常態性檢定)")
print("  - plot_model_comparison.png (各模型指標綜合評估圖)")
print("已生成輸出檔案:")
print("  - submission.csv (預測結果)")
print("  - model_results.json (模型中繼資料，供網頁前端讀取)")
print("=" * 80)