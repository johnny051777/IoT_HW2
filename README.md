# HydroPredict AI - 洪水回歸分析與互動預測工作台

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Vercel: Deployed](https://img.shields.io/badge/Vercel-Deployment-black.svg)](https://vercel.com)

基於 **CRISP-DM** 數據採掘標準流程開發之洪水機率預測系統，整合**線性回歸 (Linear Regression)**、**多元線性回歸 (MLR)**、**自回歸模型 (Auto Regression AR)**，並提供嚴謹的**特徵選擇 (Feature Selection)**、**模型評估 (Model Evaluation)** 與 **95% 信賴區間 (CI) / 預測區間 (PI)** 視覺化。

---

## 🌟 核心特色

1. **多元回歸模型對比**：
   - 單純線性回歸 (Simple Linear Regression)
   - 多元線性回歸 (全特徵 vs 特徵選擇子集)
   - 時序自回歸 (Auto Regression AR(p))
2. **統計特徵選擇**：
   - F-Statistic (SelectKBest) 檢定
   - Pearson 相關係數排名
   - 遞迴特徵消除 (RFE)
3. **95% 信賴區間 (CI) 與 預測區間 (PI)**：
   - 區分母體期望均值 (Mean Response CI) 與單一觀測波動 (Individual Observation PI)
   - 支援 4 種切換視角 (單純回歸散布帶、多元排序包絡線、AR 時序外推、殘差分佈檢定)
4. **現代互動式 Web 工作台**：
   - 原生 Vanilla CSS 深色玻璃擬態 (Glassmorphism) 設計
   - 即時 What-If 預測模擬器與動態圓形儀表
   - 支援拖曳自訂 CSV 檔案上傳並自動重算

---

## 🚀 本地快速啟動

### 1. 執行 Python 完整分析管道
```bash
python main.py
```
執行後將自動產製以下成果：
- `plot_1_simple_linear_regression_ci_pi.png`
- `plot_2_multiple_linear_regression_ci_pi.png`
- `plot_3_autoregression_forecast.png`
- `plot_4_feature_selection.png`
- `plot_5_residuals_diagnostics.png`
- `plot_model_comparison.png`
- `submission.csv`
- `model_results.json`

### 2. 開啟互動式網頁工作台
```bash
python -m http.server 8080
```
於瀏覽器開啟 `http://localhost:8080` 即可體驗完整的互動視覺化介面。

---

## 📊 模型評估結果指標

| 模型架構 | 特徵數 | MAE | RMSE | $R^2$ | 調整後 $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **多元線性回歸 (全 20 特徵)** | 20 | **0.01213** | **0.01523** | **0.9207** | **0.9191** |
| 多元線性回歸 (Top 10 特徵) | 10 | 0.02947 | 0.03700 | 0.5323 | 0.5276 |
| 單純線性回歸 (`Landslides`) | 1 | 0.04181 | 0.05282 | 0.0470 | 0.0461 |
| 自回歸模型 AR(3) | 3 | 0.04272 | 0.05411 | 基準時序 | 基準時序 |

---

## 🛠️ 技術棧
- **Python Pipeline**: `scikit-learn`, `statsmodels`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`
- **Frontend Web**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+ OLS Statistical Math Engine), Chart.js
