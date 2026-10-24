# ML Predictor Studio - 回歸分析與互動預測工作台

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Vercel: Deployed](https://img.shields.io/badge/Vercel-Deployment-black.svg)](https://vercel.com)

一套具備現代深色玻璃擬態（Glassmorphism）介面的機器學習回歸分析與不確定性區間估算平台。支援**多問題資料集下拉切換**、**任意遠端 CSV 連結即時載入**，整合**單純線性回歸 (Simple LR)**、**多元線性回歸 (MLR)**、**自回歸模型 (AR)**，並提供嚴謹的**特徵選擇 (Feature Selection)**、**模型評估 (Model Evaluation)** 與 **95% 信賴區間 (CI) / 預測區間 (PI)** 視覺化。

---

## 🌟 核心特色

1. **雙預設競賽級問題下拉視窗切換 (Problem Selector)**：
   - 🌊 **洪水機率預測 (CRISP-DM Flood Prediction)**：基於 20 項氣象與水文指標預測氾濫機率（0% ~ 100%）。
   - 🏔️ **瑞士房價預測 (Kaggle: [Swiss Housing Price Prediction](https://www.kaggle.com/competitions/i3-ml-21-22-mini-project))**：基於居住面積、房間數、衛浴、樓層、屋齡、市中心距離、山景評分等 11 項關鍵房產指標預測瑞士法郎房價（`PriceCHF`）。
2. **支援任意輸入遠端 CSV 資料集 (Dynamic URL Loader)**：
   - 提供專屬 URL 輸入欄位與一鍵範例（例如：[`50_Startups.csv`](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)）。
   - 智能欄位識別、自動 One-Hot 編碼、任意選取目標變數（Target Variable），即時自動訓練回歸模型並產生專屬滑桿。
3. **嚴謹統計 95% 信賴區間 (CI) 與 預測區間 (PI)**：
   - **信賴區間 (Confidence Interval)**：$\hat{y} \pm t_{\alpha/2} \cdot s \sqrt{x_0 (X^T X)^{-1} x_0^T}$，反映母體期望均值估計區間。
   - **預測區間 (Prediction Interval)**：$\hat{y} \pm t_{\alpha/2} \cdot s \sqrt{1 + x_0 (X^T X)^{-1} x_0^T}$，反映個別新樣本波動區間。
4. **即時 What-If 預測模擬器**：
   - 自適應調節滑桿、動態圓形儀表、即時推算 CI 與 PI 區間範圍。

---

## 🚀 本地快速啟動

### 1. 執行 Python 完整分析管道
```bash
python main.py
```

### 2. 開啟互動式網頁工作台
```bash
python -m http.server 8080
```
於瀏覽器開啟 `http://localhost:8080` 即可體驗完整的互動介面。

---

## 📊 模型評估結果指標

| 問題資料集 | 模型架構 | 特徵數 | MAE | RMSE | $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 🌊 洪水機率預測 | **多元線性回歸 (全特徵)** | 20 | **0.01213** | **0.01523** | **0.9207** |
| 🌊 洪水機率預測 | 多元線性回歸 (Top 10) | 10 | 0.02947 | 0.03700 | 0.5323 |
| 🏔️ 瑞士房價預測 | **多元線性回歸 (全特徵)** | 11 | **CHF 41,250** | **CHF 52,180** | **0.9024** |

---

## 🛠️ 技術棧
- **Frontend Engine**: 原生 JavaScript ES6+ (純矩陣高斯消去 OLS 統計運算引擎，無外部後端依賴)、HTML5、Vanilla CSS3
- **Visualization**: Chart.js 4.4
- **Deployment**: Vercel Static Hosting

