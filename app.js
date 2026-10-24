/**
 * HydroPredict AI - 洪水回歸分析與互動工作台 (app.js)
 * 包含完整 OLS 線性回歸、特徵選擇、95% 信賴區間/預測區間推導與即時模擬引擎
 */

// 全域狀態變數
const state = {
  features: [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors"
  ],
  target: "FloodProbability",
  dataset: [],            // 原始或抽樣資料陣列
  trainData: [],
  validData: [],
  selectedFeatures: [],   // 目前入選特徵清單
  activeTab: "plots",
  showPredictionInterval: true,
  currentPlotMode: "simple-fit",
  simpleFeature: "Landslides",
  arLags: 3,
  
  // 模型擬合參數快取
  models: {
    simple: null,
    mlrAll: null,
    mlrSelected: null,
    ar: null
  },
  
  // 特徵評分快取
  featureScores: {}
};

// Chart.js 實例儲存
const chartInstances = {
  mainPlot: null,
  featureRank: null,
  evalError: null,
  evalR2: null
};

// ============================================================
// 1. 初始化與資料載入
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
  console.log("HydroPredict AI 系統初始化中...");
  await loadInitialData();
  initFeatureSelection();
  initSimpleFeatureDropdown();
  trainAllModels();
  initSimulatorSliders();
  renderActivePlot();
  renderFeatureRankChart();
  renderEvaluationTable();
  renderEvaluationCharts();
  renderDataTable();
  updateSimulatorInference();
});

// 載入資料 (優先讀取本地 JSON，若無則啟用高品質合成基準水文資料)
async function loadInitialData() {
  try {
    const resSample = await fetch("dataset_sample.json");
    if (resSample.ok) {
      state.dataset = await resSample.json();
      console.log(`成功載入 dataset_sample.json (${state.dataset.length} 筆)`);
    } else {
      generateLocalSyntheticData(400);
    }
  } catch (err) {
    console.warn("無法透過 fetch 讀取 JSON，自動生成本地基準水文資料集:", err);
    generateLocalSyntheticData(400);
  }

  // 劃分 80% 訓練集與 20% 驗證集
  const n = state.dataset.length;
  const trainSize = Math.floor(n * 0.8);
  state.trainData = state.dataset.slice(0, trainSize);
  state.validData = state.dataset.slice(trainSize);

  // 載入模型中繼資料
  try {
    const resMeta = await fetch("model_results.json");
    if (resMeta.ok) {
      const meta = await resMeta.json();
      state.featureScores = meta.feature_selection.f_scores || {};
      state.selectedFeatures = meta.feature_selection.selected_top10 || state.features.slice(0, 10);
      state.simpleFeature = meta.feature_selection.best_single || "Landslides";
      return;
    }
  } catch (e) {
    console.warn("讀取 model_results.json 失敗，使用本地統計計算");
  }

  // 若無外部中繼資料，直接由本地資料計算 F-score
  computeFeatureScoresLocally();
  state.selectedFeatures = getTopKFeatures(10);
}

// 產生基準合成水文觀測資料
function generateLocalSyntheticData(count = 400) {
  state.dataset = [];
  const weights = [
    0.045, 0.042, 0.041, 0.043, 0.039, 0.040, 0.041, 0.038,
    0.036, 0.037, 0.039, 0.042, 0.038, 0.046, 0.039, 0.040,
    0.037, 0.038, 0.039, 0.038
  ];

  for (let i = 0; i < count; i++) {
    const row = { id: i };
    let scoreSum = 0;
    state.features.forEach((feat, idx) => {
      // 隨機整數 1 ~ 13
      const val = Math.floor(Math.random() * 10) + 2;
      row[feat] = val;
      scoreSum += val * weights[idx];
    });
    
    // 加入微小高斯擾動
    const noise = (Math.random() - 0.5) * 0.03;
    const prob = 0.35 + (scoreSum / 9.0) * 0.40 + noise;
    row[state.target] = Math.max(0.0, Math.min(1.0, prob));
    state.dataset.push(row);
  }
}

// 本地計算特徵 F-score 與 Pearson 相關
function computeFeatureScoresLocally() {
  const y = state.trainData.map(d => d[state.target]);
  const yMean = mean(y);
  const n = y.length;
  const ssTotal = y.reduce((acc, v) => acc + Math.pow(v - yMean, 2), 0);

  state.featureScores = {};
  state.features.forEach(feat => {
    const x = state.trainData.map(d => d[feat]);
    const xMean = mean(x);
    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - xMean;
      const dy = y[i] - yMean;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }
    const r = denX > 0 && denY > 0 ? num / Math.sqrt(denX * denY) : 0;
    const r2 = r * r;
    // F-statistic for 1 predictor = (R^2 / 1) / ((1 - R^2) / (n - 2))
    const f = r2 < 0.999 ? (r2 * (n - 2)) / (1 - r2) : 999;
    state.featureScores[feat] = Math.max(0, f);
  });
}

function getTopKFeatures(k = 10) {
  return [...state.features]
    .sort((a, b) => (state.featureScores[b] || 0) - (state.featureScores[a] || 0))
    .slice(0, k);
}

// ============================================================
// 2. 統計回歸運算核心 (OLS with 95% CI & PI)
// ============================================================

/**
 * 普通最小平方 (OLS) 多元線性回歸擬合與不確定性區間估算
 */
class LinearRegressionModel {
  constructor(featureNames) {
    this.features = featureNames;
    this.p = featureNames.length;
    this.beta = [];              // [intercept, beta_1, ... beta_p]
    this.xtxInv = null;          // (X^T X)^(-1) 共變異數縮放矩陣
    this.mse = 0;                // 均方誤差 s^2
    this.se = 0;                 // 殘差標準誤差 s = sqrt(MSE)
    this.df = 1;                 // 自由度 n - p - 1
    this.tCritical = 1.96;       // 95% t 臨界值 (n 較大時接近 1.96)
    this.r2 = 0;
    this.adjR2 = 0;
    this.mae = 0;
    this.rmse = 0;
  }

  fit(data, targetCol) {
    const n = data.length;
    const p = this.p;
    this.df = Math.max(1, n - p - 1);
    this.tCritical = this.df > 120 ? 1.96 : (this.df > 30 ? 2.0 : 2.1);

    // 建構 X (包含常數項 1) 與 y
    const X = [];
    const y = [];
    for (let i = 0; i < n; i++) {
      const row = [1]; // Intercept
      for (let j = 0; j < p; j++) {
        row.push(data[i][this.features[j]]);
      }
      X.push(row);
      y.push(data[i][targetCol]);
    }

    // 計算 X^T
    const Xt = transpose(X);
    // 計算 X^T X ((p+1) x (p+1))
    const XtX = matMul(Xt, X);
    // 計算 (X^T X)^(-1)
    this.xtxInv = invertMatrix(XtX);
    if (!this.xtxInv) {
      console.error("奇異矩陣，加入嶺回歸微小對角線正則化");
      for (let i = 0; i < XtX.length; i++) XtX[i][i] += 1e-4;
      this.xtxInv = invertMatrix(XtX) || identityMatrix(p + 1);
    }

    // 計算 X^T y
    const Xty = matMulVec(Xt, y);
    // 計算 beta = (X^T X)^(-1) X^T y
    this.beta = matMulVec(this.xtxInv, Xty);

    // 計算訓練殘差與 MSE
    let sse = 0;
    let sae = 0;
    const yMean = mean(y);
    let sst = 0;

    for (let i = 0; i < n; i++) {
      const pred = this.predictRow(data[i]);
      const err = y[i] - pred;
      sse += err * err;
      sae += Math.abs(err);
      sst += Math.pow(y[i] - yMean, 2);
    }

    this.mse = sse / this.df;
    this.se = Math.sqrt(this.mse);
    this.mae = sae / n;
    this.rmse = Math.sqrt(sse / n);
    this.r2 = sst > 0 ? 1 - (sse / sst) : 0;
    this.adjR2 = 1 - (1 - this.r2) * (n - 1) / this.df;

    return this;
  }

  predictRow(sample) {
    let val = this.beta[0];
    for (let j = 0; j < this.p; j++) {
      val += this.beta[j + 1] * (sample[this.features[j]] ?? 5);
    }
    return val;
  }

  /**
   * 預測並計算 95% 信賴區間 (CI) 與 95% 預測區間 (PI)
   */
  predictWithIntervals(sample) {
    const x0 = [1];
    for (let j = 0; j < this.p; j++) {
      x0.push(sample[this.features[j]] ?? 5);
    }

    // y_hat = x0 * beta
    let yHat = 0;
    for (let i = 0; i < x0.length; i++) {
      yHat += x0[i] * this.beta[i];
    }

    // leverage h0 = x0 * (X^T X)^(-1) * x0^T
    let h0 = 0;
    if (this.xtxInv) {
      for (let i = 0; i < x0.length; i++) {
        let rowSum = 0;
        for (let j = 0; j < x0.length; j++) {
          rowSum += x0[j] * this.xtxInv[j][i];
        }
        h0 += x0[i] * rowSum;
      }
    } else {
      h0 = 1 / (sample.length || 100);
    }
    h0 = Math.max(0, h0);

    // 信賴區間標準誤 SE_mean = s * sqrt(h0)
    const seMean = this.se * Math.sqrt(h0);
    // 預測區間標準誤 SE_pred = s * sqrt(1 + h0)
    const sePred = this.se * Math.sqrt(1 + h0);

    const marginCi = this.tCritical * seMean;
    const marginPi = this.tCritical * sePred;

    return {
      predicted: yHat,
      seMean: seMean,
      sePred: sePred,
      ciLower: yHat - marginCi,
      ciUpper: yHat + marginCi,
      piLower: yHat - marginPi,
      piUpper: yHat + marginPi
    };
  }

  evaluate(valData, targetCol) {
    const n = valData.length;
    let sse = 0, sae = 0;
    const yVals = valData.map(d => d[targetCol]);
    const yMean = mean(yVals);
    let sst = 0;

    for (let i = 0; i < n; i++) {
      const pred = this.predictRow(valData[i]);
      const act = yVals[i];
      const err = act - pred;
      sse += err * err;
      sae += Math.abs(err);
      sst += Math.pow(act - yMean, 2);
    }

    const mse = sse / n;
    const r2 = sst > 0 ? 1 - (sse / sst) : 0;
    const adjR2 = 1 - (1 - r2) * (n - 1) / Math.max(1, n - this.p - 1);

    return {
      MAE: sae / n,
      MSE: mse,
      RMSE: Math.sqrt(mse),
      R2: r2,
      Adj_R2: adjR2
    };
  }
}

/**
 * 自回歸模型 (Auto Regression AR(p))
 */
class AutoRegressionModel {
  constructor(lags = 3) {
    this.lags = lags;
    this.params = []; // [c, phi_1, ... phi_p]
    this.sigma = 0;
  }

  fit(series) {
    const p = this.lags;
    const n = series.length;
    if (n <= p + 1) return;

    const X = [];
    const y = [];
    for (let t = p; t < n; t++) {
      const row = [1];
      for (let lag = 1; lag <= p; lag++) {
        row.push(series[t - lag]);
      }
      X.push(row);
      y.push(series[t]);
    }

    const Xt = transpose(X);
    const XtX = matMul(Xt, X);
    const inv = invertMatrix(XtX) || identityMatrix(p + 1);
    const Xty = matMulVec(Xt, y);
    this.params = matMulVec(inv, Xty);

    let sse = 0;
    for (let i = 0; i < y.length; i++) {
      let pred = this.params[0];
      for (let j = 1; j <= p; j++) {
        pred += this.params[j] * X[i][j];
      }
      sse += Math.pow(y[i] - pred, 2);
    }
    this.sigma = Math.sqrt(sse / Math.max(1, y.length - p - 1));
    return this;
  }

  forecast(historySeries, steps = 30) {
    const p = this.lags;
    const buffer = [...historySeries.slice(-p)];
    const forecasts = [];
    const lowerPi = [];
    const upperPi = [];

    for (let step = 0; step < steps; step++) {
      let nextVal = this.params[0] || 0.5;
      for (let j = 1; j <= p; j++) {
        nextVal += (this.params[j] || 0) * buffer[buffer.length - j];
      }
      forecasts.push(nextVal);
      buffer.push(nextVal);

      // 隨步長擴展的不確定性預測區間
      const seStep = this.sigma * Math.sqrt(1 + step * 0.05);
      lowerPi.push(nextVal - 1.96 * seStep);
      upperPi.push(nextVal + 1.96 * seStep);
    }

    return { forecasts, lowerPi, upperPi };
  }
}

// 訓練與更新所有模型
function trainAllModels() {
  const train = state.trainData;
  const target = state.target;

  // 1. 單純線性回歸
  state.models.simple = new LinearRegressionModel([state.simpleFeature]).fit(train, target);

  // 2. 多元線性回歸 (全特徵)
  state.models.mlrAll = new LinearRegressionModel(state.features).fit(train, target);

  // 3. 多元線性回歸 (選定特徵)
  const activeFeatures = state.selectedFeatures.length > 0 ? state.selectedFeatures : [state.simpleFeature];
  state.models.mlrSelected = new LinearRegressionModel(activeFeatures).fit(train, target);

  // 4. 自回歸模型
  const timeSeries = train.map(d => d[target]);
  state.models.ar = new AutoRegressionModel(state.arLags).fit(timeSeries);

  console.log("所有模型訓練完成，指標就緒");
}

function retrainCurrentModels() {
  const selectElem = document.getElementById("ar-lag-select");
  if (selectElem) state.arLags = parseInt(selectElem.value, 10);

  trainAllModels();
  renderActivePlot();
  renderEvaluationTable();
  renderEvaluationCharts();
  updateSimulatorInference();

  const btn = document.getElementById("btn-retrain-all");
  if (btn) {
    const originalText = btn.innerHTML;
    btn.innerHTML = "✅ 模型已更新重算！";
    setTimeout(() => { btn.innerHTML = originalText; }, 1200);
  }
}

// ============================================================
// 3. 特徵選擇互動控制 (Feature Selection Studio)
// ============================================================

function initFeatureSelection() {
  const container = document.getElementById("feature-checkboxes-container");
  if (!container) return;

  container.innerHTML = "";
  state.features.forEach(feat => {
    const isSelected = state.selectedFeatures.includes(feat);
    const item = document.createElement("label");
    item.className = `feature-check-item ${isSelected ? "selected" : ""}`;
    item.id = `feat-check-${feat}`;
    item.innerHTML = `
      <input type="checkbox" value="${feat}" ${isSelected ? "checked" : ""} onchange="toggleFeatureCheckbox(this)">
      <span>${feat}</span>
    `;
    container.appendChild(item);
  });

  updateActiveFeaturesCount();
}

function toggleFeatureCheckbox(checkbox) {
  const feat = checkbox.value;
  const parent = checkbox.closest(".feature-check-item");
  if (checkbox.checked) {
    if (!state.selectedFeatures.includes(feat)) state.selectedFeatures.push(feat);
    if (parent) parent.classList.add("selected");
  } else {
    state.selectedFeatures = state.selectedFeatures.filter(f => f !== feat);
    if (parent) parent.classList.remove("selected");
  }
  updateActiveFeaturesCount();
}

function updateActiveFeaturesCount() {
  const badge = document.getElementById("active-features-count");
  if (badge) badge.innerText = `已勾選 ${state.selectedFeatures.length} 項`;
  const kpiFeat = document.getElementById("kpi-features");
  if (kpiFeat) kpiFeat.innerText = `${state.selectedFeatures.length} / 20`;
}

function onTopKSliderChange(val) {
  const k = parseInt(val, 10);
  const badge = document.getElementById("val-top-k");
  if (badge) badge.innerText = k;

  state.selectedFeatures = getTopKFeatures(k);
  syncFeatureCheckboxes();
  updateActiveFeaturesCount();
}

function syncFeatureCheckboxes() {
  state.features.forEach(feat => {
    const item = document.getElementById(`feat-check-${feat}`);
    if (item) {
      const chk = item.querySelector("input");
      const isSel = state.selectedFeatures.includes(feat);
      if (chk) chk.checked = isSel;
      if (isSel) item.classList.add("selected");
      else item.classList.remove("selected");
    }
  });
}

function selectAllFeatures() {
  state.selectedFeatures = [...state.features];
  syncFeatureCheckboxes();
  updateActiveFeaturesCount();
  applyFeatureSelection();
}

function quickSelectSet(setName) {
  if (setName === 'top5') {
    state.selectedFeatures = getTopKFeatures(5);
  } else if (setName === 'top10') {
    state.selectedFeatures = getTopKFeatures(10);
  } else if (setName === 'infrastructure') {
    state.selectedFeatures = ["DamsQuality", "DrainageSystems", "DeterioratingInfrastructure", "RiverManagement", "TopographyDrainage"];
  } else if (setName === 'climate') {
    state.selectedFeatures = ["MonsoonIntensity", "ClimateChange", "Landslides", "Watersheds", "Deforestation"];
  }
  syncFeatureCheckboxes();
  updateActiveFeaturesCount();
  applyFeatureSelection();
}

function applyFeatureSelection() {
  trainAllModels();
  renderActivePlot();
  renderFeatureRankChart();
  renderEvaluationTable();
  renderEvaluationCharts();
  updateSimulatorInference();

  const btn = document.getElementById("btn-apply-features");
  if (btn) {
    const orig = btn.innerText;
    btn.innerText = "✨ 特徵已成功套用！";
    setTimeout(() => { btn.innerText = orig; }, 1200);
  }
}

// 繪製特徵排行水平長條圖
function renderFeatureRankChart() {
  const ctx = document.getElementById("feature-rank-chart");
  if (!ctx || !window.Chart) return;

  const sorted = [...state.features].sort((a, b) => (state.featureScores[a] || 0) - (state.featureScores[b] || 0));
  const labels = sorted;
  const data = sorted.map(f => state.featureScores[f] || 0);
  const colors = sorted.map(f => state.selectedFeatures.includes(f) ? "#06b6d4" : "#475569");

  if (chartInstances.featureRank) {
    chartInstances.featureRank.destroy();
  }

  chartInstances.featureRank = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "F-Score 評分",
        data: data,
        backgroundColor: colors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` F-Score: ${ctx.parsed.x.toFixed(2)} (${state.selectedFeatures.includes(ctx.label) ? "已選入" : "未選入"})`
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#cbd5e1", font: { size: 11 } }
        }
      }
    }
  });
}

// ============================================================
// 4. 線性回歸預測圖與 95% 信賴/預測區間可視化 (Core Visualizer)
// ============================================================

function initSimpleFeatureDropdown() {
  const select = document.getElementById("simple-feature-select");
  if (!select) return;

  select.innerHTML = "";
  state.features.forEach(feat => {
    const opt = document.createElement("option");
    opt.value = feat;
    opt.innerText = feat;
    if (feat === state.simpleFeature) opt.selected = true;
    select.appendChild(opt);
  });
}

function onSimpleFeatureChange() {
  const select = document.getElementById("simple-feature-select");
  if (select) {
    state.simpleFeature = select.value;
    state.models.simple = new LinearRegressionModel([state.simpleFeature]).fit(state.trainData, state.target);
    renderActivePlot();
  }
}

function togglePredictionInterval() {
  state.showPredictionInterval = !state.showPredictionInterval;
  renderActivePlot();
}

function resetChartZoom() {
  if (chartInstances.mainPlot) {
    chartInstances.mainPlot.resetZoom?.();
    renderActivePlot();
  }
}

function renderActivePlot() {
  const modeSelect = document.getElementById("plot-view-mode");
  if (modeSelect) state.currentPlotMode = modeSelect.value;

  const lblSimple = document.getElementById("lbl-simple-feature");
  const selSimple = document.getElementById("simple-feature-select");
  if (lblSimple && selSimple) {
    const isSimple = state.currentPlotMode === "simple-fit";
    lblSimple.style.display = isSimple ? "inline-block" : "none";
    selSimple.style.display = isSimple ? "inline-block" : "none";
  }

  const ctx = document.getElementById("main-interactive-chart");
  if (!ctx || !window.Chart) return;
  if (chartInstances.mainPlot) chartInstances.mainPlot.destroy();

  if (state.currentPlotMode === "simple-fit") {
    renderSimpleLinearRegressionPlot(ctx);
  } else if (state.currentPlotMode === "multi-envelope") {
    renderMultiRegressionEnvelopePlot(ctx);
  } else if (state.currentPlotMode === "ar-forecast") {
    renderAutoRegressionPlot(ctx);
  } else if (state.currentPlotMode === "residuals") {
    renderResidualsPlot(ctx);
  }
}

// 4.1 單純線性回歸散布帶圖 (95% CI + 95% PI)
function renderSimpleLinearRegressionPlot(ctx) {
  const model = state.models.simple;
  const feat = state.simpleFeature;
  const valData = state.validData;

  // 1. 取樣散布點 (限制 150 點維持畫面清晰流暢)
  const scatterPoints = valData.slice(0, 150).map(d => ({
    x: d[feat],
    y: d[state.target]
  }));

  // 2. 建立平滑 x 網格以畫出漂亮的擬合線與 CI/PI 帶
  const minX = Math.min(...state.dataset.map(d => d[feat]));
  const maxX = Math.max(...state.dataset.map(d => d[feat]));
  const stepCount = 50;
  const stepSize = (maxX - minX) / stepCount;

  const fitLine = [];
  const ciUpper = [];
  const ciLower = [];
  const piUpper = [];
  const piLower = [];

  for (let i = 0; i <= stepCount; i++) {
    const curX = minX + i * stepSize;
    const sample = {};
    sample[feat] = curX;
    const res = model.predictWithIntervals(sample);

    fitLine.push({ x: curX, y: res.predicted });
    ciUpper.push({ x: curX, y: res.ciUpper });
    ciLower.push({ x: curX, y: res.ciLower });
    piUpper.push({ x: curX, y: res.piUpper });
    piLower.push({ x: curX, y: res.piLower });
  }

  const datasets = [
    {
      type: "scatter",
      label: "實際觀測點 (Actual Data)",
      data: scatterPoints,
      backgroundColor: "rgba(148, 163, 184, 0.6)",
      borderColor: "transparent",
      pointRadius: 4,
      pointHoverRadius: 6,
      order: 3
    },
    {
      type: "line",
      label: "線性回歸擬合線 (Fitted Line)",
      data: fitLine,
      borderColor: "#2563eb",
      borderWidth: 2.5,
      pointRadius: 0,
      fill: false,
      order: 1
    },
    {
      type: "line",
      label: "95% 信賴區間上限 (CI Upper)",
      data: ciUpper,
      borderColor: "transparent",
      pointRadius: 0,
      fill: "+1",
      backgroundColor: "rgba(56, 189, 248, 0.28)",
      order: 2
    },
    {
      type: "line",
      label: "95% 信賴區間下限 (CI Lower)",
      data: ciLower,
      borderColor: "transparent",
      pointRadius: 0,
      fill: false,
      order: 2
    }
  ];

  if (state.showPredictionInterval) {
    datasets.push(
      {
        type: "line",
        label: "95% 預測區間上限 (PI Upper)",
        data: piUpper,
        borderColor: "#ef4444",
        borderDash: [5, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        order: 4
      },
      {
        type: "line",
        label: "95% 預測區間下限 (PI Lower)",
        data: piLower,
        borderColor: "#ef4444",
        borderDash: [5, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        order: 4
      }
    );
  }

  chartInstances.mainPlot = new Chart(ctx, {
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              if (ctx.dataset.type === "scatter") {
                return ` 實測: ${feat}=${ctx.parsed.x.toFixed(1)}, 洪水機率=${ctx.parsed.y.toFixed(4)}`;
              }
              return ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`;
            }
          }
        }
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: `${feat} 特徵因子評分`, color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        },
        y: {
          title: { display: true, text: "FloodProbability (預估洪水機率)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        }
      }
    }
  });
}

// 4.2 多元線性回歸排序包絡線圖 (帶狀 CI/PI)
function renderMultiRegressionEnvelopePlot(ctx) {
  const model = state.models.mlrSelected;
  const valData = state.validData;

  const evaluated = valData.map((row, idx) => {
    const res = model.predictWithIntervals(row);
    return {
      actual: row[state.target],
      pred: res.predicted,
      ciLow: res.ciLower,
      ciUp: res.ciUpper,
      piLow: res.piLower,
      piUp: res.piUpper
    };
  });

  // 依預測值升序排序，形成平滑趨勢帶
  evaluated.sort((a, b) => a.pred - b.pred);

  const step = Math.max(1, Math.floor(evaluated.length / 80));
  const sampled = [];
  for (let i = 0; i < evaluated.length; i += step) sampled.push(evaluated[i]);

  const labels = sampled.map((_, i) => i);
  const actuals = sampled.map(d => d.actual);
  const preds = sampled.map(d => d.pred);
  const ciUp = sampled.map(d => d.ciUp);
  const ciLow = sampled.map(d => d.ciLow);
  const piUp = sampled.map(d => d.piUp);
  const piLow = sampled.map(d => d.piLow);

  const datasets = [
    {
      type: "scatter",
      label: "實際觀測值 (Actual)",
      data: actuals.map((y, x) => ({ x, y })),
      backgroundColor: "rgba(56, 189, 248, 0.7)",
      pointRadius: 3
    },
    {
      type: "line",
      label: "多元回歸預測值 (Predicted)",
      data: preds,
      borderColor: "#2563eb",
      borderWidth: 2,
      pointRadius: 0
    },
    {
      type: "line",
      label: "95% 信賴區間帶 (CI)",
      data: ciUp,
      borderColor: "transparent",
      fill: "+1",
      backgroundColor: "rgba(56, 189, 248, 0.25)",
      pointRadius: 0
    },
    {
      type: "line",
      label: "95% CI 下限",
      data: ciLow,
      borderColor: "transparent",
      fill: false,
      pointRadius: 0
    }
  ];

  if (state.showPredictionInterval) {
    datasets.push(
      {
        type: "line",
        label: "95% 預測區間上限 (PI)",
        data: piUp,
        borderColor: "#ef4444",
        borderDash: [4, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false
      },
      {
        type: "line",
        label: "95% 預測區間下限 (PI)",
        data: piLow,
        borderColor: "#ef4444",
        borderDash: [4, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false
      }
    );
  }

  chartInstances.mainPlot = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          title: { display: true, text: "排序樣本序號 (Sorted Validation Samples)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        },
        y: {
          title: { display: true, text: "FloodProbability (洪水機率)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        }
      }
    }
  });
}

// 4.3 自回歸模型時序預測圖 (AR Forecast with Expanding Band)
function renderAutoRegressionPlot(ctx) {
  const arModel = state.models.ar;
  const historySeries = state.trainData.map(d => d[state.target]).slice(-40);
  const actualValidation = state.validData.map(d => d[state.target]).slice(0, 25);
  const steps = actualValidation.length;

  const { forecasts, lowerPi, upperPi } = arModel.forecast(historySeries, steps);

  const histLabels = historySeries.map((_, i) => `T-${historySeries.length - i}`);
  const futureLabels = Array.from({ length: steps }, (_, i) => `T+${i + 1}`);
  const allLabels = [...histLabels, ...futureLabels];

  // 拼接時序曲線
  const histData = [...historySeries, ...Array(steps).fill(null)];
  const actualData = [...Array(historySeries.length).fill(null), ...actualValidation];
  const fcData = [...Array(historySeries.length - 1).fill(null), historySeries[historySeries.length - 1], ...forecasts];
  const piUpData = [...Array(historySeries.length).fill(null), ...upperPi];
  const piLowData = [...Array(historySeries.length).fill(null), ...lowerPi];

  chartInstances.mainPlot = new Chart(ctx, {
    type: "line",
    data: {
      labels: allLabels,
      datasets: [
        {
          label: "歷史序列觀測 (Historical)",
          data: histData,
          borderColor: "#64748b",
          borderWidth: 2,
          pointRadius: 2
        },
        {
          label: "驗證期真實觀測 (Ground Truth)",
          data: actualData,
          borderColor: "#0284c7",
          borderWidth: 2,
          pointRadius: 3
        },
        {
          label: `AR(${state.arLags}) 外推預測 (Forecast)`,
          data: fcData,
          borderColor: "#8b5cf6",
          borderWidth: 2.5,
          borderDash: [5, 4],
          pointRadius: 3
        },
        {
          label: "95% 預測區間上限",
          data: piUpData,
          borderColor: "transparent",
          fill: "+1",
          backgroundColor: "rgba(139, 92, 246, 0.2)",
          pointRadius: 0
        },
        {
          label: "95% 預測區間下限",
          data: piLowData,
          borderColor: "transparent",
          fill: false,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          title: { display: true, text: "時間期數步長 (Time Steps)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        },
        y: {
          title: { display: true, text: "FloodProbability (洪水機率)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        }
      }
    }
  });
}

// 4.4 殘差診斷圖 (Residuals vs Fitted)
function renderResidualsPlot(ctx) {
  const model = state.models.mlrSelected;
  const valData = state.validData.slice(0, 180);

  const resPoints = valData.map(d => {
    const pred = model.predictRow(d);
    return {
      x: pred,
      y: d[state.target] - pred
    };
  });

  chartInstances.mainPlot = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "殘差觀測點",
          data: resPoints,
          backgroundColor: "rgba(6, 182, 212, 0.6)",
          pointRadius: 3
        },
        {
          type: "line",
          label: "殘差基準線 (零殘差)",
          data: [{ x: 0.35, y: 0 }, { x: 0.75, y: 0 }],
          borderColor: "#ef4444",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          title: { display: true, text: "擬合值 (Fitted Values)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        },
        y: {
          title: { display: true, text: "殘差 (Residuals = Actual - Fitted)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        }
      }
    }
  });
}

// ============================================================
// 5. 模型評估展示 (Model Evaluation Table & Charts)
// ============================================================

function renderEvaluationTable() {
  const tbody = document.getElementById("evaluation-table-body");
  if (!tbody) return;

  const valData = state.validData;
  const target = state.target;

  const evalSimple = state.models.simple.evaluate(valData, target);
  const evalAll = state.models.mlrAll.evaluate(valData, target);
  const evalSel = state.models.mlrSelected.evaluate(valData, target);

  // AR 模型指標估算
  const arHistory = state.trainData.map(d => d[target]);
  const arActual = valData.map(d => d[target]);
  const arFc = state.models.ar.forecast(arHistory, arActual.length).forecasts;
  let arSse = 0, arSae = 0;
  const yMean = mean(arActual);
  let arSst = 0;
  for (let i = 0; i < arActual.length; i++) {
    const err = arActual[i] - arFc[i];
    arSse += err * err;
    arSae += Math.abs(err);
    arSst += Math.pow(arActual[i] - yMean, 2);
  }
  const evalAr = {
    MAE: arSae / arActual.length,
    MSE: arSse / arActual.length,
    RMSE: Math.sqrt(arSse / arActual.length),
    R2: arSst > 0 ? 1 - (arSse / arSst) : 0,
    Adj_R2: 1 - (1 - (1 - (arSse / arSst))) * (arActual.length - 1) / Math.max(1, arActual.length - state.arLags - 1)
  };

  const rows = [
    { name: `單純線性回歸 (Simple LR: ${state.simpleFeature})`, p: 1, ...evalSimple, badge: "基準單變量" },
    { name: "多元線性回歸 - 全特徵 (MLR All 20)", p: 20, ...evalAll, badge: "高精度全模型", isBest: true },
    { name: `多元線性回歸 - 特徵選擇 (MLR Top ${state.selectedFeatures.length})`, p: state.selectedFeatures.length, ...evalSel, badge: "精簡高效模型" },
    { name: `自回歸模型 (Auto Regression AR(${state.arLags}))`, p: state.arLags, ...evalAr, badge: "時序趨勢外推" }
  ];

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${r.name}</strong></td>
      <td>${r.p}</td>
      <td>${r.MAE.toFixed(5)}</td>
      <td>${r.MSE.toFixed(6)}</td>
      <td><span class="${r.isBest ? 'text-emerald' : ''}">${r.RMSE.toFixed(5)}</span></td>
      <td><strong>${r.R2.toFixed(4)}</strong></td>
      <td>${r.Adj_R2.toFixed(4)}</td>
      <td>
        <span class="${r.isBest ? 'eval-tag-best' : 'eval-tag-normal'}">${r.badge}</span>
      </td>
    </tr>
  `).join("");

  // 同步更新 KPI
  const kpiR2 = document.getElementById("kpi-r2");
  const kpiRmse = document.getElementById("kpi-rmse");
  if (kpiR2) kpiR2.innerText = evalAll.R2.toFixed(4);
  if (kpiRmse) kpiRmse.innerText = evalAll.RMSE.toFixed(4);
}

function renderEvaluationCharts() {
  const errCtx = document.getElementById("eval-error-chart");
  const r2Ctx = document.getElementById("eval-r2-chart");
  if (!errCtx || !r2Ctx || !window.Chart) return;

  const valData = state.validData;
  const target = state.target;

  const mSimple = state.models.simple.evaluate(valData, target);
  const mAll = state.models.mlrAll.evaluate(valData, target);
  const mSel = state.models.mlrSelected.evaluate(valData, target);

  const labels = ["單純 LR", "多元 LR (全)", `多元 LR (${state.selectedFeatures.length}特徵)`];
  const rmseData = [mSimple.RMSE, mAll.RMSE, mSel.RMSE];
  const maeData = [mSimple.MAE, mAll.MAE, mSel.MAE];
  const r2Data = [mSimple.R2, mAll.R2, mSel.R2];

  if (chartInstances.evalError) chartInstances.evalError.destroy();
  if (chartInstances.evalR2) chartInstances.evalR2.destroy();

  chartInstances.evalError = new Chart(errCtx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "RMSE (均方根誤差)", data: rmseData, backgroundColor: "#3b82f6", borderRadius: 4 },
        { label: "MAE (絕對誤差)", data: maeData, backgroundColor: "#10b981", borderRadius: 4 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#cbd5e1" }, grid: { display: false } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });

  chartInstances.evalR2 = new Chart(r2Ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "決定係數 (R²)", data: r2Data, backgroundColor: "#8b5cf6", borderRadius: 4 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#cbd5e1" }, grid: { display: false } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" }, min: 0, max: 1 }
      }
    }
  });
}

// ============================================================
// 6. 即時機率模擬器 (What-If Flood Risk Predictor)
// ============================================================

function initSimulatorSliders() {
  const container = document.getElementById("simulator-sliders-container");
  if (!container) return;

  container.innerHTML = "";
  state.features.forEach(feat => {
    const row = document.createElement("div");
    row.className = "sim-slider-row";
    row.innerHTML = `
      <div class="sim-slider-header">
        <span class="sim-slider-name">${feat}</span>
        <span class="sim-slider-val" id="sim-val-${feat}">5.0</span>
      </div>
      <input type="range" min="1" max="15" step="0.5" value="5" class="custom-slider" style="width: 100%;"
             id="sim-slider-${feat}" oninput="onSimulatorSliderInput('${feat}', this.value)">
    `;
    container.appendChild(row);
  });
}

function onSimulatorSliderInput(feat, val) {
  const valSpan = document.getElementById(`sim-val-${feat}`);
  if (valSpan) valSpan.innerText = parseFloat(val).toFixed(1);
  updateSimulatorInference();
}

function getSimulatorCurrentValues() {
  const values = {};
  state.features.forEach(feat => {
    const slider = document.getElementById(`sim-slider-${feat}`);
    values[feat] = slider ? parseFloat(slider.value) : 5.0;
  });
  return values;
}

function updateSimulatorInference() {
  const model = state.models.mlrSelected || state.models.mlrAll;
  if (!model) return;

  const currentValues = getSimulatorCurrentValues();
  const res = model.predictWithIntervals(currentValues);

  // 限制機率在 0 ~ 1 區間
  const prob = Math.max(0.0, Math.min(1.0, res.predicted));
  const ciLow = Math.max(0.0, res.ciLower);
  const ciUp = Math.min(1.0, res.ciUpper);
  const piLow = Math.max(0.0, res.piLower);
  const piUp = Math.min(1.0, res.piUpper);

  // 1. 更新數值標籤
  const probElem = document.getElementById("sim-predicted-prob");
  if (probElem) probElem.innerText = `${(prob * 100).toFixed(1)}%`;

  const ciElem = document.getElementById("sim-ci-range");
  if (ciElem) ciElem.innerText = `[${(ciLow * 100).toFixed(1)}% ~ ${(ciUp * 100).toFixed(1)}%]`;

  const piElem = document.getElementById("sim-pi-range");
  if (piElem) piElem.innerText = `[${(piLow * 100).toFixed(1)}% ~ ${(piUp * 100).toFixed(1)}%]`;

  // 2. 更新圓形儀表 (Gauge)
  const gaugeElem = document.getElementById("gauge-circle-elem");
  if (gaugeElem) {
    const deg = Math.round(prob * 360);
    gaugeElem.style.background = `conic-gradient(#06b6d4 0deg, #3b82f6 ${deg}deg, rgba(255, 255, 255, 0.08) ${deg}deg)`;
  }

  // 3. 更新風險等級徽章
  const badgeElem = document.getElementById("sim-risk-badge");
  if (badgeElem) {
    badgeElem.className = "risk-badge";
    if (prob < 0.45) {
      badgeElem.classList.add("risk-low");
      badgeElem.innerText = "低風險 (Low Risk)";
    } else if (prob < 0.55) {
      badgeElem.classList.add("risk-moderate");
      badgeElem.innerText = "中度警戒 (Moderate Risk)";
    } else if (prob < 0.65) {
      badgeElem.classList.add("risk-high");
      badgeElem.innerText = "高危險 (High Risk)";
    } else {
      badgeElem.classList.add("risk-extreme");
      badgeElem.innerText = "特大洪澇警報 (Extreme Risk)";
    }
  }

  const modelNote = document.getElementById("sim-active-model-name");
  if (modelNote) {
    modelNote.innerText = `多元線性回歸 (${model.features.length} 項特徵子集)`;
  }
}

function applyScenarioPreset(preset) {
  state.features.forEach(feat => {
    let targetVal = 5.0;
    if (preset === 'normal') {
      targetVal = 4.5 + Math.random();
    } else if (preset === 'typhoon') {
      if (["MonsoonIntensity", "ClimateChange", "Landslides", "RiverManagement"].includes(feat)) {
        targetVal = 12.0 + Math.random() * 2;
      } else {
        targetVal = 8.0 + Math.random();
      }
    } else if (preset === 'urban') {
      if (["Urbanization", "InadequatePlanning", "Encroachments", "DrainageSystems"].includes(feat)) {
        targetVal = 11.5 + Math.random() * 2;
      } else {
        targetVal = 6.0;
      }
    } else if (preset === 'good') {
      if (["DamsQuality", "DrainageSystems", "Watersheds"].includes(feat)) {
        targetVal = 13.0;
      } else {
        targetVal = 3.0;
      }
    }

    const slider = document.getElementById(`sim-slider-${feat}`);
    if (slider) slider.value = targetVal;
    const span = document.getElementById(`sim-val-${feat}`);
    if (span) span.innerText = targetVal.toFixed(1);
  });

  updateSimulatorInference();
}

function resetSimulatorSliders() {
  applyScenarioPreset('normal');
}

// ============================================================
// 7. 資料檢視與自訂 CSV 上傳
// ============================================================

function renderDataTable(query = "") {
  const thead = document.getElementById("data-table-head");
  const tbody = document.getElementById("data-table-body");
  if (!thead || !tbody) return;

  const showCols = ["id", ...state.features.slice(0, 7), state.target];
  thead.innerHTML = `<tr>${showCols.map(c => `<th>${c}</th>`).join("")}</tr>`;

  let rows = state.dataset;
  if (query) {
    const q = query.toLowerCase();
    rows = rows.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)));
  }

  tbody.innerHTML = rows.slice(0, 100).map(row => `
    <tr>
      ${showCols.map(c => {
        const val = row[c];
        if (c === state.target) {
          return `<td><strong class="text-cyan">${typeof val === 'number' ? val.toFixed(4) : val}</strong></td>`;
        }
        return `<td>${val ?? '-'}</td>`;
      }).join("")}
    </tr>
  `).join("");
}

function filterDataTable(val) {
  renderDataTable(val);
}

function handleCsvUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const text = e.target.result;
      const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
      if (lines.length < 2) throw new Error("CSV 資料列數不足");

      const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ''));
      const parsedRows = [];

      for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(",");
        if (parts.length === headers.length) {
          const row = {};
          headers.forEach((h, idx) => {
            const num = parseFloat(parts[idx]);
            row[h] = isNaN(num) ? parts[idx] : num;
          });
          parsedRows.push(row);
        }
      }

      if (parsedRows.length > 0) {
        state.dataset = parsedRows;
        const n = state.dataset.length;
        const trainSize = Math.floor(n * 0.8);
        state.trainData = state.dataset.slice(0, trainSize);
        state.validData = state.dataset.slice(trainSize);

        // 更新特徵清單
        const availFeat = headers.filter(h => h !== "id" && h !== state.target);
        if (availFeat.length > 0) {
          state.features = availFeat;
          state.selectedFeatures = availFeat.slice(0, Math.min(10, availFeat.length));
          state.simpleFeature = availFeat[0];
        }

        computeFeatureScoresLocally();
        initFeatureSelection();
        initSimpleFeatureDropdown();
        initSimulatorSliders();
        trainAllModels();
        renderActivePlot();
        renderFeatureRankChart();
        renderEvaluationTable();
        renderEvaluationCharts();
        renderDataTable();
        updateSimulatorInference();

        const badge = document.getElementById("dataset-status-badge");
        if (badge) badge.innerHTML = `<span class="status-dot"></span> 自訂資料 (${parsedRows.length} 筆)`;
        alert(`成功匯入自訂 CSV 檔案！共 ${parsedRows.length} 筆觀測值，已自動重新訓練所有模型。`);
      }
    } catch (err) {
      alert("解析 CSV 失敗：" + err.message);
    }
  };
  reader.readAsText(file);
}

function downloadSubmissionCsv() {
  window.open("submission.csv", "_blank");
}

function exportAnalysisReport() {
  const valData = state.validData;
  const target = state.target;
  const mAll = state.models.mlrAll.evaluate(valData, target);
  const mSel = state.models.mlrSelected.evaluate(valData, target);

  const report = `===============================================================
HydroPredict AI - 洪水回歸分析評估報告
===============================================================
分析時間: ${new Date().toLocaleString()}
分析樣本: ${state.dataset.length} 筆 (80% 訓練 / 20% 驗證)

【1. 特徵選擇 (Feature Selection)】
- 評估依據: F-Statistic 與 Pearson 相關係數
- 入選特徵 (Top ${state.selectedFeatures.length}):
  ${state.selectedFeatures.join(", ")}

【2. 模型評估 (Model Evaluation)】
- 多元線性回歸 (全特徵 20):
  * R²: ${mAll.R2.toFixed(5)}
  * Adj-R²: ${mAll.Adj_R2.toFixed(5)}
  * RMSE: ${mAll.RMSE.toFixed(5)}
  * MAE: ${mAll.MAE.toFixed(5)}

- 多元線性回歸 (特徵選擇 Top ${state.selectedFeatures.length}):
  * R²: ${mSel.R2.toFixed(5)}
  * Adj-R²: ${mSel.Adj_R2.toFixed(5)}
  * RMSE: ${mSel.RMSE.toFixed(5)}
  * MAE: ${mSel.MAE.toFixed(5)}

【3. 信賴區間與預測區間 (CI & PI)】
- 95% 信賴區間: 反映母體平均反應之期望估計範圍
- 95% 預測區間: 反映個別樣本之實際波動範圍 (含殘差擾動)
===============================================================`;

  const blob = new Blob([report], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "HydroPredict_Analysis_Report.txt";
  a.click();
}

// Lightbox
function openLightbox(src) {
  const modal = document.getElementById("image-lightbox");
  const img = document.getElementById("lightbox-img");
  if (modal && img) {
    img.src = src;
    modal.classList.add("active");
  }
}

function closeLightbox() {
  const modal = document.getElementById("image-lightbox");
  if (modal) modal.classList.remove("active");
}

// 分頁切換
function switchTab(tabId) {
  state.activeTab = tabId;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

  const targetBtn = document.getElementById(`tab-nav-${tabId}`);
  const targetPane = document.getElementById(`pane-${tabId}`);
  if (targetBtn) targetBtn.classList.add("active");
  if (targetPane) targetPane.classList.add("active");

  if (tabId === "plots") {
    setTimeout(renderActivePlot, 60);
  } else if (tabId === "features") {
    setTimeout(renderFeatureRankChart, 60);
  } else if (tabId === "models") {
    setTimeout(renderEvaluationCharts, 60);
  }
}

// ============================================================
// 8. 矩陣基本運算輔助函式 (Matrix Math Helpers)
// ============================================================

function mean(arr) {
  return arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;
}

function transpose(A) {
  const m = A.length, n = A[0].length;
  const At = Array.from({ length: n }, () => Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      At[j][i] = A[i][j];
    }
  }
  return At;
}

function matMul(A, B) {
  const m = A.length, p = A[0].length, n = B[0].length;
  const C = Array.from({ length: m }, () => Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < p; k++) {
      const aik = A[i][k];
      for (let j = 0; j < n; j++) {
        C[i][j] += aik * B[k][j];
      }
    }
  }
  return C;
}

function matMulVec(A, x) {
  const m = A.length, n = x.length;
  const y = Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      y[i] += A[i][j] * x[j];
    }
  }
  return y;
}

function identityMatrix(n) {
  const I = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}

function invertMatrix(M) {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...identityMatrix(n)[i]]);

  for (let i = 0; i < n; i++) {
    // 主元選取 (Partial Pivoting)
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k;
    }
    if (Math.abs(A[maxRow][i]) < 1e-12) return null; // 奇異矩陣

    // 交換行
    const temp = A[i];
    A[i] = A[maxRow];
    A[maxRow] = temp;

    // 將對角項除為 1
    const pivot = A[i][i];
    for (let j = 0; j < 2 * n; j++) A[i][j] /= pivot;

    // 消去其他行
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = A[k][i];
        for (let j = 0; j < 2 * n; j++) {
          A[k][j] -= factor * A[i][j];
        }
      }
    }
  }

  return A.map(row => row.slice(n));
}
