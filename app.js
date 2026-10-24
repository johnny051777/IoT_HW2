/**
 * ML Predictor Studio (app.js)
 * 支援多資料集問題切換 (CRISP-DM 洪水機率預測 & Kaggle 瑞士房價預測)
 * 支援任意遠端 CSV URL 載入 (如 50_Startups.csv) 與本地 CSV 上傳
 * 具備 OLS 線性回歸、特徵選擇、95% 信賴/預測區間推導與即時 What-If 模擬引擎
 */

// ============================================================
// 問題與資料集中繼定義 (Problem Configurations)
// ============================================================

const PROBLEMS = {
  flood: {
    id: "flood",
    name: "洪水機率預測 (Flood Probability)",
    title: "HydroPredict AI - 洪水回歸分析與互動預測工作台",
    subtitle: "線性回歸・特徵選擇・95% 信賴/預測區間・CRISP-DM 流程",
    dataFile: "dataset_sample.json",
    csvFile: "train.csv",
    target: "FloodProbability",
    targetLabel: "洪水機率 (Flood Probability)",
    targetUnit: "%",
    isProbability: true,
    isCurrency: false,
    kaggleUrl: "https://www.kaggle.com/competitions/playground-series-s4e5",
    badge: "基準水文資料集 (5,000 筆)",
    features: [
      "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
      "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
      "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
      "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
      "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
      "InadequatePlanning", "PoliticalFactors"
    ],
    simpleFeature: "Landslides",
    scenarioPresets: [
      { id: "normal", name: "🌿 正常平水情境", desc: "各項因子處於標準常態平均" },
      { id: "typhoon", name: "🌀 暴雨颱風警報", desc: "季風強度、氣候變遷與山崩風險飆高" },
      { id: "urban", name: "🏙️ 都市排水失效", desc: "過度都市化、規劃不足且排水不良" },
      { id: "good", name: "🛡️ 完善防汛水利", desc: "水壩與集水區設施維護優良" }
    ]
  },

  swiss: {
    id: "swiss",
    name: "瑞士房價預測 (Swiss Housing Price)",
    title: "SwissHousing AI - 瑞士房價回歸預測工作台",
    subtitle: "Kaggle i3-ml-21-22-mini-project 房地產回歸模型與區間預測",
    dataFile: "swiss_housing_sample.json",
    csvFile: "swiss_housing_sample.csv",
    target: "PriceCHF",
    targetLabel: "房屋售價 (Price CHF 瑞士法郎)",
    targetUnit: "CHF",
    isProbability: false,
    isCurrency: true,
    kaggleUrl: "https://www.kaggle.com/competitions/i3-ml-21-22-mini-project",
    badge: "瑞士房價 Kaggle 競賽 (1,200 筆)",
    features: [
      "LivingSpace", "Rooms", "Bathrooms", "Floor", "YearBuilt",
      "DistanceCityCenter", "HasBalcony", "ParkingSpaces",
      "MountainViewScore", "EnergyEfficiency", "TaxMultiplier"
    ],
    simpleFeature: "LivingSpace",
    scenarioPresets: [
      { id: "luxury", name: "🏰 阿爾卑斯山景豪宅", desc: "大坪數、絕佳山景、高能源效率" },
      { id: "urban_apt", name: "🏙️ 蘇黎世市區公寓", desc: "近市中心、中坪數生活機能佳" },
      { id: "suburban", name: "🏡 近郊花園住宅", desc: "透天大空間、附車位與陽台" },
      { id: "studio", name: "🏢 單身/學生小套房", desc: "低坪數、小巧實用市中心套房" }
    ]
  },

  custom: {
    id: "custom",
    name: "自訂資料集 (Custom Dataset)",
    title: "ML Custom Dataset Predictor",
    subtitle: "支援任意遠端 CSV URL 或本地檔案之回歸模型推算",
    dataFile: "",
    csvFile: "",
    target: "",
    targetLabel: "目標預測值",
    targetUnit: "",
    isProbability: false,
    isCurrency: false,
    kaggleUrl: "",
    badge: "自訂外部資料集",
    features: [],
    simpleFeature: "",
    scenarioPresets: [
      { id: "mean", name: "📊 樣本均值設定", desc: "所有數值設為資料集平均值" },
      { id: "high", name: "📈 高數值評估", desc: "所有自變數調整至 75 百分位" },
      { id: "low", name: "📉 低數值評估", desc: "所有自變數調整至 25 百分位" }
    ]
  }
};

// ============================================================
// 全域狀態管理 (Global State)
// ============================================================

const state = {
  currentProblemId: "flood",
  features: [...PROBLEMS.flood.features],
  target: PROBLEMS.flood.target,
  dataset: [],            // 完整資料列
  trainData: [],
  validData: [],
  selectedFeatures: [],   // 目前選取的特徵子集
  activeTab: "plots",
  showPredictionInterval: true,
  currentPlotMode: "simple-fit",
  simpleFeature: PROBLEMS.flood.simpleFeature,
  arLags: 3,

  // 數值範圍快取 (用於滑桿動態範圍配置)
  featureStats: {},
  targetStats: {},

  // 模型擬合實例快取
  models: {
    simple: null,
    mlrAll: null,
    mlrSelected: null,
    ar: null
  },

  // 特徵 F-score 統計快取
  featureScores: {}
};

// Chart.js 實例字典
const chartInstances = {
  mainPlot: null,
  featureRank: null,
  evalError: null,
  evalR2: null
};

// ============================================================
// 1. 初始化與問題載入主控
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
  console.log("ML Predictor Studio 初始化中...");
  await handleProblemChange("flood", false);
});

/**
 * 切換分析題目 / 資料集
 */
async function handleProblemChange(problemId, updateSelect = true) {
  state.currentProblemId = problemId;
  const cfg = PROBLEMS[problemId] || PROBLEMS.flood;

  if (updateSelect) {
    const sel = document.getElementById("problem-select");
    if (sel) sel.value = problemId;
  }

  // 更新標題與副標
  const titleElem = document.getElementById("app-main-title");
  const subElem = document.getElementById("app-main-subtitle");
  if (titleElem) titleElem.innerText = cfg.title;
  if (subElem) subElem.innerText = cfg.subtitle;

  // 更新 Kaggle 競賽連結按鈕
  const kLinkHeader = document.getElementById("link-kaggle-header");
  if (kLinkHeader) {
    if (cfg.kaggleUrl) {
      kLinkHeader.href = cfg.kaggleUrl;
      kLinkHeader.style.display = "inline-flex";
    } else {
      kLinkHeader.style.display = "none";
    }
  }

  // 載入對應資料
  if (problemId === "flood") {
    await loadFloodProblemData();
  } else if (problemId === "swiss") {
    await loadSwissProblemData();
  } else if (problemId === "custom") {
    const inputUrl = document.getElementById("dataset-url-input");
    if (inputUrl && !inputUrl.value.trim()) {
      inputUrl.value = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv";
    }
    await loadDatasetFromUrl();
    return;
  }

  updateQuickChipsHighlight();
  rebuildWorkspace();
}

/**
 * 快捷切換預設資料集
 */
function quickLoadPreset(type) {
  if (type === "flood") {
    handleProblemChange("flood");
  } else if (type === "swiss") {
    handleProblemChange("swiss");
  } else if (type === "startups") {
    const inputUrl = document.getElementById("dataset-url-input");
    if (inputUrl) {
      inputUrl.value = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv";
    }
    loadDatasetFromUrl();
  }
}

function updateQuickChipsHighlight() {
  const p = state.currentProblemId;
  document.getElementById("chip-preset-flood")?.classList.toggle("active", p === "flood");
  document.getElementById("chip-preset-swiss")?.classList.toggle("active", p === "swiss");
  document.getElementById("chip-preset-startups")?.classList.toggle("active", p === "custom");
}

/**
 * 載入洪水資料
 */
async function loadFloodProblemData() {
  const cfg = PROBLEMS.flood;
  state.features = [...cfg.features];
  state.target = cfg.target;
  state.simpleFeature = cfg.simpleFeature;

  try {
    const res = await fetch("dataset_sample.json");
    if (res.ok) {
      state.dataset = await res.json();
    } else {
      generateLocalFloodSyntheticData(400);
    }
  } catch (e) {
    generateLocalFloodSyntheticData(400);
  }

  splitTrainValidationData();

  try {
    const resMeta = await fetch("model_results.json");
    if (resMeta.ok) {
      const meta = await resMeta.json();
      state.featureScores = meta.feature_selection.f_scores || {};
      state.selectedFeatures = meta.feature_selection.selected_top10 || state.features.slice(0, 10);
      return;
    }
  } catch (e) {}

  computeFeatureScoresLocally();
  state.selectedFeatures = getTopKFeatures(10);
}

/**
 * 載入瑞士房價資料 (Kaggle i3-ml-21-22)
 */
async function loadSwissProblemData() {
  const cfg = PROBLEMS.swiss;
  state.features = [...cfg.features];
  state.target = cfg.target;
  state.simpleFeature = cfg.simpleFeature;

  try {
    const res = await fetch("swiss_housing_sample.json");
    if (res.ok) {
      state.dataset = await res.json();
    } else {
      await loadSwissCsvFallback();
    }
  } catch (e) {
    await loadSwissCsvFallback();
  }

  splitTrainValidationData();
  computeFeatureScoresLocally();
  state.selectedFeatures = getTopKFeatures(Math.min(8, state.features.length));
}

async function loadSwissCsvFallback() {
  try {
    const res = await fetch("swiss_housing_sample.csv");
    if (res.ok) {
      const text = await res.text();
      const parsed = parseCsvText(text);
      if (parsed.rows.length > 0) {
        state.dataset = parsed.rows;
        return;
      }
    }
  } catch (e) {}
  generateLocalSwissSyntheticData(500);
}

/**
 * 劃分 80% 訓練集與 20% 驗證集
 */
function splitTrainValidationData() {
  const n = state.dataset.length;
  const trainSize = Math.max(10, Math.floor(n * 0.8));
  state.trainData = state.dataset.slice(0, trainSize);
  state.validData = state.dataset.slice(trainSize);
  calculateDatasetStats();
}

/**
 * 計算各特徵與目標之統計極值 (Min, Max, Mean, Step)
 */
function calculateDatasetStats() {
  state.featureStats = {};
  state.features.forEach(feat => {
    const vals = state.dataset.map(d => Number(d[feat])).filter(v => !isNaN(v));
    if (vals.length === 0) {
      state.featureStats[feat] = { min: 0, max: 10, mean: 5, step: 1 };
      return;
    }
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const m = mean(vals);
    const span = max - min;
    const step = span > 50 ? 1 : (span > 5 ? 0.5 : 0.05);
    state.featureStats[feat] = { min, max, mean: m, step };
  });

  const tVals = state.dataset.map(d => Number(d[state.target])).filter(v => !isNaN(v));
  if (tVals.length > 0) {
    state.targetStats = {
      min: Math.min(...tVals),
      max: Math.max(...tVals),
      mean: mean(tVals)
    };
  }
}

/**
 * 重建整個視覺工作台 (訓練模型、渲染圖表與介面)
 */
function rebuildWorkspace() {
  updateTargetSelectOptions();
  updateHeaderBadges();
  initFeatureSelection();
  initSimpleFeatureDropdown();
  trainAllModels();
  initScenarioButtons();
  initSimulatorSliders();
  renderActivePlot();
  renderFeatureRankChart();
  renderEvaluationTable();
  renderEvaluationCharts();
  renderDataTable();
  updateSimulatorInference();
}

/**
 * 更新頂端目標變數選單
 */
function updateTargetSelectOptions() {
  const sel = document.getElementById("target-var-select");
  if (!sel) return;

  sel.innerHTML = "";
  const numericCols = getNumericColumnNames();
  numericCols.forEach(col => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.innerText = col;
    if (col === state.target) opt.selected = true;
    sel.appendChild(opt);
  });
}

function handleTargetVariableChange(newTarget) {
  if (newTarget === state.target) return;
  state.target = newTarget;
  state.features = getNumericColumnNames().filter(c => c !== "id" && c !== newTarget);
  state.selectedFeatures = state.features.slice(0, Math.min(10, state.features.length));
  state.simpleFeature = state.features[0] || "";

  calculateDatasetStats();
  computeFeatureScoresLocally();
  rebuildWorkspace();
}

/**
 * 取得資料集中所有數值型欄位名稱
 */
function getNumericColumnNames() {
  if (state.dataset.length === 0) return [state.target];
  const sample = state.dataset[0];
  return Object.keys(sample).filter(k => {
    const val = sample[k];
    return typeof val === "number" || (!isNaN(parseFloat(val)) && isFinite(val));
  });
}

/**
 * 更新頂端 Badge
 */
function updateHeaderBadges() {
  const badge = document.getElementById("dataset-status-badge");
  if (badge) {
    const cfg = PROBLEMS[state.currentProblemId];
    const n = state.dataset.length.toLocaleString();
    if (state.currentProblemId === "flood") {
      badge.innerHTML = `<span class="status-dot"></span> 基準水文資料集 (${n} 筆)`;
    } else if (state.currentProblemId === "swiss") {
      badge.innerHTML = `<span class="status-dot"></span> 瑞士房價 Kaggle (${n} 筆)`;
    } else {
      badge.innerHTML = `<span class="status-dot"></span> 自訂資料集 (${n} 筆)`;
    }
  }

  // 更新 KPI 卡片樣本數
  const kpiSamples = document.getElementById("kpi-samples");
  if (kpiSamples) kpiSamples.innerText = state.dataset.length.toLocaleString();
}

// ============================================================
// 2. 任意遠端 CSV URL 載入與本地 CSV 解析器
// ============================================================

/**
 * 從任意 URL 載入 CSV (如 50_Startups.csv)
 */
async function loadDatasetFromUrl() {
  const input = document.getElementById("dataset-url-input");
  const btn = document.getElementById("btn-load-url");
  if (!input) return;

  const url = input.value.trim();
  if (!url) {
    alert("請輸入有效的 CSV 檔案網址！");
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerText = "載入中...";
  }

  try {
    console.log(`正在從 URL 擷取資料集: ${url}`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status} 讀取失敗`);

    const text = await response.text();
    const { headers, rows } = parseCsvText(text);

    if (rows.length < 5) throw new Error("CSV 資料列數過少 (至少需要 5 筆觀測值)");

    state.currentProblemId = "custom";
    document.getElementById("problem-select").value = "custom";
    updateQuickChipsHighlight();

    // 智能識別目標變數 Target
    let detectedTarget = headers[headers.length - 1]; // 預設為最後一欄
    const priorityTargets = ["Profit", "Price", "price", "Target", "target", "PriceCHF", "FloodProbability", "y", "label"];
    for (const pt of priorityTargets) {
      if (headers.includes(pt)) {
        detectedTarget = pt;
        break;
      }
    }

    state.target = detectedTarget;
    state.dataset = rows;

    // 篩選自變數特徵 (排除 id 與 target)
    state.features = headers.filter(h => h !== "id" && h !== detectedTarget);
    state.selectedFeatures = state.features.slice(0, Math.min(10, state.features.length));
    state.simpleFeature = state.features[0] || "";

    splitTrainValidationData();
    computeFeatureScoresLocally();
    rebuildWorkspace();

    alert(`🎉 成功載入自訂遠端資料集！共解析 ${rows.length} 筆觀測值、${state.features.length} 項自變數特徵，預設目標為 [${state.target}]。已自動重新訓練所有模型！`);
  } catch (err) {
    console.error("載入遠端 CSV 失敗:", err);
    alert(`無法載入遠端資料集 (${err.message})。\n提示：若遭遇跨來源 (CORS) 限制，請確保使用 GitHub Raw 或支援 CORS 的 CDN 連結。`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = "載入 URL";
    }
  }
}

/**
 * 處理使用者本地檔案上傳 CSV
 */
function handleCsvUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const text = e.target.result;
      const { headers, rows } = parseCsvText(text);
      if (rows.length < 5) throw new Error("CSV 資料列數不足");

      state.currentProblemId = "custom";
      document.getElementById("problem-select").value = "custom";
      updateQuickChipsHighlight();

      // 目標識別
      let detectedTarget = headers[headers.length - 1];
      const priorityTargets = ["Profit", "Price", "price", "Target", "target", "PriceCHF", "FloodProbability"];
      for (const pt of priorityTargets) {
        if (headers.includes(pt)) {
          detectedTarget = pt;
          break;
        }
      }

      state.target = detectedTarget;
      state.dataset = rows;
      state.features = headers.filter(h => h !== "id" && h !== detectedTarget);
      state.selectedFeatures = state.features.slice(0, Math.min(10, state.features.length));
      state.simpleFeature = state.features[0] || "";

      splitTrainValidationData();
      computeFeatureScoresLocally();
      rebuildWorkspace();

      alert(`✅ 成功上傳本地 CSV 檔案！共匯入 ${rows.length} 筆觀測值，已自動重算所有回歸指標與區間帶。`);
    } catch (err) {
      alert("解析 CSV 失敗：" + err.message);
    }
  };
  reader.readAsText(file);
}

/**
 * 通用 CSV 字串解析器 (支援雙引號、類別變數 One-Hot 編碼轉換)
 */
function parseCsvText(csvText) {
  const lines = csvText.split(/\r?\n/).filter(l => l.trim().length > 0);
  if (lines.length < 2) return { headers: [], rows: [] };

  // 解析第一行 Header
  const rawHeaders = parseCsvLine(lines[0]);
  const rawRows = [];

  for (let i = 1; i < lines.length; i++) {
    const parts = parseCsvLine(lines[i]);
    if (parts.length === rawHeaders.length) {
      const row = {};
      rawHeaders.forEach((h, idx) => {
        row[h] = parts[idx];
      });
      rawRows.push(row);
    }
  }

  // 檢查欄位型別並將類別特徵 One-Hot 編碼 (例如 50_Startups.csv 中的 State 欄位)
  const finalHeaders = [];
  const processedRows = rawRows.map(r => ({ ...r }));

  rawHeaders.forEach(h => {
    // 檢查是否皆為數值
    let isNumeric = true;
    const distinctStrValues = new Set();

    for (let r of rawRows) {
      const rawVal = r[h];
      const num = parseFloat(rawVal);
      if (isNaN(num)) {
        isNumeric = false;
        distinctStrValues.add(String(rawVal).trim());
      }
    }

    if (isNumeric) {
      finalHeaders.push(h);
      processedRows.forEach(r => {
        r[h] = parseFloat(r[h]) || 0;
      });
    } else if (distinctStrValues.size <= 8 && distinctStrValues.size > 1) {
      // 類別項 One-Hot 編碼
      distinctStrValues.forEach(cat => {
        const dummyCol = `${h}_${cat.replace(/\s+/g, '_')}`;
        finalHeaders.push(dummyCol);
        processedRows.forEach(r => {
          r[dummyCol] = (String(r[h]).trim() === cat) ? 1 : 0;
        });
      });
    }
    // 其餘高基數文字欄位直接忽略
  });

  return { headers: finalHeaders, rows: processedRows };
}

/**
 * 輔助解析單行 CSV (處理雙引號與逗號)
 */
function parseCsvLine(line) {
  const result = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"' || char === "'") {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(cur.trim().replace(/^["']|["']$/g, ''));
      cur = "";
    } else {
      cur += char;
    }
  }
  result.push(cur.trim().replace(/^["']|["']$/g, ''));
  return result;
}

/**
 * 下載目前資料集為 CSV
 */
function downloadCurrentDatasetCsv() {
  if (state.dataset.length === 0) return;
  const cols = ["id", ...state.features, state.target];
  const headerLine = cols.join(",");
  const rowLines = state.dataset.map((row, idx) => {
    return cols.map(c => c === "id" ? (row.id ?? idx + 1) : (row[c] ?? "")).join(",");
  });

  const content = [headerLine, ...rowLines].join("\n");
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${state.currentProblemId}_dataset.csv`;
  a.click();
}

function downloadSubmissionCsv() {
  window.open("submission.csv", "_blank");
}

// ============================================================
// 3. 統計回歸運算核心 (OLS Regression with 95% CI & PI)
// ============================================================

class LinearRegressionModel {
  constructor(featureNames) {
    this.features = featureNames;
    this.p = featureNames.length;
    this.beta = [];              // [intercept, beta_1, ... beta_p]
    this.xtxInv = null;          // (X^T X)^(-1) 共變異數縮放矩陣
    this.mse = 0;                // 均方誤差 s^2
    this.se = 0;                 // 殘差標準誤差 s = sqrt(MSE)
    this.df = 1;                 // 自由度 n - p - 1
    this.tCritical = 1.96;
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

    if (n === 0 || p === 0) return this;

    // 建構 X (常數項 1) 與 y
    const X = [];
    const y = [];
    for (let i = 0; i < n; i++) {
      const row = [1];
      for (let j = 0; j < p; j++) {
        row.push(Number(data[i][this.features[j]]) || 0);
      }
      X.push(row);
      y.push(Number(data[i][targetCol]) || 0);
    }

    const Xt = transpose(X);
    const XtX = matMul(Xt, X);
    this.xtxInv = invertMatrix(XtX);

    // 奇異矩陣嶺回歸擾動保護
    if (!this.xtxInv) {
      for (let i = 0; i < XtX.length; i++) XtX[i][i] += 1e-4;
      this.xtxInv = invertMatrix(XtX) || identityMatrix(p + 1);
    }

    const Xty = matMulVec(Xt, y);
    this.beta = matMulVec(this.xtxInv, Xty);

    // 計算擬合殘差與指標
    let sse = 0;
    let sae = 0;
    const yMean = mean(y);
    let sst = 0;

    for (let i = 0; i < n; i++) {
      const pred = this.predictRow(data[i]);
      const act = y[i];
      const err = act - pred;
      sse += err * err;
      sae += Math.abs(err);
      sst += Math.pow(act - yMean, 2);
    }

    this.mse = sse / this.df;
    this.se = Math.sqrt(Math.max(1e-12, this.mse));
    this.mae = sae / n;
    this.rmse = Math.sqrt(sse / n);
    this.r2 = sst > 0 ? Math.max(0, 1 - (sse / sst)) : 0;
    this.adjR2 = Math.max(0, 1 - (1 - this.r2) * (n - 1) / this.df);

    return this;
  }

  predictRow(sample) {
    if (!this.beta || this.beta.length === 0) return 0;
    let val = this.beta[0];
    for (let j = 0; j < this.p; j++) {
      val += this.beta[j + 1] * (Number(sample[this.features[j]]) || 0);
    }
    return val;
  }

  /**
   * 推導 95% 信賴區間 (CI) 與 95% 預測區間 (PI)
   */
  predictWithIntervals(sample) {
    const x0 = [1];
    for (let j = 0; j < this.p; j++) {
      x0.push(Number(sample[this.features[j]]) || 0);
    }

    let yHat = this.beta[0];
    for (let i = 1; i < x0.length; i++) {
      yHat += x0[i] * (this.beta[i] || 0);
    }

    // Leverage 槓桿值 h0 = x0 (X^T X)^(-1) x0^T
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
      h0 = 1 / Math.max(10, state.trainData.length);
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
    if (n === 0) return { MAE: 0, MSE: 0, RMSE: 0, R2: 0, Adj_R2: 0 };

    let sse = 0, sae = 0;
    const yVals = valData.map(d => Number(d[targetCol]) || 0);
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
    const r2 = sst > 0 ? Math.max(0, 1 - (sse / sst)) : 0;
    const adjR2 = Math.max(0, 1 - (1 - r2) * (n - 1) / Math.max(1, n - this.p - 1));

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
    this.params = [];
    this.sigma = 0;
  }

  fit(series) {
    const p = this.lags;
    const n = series.length;
    if (n <= p + 1) return this;

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
        pred += (this.params[j] || 0) * X[i][j];
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

    const defaultMean = mean(historySeries) || 0.5;

    for (let step = 0; step < steps; step++) {
      let nextVal = this.params[0] || defaultMean;
      for (let j = 1; j <= p; j++) {
        nextVal += (this.params[j] || 0) * (buffer[buffer.length - j] || defaultMean);
      }
      forecasts.push(nextVal);
      buffer.push(nextVal);

      const seStep = (this.sigma || 0.05) * Math.sqrt(1 + step * 0.05);
      lowerPi.push(nextVal - 1.96 * seStep);
      upperPi.push(nextVal + 1.96 * seStep);
    }

    return { forecasts, lowerPi, upperPi };
  }
}

/**
 * 本地計算各特徵之 F-score
 */
function computeFeatureScoresLocally() {
  const y = state.trainData.map(d => Number(d[state.target]) || 0);
  const yMean = mean(y);
  const n = y.length;
  if (n < 3) return;

  state.featureScores = {};
  state.features.forEach(feat => {
    const x = state.trainData.map(d => Number(d[feat]) || 0);
    const xMean = mean(x);
    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - xMean;
      const dy = y[i] - yMean;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }
    const r = (denX > 0 && denY > 0) ? num / Math.sqrt(denX * denY) : 0;
    const r2 = r * r;
    const f = r2 < 0.999 ? (r2 * (n - 2)) / Math.max(1e-9, (1 - r2)) : 999;
    state.featureScores[feat] = Math.max(0, f);
  });
}

function getTopKFeatures(k = 10) {
  return [...state.features]
    .sort((a, b) => (state.featureScores[b] || 0) - (state.featureScores[a] || 0))
    .slice(0, Math.min(k, state.features.length));
}

/**
 * 訓練全架構模型
 */
function trainAllModels() {
  const tr = state.trainData;
  const tgt = state.target;

  // 1. 單純線性回歸 (Simple Linear Regression)
  const sf = state.simpleFeature || state.features[0];
  state.models.simple = new LinearRegressionModel([sf]).fit(tr, tgt);

  // 2. 多元線性回歸 - 全特徵 (MLR All)
  state.models.mlrAll = new LinearRegressionModel(state.features).fit(tr, tgt);

  // 3. 多元線性回歸 - 特徵選擇子集 (MLR Selected)
  const selFeats = state.selectedFeatures.length > 0 ? state.selectedFeatures : state.features.slice(0, 5);
  state.models.mlrSelected = new LinearRegressionModel(selFeats).fit(tr, tgt);

  // 4. 時序自回歸 (AutoReg AR)
  const series = tr.map(d => Number(d[tgt]) || 0);
  state.models.ar = new AutoRegressionModel(state.arLags).fit(series);

  // 更新 KPI 卡片
  const evalAll = state.models.mlrAll.evaluate(state.validData, tgt);
  const r2Elem = document.getElementById("kpi-r2");
  const rmseElem = document.getElementById("kpi-rmse");
  const featElem = document.getElementById("kpi-features");
  const featSub = document.getElementById("kpi-features-sub");

  if (r2Elem) r2Elem.innerText = evalAll.R2.toFixed(4);
  if (rmseElem) {
    if (evalAll.RMSE > 100) {
      rmseElem.innerText = Math.round(evalAll.RMSE).toLocaleString();
    } else {
      rmseElem.innerText = evalAll.RMSE.toFixed(4);
    }
  }
  if (featElem) featElem.innerText = `${state.selectedFeatures.length} / ${state.features.length}`;
  if (featSub) featSub.innerText = `選入 Top ${state.selectedFeatures.length} 特徵子集`;
}

// ============================================================
// 4. 特徵選擇面板 (Feature Selection Lab)
// ============================================================

function initFeatureSelection() {
  const container = document.getElementById("feature-checkbox-grid");
  if (!container) return;

  container.innerHTML = "";
  state.features.forEach(feat => {
    const isSelected = state.selectedFeatures.includes(feat);
    const score = (state.featureScores[feat] || 0).toFixed(1);

    const item = document.createElement("div");
    item.className = `feature-check-item ${isSelected ? "selected" : ""}`;
    item.id = `feat-check-${feat}`;
    item.onclick = (e) => {
      if (e.target.tagName !== "INPUT") {
        toggleFeatureCheckbox(feat);
      }
    };

    item.innerHTML = `
      <input type="checkbox" id="chk-${feat}" ${isSelected ? "checked" : ""} onchange="toggleFeatureCheckbox('${feat}')">
      <div class="feat-info">
        <label for="chk-${feat}" class="feat-name">${feat}</label>
        <span class="feat-score">F-score: ${score}</span>
      </div>
    `;
    container.appendChild(item);
  });

  updateActiveFeaturesCount();
}

function toggleFeatureCheckbox(feat) {
  const idx = state.selectedFeatures.indexOf(feat);
  if (idx >= 0) {
    if (state.selectedFeatures.length <= 1) {
      alert("至少需保留 1 項入選特徵！");
      return;
    }
    state.selectedFeatures.splice(idx, 1);
  } else {
    state.selectedFeatures.push(feat);
  }

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
      item.classList.toggle("selected", isSel);
    }
  });
}

function updateActiveFeaturesCount() {
  const span = document.getElementById("active-features-count");
  if (span) span.innerText = state.selectedFeatures.length;
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
  } else {
    state.selectedFeatures = getTopKFeatures(Math.min(7, state.features.length));
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

/**
 * 繪製特徵 F-score 長條排行圖
 */
function renderFeatureRankChart() {
  const ctx = document.getElementById("feature-rank-chart");
  if (!ctx || !window.Chart) return;

  const sorted = [...state.features].sort((a, b) => (state.featureScores[a] || 0) - (state.featureScores[b] || 0));
  const labels = sorted;
  const data = sorted.map(f => state.featureScores[f] || 0);
  const colors = sorted.map(f => state.selectedFeatures.includes(f) ? "#06b6d4" : "#475569");

  if (chartInstances.featureRank) chartInstances.featureRank.destroy();

  chartInstances.featureRank = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "F-Score 統計檢定分數",
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
            label: (c) => ` F-Score: ${c.parsed.x.toFixed(2)} (${state.selectedFeatures.includes(c.label) ? "已選入" : "未選入"})`
          }
        }
      },
      scales: {
        x: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#94a3b8" } },
        y: { grid: { display: false }, ticks: { color: "#cbd5e1", font: { size: 11 } } }
      }
    }
  });
}

// ============================================================
// 5. 線性回歸預測圖與 95% 信賴/預測區間可視化
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

  const canvas = document.getElementById("main-prediction-chart");
  if (!canvas || !window.Chart) return;
  const ctx = canvas.getContext("2d");

  if (chartInstances.mainPlot) chartInstances.mainPlot.destroy();

  if (state.currentPlotMode === "simple-fit") {
    renderSimpleRegressionPlot(ctx);
  } else if (state.currentPlotMode === "multi-envelope") {
    renderMultipleRegressionEnvelopePlot(ctx);
  } else if (state.currentPlotMode === "ar-forecast") {
    renderAutoRegressionPlot(ctx);
  } else if (state.currentPlotMode === "residuals") {
    renderResidualsPlot(ctx);
  }
}

/**
 * 5.1 單純線性回歸散布帶圖 (95% CI + 95% PI)
 */
function renderSimpleRegressionPlot(ctx) {
  const sf = state.simpleFeature || state.features[0];
  const model = state.models.simple;
  if (!model) return;

  const dataSubset = state.validData.slice(0, 180);
  const scatterPoints = dataSubset.map(d => ({
    x: Number(d[sf]) || 0,
    y: Number(d[state.target]) || 0
  }));

  const xVals = dataSubset.map(d => Number(d[sf]) || 0);
  const minX = Math.min(...xVals);
  const maxX = Math.max(...xVals);
  const numSteps = 50;
  const stepSize = (maxX - minX) / numSteps || 1;

  const fitLine = [];
  const ciLowerLine = [];
  const ciUpperLine = [];
  const piLowerLine = [];
  const piUpperLine = [];

  for (let i = 0; i <= numSteps; i++) {
    const xVal = minX + i * stepSize;
    const sample = {};
    sample[sf] = xVal;
    const res = model.predictWithIntervals(sample);

    fitLine.push({ x: xVal, y: res.predicted });
    ciLowerLine.push({ x: xVal, y: res.ciLower });
    ciUpperLine.push({ x: xVal, y: res.ciUpper });
    piLowerLine.push({ x: xVal, y: res.piLower });
    piUpperLine.push({ x: xVal, y: res.piUpper });
  }

  const datasets = [
    {
      type: "scatter",
      label: "驗證集真實觀測",
      data: scatterPoints,
      backgroundColor: "rgba(59, 130, 246, 0.65)",
      borderColor: "transparent",
      pointRadius: 3.5
    },
    {
      type: "line",
      label: `擬合回歸線 (${sf})`,
      data: fitLine,
      borderColor: "#06b6d4",
      borderWidth: 2.5,
      pointRadius: 0
    },
    {
      type: "line",
      label: "95% 信賴區間上限 (CI)",
      data: ciUpperLine,
      borderColor: "transparent",
      fill: "+1",
      backgroundColor: "rgba(6, 182, 212, 0.22)",
      pointRadius: 0
    },
    {
      type: "line",
      label: "95% 信賴區間下限 (CI)",
      data: ciLowerLine,
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
        data: piUpperLine,
        borderColor: "#ef4444",
        borderDash: [5, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false
      },
      {
        type: "line",
        label: "95% 預測區間下限 (PI)",
        data: piLowerLine,
        borderColor: "#ef4444",
        borderDash: [5, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false
      }
    );
  }

  chartInstances.mainPlot = new Chart(ctx, {
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: `${sf} 自變數`, color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        },
        y: {
          title: { display: true, text: `${state.target} (目標變數)`, color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        }
      }
    }
  });
}

/**
 * 5.2 多元線性回歸排序預測包絡線圖
 */
function renderMultipleRegressionEnvelopePlot(ctx) {
  const model = state.models.mlrSelected || state.models.mlrAll;
  if (!model) return;

  const valData = state.validData.slice(0, 100);
  const rowsWithPred = valData.map(d => {
    const res = model.predictWithIntervals(d);
    return {
      actual: Number(d[state.target]) || 0,
      pred: res.predicted,
      ciLow: res.ciLower,
      ciUp: res.ciUpper,
      piLow: res.piLower,
      piUp: res.piUpper
    };
  });

  rowsWithPred.sort((a, b) => a.pred - b.pred);

  const labels = rowsWithPred.map((_, i) => `#${i + 1}`);
  const actuals = rowsWithPred.map(r => r.actual);
  const preds = rowsWithPred.map(r => r.pred);
  const ciLow = rowsWithPred.map(r => r.ciLow);
  const ciUp = rowsWithPred.map(r => r.ciUp);
  const piLow = rowsWithPred.map(r => r.piLow);
  const piUp = rowsWithPred.map(r => r.piUp);

  const datasets = [
    {
      type: "scatter",
      label: "真實觀測值",
      data: actuals.map((y, idx) => ({ x: idx, y })),
      backgroundColor: "rgba(16, 185, 129, 0.75)",
      pointRadius: 3
    },
    {
      type: "line",
      label: "多元回歸擬合值",
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
      plugins: { legend: { display: false } },
      scales: {
        x: {
          title: { display: true, text: "排序樣本序號 (Sorted Validation Samples)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        },
        y: {
          title: { display: true, text: `${state.target} 目標數值`, color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        }
      }
    }
  });
}

/**
 * 5.3 自回歸模型時序預測圖 (AR)
 */
function renderAutoRegressionPlot(ctx) {
  const arModel = state.models.ar;
  if (!arModel) return;

  const historySeries = state.trainData.map(d => Number(d[state.target]) || 0).slice(-40);
  const actualValidation = state.validData.map(d => Number(d[state.target]) || 0).slice(0, 25);
  const steps = actualValidation.length || 20;

  const { forecasts, lowerPi, upperPi } = arModel.forecast(historySeries, steps);

  const histLabels = historySeries.map((_, i) => `T-${historySeries.length - i}`);
  const futureLabels = Array.from({ length: steps }, (_, i) => `T+${i + 1}`);
  const allLabels = [...histLabels, ...futureLabels];

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
          title: { display: true, text: `${state.target} 預測序列`, color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        }
      }
    }
  });
}

/**
 * 5.4 殘差診斷圖
 */
function renderResidualsPlot(ctx) {
  const model = state.models.mlrSelected || state.models.mlrAll;
  if (!model) return;

  const valData = state.validData.slice(0, 150);
  const resPoints = valData.map(d => {
    const pred = model.predictRow(d);
    return {
      x: pred,
      y: (Number(d[state.target]) || 0) - pred
    };
  });

  const predVals = resPoints.map(p => p.x);
  const minPred = Math.min(...predVals);
  const maxPred = Math.max(...predVals);

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
          label: "零殘差基準線",
          data: [{ x: minPred, y: 0 }, { x: maxPred, y: 0 }],
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
          title: { display: true, text: "擬合預測值 (Fitted Values)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        },
        y: {
          title: { display: true, text: "回歸殘差 (Residuals = Actual - Fitted)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#cbd5e1" }
        }
      }
    }
  });
}

// ============================================================
// 6. 模型評估展示 (Model Evaluation Table & Charts)
// ============================================================

function renderEvaluationTable() {
  const tbody = document.getElementById("evaluation-table-body");
  if (!tbody) return;

  const valData = state.validData;
  const target = state.target;

  const evalSimple = state.models.simple.evaluate(valData, target);
  const evalAll = state.models.mlrAll.evaluate(valData, target);
  const evalSel = state.models.mlrSelected.evaluate(valData, target);

  const sf = state.simpleFeature || state.features[0];
  const rows = [
    { name: `單純線性回歸 (Simple LR: ${sf})`, p: 1, ...evalSimple, badge: "基準單變量" },
    { name: `多元線性回歸 - 全特徵 (MLR All ${state.features.length})`, p: state.features.length, ...evalAll, badge: "高精度全模型", isBest: true },
    { name: `多元線性回歸 - 特徵選擇 (MLR Top ${state.selectedFeatures.length})`, p: state.selectedFeatures.length, ...evalSel, badge: "精簡高效模型" }
  ];

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${r.name}</strong></td>
      <td>${r.p}</td>
      <td>${formatMetric(r.MAE)}</td>
      <td>${formatMetric(r.MSE)}</td>
      <td><span class="${r.isBest ? 'text-emerald' : ''}">${formatMetric(r.RMSE)}</span></td>
      <td><strong>${r.R2.toFixed(4)}</strong></td>
      <td>${r.Adj_R2.toFixed(4)}</td>
      <td>
        <span class="${r.isBest ? 'eval-tag-best' : 'eval-tag-normal'}">${r.badge}</span>
      </td>
    </tr>
  `).join("");
}

function formatMetric(val) {
  if (val > 1000) return Math.round(val).toLocaleString();
  if (val > 1) return val.toFixed(2);
  return val.toFixed(5);
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
        { label: "MAE (平均絕對誤差)", data: maeData, backgroundColor: "#10b981", borderRadius: 4 }
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
// 7. 即時 What-If 模擬器與情境切換
// ============================================================

/**
 * 初始化情境按鈕 (根據當前問題動態產製)
 */
function initScenarioButtons() {
  const container = document.getElementById("scenario-presets-container");
  if (!container) return;

  const cfg = PROBLEMS[state.currentProblemId] || PROBLEMS.flood;
  const presets = cfg.scenarioPresets || [];

  container.innerHTML = `
    <span class="text-muted">情境快選：</span>
    ${presets.map(p => `
      <button class="chip-btn" id="scenario-${p.id}" onclick="applyScenarioPreset('${p.id}')">${p.name}</button>
    `).join("")}
    <button class="btn btn-outline btn-sm" id="btn-reset-sliders" onclick="resetSimulatorSliders()">重置滑桿</button>
  `;
}

/**
 * 初始化滑桿 (依據資料集實際 Min / Max / Step 自適應)
 */
function initSimulatorSliders() {
  const container = document.getElementById("simulator-sliders-container");
  if (!container) return;

  container.innerHTML = "";
  state.features.forEach(feat => {
    const stat = state.featureStats[feat] || { min: 0, max: 10, mean: 5, step: 1 };
    const initialVal = Number(stat.mean.toFixed(1));

    const row = document.createElement("div");
    row.className = "sim-slider-row";
    row.innerHTML = `
      <div class="sim-slider-header">
        <span class="sim-slider-name">${feat}</span>
        <span class="sim-slider-val" id="sim-val-${feat}">${initialVal}</span>
      </div>
      <input type="range" 
             min="${stat.min}" 
             max="${stat.max}" 
             step="${stat.step}" 
             value="${initialVal}" 
             class="custom-slider" 
             style="width: 100%;"
             id="sim-slider-${feat}" 
             oninput="onSimulatorSliderInput('${feat}', this.value)">
    `;
    container.appendChild(row);
  });

  // 更新模擬器標籤與單位
  const cfg = PROBLEMS[state.currentProblemId] || PROBLEMS.flood;
  const gaugeSub = document.getElementById("sim-gauge-sublabel");
  if (gaugeSub) gaugeSub.innerText = `預估 ${cfg.targetLabel}`;
  const sliderTitle = document.getElementById("sim-sliders-title");
  if (sliderTitle) sliderTitle.innerText = `特徵因子調節盤 (${state.features.length} 項自變數)`;
}

function onSimulatorSliderInput(feat, val) {
  const valSpan = document.getElementById(`sim-val-${feat}`);
  if (valSpan) valSpan.innerText = parseFloat(val).toLocaleString();
  updateSimulatorInference();
}

function getSimulatorCurrentValues() {
  const values = {};
  state.features.forEach(feat => {
    const slider = document.getElementById(`sim-slider-${feat}`);
    const defaultVal = state.featureStats[feat]?.mean || 0;
    values[feat] = slider ? parseFloat(slider.value) : defaultVal;
  });
  return values;
}

/**
 * 即時推算預測數值與 95% CI / PI 區間
 */
function updateSimulatorInference() {
  const model = state.models.mlrSelected || state.models.mlrAll;
  if (!model) return;

  const currentValues = getSimulatorCurrentValues();
  const res = model.predictWithIntervals(currentValues);
  const cfg = PROBLEMS[state.currentProblemId] || PROBLEMS.flood;

  let pred = res.predicted;
  let ciLow = res.ciLower;
  let ciUp = res.ciUpper;
  let piLow = res.piLower;
  let piUp = res.piUpper;

  // 格式化顯示字串
  let displayPred = "";
  let displayCi = "";
  let displayPi = "";

  if (cfg.isProbability) {
    pred = Math.max(0.0, Math.min(1.0, pred));
    ciLow = Math.max(0.0, Math.min(1.0, ciLow));
    ciUp = Math.max(0.0, Math.min(1.0, ciUp));
    piLow = Math.max(0.0, Math.min(1.0, piLow));
    piUp = Math.max(0.0, Math.min(1.0, piUp));

    displayPred = `${(pred * 100).toFixed(1)}%`;
    displayCi = `[${(ciLow * 100).toFixed(1)}% ~ ${(ciUp * 100).toFixed(1)}%]`;
    displayPi = `[${(piLow * 100).toFixed(1)}% ~ ${(piUp * 100).toFixed(1)}%]`;
  } else if (cfg.isCurrency) {
    displayPred = `CHF ${Math.round(pred).toLocaleString()}`;
    displayCi = `[CHF ${Math.round(ciLow).toLocaleString()} ~ ${Math.round(ciUp).toLocaleString()}]`;
    displayPi = `[CHF ${Math.round(piLow).toLocaleString()} ~ ${Math.round(piUp).toLocaleString()}]`;
  } else {
    displayPred = pred > 1000 ? Math.round(pred).toLocaleString() : pred.toFixed(2);
    displayCi = `[${formatMetric(ciLow)} ~ ${formatMetric(ciUp)}]`;
    displayPi = `[${formatMetric(piLow)} ~ ${formatMetric(piUp)}]`;
  }

  const probElem = document.getElementById("sim-predicted-prob");
  if (probElem) probElem.innerText = displayPred;

  const ciElem = document.getElementById("sim-ci-range");
  if (ciElem) ciElem.innerText = displayCi;

  const piElem = document.getElementById("sim-pi-range");
  if (piElem) piElem.innerText = displayPi;

  // 更新儀表色相
  const gaugeElem = document.getElementById("gauge-circle-elem");
  if (gaugeElem) {
    let deg = 180;
    if (cfg.isProbability) {
      deg = Math.round(pred * 360);
    } else if (state.targetStats.max > state.targetStats.min) {
      const frac = (pred - state.targetStats.min) / (state.targetStats.max - state.targetStats.min);
      deg = Math.max(20, Math.min(360, Math.round(frac * 360)));
    }
    gaugeElem.style.background = `conic-gradient(#06b6d4 0deg, #3b82f6 ${deg}deg, rgba(255, 255, 255, 0.08) ${deg}deg)`;
  }

  // 更新風險 / 等級徽章
  const badgeElem = document.getElementById("sim-risk-badge");
  if (badgeElem) {
    badgeElem.className = "risk-badge";
    if (cfg.isProbability) {
      if (pred < 0.45) {
        badgeElem.classList.add("risk-low");
        badgeElem.innerText = "低風險 (Low Risk)";
      } else if (pred < 0.55) {
        badgeElem.classList.add("risk-moderate");
        badgeElem.innerText = "中度警戒 (Moderate Risk)";
      } else if (pred < 0.65) {
        badgeElem.classList.add("risk-high");
        badgeElem.innerText = "高危險 (High Risk)";
      } else {
        badgeElem.classList.add("risk-extreme");
        badgeElem.innerText = "特大洪澇警報 (Extreme Risk)";
      }
    } else if (cfg.isCurrency) {
      if (pred > 1800000) {
        badgeElem.classList.add("risk-low");
        badgeElem.innerText = "豪華頂級住宅 (Luxury Property)";
      } else if (pred > 1200000) {
        badgeElem.classList.add("risk-moderate");
        badgeElem.innerText = "中高階房產 (Prime Standard)";
      } else {
        badgeElem.classList.add("risk-high");
        badgeElem.innerText = "平價合宜宅 (Entry Budget)";
      }
    } else {
      badgeElem.classList.add("risk-moderate");
      badgeElem.innerText = "模型預測點 (Predicted Value)";
    }
  }

  const modelNote = document.getElementById("sim-active-model-name");
  if (modelNote) {
    modelNote.innerText = `多元線性回歸 (${model.features.length} 項特徵子集)`;
  }
}

/**
 * 套用情境預設設定
 */
function applyScenarioPreset(presetId) {
  const cfg = PROBLEMS[state.currentProblemId];

  if (state.currentProblemId === "flood") {
    state.features.forEach(feat => {
      let v = 5.0;
      if (presetId === "normal") v = 4.5 + Math.random();
      else if (presetId === "typhoon") {
        v = ["MonsoonIntensity", "ClimateChange", "Landslides", "RiverManagement"].includes(feat) ? (12.0 + Math.random() * 2) : 8.0;
      } else if (presetId === "urban") {
        v = ["Urbanization", "InadequatePlanning", "Encroachments", "DrainageSystems"].includes(feat) ? (11.5 + Math.random() * 2) : 6.0;
      } else if (presetId === "good") {
        v = ["DamsQuality", "DrainageSystems", "Watersheds"].includes(feat) ? 13.0 : 3.0;
      }
      setSliderVal(feat, v);
    });
  } else if (state.currentProblemId === "swiss") {
    const pMap = {
      luxury: { LivingSpace: 240, Rooms: 6.5, Bathrooms: 3, Floor: 3, YearBuilt: 2018, DistanceCityCenter: 4.5, HasBalcony: 1, ParkingSpaces: 2, MountainViewScore: 10, EnergyEfficiency: 5, TaxMultiplier: 0.85 },
      urban_apt: { LivingSpace: 95, Rooms: 3.5, Bathrooms: 2, Floor: 4, YearBuilt: 2012, DistanceCityCenter: 1.5, HasBalcony: 1, ParkingSpaces: 1, MountainViewScore: 4, EnergyEfficiency: 4, TaxMultiplier: 1.15 },
      suburban: { LivingSpace: 160, Rooms: 5.5, Bathrooms: 2, Floor: 1, YearBuilt: 1995, DistanceCityCenter: 12.0, HasBalcony: 1, ParkingSpaces: 2, MountainViewScore: 7, EnergyEfficiency: 3, TaxMultiplier: 0.95 },
      studio: { LivingSpace: 45, Rooms: 1.5, Bathrooms: 1, Floor: 2, YearBuilt: 1980, DistanceCityCenter: 2.0, HasBalcony: 0, ParkingSpaces: 0, MountainViewScore: 2, EnergyEfficiency: 2, TaxMultiplier: 1.25 }
    };
    const vals = pMap[presetId] || pMap.urban_apt;
    state.features.forEach(feat => {
      if (vals[feat] !== undefined) setSliderVal(feat, vals[feat]);
    });
  } else {
    // 自訂資料集
    state.features.forEach(feat => {
      const stat = state.featureStats[feat];
      if (!stat) return;
      let v = stat.mean;
      if (presetId === "high") v = stat.min + (stat.max - stat.min) * 0.75;
      if (presetId === "low") v = stat.min + (stat.max - stat.min) * 0.25;
      setSliderVal(feat, v);
    });
  }

  updateSimulatorInference();
}

function setSliderVal(feat, val) {
  const slider = document.getElementById(`sim-slider-${feat}`);
  if (slider) slider.value = val;
  const span = document.getElementById(`sim-val-${feat}`);
  if (span) span.innerText = parseFloat(val).toLocaleString();
}

function resetSimulatorSliders() {
  state.features.forEach(feat => {
    const defaultVal = state.featureStats[feat]?.mean || 5.0;
    setSliderVal(feat, Number(defaultVal.toFixed(1)));
  });
  updateSimulatorInference();
}

// ============================================================
// 8. 資料預覽表格與報告導出
// ============================================================

function renderDataTable(query = "") {
  const thead = document.getElementById("data-table-head");
  const tbody = document.getElementById("data-table-body");
  if (!thead || !tbody) return;

  const showCols = ["id", ...state.features.slice(0, 8), state.target];
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
          const num = Number(val);
          const disp = isNaN(num) ? val : (num > 1000 ? Math.round(num).toLocaleString() : num.toFixed(4));
          return `<td><strong class="text-cyan">${disp}</strong></td>`;
        }
        return `<td>${val ?? '-'}</td>`;
      }).join("")}
    </tr>
  `).join("");
}

function filterDataTable(val) {
  renderDataTable(val);
}

function exportAnalysisReport() {
  const valData = state.validData;
  const target = state.target;
  const mAll = state.models.mlrAll.evaluate(valData, target);
  const mSel = state.models.mlrSelected.evaluate(valData, target);
  const cfg = PROBLEMS[state.currentProblemId] || PROBLEMS.flood;

  const report = `===============================================================
ML Predictor Studio - 回歸分析評估報告
===============================================================
分析題目: ${cfg.name}
目標變數: ${target}
分析時間: ${new Date().toLocaleString()}
分析樣本: ${state.dataset.length.toLocaleString()} 筆 (80% 訓練 / 20% 驗證)

【1. 特徵選擇 (Feature Selection)】
- 評估依據: F-Statistic 與 Pearson 相關係數
- 入選特徵 (Top ${state.selectedFeatures.length}):
  ${state.selectedFeatures.join(", ")}

【2. 模型評估指標 (Model Evaluation)】
- 多元線性回歸 (全特徵 ${state.features.length}):
  * R²: ${mAll.R2.toFixed(5)}
  * 調整後 R²: ${mAll.Adj_R2.toFixed(5)}
  * RMSE: ${formatMetric(mAll.RMSE)}
  * MAE: ${formatMetric(mAll.MAE)}

- 多元線性回歸 (特徵選擇 Top ${state.selectedFeatures.length}):
  * R²: ${mSel.R2.toFixed(5)}
  * 調整後 R²: ${mSel.Adj_R2.toFixed(5)}
  * RMSE: ${formatMetric(mSel.RMSE)}
  * MAE: ${formatMetric(mSel.MAE)}

【3. 95% 信賴區間 (CI) 與 95% 預測區間 (PI)】
- 95% 信賴區間: 反映母體平均期望反應 E(Y|X) 之估計誤差帶
- 95% 預測區間: 反映個別樣本真實觀測 Y_new 之波動界限 (包含殘差誤差 s)
===============================================================`;

  const blob = new Blob([report], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${state.currentProblemId}_Analysis_Report.txt`;
  a.click();
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

  if (tabId === "plots") setTimeout(renderActivePlot, 60);
  else if (tabId === "features") setTimeout(renderFeatureRankChart, 60);
  else if (tabId === "models") setTimeout(renderEvaluationCharts, 60);
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

// ============================================================
// 9. 本地合成資料備援 (Synthetic Generators)
// ============================================================

function generateLocalFloodSyntheticData(count = 400) {
  state.dataset = [];
  const weights = [
    0.045, 0.042, 0.041, 0.043, 0.039, 0.040, 0.041, 0.038,
    0.036, 0.037, 0.039, 0.042, 0.038, 0.046, 0.039, 0.040,
    0.037, 0.038, 0.039, 0.038
  ];
  for (let i = 0; i < count; i++) {
    const row = { id: i + 1 };
    let scoreSum = 0;
    state.features.forEach((feat, idx) => {
      const val = Math.floor(Math.random() * 10) + 2;
      row[feat] = val;
      scoreSum += val * weights[idx];
    });
    const noise = (Math.random() - 0.5) * 0.03;
    const prob = 0.35 + (scoreSum / 9.0) * 0.40 + noise;
    row[state.target] = Math.max(0.0, Math.min(1.0, prob));
    state.dataset.push(row);
  }
}

function generateLocalSwissSyntheticData(count = 500) {
  state.dataset = [];
  for (let i = 0; i < count; i++) {
    const space = Math.round(40 + Math.random() * 220);
    const rooms = Math.round((space / 28 + (Math.random() - 0.5)) * 2) / 2;
    const baths = Math.min(4, Math.max(1, Math.round(1 + space / 90)));
    const floor = Math.floor(Math.random() * 8);
    const year = 1960 + Math.floor(Math.random() * 64);
    const dist = Math.round((1 + Math.random() * 20) * 10) / 10;
    const balcony = Math.random() > 0.3 ? 1 : 0;
    const parking = Math.floor(Math.random() * 3);
    const view = Math.floor(Math.random() * 10) + 1;
    const energy = Math.floor(Math.random() * 5) + 1;
    const tax = Math.round((0.8 + Math.random() * 0.5) * 100) / 100;

    const price = Math.round(
      180000 +
      space * 6450 +
      rooms * 32000 +
      baths * 41000 +
      floor * 7200 +
      (year - 1960) * 1150 -
      dist * 19500 +
      balcony * 38000 +
      parking * 45000 +
      view * 22000 +
      energy * 16500 -
      (tax - 1.0) * 140000 +
      (Math.random() - 0.5) * 80000
    );

    state.dataset.push({
      id: i + 1,
      LivingSpace: space,
      Rooms: rooms,
      Bathrooms: baths,
      Floor: floor,
      YearBuilt: year,
      DistanceCityCenter: dist,
      HasBalcony: balcony,
      ParkingSpaces: parking,
      MountainViewScore: view,
      EnergyEfficiency: energy,
      TaxMultiplier: tax,
      PriceCHF: Math.max(280000, price)
    });
  }
}

// ============================================================
// 10. 矩陣運算核心演算法 (Matrix Math Engine)
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
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k;
    }
    if (Math.abs(A[maxRow][i]) < 1e-12) return null;

    const temp = A[i];
    A[i] = A[maxRow];
    A[maxRow] = temp;

    const pivot = A[i][i];
    for (let j = 0; j < 2 * n; j++) A[i][j] /= pivot;

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
