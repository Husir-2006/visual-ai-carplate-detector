let fleetRecords = [];
let passRecords = [];
let accessLists = { white: [], black: [] };
let systemSettings = {};
let selectedFile = null;
let lastVehicleImages = [];

const imageInput = document.querySelector("#imageInput");
const detectBtn = document.querySelector("#detectBtn");
const resetBtn = document.querySelector("#resetBtn");
const originalPreview = document.querySelector("#originalPreview");
const resultPreview = document.querySelector("#resultPreview");
const mode = document.querySelector("#mode");
const vehicleCount = document.querySelector("#vehicleCount");
const plateCount = document.querySelector("#plateCount");
const avgConfidence = document.querySelector("#avgConfidence");
const plateNumber = document.querySelector("#plateNumber");
const detectionList = document.querySelector("#detectionList");
const plateGallery = document.querySelector("#plateGallery");
const vehicleGallery = document.querySelector("#vehicleGallery");
const vehicleProfile = document.querySelector("#vehicleProfile");
const matchHint = document.querySelector("#matchHint");
const dropzone = document.querySelector("#dropzone");
const fleetTable = document.querySelector("#fleetTable");
const fleetViewTable = document.querySelector("#fleetViewTable");
const recordTable = document.querySelector("#recordTable");
const tableSearch = document.querySelector("#tableSearch");
const fleetViewSearch = document.querySelector("#fleetViewSearch");
const recordSearch = document.querySelector("#recordSearch");
const manualPlate = document.querySelector("#manualPlate");
const manualSearchBtn = document.querySelector("#manualSearchBtn");
const registeredCount = document.querySelector("#registeredCount");
const todayPassCount = document.querySelector("#todayPassCount");
const pageTitle = document.querySelector("#pageTitle");
const navButtons = document.querySelectorAll(".side-nav button");

const viewTitles = {
  workbench: "企业车牌识别与车辆查询系统",
  fleet: "车辆档案",
  records: "通行记录",
  lists: "黑白名单",
  settings: "系统设置",
};

init();

async function init() {
  bindEvents();
  await loadSystemData();
  renderAllData();
  clearResult();
}

function bindEvents() {
  navButtons.forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });

  imageInput.addEventListener("change", () => setSelectedFile(imageInput.files[0]));

  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));

  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
    setSelectedFile(event.dataTransfer.files[0]);
  });

  detectBtn.addEventListener("click", detectImage);
  resetBtn.addEventListener("click", resetWorkspace);

  tableSearch.addEventListener("input", () => renderFleetTable(filterFleet(tableSearch.value), fleetTable));
  fleetViewSearch.addEventListener("input", () => renderFleetTable(filterFleet(fleetViewSearch.value), fleetViewTable, true));
  recordSearch.addEventListener("input", () => renderPassRecords(filterRecords(recordSearch.value)));

  manualSearchBtn.addEventListener("click", () => {
    const plate = normalize(manualPlate.value);
    plateNumber.textContent = plate || "未识别";
    lookupVehicle(plate, true);
  });

  manualPlate.addEventListener("keydown", (event) => {
    if (event.key === "Enter") manualSearchBtn.click();
  });
}

async function loadSystemData() {
  const [fleet, records, lists, settings] = await Promise.all([
    fetchJson("/api/fleet", []),
    fetchJson("/api/pass-records", []),
    fetchJson("/api/access-lists", { white: [], black: [] }),
    fetchJson("/api/settings", {}),
  ]);
  fleetRecords = fleet;
  passRecords = records;
  accessLists = lists;
  systemSettings = settings;
}

async function fetchJson(url, fallback) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(url);
    return await response.json();
  } catch (error) {
    console.warn("数据加载失败", url, error);
    return fallback;
  }
}

function renderAllData() {
  registeredCount.textContent = fleetRecords.length;
  todayPassCount.textContent = passRecords.filter((record) => record.time?.includes("2026-07-19")).length || passRecords.length;
  renderFleetTable(fleetRecords, fleetTable);
  renderFleetTable(fleetRecords, fleetViewTable, true);
  renderPassRecords(passRecords);
  renderAccessLists();
  renderSettings();
}

function switchView(view) {
  navButtons.forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  document.querySelectorAll("[data-view-panel]").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.viewPanel === view);
  });
  pageTitle.textContent = viewTitles[view] || viewTitles.workbench;
  resetBtn.style.display = view === "workbench" ? "inline-flex" : "none";
}

async function detectImage() {
  if (!selectedFile) {
    mode.textContent = "请先上传图片";
    return;
  }

  detectBtn.disabled = true;
  mode.textContent = "识别中";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/detect", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "识别失败");
    renderResult(data);
  } catch (error) {
    mode.textContent = error.message;
  } finally {
    detectBtn.disabled = false;
  }
}

function resetWorkspace() {
  selectedFile = null;
  lastVehicleImages = [];
  imageInput.value = "";
  originalPreview.removeAttribute("src");
  resultPreview.removeAttribute("src");
  originalPreview.classList.remove("visible");
  resultPreview.classList.remove("visible");
  mode.textContent = "等待识别";
  clearResult();
}

function setSelectedFile(file) {
  if (!file) return;
  selectedFile = file;
  originalPreview.src = URL.createObjectURL(file);
  originalPreview.classList.add("visible");
  resultPreview.classList.remove("visible");
  mode.textContent = "图片已选择";
  clearResult();
}

function renderResult(data) {
  mode.textContent = data.message || data.mode;
  mode.dataset.status = data.status || "recognized";
  vehicleCount.textContent = data.summary.vehicleCount;
  plateCount.textContent = data.summary.plateCount;
  avgConfidence.textContent = data.summary.avgConfidence;
  lastVehicleImages = data.vehicleImages || [];

  const recognizedPlate = data.summary.plateNumbers?.[0] || "";
  plateNumber.textContent = recognizedPlate || "未识别";
  manualPlate.value = recognizedPlate;
  resultPreview.src = data.resultImage;
  resultPreview.classList.add("visible");

  if (!recognizedPlate && data.message) {
    matchHint.textContent = data.message;
  }
  lookupVehicle(recognizedPlate, false, data.vehicleType);
  renderDetectionList(data);
  renderGallery(vehicleGallery, data.vehicleImages || [], "车辆返回图", "暂无车辆图片");
  renderGallery(plateGallery, data.plateImages || [], "车牌裁剪结果", "暂无车牌区域");
}

function lookupVehicle(plate, manual = false, detectedType = "") {
  const normalizedPlate = normalize(plate);
  if (!normalizedPlate) {
    matchHint.textContent = "未识别到车牌，无法查询车辆档案";
    vehicleProfile.innerHTML = `<div class="profile-empty">暂无匹配车辆</div>`;
    return;
  }

  const record = fleetRecords.find((item) => normalize(item.plate) === normalizedPlate);
  if (!record) {
    matchHint.textContent = manual ? "档案库中没有该车牌" : "识别成功，但未在档案库中找到车辆";
    vehicleProfile.innerHTML = renderUnknownVehicle(normalizedPlate, detectedType);
    return;
  }

  matchHint.textContent = "已匹配车辆档案";
  vehicleProfile.innerHTML = renderVehicleProfile(record);
}

function renderVehicleProfile(record) {
  const statusClass = statusToClass(record.status);
  const vehicleImage = lastVehicleImages[0]
    ? `<img src="${lastVehicleImages[0]}" alt="匹配车辆图片" />`
    : `<div class="profile-image-placeholder">无车辆图</div>`;
  return `
    <div class="profile-image">${vehicleImage}</div>
    <div class="profile-title">
      <div><span>匹配车辆</span><strong>${record.plate}</strong></div>
      <em class="badge ${statusClass}">${record.status}</em>
    </div>
    <dl class="profile-list">
      <div><dt>车主/部门</dt><dd>${record.owner}</dd></div>
      <div><dt>车辆型号</dt><dd>${record.brand}</dd></div>
      <div><dt>车辆类型</dt><dd>${record.type}</dd></div>
      <div><dt>车辆颜色</dt><dd>${record.color}</dd></div>
      <div><dt>通行权限</dt><dd>${record.permit}</dd></div>
      <div><dt>联系电话</dt><dd>${record.phone}</dd></div>
      <div><dt>使用场景</dt><dd>${record.purpose}</dd></div>
      <div><dt>最近通行</dt><dd>${record.lastSeen}</dd></div>
    </dl>
  `;
}

function renderUnknownVehicle(plate, detectedType) {
  const vehicleImage = lastVehicleImages[0]
    ? `<img src="${lastVehicleImages[0]}" alt="未登记车辆图片" />`
    : `<div class="profile-image-placeholder">无车辆图</div>`;
  return `
    <div class="profile-image">${vehicleImage}</div>
    <div class="profile-title"><div><span>未登记车辆</span><strong>${plate}</strong></div><em class="badge danger">未授权</em></div>
    <dl class="profile-list">
      <div><dt>车辆类型</dt><dd>${detectedType || "未知"}</dd></div>
      <div><dt>处理建议</dt><dd>请人工核验并登记访客信息</dd></div>
      <div><dt>通行权限</dt><dd>待审批</dd></div>
    </dl>
  `;
}

function renderDetectionList(data) {
  const items = [...data.vehicles, ...data.plates];
  detectionList.classList.toggle("empty", items.length === 0);
  detectionList.innerHTML = items.length
    ? items.map((item, index) => {
        const text = item.text && item.text !== "未识别" ? `<b>车牌号：${item.text}</b>` : "";
        const ocr = item.ocrMethod ? `<small>OCR 方法：${item.ocrMethod}</small>` : "";
        return `<article><strong>${index + 1}. ${item.label}</strong>${text}<span>置信度 ${(item.confidence * 100).toFixed(1)}%</span><small>坐标 [${item.box.join(", ")}]</small>${ocr}</article>`;
      }).join("")
    : "暂无检测结果";
}

function renderFleetTable(records, target, full = false) {
  target.innerHTML = records.map((record) => {
    const statusClass = statusToClass(record.status);
    const cells = full
      ? `<td>${record.brand}</td><td>${record.owner}</td><td>${record.permit}</td><td><span class="badge ${statusClass}">${record.status}</span></td><td>${record.lastSeen}</td>`
      : `<td>${record.owner}</td><td>${record.permit}</td><td><span class="badge ${statusClass}">${record.status}</span></td><td>${record.lastSeen}</td>`;
    return `<tr data-plate="${record.plate}"><td><strong>${record.plate}</strong></td><td>${record.type}</td>${cells}</tr>`;
  }).join("");

  target.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", () => {
      switchView("workbench");
      manualPlate.value = row.dataset.plate;
      plateNumber.textContent = row.dataset.plate;
      lookupVehicle(row.dataset.plate, true);
    });
  });
}

function renderPassRecords(records) {
  recordTable.innerHTML = records.map((record) => `
    <tr>
      <td>${record.time}</td><td>${record.gate}</td><td><strong>${record.plate}</strong></td>
      <td>${record.owner}</td><td>${record.direction}</td><td>${record.result}</td>
    </tr>
  `).join("");
}

function renderAccessLists() {
  document.querySelector("#whiteList").innerHTML = renderCards(accessLists.white || [], "ok");
  document.querySelector("#blackList").innerHTML = renderCards(accessLists.black || [], "danger");
}

function renderCards(records, badgeClass) {
  return records.length ? records.map((item) => `
    <article class="record-card">
      <strong>${item.plate}</strong>
      <span class="badge ${badgeClass}">${item.expire || item.created}</span>
      <p>${item.reason}</p>
    </article>
  `).join("") : `<div class="profile-empty">暂无数据</div>`;
}

function renderSettings() {
  const labels = {
    gate: "门岗编号",
    confidenceThreshold: "置信度阈值",
    ocrEngine: "OCR 引擎",
    modelPriority: "模型优先级",
    saveEvidence: "保存识别证据",
    manualReview: "异常人工复核",
  };
  document.querySelector("#settingsList").innerHTML = Object.entries(labels).map(([key, label]) => {
    const value = systemSettings[key];
    return `<div><dt>${label}</dt><dd>${formatValue(value)}</dd></div>`;
  }).join("");
}

function renderGallery(container, images, alt, emptyText) {
  container.classList.toggle("empty", images.length === 0);
  container.innerHTML = images.length ? images.map((src) => `<img src="${src}" alt="${alt}" />`).join("") : emptyText;
}

function clearResult() {
  vehicleCount.textContent = "0";
  plateCount.textContent = "0";
  avgConfidence.textContent = "0";
  plateNumber.textContent = "未识别";
  matchHint.textContent = "上传图片后自动查询车辆档案";
  vehicleProfile.innerHTML = `<div class="profile-empty">暂无匹配车辆</div>`;
  detectionList.classList.add("empty");
  detectionList.textContent = "暂无检测结果";
  plateGallery.classList.add("empty");
  plateGallery.textContent = "暂无车牌区域";
  vehicleGallery.classList.add("empty");
  vehicleGallery.textContent = "暂无车辆图片";
}

function filterFleet(keyword) {
  const value = normalize(keyword);
  return fleetRecords.filter((record) => [record.plate, record.type, record.owner, record.status, record.permit, record.brand].some((field) => normalize(field).includes(value)));
}

function filterRecords(keyword) {
  const value = normalize(keyword);
  return passRecords.filter((record) => [record.time, record.gate, record.plate, record.owner, record.result].some((field) => normalize(field).includes(value)));
}

function normalize(value) {
  return String(value || "").trim().toUpperCase().replace(/\s+/g, "");
}

function statusToClass(status) {
  return status === "正常" ? "ok" : status === "异常" || status === "未授权" ? "danger" : "warn";
}

function formatValue(value) {
  if (typeof value === "boolean") return value ? "开启" : "关闭";
  return value ?? "未设置";
}
