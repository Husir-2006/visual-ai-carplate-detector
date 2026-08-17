let fleetRecords = [];
let passRecords = [];
let accessLists = { white: [], black: [], pending: [] };
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

const editModal = document.querySelector("#editModal");
const editModalClose = document.querySelector("#editModalClose");
const editModalCancel = document.querySelector("#editModalCancel");
const editForm = document.querySelector("#editForm");
const editMode = document.querySelector("#editMode");
const editPlate = document.querySelector("#editPlate");
const editPlateInput = document.querySelector("#editPlateInput");
const editType = document.querySelector("#editType");
const editOwner = document.querySelector("#editOwner");
const editBrand = document.querySelector("#editBrand");
const editPermit = document.querySelector("#editPermit");
const editPhone = document.querySelector("#editPhone");
const editColor = document.querySelector("#editColor");
const editPurpose = document.querySelector("#editPurpose");
const editStatus = document.querySelector("#editStatus");
const toast = document.querySelector("#toast");

const accessEditModal = document.querySelector("#accessEditModal");
const accessEditClose = document.querySelector("#accessEditClose");
const accessEditCancel = document.querySelector("#accessEditCancel");
const accessEditForm = document.querySelector("#accessEditForm");
const accessEditList = document.querySelector("#accessEditList");
const accessEditPlate = document.querySelector("#accessEditPlate");
const accessEditPlateInput = document.querySelector("#accessEditPlateInput");
const accessEditReason = document.querySelector("#accessEditReason");
const accessWhiteField = document.querySelector("#accessWhiteField");
const accessExpireType = document.querySelector("#accessExpireType");
const accessExpireDateWrap = document.querySelector("#accessExpireDateWrap");
const accessExpireDate = document.querySelector("#accessExpireDate");

const viewTitles = {
  workbench: "企业车牌识别与车辆查询系统",
  fleet: "车辆档案",
  records: "通行记录",
  lists: "黑白名单",
  settings: "系统设置展示",
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

  editModalClose.addEventListener("click", closeEditModal);
  editModalCancel.addEventListener("click", closeEditModal);
  editModal.addEventListener("click", (event) => {
    if (event.target === editModal) closeEditModal();
  });
  editForm.addEventListener("submit", saveFleetForm);
  accessEditClose.addEventListener("click", closeAccessEditModal);
  accessEditCancel.addEventListener("click", closeAccessEditModal);
  accessEditModal.addEventListener("click", (event) => {
    if (event.target === accessEditModal) closeAccessEditModal();
  });
  accessEditForm.addEventListener("submit", saveAccessEdit);
  accessExpireType.addEventListener("change", toggleExpireDateInput);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !editModal.hidden) closeEditModal();
    if (event.key === "Escape" && !accessEditModal.hidden) closeAccessEditModal();
  });
}

async function loadSystemData() {
  const [fleet, records, lists, settings] = await Promise.all([
    fetchJson("/api/fleet", []),
    fetchJson("/api/pass-records", []),
    fetchJson("/api/access-lists", { white: [], black: [], pending: [] }),
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
    const recognizedPlate = data.primaryPlate || data.summary?.plateNumbers?.[0] || "";
    if (recognizedPlate) {
      // 自动登记完成后刷新档案与通行记录，并重新匹配展示
      await loadSystemData();
      renderAllData();
      lookupVehicle(recognizedPlate, false, data.vehicleType);
    }
    showAutoRecordToast(data.auto, recognizedPlate);
  } catch (error) {
    mode.textContent = error.message;
  } finally {
    detectBtn.disabled = false;
  }
}

function showAutoRecordToast(auto, plate) {
  if (!auto) return;
  const parts = [];
  if (auto.fleetAdded?.length) parts.push(`新增 ${auto.fleetAdded.length} 辆待核验档案`);
  if (auto.fleetUpdated?.length) parts.push(`更新 ${auto.fleetUpdated.length} 条档案最近通行`);
  if (auto.pendingAdded?.length) parts.push("已加入待审核列表");
  if (auto.passRecordSaved) parts.push("已记入通行记录");
  if (parts.length) showToast(`${plate || "车辆"}：${parts.join("；")}`);
}

let toastTimer = null;
function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("show");
    toast.hidden = true;
  }, 3200);
}

function openEditModal(record, create = false) {
  editMode.value = create ? "create" : "edit";
  editPlate.value = record.plate || "";
  editPlateInput.value = record.plate || "";
  editPlateInput.readOnly = !create;
  editType.value = record.type || "";
  editOwner.value = record.owner || "";
  editBrand.value = record.brand || "";
  editPermit.value = record.permit || "";
  editPhone.value = record.phone || "";
  editColor.value = record.color || "";
  editPurpose.value = record.purpose || "";
  editStatus.value = record.status || "需核验";
  document.querySelector("#editModalTitle").textContent = create ? "登记车辆档案" : "编辑车辆档案";
  document.querySelector("#editModalSubtitle").textContent = create
    ? "补充车辆信息后保存，新档案状态默认为「需核验」"
    : "完善车辆信息后保存，状态可在核验后更新";
  document.querySelector("#editModalSave").textContent = create ? "登记" : "保存";
  editModal.hidden = false;
  (create ? editPlateInput : editOwner).focus();
}

function closeEditModal() {
  editModal.hidden = true;
}

function editFleetByPlate(plate) {
  const record = fleetRecords.find((item) => normalize(item.plate) === normalize(plate));
  if (record) openEditModal(record);
}

function openFleetCreate(plate) {
  openEditModal({ plate, status: "需核验" }, true);
}

async function saveFleetForm(event) {
  event.preventDefault();
  const create = editMode.value === "create";
  const payload = {
    type: editType.value.trim(),
    owner: editOwner.value.trim(),
    brand: editBrand.value.trim(),
    permit: editPermit.value.trim(),
    status: editStatus.value,
    phone: editPhone.value.trim(),
    color: editColor.value.trim(),
    purpose: editPurpose.value.trim(),
  };
  if (!create && !editPlate.value) {
    showToast("缺少车牌号，无法保存");
    return;
  }
  if (create && !normalize(editPlateInput.value)) {
    showToast("请输入车牌号");
    return;
  }

  const url = create ? "/api/fleet" : `/api/fleet/${encodeURIComponent(editPlate.value)}`;
  const method = create ? "POST" : "PUT";
  if (create) payload.plate = editPlateInput.value.trim();

  const saveBtn = document.querySelector("#editModalSave");
  saveBtn.disabled = true;
  try {
    const response = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "保存失败");
    closeEditModal();
    await loadSystemData();
    renderAllData();
    const savedPlate = data.record?.plate || payload.plate || editPlate.value;
    showToast(create ? `已登记车辆档案：${savedPlate}` : `已保存档案：${savedPlate}`);
    if (savedPlate && normalize(manualPlate.value) === normalize(savedPlate)) {
      lookupVehicle(savedPlate, true);
    }
  } catch (error) {
    showToast(error.message);
  } finally {
    saveBtn.disabled = false;
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

  const recognizedPlate = data.primaryPlate || data.summary.plateNumbers?.[0] || "";
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
    <div class="profile-actions">
      <button type="button" class="edit-btn" onclick="editFleetByPlate('${escapeJs(record.plate)}')">编辑档案</button>
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
    <div class="profile-actions">
      <button type="button" class="edit-btn" onclick="openFleetCreate('${escapeJs(plate)}')">登记到档案</button>
    </div>
    <dl class="profile-list">
      <div><dt>车辆类型</dt><dd>${detectedType || "未知"}</dd></div>
      <div><dt>处理建议</dt><dd>请人工核验并登记访客信息</dd></div>
      <div><dt>通行权限</dt><dd>待审批</dd></div>
    </dl>
  `;
}

function escapeJs(value) {
  return String(value || "").replace(/\\/g, "\\\\").replace(/'/g, "\\'");
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
    return `<tr data-plate="${record.plate}"><td><strong>${record.plate}</strong></td><td>${record.type}</td>${cells}<td><button type="button" class="edit-btn" data-plate="${record.plate}">编辑</button></td></tr>`;
  }).join("");

  target.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", () => {
      switchView("workbench");
      manualPlate.value = row.dataset.plate;
      plateNumber.textContent = row.dataset.plate;
      lookupVehicle(row.dataset.plate, true);
    });
  });

  target.querySelectorAll(".edit-btn").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      const record = fleetRecords.find((item) => normalize(item.plate) === normalize(btn.dataset.plate));
      if (record) openEditModal(record);
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
  document.querySelector("#pendingList").innerHTML = renderCards(accessLists.pending || [], "warn", [
    { label: "加入白名单", action: "move-white" },
    { label: "加入黑名单", action: "move-black" },
  ]);
  document.querySelector("#whiteList").innerHTML = renderCards(accessLists.white || [], "ok", [
    { label: "编辑信息", action: "edit", list: "white" },
    { label: "移回待审核", action: "move-pending" },
  ]);
  document.querySelector("#blackList").innerHTML = renderCards(accessLists.black || [], "danger", [
    { label: "移回待审核", action: "move-pending" },
  ]);
  document.querySelectorAll(".list-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.action === "edit") openAccessEditModal(btn.dataset.plate, btn.dataset.list);
      else moveAccessVehicle(btn.dataset.plate, btn.dataset.action);
    });
  });
}

function renderCards(records, badgeClass, actions = []) {
  if (!records.length) return `<div class="profile-empty">暂无数据</div>`;
  return records.map((item) => {
    const buttons = actions.length
      ? `<div class="record-actions">${actions.map((action) => `
          <button type="button" class="list-btn ${action.action}" data-plate="${escapeAttr(item.plate)}" data-action="${action.action}"${action.list ? ` data-list="${action.list}"` : ""}>${action.label}</button>
        `).join("")}</div>`
      : "";
    return `<article class="record-card">
      <strong>${item.plate}</strong>
      <span class="badge ${badgeClass}">${formatAccessBadge(item.expire || item.created)}</span>
      <p>${item.reason}</p>
      ${buttons}
    </article>`;
  }).join("");
}

function formatAccessBadge(value) {
  const v = String(value || "").trim();
  if (/^\d{8}$/.test(v)) return `${v.slice(0, 4)}-${v.slice(4, 6)}-${v.slice(6, 8)}`;
  if (/^\d{4}-\d{2}-\d{2}/.test(v)) return v.slice(0, 10);
  return v;
}

const LIST_LABELS = { white: "白名单", black: "黑名单", pending: "待审核列表" };
const REASON_PRESETS = ["员工车辆", "访客业务车辆", "施工作业车辆", "特殊任务车辆（如消防、公安、急救）", "货运车辆"];

function openAccessEditModal(plate, listName) {
  const list = accessLists[listName] || [];
  const record = list.find((item) => normalize(item.plate) === normalize(plate));
  if (!record) {
    showToast("未找到该车辆");
    return;
  }
  accessEditList.value = listName;
  accessEditPlate.value = record.plate || "";
  accessEditPlateInput.value = record.plate || "";
  // 用途说明下拉：默认项为"待补充"，其余为预设选项
  const currentReason = String(record.reason || "").trim();
  accessEditReason.value = REASON_PRESETS.includes(currentReason) ? currentReason : "待补充";
  const isWhite = listName === "white";
  document.querySelector("#accessEditTitle").textContent = `编辑${LIST_LABELS[listName]}车辆信息`;
  document.querySelector("#accessEditSubtitle").textContent = isWhite
    ? "绿底徽章显示有效期：默认 / 具体日期（8位数字，年月日）/ 长期"
    : "登记/进入时间仅到日期，可补充用途说明";
  accessWhiteField.hidden = !isWhite;
  if (isWhite) {
    const exp = String(record.expire || "").trim();
    if (exp === "长期") {
      accessExpireType.value = "long";
      accessExpireDate.value = "";
    } else if (/^\d{8}$/.test(exp)) {
      accessExpireType.value = "date";
      accessExpireDate.value = exp;
    } else if (/^\d{4}-\d{2}-\d{2}$/.test(exp)) {
      accessExpireType.value = "date";
      accessExpireDate.value = exp.replace(/-/g, "");
    } else {
      accessExpireType.value = "default";
      accessExpireDate.value = "";
    }
  }
  toggleExpireDateInput();
  accessEditModal.hidden = false;
  accessEditReason.focus();
}

function toggleExpireDateInput() {
  const show = accessExpireType.value === "date";
  accessExpireDateWrap.hidden = !show;
  if (show) accessExpireDate.focus();
}

function closeAccessEditModal() {
  accessEditModal.hidden = true;
}

async function saveAccessEdit(event) {
  event.preventDefault();
  const plate = accessEditPlate.value;
  const listName = accessEditList.value;
  const payload = {};
  const reasonVal = accessEditReason.value;
  // 用途说明按所选选项保存（含"待补充"）
  if (reasonVal) payload.reason = reasonVal;
  if (listName === "white") {
    const type = accessExpireType.value;
    if (type === "date") {
      const d = accessExpireDate.value.trim();
      if (!isValidDate8(d)) {
        showToast("到期日期需为 8 位数字（YYYYMMDD，如 20260818）");
        return;
      }
      payload.expire = d;
    } else {
      payload.expire = type === "long" ? "长期" : "默认";
    }
  }
  // 黑名单/待审核仅更新用途说明，日期由系统自动记录
  try {
    const response = await fetch(`/api/access-lists/${encodeURIComponent(plate)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "保存失败");
    closeAccessEditModal();
    await loadSystemData();
    renderAllData();
    showToast(`${plate} 信息已更新`);
  } catch (error) {
    showToast(error.message);
  }
}

function isValidDate8(value) {
  if (!/^\d{8}$/.test(value)) return false;
  const y = +value.slice(0, 4);
  const m = +value.slice(4, 6);
  const d = +value.slice(6, 8);
  const dt = new Date(y, m - 1, d);
  return dt.getFullYear() === y && dt.getMonth() === m - 1 && dt.getDate() === d;
}

async function moveAccessVehicle(plate, action) {
  const target = action === "move-white" ? "white" : action === "move-black" ? "black" : "pending";
  try {
    const response = await fetch("/api/access-lists", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plate, target }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "操作失败");
    await loadSystemData();
    renderAllData();
    showToast(`${plate} 已移入${LIST_LABELS[target]}`);
  } catch (error) {
    showToast(error.message);
  }
}

function escapeAttr(value) {
  return String(value || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
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
