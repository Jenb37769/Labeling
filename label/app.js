const state = {
  entries: [],
  currentIndex: -1,
  currentStem: "",
  currentSource: "",
  document: null,
  selectedId: null,
  zoom: 1,
  createMode: false,
  readOnly: false,
  loadCount: 0,
  histTop: [],
  dragMode: null,
  dragStart: null,
  activeHandle: null,
  draftBox: null,
};

const els = {
  frameList: document.getElementById("frameList"),
  boxList: document.getElementById("boxList"),
  statusText: document.getElementById("statusText"),
  mainImage: document.getElementById("mainImage"),
  boxLayer: document.getElementById("boxLayer"),
  canvasStage: document.getElementById("canvasStage"),
  canvasScroll: document.getElementById("canvasScroll"),
  draftBox: document.getElementById("draftBox"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  saveBtn: document.getElementById("saveBtn"),
  saveNextBtn: document.getElementById("saveNextBtn"),
  deleteImageBtn: document.getElementById("deleteImageBtn"),
  newBoxBtn: document.getElementById("newBoxBtn"),
  deleteBoxBtn: document.getElementById("deleteBoxBtn"),
  prevIconBtn: document.getElementById("prevIconBtn"),
  nextIconBtn: document.getElementById("nextIconBtn"),
  zoomRange: document.getElementById("zoomRange"),
  zoomValue: document.getElementById("zoomValue"),
  fieldId: document.getElementById("fieldId"),
  fieldName: document.getElementById("fieldName"),
  fieldType: document.getElementById("fieldType"),
  fieldClickable: document.getElementById("fieldClickable"),
  fieldX1: document.getElementById("fieldX1"),
  fieldY1: document.getElementById("fieldY1"),
  fieldX2: document.getElementById("fieldX2"),
  fieldY2: document.getElementById("fieldY2"),
  applyFieldsBtn: document.getElementById("applyFieldsBtn"),
  loadCountValue: document.getElementById("loadCountValue"),
  previewCanvas: document.getElementById("previewCanvas"),
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function updateStatus(extra = "") {
  if (!state.entries.length || state.currentIndex < 0) {
    els.statusText.textContent = "No frames available";
    return;
  }
  const entry = state.entries[state.currentIndex];
  const prefix = `${state.currentIndex + 1}/${state.entries.length}  ${entry.image_name}  [${state.currentSource}]`;
  els.statusText.textContent = extra ? `${prefix}  ${extra}` : prefix;
}

function colorForType(type) {
  const palette = {
    tool_icon: "#265d9b",
    menu_item: "#7b3fa1",
    submenu_item: "#7b3fa1",
    panel_tab: "#157f52",
    panel_item: "#a85a1b",
    icon_button: "#c84a2f",
    text_button: "#b64d78",
    dropdown: "#5d53aa",
    input_field: "#1f7c86",
    slider: "#8a5f2f",
    toggle: "#8d3446",
    canvas_target: "#0f6c78",
  };
  return palette[type] || "#c84a2f";
}

function clampBox(box) {
  const maxX = els.mainImage.naturalWidth || 1;
  const maxY = els.mainImage.naturalHeight || 1;
  const x1 = Math.max(0, Math.min(box[0], maxX - 1));
  const y1 = Math.max(0, Math.min(box[1], maxY - 1));
  const x2 = Math.max(x1 + 1, Math.min(box[2], maxX));
  const y2 = Math.max(y1 + 1, Math.min(box[3], maxY));
  return [x1, y1, x2, y2].map((value) => Math.round(value));
}

function computeCenter(bbox) {
  return [
    Math.round((bbox[0] + bbox[2]) / 2),
    Math.round((bbox[1] + bbox[3]) / 2),
  ];
}

function selectedElement() {
  return state.document?.elements.find((element) => element.id === state.selectedId) || null;
}

function markSelectedTouched() {
  const element = selectedElement();
  if (!element || state.readOnly) {
    return;
  }
  element.touched = true;
}

function selectElementById(id) {
  if (!state.document) {
    return;
  }
  const exists = state.document.elements.some((element) => element.id === id);
  if (!exists) {
    return;
  }
  state.selectedId = id;
  markSelectedTouched();
  renderAll();
  focusSelected();
}

function selectElementByDelta(delta) {
  if (!state.document || !state.document.elements.length) {
    return;
  }
  const elements = state.document.elements;
  const currentIndex = elements.findIndex((element) => element.id === state.selectedId);
  const nextIndex = currentIndex === -1 ? 0 : (currentIndex + delta + elements.length) % elements.length;
  selectElementById(elements[nextIndex].id);
}

function renderFrameList() {
  els.frameList.innerHTML = "";
  state.entries.forEach((entry, index) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `frame-item${entry.is_labeled ? " done" : ""}${index === state.currentIndex ? " active" : ""}`;
    item.textContent = formatFrameName(entry.image_name);
    item.addEventListener("click", () => loadEntry(index));
    els.frameList.appendChild(item);
  });
}

function formatFrameName(imageName) {
  const matched = imageName.match(/_(\d{6})_(\d+\.\d+s)\.png$/);
  if (matched) {
    return `${matched[1]} | ${matched[2]}`;
  }
  return imageName;
}

function renderBoxList() {
  els.boxList.innerHTML = "";
  if (!state.document) {
    return;
  }
  state.document.elements.forEach((element) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `box-item${element.id === state.selectedId ? " active" : ""}`;
    item.textContent = `${element.id} | ${element.name || "(unnamed)"} | ${element.type}`;
    item.addEventListener("click", () => {
      selectElementById(element.id);
    });
    els.boxList.appendChild(item);
  });
}

function renderBoxes() {
  els.boxLayer.innerHTML = "";
  if (!state.document) {
    return;
  }
  state.document.elements.forEach((element) => {
    const box = document.createElement("div");
    const [x1, y1, x2, y2] = element.bbox;
    const color = colorForType(element.type);
    const isSelected = element.id === state.selectedId;
    const isPrefill = Boolean(element.prefill_match);
    const isTouched = Boolean(element.touched);
    box.className = `anno-box${isSelected ? " selected" : ""}${isPrefill ? " prefill" : ""}`;
    if (isTouched) {
      box.className += " touched";
    }
    box.style.left = `${x1}px`;
    box.style.top = `${y1}px`;
    box.style.width = `${x2 - x1}px`;
    box.style.height = `${y2 - y1}px`;
    box.style.color = color;
    box.dataset.id = element.id;

    ["nw", "ne", "sw", "se"].forEach((direction) => {
      const handle = document.createElement("div");
      handle.className = `resize-handle ${direction}`;
      handle.dataset.handle = direction;
      handle.addEventListener("pointerdown", (event) => startResize(event, element.id, direction));
      box.appendChild(handle);
    });

    box.addEventListener("pointerdown", (event) => startMove(event, element.id));
    els.boxLayer.appendChild(box);
  });
}

function renderFields() {
  const element = selectedElement();
  const readOnly = state.readOnly;
  els.fieldId.value = element?.id || "";
  els.fieldName.value = element?.name || "";
  els.fieldType.value = element?.type || "icon_button";
  els.fieldClickable.checked = element?.clickable ?? true;
  els.fieldX1.value = element?.bbox?.[0] ?? "";
  els.fieldY1.value = element?.bbox?.[1] ?? "";
  els.fieldX2.value = element?.bbox?.[2] ?? "";
  els.fieldY2.value = element?.bbox?.[3] ?? "";

  [
    els.fieldId,
    els.fieldName,
    els.fieldType,
    els.fieldClickable,
    els.fieldX1,
    els.fieldY1,
    els.fieldX2,
    els.fieldY2,
    els.applyFieldsBtn,
  ].forEach((el) => {
    if (!el) {
      return;
    }
    el.disabled = readOnly;
  });
}

async function ensureHistTop() {
  if (state.histTop.length) {
    return state.histTop;
  }
  const data = await requestJson("/api/hist_top?limit=1000");
  state.histTop = data.items || [];
  return state.histTop;
}

function elementSize(bbox) {
  const width = Math.max(1, bbox[2] - bbox[0]);
  const height = Math.max(1, bbox[3] - bbox[1]);
  return { width, height, area: width * height };
}

function isCloseRatio(reference, current, tolerance) {
  if (!reference) {
    return false;
  }
  return Math.abs(current - reference) / reference <= tolerance;
}

function centerDistanceRatio(centerA, centerB, baseline) {
  if (!centerA || !centerB || centerA.length !== 2 || centerB.length !== 2) {
    return 1;
  }
  if (!baseline) {
    return 1;
  }
  const dx = centerA[0] - centerB[0];
  const dy = centerA[1] - centerB[1];
  return Math.hypot(dx, dy) / baseline;
}

function pickBestHistMatch(element, histItems) {
  const tolerance = 0.10;
  const centerTolerance = 0.08;
  const { width, height, area } = elementSize(element.bbox);
  const centerBaseline = Math.min(width, height);

  let best = null;
  let bestCount = -1;
  for (const item of histItems) {
    if (item.type !== element.type) {
      continue;
    }
    if (!isCloseRatio(item.width, width, tolerance)) {
      continue;
    }
    if (!isCloseRatio(item.height, height, tolerance)) {
      continue;
    }
    if (!isCloseRatio(item.area, area, tolerance)) {
      continue;
    }
    if (centerDistanceRatio(item.center, element.center, centerBaseline) > centerTolerance) {
      continue;
    }
    const itemCount = Number(item.count || 0);
    if (itemCount > bestCount) {
      best = item;
      bestCount = itemCount;
    }
  }

  return best;
}

function applyHistPrefill() {
  if (!state.document || !state.histTop.length) {
    return;
  }
  state.document.elements = state.document.elements.map((element) => {
    if (!element?.bbox || !element?.center) {
      return element;
    }
    const match = pickBestHistMatch(element, state.histTop);
    if (!match) {
      return { ...element, prefill_match: null };
    }
    return {
      ...element,
      name: match.name || element.name,
      raw_type: match.raw_type || element.raw_type,
      region: match.region || element.region,
      clickable: typeof match.clickable === "boolean" ? match.clickable : element.clickable,
      prefill_match: {
        name: match.name,
        type: match.type,
        count: match.count,
      },
    };
  });
}

function renderAll(extraStatus = "") {
  renderFrameList();
  renderBoxes();
  renderBoxList();
  renderFields();
  renderPreview();
  if (els.loadCountValue) {
    els.loadCountValue.textContent = String(state.loadCount || 0);
  }
  updateStatus(extraStatus);
}

function focusSelected() {
  const element = selectedElement();
  if (!element || !els.canvasScroll || !els.mainImage?.naturalWidth) {
    return;
  }
  const [x1, y1, x2, y2] = element.bbox;
  const boxWidth = Math.max(1, x2 - x1);
  const boxHeight = Math.max(1, y2 - y1);
  const viewportWidth = els.canvasScroll.clientWidth;
  const viewportHeight = els.canvasScroll.clientHeight;
  if (!viewportWidth || !viewportHeight) {
    return;
  }

  const targetScale = Math.min((viewportWidth * 0.5) / boxWidth, (viewportHeight * 0.5) / boxHeight);
  const clampedScale = Math.max(0.5, Math.min(2.0, targetScale));
  setZoom(clampedScale);

  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;
  const scrollLeft = Math.max(0, centerX * clampedScale - viewportWidth / 2);
  const scrollTop = Math.max(0, centerY * clampedScale - viewportHeight / 2);
  els.canvasScroll.scrollLeft = scrollLeft;
  els.canvasScroll.scrollTop = scrollTop;
}

function renderPreview() {
  const canvas = els.previewCanvas;
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== width) {
    canvas.width = width;
  }
  if (canvas.height !== height) {
    canvas.height = height;
  }

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.fillRect(0, 0, width, height);

  const element = selectedElement();
  if (!element || !els.mainImage?.naturalWidth) {
    return;
  }

  const [x1, y1, x2, y2] = element.bbox;
  const cropWidth = Math.max(1, x2 - x1);
  const cropHeight = Math.max(1, y2 - y1);
  const padX = Math.round(cropWidth * 0.2);
  const padY = Math.round(cropHeight * 0.2);
  const sx = Math.max(0, x1 - padX);
  const sy = Math.max(0, y1 - padY);
  const sWidth = Math.min(els.mainImage.naturalWidth - sx, cropWidth + padX * 2);
  const sHeight = Math.min(els.mainImage.naturalHeight - sy, cropHeight + padY * 2);

  const scale = Math.min(width / sWidth, height / sHeight);
  const drawWidth = Math.max(1, Math.floor(sWidth * scale));
  const drawHeight = Math.max(1, Math.floor(sHeight * scale));
  const dx = Math.floor((width - drawWidth) / 2);
  const dy = Math.floor((height - drawHeight) / 2);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(els.mainImage, sx, sy, sWidth, sHeight, dx, dy, drawWidth, drawHeight);
}

function setReadOnly(isReadOnly) {
  state.readOnly = isReadOnly;
  els.saveBtn.disabled = isReadOnly;
  els.saveNextBtn.disabled = isReadOnly;
  els.newBoxBtn.disabled = isReadOnly;
  els.deleteBoxBtn.disabled = isReadOnly;
  els.deleteImageBtn.disabled = isReadOnly;
  if (els.prevIconBtn) {
    els.prevIconBtn.disabled = isReadOnly;
  }
  if (els.nextIconBtn) {
    els.nextIconBtn.disabled = isReadOnly;
  }
  if (isReadOnly) {
    state.createMode = false;
  }
}

function setZoom(value) {
  state.zoom = value;
  els.canvasStage.style.transform = `scale(${value})`;
  els.zoomValue.textContent = `${Math.round(value * 100)}%`;
  els.zoomRange.value = String(Math.round(value * 100));
}

function nextElementId() {
  const used = new Set((state.document?.elements || []).map((element) => element.id));
  let index = state.document?.elements?.length || 0;
  while (used.has(String(index + 1))) {
    index += 1;
  }
  return String(index + 1);
}

function updateElement(id, updater) {
  if (!state.document) {
    return;
  }
  state.document.elements = state.document.elements.map((element) => {
    if (element.id !== id) {
      return element;
    }
    const updated = updater({ ...element });
    updated.bbox = clampBox(updated.bbox);
    updated.center = computeCenter(updated.bbox);
    return updated;
  });
  renderAll();
}

function pointerToImage(event) {
  const rect = els.canvasStage.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / state.zoom,
    y: (event.clientY - rect.top) / state.zoom,
  };
}

function startMove(event, id) {
  if (state.readOnly) {
    return;
  }
  if (event.target.dataset.handle) {
    return;
  }
  event.preventDefault();
  state.selectedId = id;
  markSelectedTouched();
  const element = selectedElement();
  if (!element) {
    return;
  }
  const point = pointerToImage(event);
  state.dragMode = "move";
  state.dragStart = {
    pointerX: point.x,
    pointerY: point.y,
    bbox: [...element.bbox],
  };
  renderAll();
}

function startResize(event, id, handle) {
  if (state.readOnly) {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  state.selectedId = id;
  markSelectedTouched();
  const element = selectedElement();
  if (!element) {
    return;
  }
  const point = pointerToImage(event);
  state.dragMode = "resize";
  state.activeHandle = handle;
  state.dragStart = {
    pointerX: point.x,
    pointerY: point.y,
    bbox: [...element.bbox],
  };
  renderAll();
}

function applyDrag(event) {
  if (!state.dragMode || !state.selectedId || !state.dragStart) {
    return;
  }
  const point = pointerToImage(event);
  const dx = point.x - state.dragStart.pointerX;
  const dy = point.y - state.dragStart.pointerY;
  updateElement(state.selectedId, (element) => {
    let [x1, y1, x2, y2] = state.dragStart.bbox;
    if (state.dragMode === "move") {
      x1 += dx;
      x2 += dx;
      y1 += dy;
      y2 += dy;
    } else if (state.dragMode === "resize") {
      if (state.activeHandle.includes("n")) {
        y1 += dy;
      }
      if (state.activeHandle.includes("s")) {
        y2 += dy;
      }
      if (state.activeHandle.includes("w")) {
        x1 += dx;
      }
      if (state.activeHandle.includes("e")) {
        x2 += dx;
      }
    }
    element.bbox = [x1, y1, x2, y2];
    return element;
  });
}

function stopDrag() {
  state.dragMode = null;
  state.dragStart = null;
  state.activeHandle = null;
}

function beginDraft(event) {
  if (state.readOnly) {
    return;
  }
  if (!state.createMode) {
    return;
  }
  const point = pointerToImage(event);
  state.draftBox = {
    x1: point.x,
    y1: point.y,
    x2: point.x,
    y2: point.y,
  };
  els.draftBox.classList.remove("hidden");
  updateDraftBox();
}

function updateDraft(event) {
  if (!state.draftBox) {
    return;
  }
  const point = pointerToImage(event);
  state.draftBox.x2 = point.x;
  state.draftBox.y2 = point.y;
  updateDraftBox();
}

function updateDraftBox() {
  if (!state.draftBox) {
    els.draftBox.classList.add("hidden");
    return;
  }
  const x1 = Math.min(state.draftBox.x1, state.draftBox.x2);
  const y1 = Math.min(state.draftBox.y1, state.draftBox.y2);
  const x2 = Math.max(state.draftBox.x1, state.draftBox.x2);
  const y2 = Math.max(state.draftBox.y1, state.draftBox.y2);
  els.draftBox.style.left = `${x1}px`;
  els.draftBox.style.top = `${y1}px`;
  els.draftBox.style.width = `${x2 - x1}px`;
  els.draftBox.style.height = `${y2 - y1}px`;
}

function finishDraft() {
  if (state.readOnly) {
    return;
  }
  if (!state.draftBox || !state.document) {
    return;
  }
  const x1 = Math.min(state.draftBox.x1, state.draftBox.x2);
  const y1 = Math.min(state.draftBox.y1, state.draftBox.y2);
  const x2 = Math.max(state.draftBox.x1, state.draftBox.x2);
  const y2 = Math.max(state.draftBox.y1, state.draftBox.y2);
  state.draftBox = null;
  els.draftBox.classList.add("hidden");
  if (x2 - x1 < 6 || y2 - y1 < 6) {
    return;
  }
  const bbox = clampBox([x1, y1, x2, y2]);
  const element = {
    id: nextElementId(),
    name: "",
    bbox,
    center: computeCenter(bbox),
    type: "icon_button",
    clickable: true,
    touched: true,
  };
  state.document.elements.push(element);
  state.selectedId = element.id;
  state.createMode = false;
  renderAll("New box created");
}

function deleteSelected() {
  if (state.readOnly) {
    return;
  }
  if (!state.document || !state.selectedId) {
    return;
  }
  const elements = state.document.elements;
  const currentIndex = elements.findIndex((element) => element.id === state.selectedId);
  state.document.elements = elements.filter((element) => element.id !== state.selectedId);
  if (!state.document.elements.length) {
    state.selectedId = null;
    renderAll("Box deleted");
    return;
  }
  const nextIndex = currentIndex === -1 ? 0 : Math.min(currentIndex, state.document.elements.length - 1);
  state.selectedId = state.document.elements[nextIndex].id;
  markSelectedTouched();
  renderAll("Box deleted");
}

function applyFields() {
  if (state.readOnly) {
    return;
  }
  const element = selectedElement();
  if (!element) {
    return;
  }
  updateElement(element.id, (draft) => {
    draft.touched = true;
    draft.id = els.fieldId.value.trim() || draft.id;
    draft.name = els.fieldName.value.trim();
    draft.type = els.fieldType.value;
    draft.clickable = els.fieldClickable.checked;
    draft.bbox = [
      Number(els.fieldX1.value || draft.bbox[0]),
      Number(els.fieldY1.value || draft.bbox[1]),
      Number(els.fieldX2.value || draft.bbox[2]),
      Number(els.fieldY2.value || draft.bbox[3]),
    ];
    return draft;
  });
}

async function saveCurrent() {
  if (state.readOnly) {
    updateStatus("Read-only: already saved");
    return;
  }
  if (!state.document || !state.currentStem) {
    return;
  }
  markSelectedTouched();
  const hasUnconfirmed = state.document.elements.some((element) => !element.touched);
  if (hasUnconfirmed) {
    updateStatus("Save failed: all boxes must be red or green");
    return;
  }
  const data = await requestJson("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      stem: state.currentStem,
      document: state.document,
    }),
  });
  const entry = state.entries[state.currentIndex];
  if (entry) {
    entry.is_labeled = true;
  }
  state.document = data.document;
  state.currentSource = "final_label";
  renderAll("Saved");
}

async function deleteCurrentImage() {
  if (state.readOnly) {
    updateStatus("Read-only: already saved");
    return;
  }
  if (!state.currentStem || state.currentIndex < 0) {
    return;
  }
  const entry = state.entries[state.currentIndex];
  const ok = window.confirm(`Delete this image and its labels?\n\n${entry.image_name}`);
  if (!ok) {
    return;
  }

  await requestJson("/api/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stem: state.currentStem }),
  });

  state.entries.splice(state.currentIndex, 1);
  if (!state.entries.length) {
    state.currentIndex = -1;
    state.currentStem = "";
    state.currentSource = "";
    state.document = null;
    state.selectedId = null;
    els.mainImage.removeAttribute("src");
    els.boxLayer.innerHTML = "";
    els.boxList.innerHTML = "";
    els.frameList.innerHTML = "";
    renderFields();
    updateStatus("Image deleted");
    return;
  }

  const nextIndex = Math.min(state.currentIndex, state.entries.length - 1);
  await loadEntry(nextIndex);
  renderAll("Image deleted");
}

async function loadEntry(index) {
  if (index < 0 || index >= state.entries.length) {
    return;
  }
  const entry = state.entries[index];
  const data = await requestJson(`/api/item?stem=${encodeURIComponent(entry.stem)}`);
  state.currentIndex = index;
  state.currentStem = entry.stem;
  state.currentSource = data.source;
  state.loadCount = Number(data.load_count || 0);
  setReadOnly(data.source === "final_label");
  state.document = data.document;
  if (state.document?.elements) {
    state.document.elements = state.document.elements.map((element) => ({ ...element, touched: false }));
  }
  state.selectedId = state.document.elements[0]?.id || null;
  state.createMode = false;
  if (data.source === "final_unlabel") {
    await ensureHistTop();
    applyHistPrefill();
  }
  const imageUrl = `${data.image_url}?t=${Date.now()}`;
  els.mainImage.onload = () => {
    els.canvasStage.style.width = `${els.mainImage.naturalWidth}px`;
    els.canvasStage.style.height = `${els.mainImage.naturalHeight}px`;
    setZoom(Number(els.zoomRange.value) / 100);
    renderAll();
    focusSelected();
  };
  els.mainImage.onerror = () => {
    els.canvasStage.style.width = "0px";
    els.canvasStage.style.height = "0px";
    renderAll(`Image load failed: ${imageUrl}`);
  };
  els.mainImage.src = imageUrl;
  renderAll(`Loading image: ${imageUrl}`);
}

async function init() {
  const data = await requestJson("/api/entries");
  state.entries = data.entries;
  renderFrameList();
  if (state.entries.length) {
    await loadEntry(0);
  } else {
    updateStatus();
  }
}

els.prevBtn.addEventListener("click", () => loadEntry(state.currentIndex - 1));
els.nextBtn.addEventListener("click", () => loadEntry(state.currentIndex + 1));
els.saveBtn.addEventListener("click", () => saveCurrent().catch((error) => updateStatus(error.message)));
els.saveNextBtn.addEventListener("click", async () => {
  try {
    await saveCurrent();
    await loadEntry(Math.min(state.currentIndex + 1, state.entries.length - 1));
  } catch (error) {
    updateStatus(error.message);
  }
});
els.deleteImageBtn.addEventListener("click", () => deleteCurrentImage().catch((error) => updateStatus(error.message)));
els.newBoxBtn.addEventListener("click", () => {
  if (state.readOnly) {
    updateStatus("Read-only: already saved");
    return;
  }
  state.createMode = !state.createMode;
  updateStatus(state.createMode ? "Draw on the image to create a box" : "");
});
els.deleteBoxBtn.addEventListener("click", deleteSelected);
els.prevIconBtn.addEventListener("click", () => selectElementByDelta(-1));
els.nextIconBtn.addEventListener("click", () => selectElementByDelta(1));
els.applyFieldsBtn.addEventListener("click", applyFields);
els.zoomRange.addEventListener("input", (event) => setZoom(Number(event.target.value) / 100));

els.canvasStage.addEventListener("pointerdown", (event) => {
  if (state.createMode) {
    beginDraft(event);
  }
});

window.addEventListener("pointermove", (event) => {
  if (state.dragMode) {
    applyDrag(event);
  } else if (state.draftBox) {
    updateDraft(event);
  }
});

window.addEventListener("pointerup", () => {
  if (state.dragMode) {
    stopDrag();
  }
  if (state.draftBox) {
    finishDraft();
  }
});

window.addEventListener("keydown", async (event) => {
  if (event.key === "Delete") {
    deleteSelected();
  }
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    selectElementByDelta(-1);
  }
  if (event.key === "ArrowRight") {
    event.preventDefault();
    selectElementByDelta(1);
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    try {
      await saveCurrent();
    } catch (error) {
      updateStatus(error.message);
    }
  }
});

els.zoomRange.value = "70";
setZoom(0.7);
init().catch((error) => {
  console.error(error);
  updateStatus(error.message);
});
