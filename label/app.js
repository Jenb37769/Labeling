const state = {
  entries: [],
  currentIndex: -1,
  currentStem: "",
  currentSource: "",
  document: null,
  selectedId: null,
  zoom: 1,
  createMode: false,
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
  draftBox: document.getElementById("draftBox"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  saveBtn: document.getElementById("saveBtn"),
  saveNextBtn: document.getElementById("saveNextBtn"),
  deleteImageBtn: document.getElementById("deleteImageBtn"),
  newBoxBtn: document.getElementById("newBoxBtn"),
  deleteBoxBtn: document.getElementById("deleteBoxBtn"),
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
      state.selectedId = element.id;
      renderAll();
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
    box.className = `anno-box${element.id === state.selectedId ? " selected" : ""}`;
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
  els.fieldId.value = element?.id || "";
  els.fieldName.value = element?.name || "";
  els.fieldType.value = element?.type || "icon_button";
  els.fieldClickable.checked = element?.clickable ?? true;
  els.fieldX1.value = element?.bbox?.[0] ?? "";
  els.fieldY1.value = element?.bbox?.[1] ?? "";
  els.fieldX2.value = element?.bbox?.[2] ?? "";
  els.fieldY2.value = element?.bbox?.[3] ?? "";
}

function renderAll(extraStatus = "") {
  renderFrameList();
  renderBoxes();
  renderBoxList();
  renderFields();
  updateStatus(extraStatus);
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
  if (event.target.dataset.handle) {
    return;
  }
  event.preventDefault();
  state.selectedId = id;
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
  event.preventDefault();
  event.stopPropagation();
  state.selectedId = id;
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
  };
  state.document.elements.push(element);
  state.selectedId = element.id;
  state.createMode = false;
  renderAll("New box created");
}

function deleteSelected() {
  if (!state.document || !state.selectedId) {
    return;
  }
  state.document.elements = state.document.elements.filter((element) => element.id !== state.selectedId);
  state.selectedId = state.document.elements[0]?.id || null;
  renderAll("Box deleted");
}

function applyFields() {
  const element = selectedElement();
  if (!element) {
    return;
  }
  updateElement(element.id, (draft) => {
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
  if (!state.document || !state.currentStem) {
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
  state.document = data.document;
  state.selectedId = state.document.elements[0]?.id || null;
  state.createMode = false;
  const imageUrl = `${data.image_url}?t=${Date.now()}`;
  els.mainImage.onload = () => {
    els.canvasStage.style.width = `${els.mainImage.naturalWidth}px`;
    els.canvasStage.style.height = `${els.mainImage.naturalHeight}px`;
    setZoom(Number(els.zoomRange.value) / 100);
    renderAll();
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
  state.createMode = !state.createMode;
  updateStatus(state.createMode ? "Draw on the image to create a box" : "");
});
els.deleteBoxBtn.addEventListener("click", deleteSelected);
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
