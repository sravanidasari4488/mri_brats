import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const AXIS_NAME = ["Sagittal", "Coronal", "Axial"];
const TUMOR_ORDER = ["enhancing", "necrosis", "edema"];
const GROUP_ORDER = [
  "Tissue",
  "Cortex",
  "Subcortical",
  "Ventricles",
  "Brainstem & Cerebellum",
  "Tumor",
  "Tracts",
  "Brain",
  "Anatomy",
];
const SLICE_SEQS = ["flair", "t1", "t1c", "t2"];
const BLOOD_NAME = "Possible blood products (experimental, T1-hyperintense)";
const ORIENT_LETTERS = {
  0: { left: "A", right: "P", top: "S", bottom: "I" },
  1: { left: "R", right: "L", top: "S", bottom: "I" },
  2: { left: "R", right: "L", top: "A", bottom: "P" },
};
const PRESETS = [
  { id: "glass-tumor", label: "Glass + tumor" },
  { id: "tumor-focus", label: "Tumor focus" },
  { id: "anatomy", label: "Anatomy" },
  { id: "tumor-tracts", label: "Tumor + Tracts" },
];
const TUMOR_LABELS = new Set(["enhancing", "necrosis", "edema"]);
const TISSUE_LABELS = {
  "Left Cortex": "Cortical grey matter (left)",
  "Right Cortex": "Cortical grey matter (right)",
  "White Matter": "White matter",
};
const DEEP_GROUPS = new Set(["Subcortical", "Ventricles"]);
const GLASS_GROUPS = new Set(["Cortex", "Brainstem & Cerebellum"]);
const GLASS_COLOR = new THREE.Color(0x9fb8d6);

const params = new URLSearchParams(location.search);
const caseId = params.get("case");
const uploadScreen = document.querySelector("#upload-screen");
const viewerRoot = document.querySelector("#viewer");
const tooltip = document.querySelector("#tooltip");
const tractNote = document.querySelector("#tract-note");

const SCAN_ROLES = {
  t2f: { name: "FLAIR", help: "Highlights swelling around the tumor.", required: true },
  t1n: { name: "T1", help: "Plain anatomy scan, taken without contrast dye.", required: true },
  t1c: { name: "T1 with contrast", help: "Taken after contrast dye. Shows the active tumor.", required: true },
  t2w: { name: "T2", help: "Shows fluid and swelling.", required: true },
  seg: { name: "Tumor outline", help: "Optional. Drawn by a radiologist; only used for comparison.", required: false },
  synthseg: { name: "Brain structure map", help: "Optional. Used with “Show brain structures”.", required: false },
};
const SCAN_ORDER = ["t2f", "t1n", "t1c", "t2w", "seg", "synthseg"];
const REQUIRED_SCANS = SCAN_ORDER.filter((role) => SCAN_ROLES[role].required);
const upload = { files: [], folder: null, caseId: null, skipped: 0, dicom: false };

setupUpload();

if (caseId) {
  uploadScreen.hidden = true;
  viewerRoot.hidden = false;
  if (params.get("preview") === "1") {
    const panel = document.querySelector("#panel");
    if (panel) panel.hidden = true;
    const slices = document.querySelector("#slices-panel");
    if (slices) slices.hidden = true;
  }
  startViewer(caseId);
}

function setupUpload() {
  const form = document.querySelector("#upload-form");
  const zone = document.querySelector("#drop-zone");
  const folderInput = document.querySelector("#folder-input");
  const filesInput = document.querySelector("#files-input");
  form.addEventListener("submit", onUpload);
  document.querySelector("#pick-folder").addEventListener("click", () => folderInput.click());
  document.querySelector("#pick-files").addEventListener("click", () => filesInput.click());
  document.querySelector("#scan-clear").addEventListener("click", () => {
    Object.assign(upload, { files: [], folder: null, caseId: null, skipped: 0, dicom: false });
    renderScanReview();
  });
  folderInput.addEventListener("change", () => {
    const files = Array.from(folderInput.files);
    const first = files[0]?.webkitRelativePath || "";
    addScanFiles(files, first.includes("/") ? first.split("/")[0] : null);
    folderInput.value = "";
  });
  filesInput.addEventListener("change", () => {
    addScanFiles(Array.from(filesInput.files), null);
    filesInput.value = "";
  });
  zone.addEventListener("click", (event) => {
    if (!event.target.closest("button")) folderInput.click();
  });
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      folderInput.click();
    }
  });
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("dragging");
    });
  }
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", async (event) => {
    event.preventDefault();
    zone.classList.remove("dragging");
    const { files, folder } = await filesFromDrop(event.dataTransfer);
    addScanFiles(files, folder);
  });
}

async function filesFromDrop(dataTransfer) {
  // Entries must be taken before the first await; the browser clears the list afterwards.
  const entries = Array.from(dataTransfer.items || [])
    .map((item) => (item.webkitGetAsEntry ? item.webkitGetAsEntry() : null))
    .filter(Boolean);
  if (!entries.length) return { files: Array.from(dataTransfer.files || []), folder: null };
  const files = [];
  const folder = entries.find((entry) => entry.isDirectory)?.name || null;
  for (const entry of entries) await walkEntry(entry, files, 0);
  return { files, folder };
}

async function walkEntry(entry, out, depth) {
  if (entry.isFile) {
    out.push(await new Promise((resolve, reject) => entry.file(resolve, reject)));
    return;
  }
  if (!entry.isDirectory || depth > 4) return;
  const reader = entry.createReader();
  for (;;) {
    const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) break;
    for (const child of batch) await walkEntry(child, out, depth + 1);
  }
}

function isNifti(name) {
  const lower = name.toLowerCase();
  return lower.endsWith(".nii") || lower.endsWith(".nii.gz");
}

async function addScanFiles(files, folder) {
  const status = document.querySelector("#upload-status");
  const nifti = files.filter((file) => isNifti(file.name));
  upload.skipped += files.length - nifti.length;
  upload.dicom ||= files.some((file) => /\.dcm$/i.test(file.name) || file.name.toUpperCase() === "DICOMDIR");
  if (folder) upload.folder = folder;
  for (const file of nifti) {
    const duplicate = upload.files.some((item) => item.file.name === file.name && item.file.size === file.size);
    if (!duplicate) upload.files.push({ file, role: null, manual: false });
  }
  status.textContent = "";
  if (upload.files.length) {
    status.textContent = "Checking the files…";
    try {
      const response = await fetch("/api/identify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ names: upload.files.map((item) => item.file.name), folder: upload.folder }),
      });
      const payload = await readJson(response);
      upload.files.forEach((item, index) => {
        if (!item.manual) item.role = payload.roles[index] || null;
      });
      upload.caseId = payload.case_id;
      status.textContent = "";
    } catch (error) {
      status.textContent = `Could not read the file names: ${error.message}`;
    }
  }
  renderScanReview();
}

function renderScanReview() {
  const review = document.querySelector("#scan-review");
  const checklist = document.querySelector("#scan-checklist");
  const unknown = document.querySelector("#scan-unknown");
  const hint = document.querySelector("#scan-hint");
  const submit = document.querySelector("#upload-submit");
  const anything = upload.files.length || upload.skipped;
  review.hidden = !anything;
  checklist.replaceChildren();
  unknown.replaceChildren();
  const byRole = (role) => upload.files.filter((item) => item.role === role);

  for (const role of SCAN_ORDER) {
    const info = SCAN_ROLES[role];
    const matches = byRole(role);
    if (!info.required && !matches.length) continue;
    const li = document.createElement("li");
    li.className = matches.length === 1 ? "found" : matches.length > 1 ? "conflict" : "missing";
    const mark = matches.length === 1 ? "✓" : matches.length > 1 ? "!" : "✗";
    const title = document.createElement("strong");
    title.textContent = `${mark} ${info.name}`;
    const detail = document.createElement("span");
    detail.textContent = matches.length ? matches.map((item) => item.file.name).join(", ") : "not found";
    const help = document.createElement("small");
    help.textContent = info.help;
    li.append(title, detail, help);
    checklist.append(li);
  }

  if (upload.files.length) {
    const heading = document.createElement("h3");
    heading.textContent = "Files";
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "If a file was matched wrongly, change it here.";
    unknown.append(heading, note);
    for (const item of upload.files) {
      const row = document.createElement("label");
      row.className = "scan-row";
      const name = document.createElement("span");
      name.textContent = item.file.name;
      const select = document.createElement("select");
      select.append(new Option(item.role ? "Not used" : "Not sure / not used", ""));
      for (const role of SCAN_ORDER) select.append(new Option(SCAN_ROLES[role].name, role));
      select.value = item.role || "";
      select.addEventListener("change", () => {
        item.role = select.value || null;
        item.manual = true;
        renderScanReview();
      });
      row.append(name, select);
      unknown.append(row);
    }
  }

  const missing = REQUIRED_SCANS.filter((role) => byRole(role).length === 0);
  const conflicts = SCAN_ORDER.filter((role) => byRole(role).length > 1);
  const messages = [];
  if (upload.dicom) {
    messages.push(
      "Some files are DICOM (.dcm), the format that comes straight from the scanner or a hospital CD. " +
        "This viewer needs NIfTI files (.nii or .nii.gz). Ask the hospital for NIfTI, or convert the folder with a tool such as dcm2niix."
    );
  }
  if (missing.length && upload.files.length) {
    messages.push(
      `Still needed: ${missing.map((role) => SCAN_ROLES[role].name).join(", ")}. ` +
        "If one of the files above is that scan, pick it from its menu. Otherwise add the missing file."
    );
  }
  if (conflicts.length) {
    messages.push(
      `More than one file is marked as ${conflicts.map((role) => SCAN_ROLES[role].name).join(" and ")}. ` +
        "Set the extra file to “Not used”."
    );
  }
  if (upload.skipped) {
    messages.push(`Skipped ${upload.skipped} file${upload.skipped === 1 ? "" : "s"} that ${upload.skipped === 1 ? "is" : "are"} not an MRI volume.`);
  }
  if (!missing.length && !conflicts.length && upload.files.length) {
    messages.unshift("All four scans found. You can build the 3D model.");
  }
  hint.textContent = messages.join(" ");
  hint.className = !missing.length && !conflicts.length && upload.files.length ? "ok" : "";
  submit.disabled = Boolean(missing.length || conflicts.length || !upload.files.length);
}

async function onUpload(event) {
  event.preventDefault();
  const status = document.querySelector("#upload-status");
  const form = event.currentTarget;
  const submit = document.querySelector("#upload-submit");
  const body = new FormData();
  for (const item of upload.files) {
    if (item.role) body.append(item.role, item.file, item.file.name);
  }
  if (upload.caseId) body.set("case_id", upload.caseId);
  body.set("anatomy", form.anatomy.checked ? "1" : "0");
  if (form.tracts.checked) body.set("tracts", "1");
  submit.disabled = true;
  renderJobProgress(null);
  status.textContent = "Starting reconstruction…";
  try {
    const response = await fetch("/api/cases", { method: "POST", body });
    const payload = await readJson(response);
    if (!response.ok) throw new Error(payload.error || response.statusText);
    if (payload.status === "done" && payload.viewer) {
      status.textContent = "Done. Opening the 3D view…";
      location.href = payload.viewer;
      return;
    }
    const jobUrl = payload.job || `/api/cases/${encodeURIComponent(payload.case_id)}/job`;
    const done = await pollJob(jobUrl, status);
    status.textContent = "Done. Opening the 3D view…";
    location.href = done.viewer || payload.viewer || `/viewer?case=${encodeURIComponent(payload.case_id)}`;
  } catch (error) {
    status.textContent = error.message;
    submit.disabled = false;
  }
}

async function pollJob(url, status) {
  for (;;) {
    const response = await fetch(url);
    const job = await readJson(response);
    if (!response.ok) throw new Error(job.error || response.statusText);
    renderJobProgress(job);
    const stage = job.stage || "Loading";
    const elapsed = formatSeconds(job.elapsed);
    status.textContent = job.status === "done"
      ? `Done in ${elapsed}.`
      : `${stage}… ${elapsed} so far. Tumor plus brain structures usually take about 10 minutes. You can leave this page open.`;
    if (job.status === "done") return job;
    if (job.status === "error") throw new Error(job.error || "Reconstruction failed");
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

function renderJobProgress(job) {
  const host = document.querySelector("#job-progress");
  const list = document.querySelector("#job-stages");
  const fill = document.querySelector("#job-bar-fill");
  if (!host || !list) return;
  if (!job) {
    host.hidden = true;
    list.replaceChildren();
    if (fill) fill.style.width = "0%";
    return;
  }
  host.hidden = false;
  const order = ["Loading", "Tumor", "Anatomy", "Lobes", "Meshing", "Done"];
  const stages = job.stages || {};
  list.replaceChildren();
  let doneCount = 0;
  for (const name of order) {
    const info = stages[name] || { status: name === job.stage ? "running" : "pending", seconds: null };
    const li = document.createElement("li");
    li.className = `is-${info.status || "pending"}`;
    const label = document.createElement("span");
    label.textContent = name;
    const time = document.createElement("span");
    time.textContent = info.status === "skipped"
      ? "skipped"
      : Number.isFinite(Number(info.seconds))
        ? formatSeconds(info.seconds)
        : info.status === "running"
          ? "…"
          : "";
    li.append(label, time);
    list.append(li);
    if (info.status === "done" || info.status === "skipped") doneCount += 1;
  }
  if (fill) {
    const fraction = job.status === "done" ? 1 : doneCount / Math.max(order.length, 1);
    fill.style.width = `${Math.round(fraction * 100)}%`;
  }
}

function formatSeconds(value) {
  const seconds = Math.max(0, Number(value) || 0);
  const whole = Math.round(seconds);
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return minutes ? `${minutes}:${String(rest).padStart(2, "0")}` : `${seconds.toFixed(1)}s`;
}

async function startViewer(id) {
  const manifest = await fetch(`/api/cases/${encodeURIComponent(id)}/manifest.json?v=anatomy1`).then(readJson);
  document.querySelector("#case-title").textContent = manifest.case_id || id;
  showAnatomyWarning(manifest);
  buildPanel(manifest);
  buildProximity(manifest);
  buildTumorCard(manifest);
  buildPresetButtons();

  const canvas = document.querySelector("#view");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x0b0f17);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.localClippingEnabled = true;
  renderer.autoClear = false;
  renderer.sortObjects = true;
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = "label-layer";
  viewerRoot.appendChild(labelRenderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0f17);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.autoRotate = false;
  controls.autoRotateSpeed = 0.6;
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  const key = new THREE.DirectionalLight(0xf2f6ff, 1.15);
  key.position.set(0.55, -1.1, 1.6);
  key.castShadow = false;
  scene.add(key);

  const planes = [0, 1, 2].map(() => new THREE.Plane());
  const sliceMeshes = planes.map(() => makeSliceMesh());
  sliceMeshes.forEach((group) => scene.add(group));
  const gizmo = makeOrientationGizmo();

  const gltf = await new GLTFLoader().loadAsync(
    `/api/cases/${encodeURIComponent(id)}/scene.glb?v=slices3`,
  );
  scene.add(gltf.scene);
  const pickable = [];
  const byName = new Map();
  gltf.scene.traverse((object) => {
    if (!object.isMesh) return;
    const name = structureName(object, manifest);
    if (!name) return;
    orientNormals(object.geometry);
    object.name = name;
    const entry = manifest.structures[name];
    const opacity = isGlass(name, entry) ? 0.7 : entry.color[3] / 255;
    const material = structureMaterial(entry, name, object);
    material.clippingPlanes = [];
    applyOpacity(material, opacity);
    object.material = material;
    object.renderOrder = renderOrderFor(name, entry);
    object.userData.entry = entry;
    object.userData.group = structureGroup(entry);
    object.userData.baseOpacity = opacity;
    object.userData.glass = isGlass(name, entry);
    object.visible = entry.hidden_by_default !== true;
    if (name === "enhancing") inflateAroundCentroid(object, 1.02);
    pickable.push(object);
    byName.set(name, object);
  });
  const labels = attachLabels(scene, byName, manifest);

  let volume = await fetch(`/api/cases/${encodeURIComponent(id)}/volume?modality=flair`).then(readJson);
  const clipState = volume.shape.map((length, axis) => ({
    enabled: false,
    index: Math.floor((length - 1) / 2),
  }));
  const sliceView = {
    axis: 2,
    index: clipState[2].index,
    overlayOpacity: 0.85,
    windows: copyWindows(volume.windows),
    selectedSeq: "flair",
    crosshair: null,
  };
  const marker = makeCrosshairMarker();
  scene.add(marker);
  const onClipChange = () => {
    const enabled = clipState.findIndex((state) => state.enabled);
    if (enabled >= 0) {
      sliceView.axis = enabled;
      sliceView.index = clipState[enabled].index;
    } else {
      sliceView.index = clipState[sliceView.axis].index;
    }
    const slider = document.querySelector("#slice-index");
    const readout = document.querySelector("#slice-index-readout");
    if (slider) {
      slider.max = String(volume.shape[sliceView.axis] - 1);
      slider.value = String(sliceView.index);
    }
    if (readout) readout.textContent = String(sliceView.index);
    document.querySelectorAll("[data-orient]").forEach((button) => {
      button.classList.toggle("is-active", Number(button.dataset.orient) === sliceView.axis);
    });
    refreshClips();
    updateOrientLetters(volume, sliceView);
    redrawSlices(volume, sliceView);
  };
  buildClipControls(volume, clipState, onClipChange);
  wireSlicePanel(volume, clipState, sliceView, marker, () => refreshClips());
  redrawSlices(volume, sliceView);

  document.querySelector("#modality").addEventListener("change", async (event) => {
    volume = await fetch(
      `/api/cases/${encodeURIComponent(id)}/volume?modality=${event.target.value}`,
    ).then(readJson);
    clipState.forEach((state, axis) => {
      state.index = Math.min(state.index, volume.shape[axis] - 1);
    });
    sliceView.index = Math.min(sliceView.index, volume.shape[sliceView.axis] - 1);
    if (!Object.keys(sliceView.windows).length) sliceView.windows = copyWindows(volume.windows);
    buildClipControls(volume, clipState, onClipChange);
    refreshClips();
    redrawSlices(volume, sliceView);
  });

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let hoveredName = null;
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(pickable, false).find((item) => item.object.visible);
    hoveredName = hit ? hit.object.name : null;
    if (!hit) {
      tooltip.hidden = true;
      return;
    }
    const entry = hit.object.userData.entry;
    tooltip.hidden = false;
    tooltip.style.left = `${event.clientX + 12}px`;
    tooltip.style.top = `${event.clientY + 12}px`;
    tooltip.textContent = `${hit.object.name}\n${Number(entry.volume_cm3).toFixed(2)} cm³`;
  });
  canvas.addEventListener("pointerleave", () => {
    hoveredName = null;
    tooltip.hidden = true;
  });

  wireLegend(byName, camera, controls);
  wirePresets(byName, camera, controls, gltf.scene);
  document.querySelector("#screenshot").addEventListener("click", () => {
    saveScreenshot(renderer, labelRenderer, scene, camera, byName, manifest);
  });
  refreshClips();
  wireGlassControls(byName);
  const autoRotate = document.querySelector("#auto-rotate");
  if (autoRotate) {
    autoRotate.addEventListener("change", () => {
      controls.autoRotate = autoRotate.checked;
    });
  }
  applyPreset("glass-tumor", byName, camera, controls, gltf.scene);
  frameTumorView(camera, controls, byName);
  window.__viewerReady = true;

  function refreshClips() {
    const active = [];
    clipState.forEach((state, axis) => {
      const normal = axisDirection(volume.affine, axis).multiplyScalar(-1);
      const center = voxelToWorld(volume.affine, centerIndex(volume.shape, axis, state.index));
      planes[axis].setFromNormalAndCoplanarPoint(normal, center);
      if (state.enabled) active.push(planes[axis]);
      updateSliceMesh(sliceMeshes[axis], volume, axis, state.index, state.enabled, sliceView);
      const clipOthers = planes.filter((_plane, other) => other !== axis && clipState[other].enabled);
      sliceMeshes[axis].userData.mri.material.clippingPlanes = clipOthers;
      sliceMeshes[axis].userData.overlay.material.clippingPlanes = clipOthers;
    });
    pickable.forEach((mesh) => {
      mesh.material.clippingPlanes = active;
      mesh.material.needsUpdate = true;
    });
    updateTractNote(byName);
  }

  function resize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    renderer.setSize(width, height, false);
    labelRenderer.setSize(width, height);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }

  function frame() {
    resize();
    controls.update();
    updateLabels(labels, byName, camera, hoveredName);
    renderer.clear();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
    renderOrientationGizmo(renderer, camera, controls, gizmo);
    requestAnimationFrame(frame);
  }
  window.addEventListener("resize", resize);
  requestAnimationFrame(frame);
}

function structureGroup(entry) {
  if (entry.group) return entry.group;
  if (entry.source === "atlas-estimated") return "Tracts";
  if (entry.source === "model") return "Tumor";
  return "Anatomy";
}

function showAnatomyWarning(manifest) {
  const el = document.querySelector("#anatomy-warning");
  if (!el) return;
  const text = manifest?.anatomy_warning;
  el.hidden = !text;
  el.textContent = text || "";
}

function buildPanel(manifest) {
  const groups = new Map();
  const names = Object.keys(manifest.structures).sort((a, b) => structureRank(a, manifest) - structureRank(b, manifest));
  for (const name of names) {
    const entry = manifest.structures[name];
    const group = structureGroup(entry);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push([name, entry]);
  }
  const ordered = [...groups.keys()].sort((a, b) => groupRank(a) - groupRank(b));
  const root = document.querySelector("#groups");
  root.replaceChildren();
  for (const title of ordered) {
    const rows = groups.get(title);
    const section = document.createElement("section");
    section.className = "group";
    section.dataset.group = title;
    const header = document.createElement("header");
    const heading = document.createElement("h2");
    heading.textContent = title;
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.01";
    slider.value = "1";
    slider.dataset.role = "group-opacity";
    slider.title = `${title} opacity`;
    slider.setAttribute("aria-label", `${title} opacity`);
    header.append(heading, slider);
    section.appendChild(header);
    for (const [name, entry] of rows) section.appendChild(legendItem(name, entry));
    root.appendChild(section);
  }
}

function structureLabel(name, entry) {
  return entry?.full_name || TISSUE_LABELS[name] || name;
}

function legendItem(name, entry) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "legend-item";
  button.dataset.name = name;
  if (entry?.volume_only) button.classList.add("is-volume");
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = `rgb(${entry.color[0]}, ${entry.color[1]}, ${entry.color[2]})`;
  const label = document.createElement("span");
  label.className = "name";
  label.textContent = structureLabel(name, entry);
  const volume = document.createElement("span");
  volume.className = "volume";
  volume.textContent = Number.isFinite(Number(entry.volume_cm3))
    ? `${Number(entry.volume_cm3).toFixed(2)} cm³`
    : "";
  button.append(swatch, label, volume);
  return button;
}

function groupRank(name) {
  const index = GROUP_ORDER.indexOf(name);
  return index === -1 ? GROUP_ORDER.length : index;
}

function buildPresetButtons() {
  const host = document.querySelector("#presets");
  host.replaceChildren();
  for (const preset of PRESETS) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.preset = preset.id;
    button.textContent = preset.label;
    host.appendChild(button);
  }
}

function buildTumorCard(manifest) {
  const host = document.querySelector("#tumor-location");
  const summary = manifest.tumor_location;
  if (!summary) {
    host.textContent = "No tumor location summary.";
    return;
  }
  host.replaceChildren();
  const note = document.createElement("p");
  note.textContent = `Overlaps, or within ${Number(summary.margin_mm || 10).toFixed(0)} mm.`;
  host.appendChild(note);
  for (const region of ["WT", "ET"]) {
    const block = document.createElement("div");
    block.className = "location-region";
    const heading = document.createElement("h3");
    heading.textContent = region === "WT" ? "Whole tumor" : "Enhancing tumor";
    const list = document.createElement("ul");
    const rows = summary[region] || [];
    if (!rows.length) {
      const item = document.createElement("li");
      item.textContent = "None within range.";
      list.appendChild(item);
    }
    for (const row of rows) {
      const item = document.createElement("li");
      const where = row.relation === "overlaps" ? "overlaps" : `${Number(row.distance_mm).toFixed(1)} mm`;
      item.textContent = `${row.name} — ${where}`;
      list.appendChild(item);
    }
    block.append(heading, list);
    host.appendChild(block);
  }
  const blood = document.querySelector("#blood-note");
  if (blood) {
    const note = manifest.blood_products_note || (manifest.structures && manifest.structures[BLOOD_NAME]?.note);
    blood.hidden = !note;
    blood.textContent = note || "";
  }
}

function structureRank(name, manifest) {
  const source = manifest.structures[name].source;
  const tumor = TUMOR_ORDER.indexOf(name);
  if (source === "model") return (tumor === -1 ? 9 : tumor);
  if (source === "atlas-estimated") return 100;
  if (name === "brain") return 80;
  return 40;
}

function renderOrderFor(name, entry) {
  if (name === "necrosis" || name === "enhancing") return 1;
  if (name === "edema") return 2;
  if (entry && isGlass(name, entry)) return 10;
  return 5;
}

function isGlass(name, entry) {
  if (!entry) return false;
  if (entry.glass) return true;
  if (name === "White Matter") return true;
  return GLASS_GROUPS.has(entry.group) || name === "brain";
}

function inflateAroundCentroid(object, scale) {
  object.geometry.computeBoundingBox();
  const center = new THREE.Vector3();
  object.geometry.boundingBox.getCenter(center);
  object.geometry.translate(-center.x, -center.y, -center.z);
  object.geometry.scale(scale, scale, scale);
  object.geometry.translate(center.x, center.y, center.z);
}

function structureKey(name, manifest) {
  if (!name) return null;
  if (manifest.structures[name]) return name;
  const spaced = name.replace(/_/g, " ");
  if (manifest.structures[spaced]) return spaced;
  return null;
}

function structureName(object, manifest) {
  const candidates = [object.name, object.parent?.name, object.parent?.parent?.name];
  for (const candidate of candidates) {
    const name = structureKey(candidate, manifest);
    if (name) return name;
  }
  return null;
}

function srgbColor(rgb) {
  return new THREE.Color().setHex((rgb[0] << 16) | (rgb[1] << 8) | rgb[2]);
}

function orientNormals(geometry) {
  if (!geometry || !geometry.getAttribute("position")) return;
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  const center = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  const pos = geometry.getAttribute("position");
  const nrm = geometry.getAttribute("normal");
  if (!nrm) return;
  const point = new THREE.Vector3();
  const normal = new THREE.Vector3();
  let inward = 0;
  let outward = 0;
  const step = Math.max(1, Math.floor(pos.count / 250));
  for (let i = 0; i < pos.count; i += step) {
    point.fromBufferAttribute(pos, i).sub(center);
    normal.fromBufferAttribute(nrm, i);
    if (normal.dot(point) < 0) inward += 1;
    else outward += 1;
  }
  if (inward <= outward) return;
  for (let i = 0; i < nrm.count; i += 1) {
    nrm.setXYZ(i, -nrm.getX(i), -nrm.getY(i), -nrm.getZ(i));
  }
  nrm.needsUpdate = true;
}

function glassMaterial(object, opacity, baseColor) {
  const hasColor = Boolean(object.geometry?.getAttribute("color"));
  return new THREE.ShaderMaterial({
    uniforms: {
      baseColor: { value: (baseColor || GLASS_COLOR).clone() },
      opacity: { value: opacity },
      intensity: { value: 1.35 },
      tintMix: { value: 0 },
    },
    vertexShader: `
      #include <clipping_planes_pars_vertex>
      varying vec3 vNormal;
      varying vec3 vViewDir;
      varying vec3 vLobe;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vViewDir = normalize(-mvPosition.xyz);
        vLobe = vec3(1.0);
        #ifdef USE_COLOR
        vLobe = color.rgb;
        #endif
        gl_Position = projectionMatrix * mvPosition;
        #include <clipping_planes_vertex>
      }
    `,
    fragmentShader: `
      #include <common>
      #include <clipping_planes_pars_fragment>
      uniform vec3 baseColor;
      uniform float opacity;
      uniform float intensity;
      uniform float tintMix;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      varying vec3 vLobe;
      void main() {
        #include <clipping_planes_fragment>
        float fresnel = pow(1.0 - abs(dot(normalize(vNormal), normalize(vViewDir))), 2.5);
        vec3 base = mix(baseColor, vLobe, tintMix);
        vec3 color = base * fresnel * intensity;
        float alpha = fresnel * opacity;
        gl_FragColor = vec4(color, alpha);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    vertexColors: hasColor,
    clipping: true,
  });
}

function structureMaterial(entry, name, object) {
  if (isGlass(name, entry)) {
    const rgb = entry.color || [159, 184, 214];
    const color = name === "White Matter" || entry.group === "Ventricles"
      ? srgbColor(rgb)
      : GLASS_COLOR;
    const opacity = name === "White Matter" ? 0.4 : entry.group === "Ventricles" ? 0.45 : 0.7;
    return glassMaterial(object, opacity, color);
  }
  if (name === BLOOD_NAME) {
    return new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x6e1620),
      roughness: 0.55,
      metalness: 0.04,
      side: THREE.DoubleSide,
    });
  }
  if (name === "edema") {
    return new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0xf5c542),
      roughness: 0.4,
      clearcoat: 0.3,
      transparent: true,
      opacity: 0.25,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
  }
  if (name === "necrosis" || name === "enhancing") {
    const hex = name === "necrosis" ? 0xc0392b : 0x3b82f6;
    return new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(hex),
      roughness: 0.4,
      clearcoat: 0.3,
      metalness: 0.05,
      side: THREE.DoubleSide,
    });
  }
  const color = srgbColor(entry.color).lerp(new THREE.Color(0xffffff), 0.28);
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.55,
    metalness: 0.04,
    side: THREE.DoubleSide,
  });
}

function applyOpacity(material, opacity) {
  if (material.uniforms && material.uniforms.opacity) {
    material.uniforms.opacity.value = opacity;
    return;
  }
  material.opacity = opacity;
  material.transparent = opacity < 0.999;
  if (material.depthWrite !== false || opacity >= 0.85) {
    material.depthWrite = opacity >= 0.85;
  }
}

function wireLegend(byName, camera, controls) {
  document.querySelectorAll(".legend-item").forEach((row) => {
    const name = row.dataset.name;
    row.addEventListener("click", () => {
      if (row.classList.contains("is-volume")) return;
      const mesh = byName.get(name);
      if (!mesh) return;
      mesh.visible = !mesh.visible;
      syncLegend(byName);
    });
    row.addEventListener("dblclick", (event) => {
      event.preventDefault();
      isolateStructure(name, byName, camera, controls);
    });
  });
  document.querySelectorAll("[data-role=group-opacity]").forEach((slider) => {
    slider.addEventListener("input", () => {
      const group = slider.closest("[data-group]").dataset.group;
      setGroupOpacity(byName, group, Number(slider.value));
    });
  });
  syncGroupSliders(byName);
  syncLegend(byName);
}

function wirePresets(byName, camera, controls, root) {
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.preset, byName, camera, controls, root));
  });
}

function applyPreset(id, byName, camera, controls, root) {
  const names = [...byName.keys()];
  if (id === "glass-tumor") {
    names.forEach((name) => {
      const mesh = byName.get(name);
      const group = mesh.userData.group;
      const hidden = mesh.userData.entry?.hidden_by_default;
      const show = !hidden && (Boolean(mesh.userData.glass) || group === "Tumor" || name === "brain");
      setVisible(byName, name, show);
      if (group === "Tumor") restoreOpacity(mesh);
    });
    frameTumorView(camera, controls, byName);
  } else if (id === "anatomy") {
    names.forEach((name) => setVisible(byName, name, true));
    syncGroupSliders(byName);
    frameCamera(camera, controls, root);
  } else if (id === "tumor-focus") {
    names.forEach((name) => {
      const mesh = byName.get(name);
      const hidden = mesh.userData.entry?.hidden_by_default;
      const show = !hidden && (mesh.userData.glass || mesh.userData.group === "Tumor" || name === "brain");
      setVisible(byName, name, show);
    });
    byName.forEach((mesh) => {
      if (mesh.userData.glass) setBaseOpacity(mesh, 0.35);
      else if (mesh.userData.group === "Tumor") restoreOpacity(mesh);
    });
    setSlider("Cortex", 1);
    setSlider("Tumor", 1);
    frameTumorView(camera, controls, byName);
  } else if (id === "tumor-tracts") {
    names.forEach((name) => {
      const mesh = byName.get(name);
      const show = mesh.userData.group === "Tumor" || mesh.userData.group === "Tracts";
      setVisible(byName, name, show);
      if (!show) return;
      if (mesh.userData.group === "Tumor") restoreOpacity(mesh);
      else restoreOpacity(mesh);
    });
    setSlider("Tumor", 1);
    setSlider("Tracts", 1);
    frameObjects(camera, controls, [
      ...meshesInGroup(byName, "Tumor"),
      ...meshesInGroup(byName, "Tracts"),
    ]);
  }
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.preset === id);
  });
  syncLegend(byName);
}

function frameTumorView(camera, controls, byName) {
  const tumors = meshesInGroup(byName, "Tumor");
  const glass = [...byName.values()].filter((mesh) => mesh.userData.glass || mesh.name === "brain");
  const frame = glass.length ? glass : [...byName.values()];
  const box = new THREE.Box3();
  frame.forEach((mesh) => {
    if (mesh.visible) box.expandByObject(mesh);
  });
  if (box.isEmpty()) {
    frame.forEach((mesh) => box.expandByObject(mesh));
  }
  const center = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
  let target = center.clone();
  if (tumors.length) {
    const tumorBox = new THREE.Box3();
    tumors.forEach((mesh) => tumorBox.expandByObject(mesh));
    if (!tumorBox.isEmpty()) {
      target.lerp(tumorBox.getCenter(new THREE.Vector3()), 0.55);
    }
  }
  const size = box.getSize(new THREE.Vector3()).length() || 100;
  controls.target.copy(target);
  camera.position.copy(target).add(new THREE.Vector3(size * 0.72, -size * 1.05, size * 0.48));
  camera.near = Math.max(size / 1000, 0.01);
  camera.far = size * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

function wireGlassControls(byName) {
  const opacity = document.querySelector("#glass-opacity");
  const intensity = document.querySelector("#glass-intensity");
  const tint = document.querySelector("#lobe-tint");
  const apply = () => {
    const op = opacity ? Number(opacity.value) : 0.7;
    const rim = intensity ? Number(intensity.value) : 1.35;
    const mix = tint && tint.checked ? 0.25 : 0;
    byName.forEach((mesh) => {
      if (!mesh.userData.glass || !mesh.material.uniforms) return;
      const name = mesh.name;
      const scale = name === "White Matter" ? 0.55 : mesh.userData.entry?.group === "Ventricles" ? 0.65 : 1;
      mesh.material.uniforms.opacity.value = op * scale;
      mesh.material.uniforms.intensity.value = rim;
      mesh.material.uniforms.tintMix.value = mesh.userData.entry?.lobe_tints ? mix : 0;
    });
  };
  if (opacity) opacity.addEventListener("input", apply);
  if (intensity) intensity.addEventListener("input", apply);
  if (tint) tint.addEventListener("change", apply);
  apply();
}

function isolateStructure(name, byName, camera, controls) {
  byName.forEach((_mesh, other) => setVisible(byName, other, other === name));
  const mesh = byName.get(name);
  if (mesh) {
    mesh.visible = true;
    frameCamera(camera, controls, mesh);
  }
  syncLegend(byName);
}

function setVisible(byName, name, visible) {
  const mesh = byName.get(name);
  if (mesh) mesh.visible = visible;
}

function restoreOpacity(mesh) {
  if (!mesh) return;
  setBaseOpacity(mesh, mesh.userData.entry.color[3] / 255);
}

function setBaseOpacity(mesh, opacity) {
  mesh.userData.baseOpacity = opacity;
  applyOpacity(mesh.material, opacity);
}

function setGroupOpacity(byName, group, value) {
  byName.forEach((mesh) => {
    if (mesh.userData.group === group) applyOpacity(mesh.material, mesh.userData.baseOpacity * value);
  });
}

function setSlider(group, value) {
  const section = document.querySelector(`[data-group="${CSS.escape(group)}"]`);
  const slider = section && section.querySelector("[data-role=group-opacity]");
  if (slider) slider.value = String(value);
}

function syncGroupSliders(byName) {
  byName.forEach((mesh) => restoreOpacity(mesh));
  document.querySelectorAll("[data-role=group-opacity]").forEach((slider) => {
    slider.value = "1";
  });
}

function meshesInGroup(byName, group) {
  return [...byName.values()].filter((mesh) => mesh.userData.group === group && mesh.visible);
}

function syncLegend(byName) {
  document.querySelectorAll(".legend-item").forEach((row) => {
    if (row.classList.contains("is-volume")) {
      row.classList.remove("is-hidden");
      return;
    }
    const mesh = byName.get(row.dataset.name);
    row.classList.toggle("is-hidden", !mesh || !mesh.visible);
  });
  updateTractNote(byName);
}

function updateTractNote(byName) {
  let visible = false;
  byName.forEach((mesh) => {
    if (mesh.userData.group === "Tracts" && mesh.visible) visible = true;
  });
  tractNote.hidden = !visible;
}

function attachLabels(scene, byName, manifest) {
  const box = new THREE.Box3();
  byName.forEach((mesh) => box.expandByObject(mesh));
  const center = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
  const size = box.isEmpty() ? 100 : box.getSize(new THREE.Vector3()).length() || 100;
  const offset = Math.max(18, size * 0.08);
  const labels = [];
  byName.forEach((mesh, name) => {
    const entry = manifest.structures[name];
    if (!entry || !entry.centroid_mm) return;
    if (!TUMOR_LABELS.has(name)) return;
    const centroid = new THREE.Vector3(...entry.centroid_mm);
    const surface = closestSurfacePoint(mesh, centroid);
    const direction = centroid.clone().sub(center);
    if (direction.lengthSq() < 1e-6) direction.set(1, 0, 0);
    direction.normalize();
    const labelPos = centroid.clone().addScaledVector(direction, offset);
    const element = document.createElement("div");
    element.className = "structure-label";
    element.textContent = name;
    element.style.color = "#e8eef6";
    const css = new CSS2DObject(element);
    css.position.copy(labelPos);
    scene.add(css);
    const geometry = new THREE.BufferGeometry().setFromPoints([surface, labelPos]);
    const material = new THREE.LineBasicMaterial({
      color: srgbColor(entry.color),
      transparent: true,
      depthTest: false,
    });
    const line = new THREE.Line(geometry, material);
    line.renderOrder = 60;
    scene.add(line);
    labels.push({ name, centroid, css, line, element });
  });
  return labels;
}

function closestSurfacePoint(mesh, centroid) {
  const positions = mesh.geometry.getAttribute("position");
  if (!positions || positions.count === 0) return centroid.clone();
  mesh.updateWorldMatrix(true, false);
  const point = new THREE.Vector3();
  let best = null;
  let bestDistance = Infinity;
  const step = Math.max(1, Math.floor(positions.count / 2500));
  for (let index = 0; index < positions.count; index += step) {
    point.fromBufferAttribute(positions, index).applyMatrix4(mesh.matrixWorld);
    const distance = point.distanceToSquared(centroid);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = point.clone();
    }
  }
  return best || centroid.clone();
}

function updateLabels(labels, byName, camera, hoveredName) {
  const mode = document.querySelector("#label-mode").value;
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  for (const label of labels) {
    const mesh = byName.get(label.name);
    const hidden = !mesh || !mesh.visible || mode === "off";
    const hovered = mode !== "hover" || hoveredName === label.name;
    const show = !hidden && hovered;
    const facing = label.centroid.clone().sub(camera.position).dot(forward);
    const faded = facing <= 0;
    label.css.visible = show;
    label.line.visible = show;
    label.element.style.opacity = faded ? "0.18" : "1";
    label.line.material.opacity = faded ? 0.12 : 0.9;
  }
}

function frameObjects(camera, controls, objects) {
  const present = objects.filter(Boolean);
  if (!present.length) return;
  const box = new THREE.Box3();
  present.forEach((mesh) => box.expandByObject(mesh));
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length() || 100;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(size * 0.9, -size * 1.3, size * 0.55));
  camera.near = Math.max(size / 1000, 0.01);
  camera.far = size * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

function saveScreenshot(renderer, labelRenderer, scene, camera, byName, manifest) {
  camera.updateMatrixWorld();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  const scale = window.devicePixelRatio || 1;
  const legendWidth = Math.round(420 * scale);
  const out = document.createElement("canvas");
  out.width = renderer.domElement.width + legendWidth;
  out.height = Math.max(renderer.domElement.height, legendPixelHeight(scale));
  const ctx = out.getContext("2d");
  ctx.fillStyle = "#f5f6f8";
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(renderer.domElement, 0, 0);
  drawLabelsOnCanvas(ctx, renderer.domElement.width, renderer.domElement.height, scene, camera, byName);
  drawLegendOnCanvas(ctx, renderer.domElement.width, legendWidth, out.height, byName, manifest);
  const link = document.createElement("a");
  link.href = out.toDataURL("image/png");
  link.download = `${document.querySelector("#case-title").textContent || "scene"}-view.png`;
  link.click();
}

function legendPixelHeight(scale) {
  let lines = 4 + tumorCardLines().length;
  if (!tractNote.hidden) lines += 2;
  document.querySelectorAll("#groups .group").forEach((section) => {
    lines += 2 + section.querySelectorAll(".legend-item").length;
  });
  return Math.ceil(lines * 20 * scale + 40 * scale);
}

function tumorCardLines() {
  return (document.querySelector("#tumor-location").innerText || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function drawLabelsOnCanvas(ctx, width, height, scene, camera, byName) {
  ctx.save();
  ctx.font = `${Math.round(12 * (window.devicePixelRatio || 1))}px Segoe UI, sans-serif`;
  scene.traverse((object) => {
    if (!object.isCSS2DObject || !object.visible) return;
    const name = object.element.textContent;
    if (!byName.has(name)) return;
    const projected = object.position.clone().project(camera);
    if (projected.z < -1 || projected.z > 1) return;
    const x = (projected.x * 0.5 + 0.5) * width;
    const y = (-projected.y * 0.5 + 0.5) * height;
    ctx.globalAlpha = Number(object.element.style.opacity || 1);
    ctx.fillStyle = object.element.style.color || "#1c2430";
    ctx.fillText(name, x + 6, y);
  });
  ctx.restore();
}

function drawLegendOnCanvas(ctx, left, width, height, byName, manifest) {
  const scale = window.devicePixelRatio || 1;
  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(left, 0, width, height);
  ctx.fillStyle = "#1c2430";
  ctx.font = `${Math.round(16 * scale)}px Segoe UI, sans-serif`;
  let y = 36 * scale;
  ctx.fillText(document.querySelector("#case-title").textContent || "Case", left + 16 * scale, y);
  y += 28 * scale;
  const cardLines = tumorCardLines();
  if (cardLines.length) {
    ctx.fillStyle = "#1c2430";
    ctx.font = `${Math.round(13 * scale)}px Segoe UI, sans-serif`;
    cardLines.forEach((line) => {
      ctx.fillText(line, left + 16 * scale, y);
      y += 18 * scale;
    });
    y += 10 * scale;
  }
  if (!tractNote.hidden) {
    ctx.fillStyle = "#6a4b00";
    ctx.font = `${Math.round(13 * scale)}px Segoe UI, sans-serif`;
    ctx.fillText("Tracts are atlas-estimated", left + 16 * scale, y);
    y += 28 * scale;
  }
  const sections = document.querySelectorAll("#groups .group");
  sections.forEach((section) => {
    ctx.fillStyle = "#1c2430";
    ctx.font = `${Math.round(14 * scale)}px Segoe UI, sans-serif`;
    ctx.fillText(section.dataset.group, left + 16 * scale, y);
    y += 22 * scale;
    section.querySelectorAll(".legend-item").forEach((row) => {
      const entry = manifest.structures[row.dataset.name];
      const mesh = byName.get(row.dataset.name);
      ctx.globalAlpha = entry.volume_only || (mesh && mesh.visible) ? 1 : 0.35;
      ctx.fillStyle = `rgb(${entry.color[0]}, ${entry.color[1]}, ${entry.color[2]})`;
      ctx.fillRect(left + 16 * scale, y - 10 * scale, 12 * scale, 12 * scale);
      ctx.fillStyle = "#1c2430";
      ctx.font = `${Math.round(12 * scale)}px Segoe UI, sans-serif`;
      ctx.fillText(structureLabel(row.dataset.name, entry), left + 36 * scale, y);
      const volume = Number.isFinite(Number(entry.volume_cm3))
        ? `${Number(entry.volume_cm3).toFixed(2)} cm³`
        : "";
      ctx.fillStyle = "#5b6b7c";
      ctx.fillText(volume, left + width - 16 * scale - ctx.measureText(volume).width, y);
      y += 18 * scale;
    });
    ctx.globalAlpha = 1;
    y += 8 * scale;
  });
  ctx.restore();
}

function buildProximity(manifest) {
  const host = document.querySelector("#proximity");
  const rows = Object.entries(manifest.tract_proximity || {});
  if (!rows.length) {
    host.textContent = "No tract distances.";
    return;
  }
  const table = document.createElement("table");
  table.innerHTML = "<thead><tr><th>Tract</th><th>WT mm</th><th>ET mm</th><th>Overlap</th></tr></thead>";
  const body = document.createElement("tbody");
  for (const [name, metrics] of rows) {
    const overlap = [
      metrics.wt_overlaps ? "WT" : "",
      metrics.et_overlaps ? "ET" : "",
    ].filter(Boolean).join(", ") || "no";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td></td><td>${formatMm(metrics.wt_distance_mm)}</td><td>${formatMm(metrics.et_distance_mm)}</td><td>${overlap}</td>`;
    tr.querySelector("td").textContent = name;
    body.appendChild(tr);
  }
  table.appendChild(body);
  host.replaceChildren(table);
}

function formatMm(value) {
  return value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(1);
}

function buildClipControls(volume, clipState, onChange) {
  const host = document.querySelector("#clip-controls");
  host.replaceChildren();
  volume.shape.forEach((length, axis) => {
    const wrap = document.createElement("div");
    wrap.className = "clip";
    const header = document.createElement("header");
    const label = document.createElement("label");
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = clipState[axis].enabled;
    label.append(toggle, document.createTextNode(` ${AXIS_NAME[axis]}`));
    const readout = document.createElement("span");
    readout.textContent = String(clipState[axis].index);
    header.append(label, readout);
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(length - 1);
    slider.value = String(clipState[axis].index);
    slider.addEventListener("input", () => {
      clipState[axis].index = Number(slider.value);
      readout.textContent = slider.value;
      onChange();
    });
    toggle.addEventListener("change", () => {
      clipState[axis].enabled = toggle.checked;
      onChange();
    });
    wrap.append(header, slider);
    host.appendChild(wrap);
  });
}

function makeSliceMesh() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(12), 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  const mri = new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      side: THREE.DoubleSide,
      transparent: true,
      depthWrite: false,
      alphaTest: 0.02,
      polygonOffset: true,
      polygonOffsetFactor: -2,
    }),
  );
  const overlay = new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      side: THREE.DoubleSide,
      transparent: true,
      depthWrite: false,
      alphaTest: 0.02,
      polygonOffset: true,
      polygonOffsetFactor: -4,
    }),
  );
  mri.renderOrder = 50;
  overlay.renderOrder = 51;
  const group = new THREE.Group();
  group.add(mri);
  group.add(overlay);
  group.userData.isSlice = true;
  group.userData.mri = mri;
  group.userData.overlay = overlay;
  group.visible = false;
  return group;
}

const sliceTokens = [0, 0, 0];

function updateSliceMesh(group, volume, axis, index, enabled, sliceView) {
  group.visible = enabled;
  if (!enabled) return;
  const corners = sliceTextureCorners(volume.affine, volume.shape, axis, index);
  const positions = group.userData.mri.geometry.attributes.position;
  corners.forEach((corner, vertex) => {
    positions.setXYZ(vertex, corner.x, corner.y, corner.z);
  });
  positions.needsUpdate = true;
  group.userData.mri.geometry.computeBoundingSphere();
  const token = ++sliceTokens[axis];
  const modality = document.querySelector("#modality").value;
  const url = sliceUrl(modality, axis, index, sliceView);
  new THREE.TextureLoader().load(url, (texture) => {
    if (sliceTokens[axis] !== token) return;
    texture.flipY = false;
    texture.colorSpace = THREE.NoColorSpace;
    texture.premultiplyAlpha = false;
    group.userData.mri.material.map = texture;
    group.userData.mri.material.transparent = true;
    group.userData.mri.material.needsUpdate = true;
  });
  new THREE.TextureLoader().load(overlayUrl(axis, index), (texture) => {
    if (sliceTokens[axis] !== token) return;
    texture.flipY = false;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.premultiplyAlpha = false;
    group.userData.overlay.material.map = texture;
    group.userData.overlay.material.opacity = sliceView?.overlayOpacity ?? 0.85;
    group.userData.overlay.material.transparent = true;
    group.userData.overlay.material.needsUpdate = true;
  });
}

function volumeBox(volume) {
  const shape = volume.shape;
  const bbox = volume.bbox;
  if (!bbox || !bbox.lo || !bbox.hi) {
    return { lo: [0, 0, 0], hi: shape.map((length) => length - 1) };
  }
  return bbox;
}

function orientedPixelToIjk(axis, index, row, col, shape) {
  const nx = shape[0];
  const ny = shape[1];
  const nz = shape[2];
  if (axis === 2) return [nx - 1 - col, ny - 1 - row, index];
  if (axis === 1) return [nx - 1 - col, index, nz - 1 - row];
  return [index, ny - 1 - col, nz - 1 - row];
}

function sliceTextureCorners(affine, shape, axis, index) {
  const nx = shape[0];
  const ny = shape[1];
  const nz = shape[2];
  const rows = axis === 2 ? ny : nz;
  const cols = axis === 0 ? ny : nx;
  return [
    [0, 0],
    [0, cols - 1],
    [rows - 1, cols - 1],
    [rows - 1, 0],
  ].map(([row, col]) => voxelToWorld(affine, orientedPixelToIjk(axis, index, row, col, shape)));
}

function centerIndex(shape, axis, index) {
  return shape.map((length, item) => (item === axis ? index : (length - 1) / 2));
}

function voxelToWorld(affine, ijk) {
  const [i, j, k] = ijk;
  return new THREE.Vector3(
    affine[0][0] * i + affine[0][1] * j + affine[0][2] * k + affine[0][3],
    affine[1][0] * i + affine[1][1] * j + affine[1][2] * k + affine[1][3],
    affine[2][0] * i + affine[2][1] * j + affine[2][2] * k + affine[2][3],
  );
}

function axisDirection(affine, axis) {
  return new THREE.Vector3(affine[0][axis], affine[1][axis], affine[2][axis]).normalize();
}

function frameCamera(camera, controls, root) {
  const box = new THREE.Box3().setFromObject(root);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length() || 100;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(size * 0.7, -size, size * 0.45));
  camera.near = Math.max(size / 1000, 0.01);
  camera.far = size * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

function copyWindows(windows) {
  const out = {};
  Object.entries(windows || {}).forEach(([name, spec]) => {
    out[name] = { lo: Number(spec.lo), hi: Number(spec.hi) };
  });
  return out;
}

function sliceUrl(modality, axis, index, sliceView) {
  const win = sliceView?.windows?.[modality];
  const params = new URLSearchParams({
    modality,
    axis: String(axis),
    index: String(index),
  });
  if (win) {
    params.set("lo", String(win.lo));
    params.set("hi", String(win.hi));
  }
  return `/api/cases/${encodeURIComponent(caseId)}/slice?${params}`;
}

function overlayUrl(axis, index) {
  return `/api/cases/${encodeURIComponent(caseId)}/overlay?axis=${axis}&index=${index}`;
}

function makeCrosshairMarker() {
  const group = new THREE.Group();
  const geom = new THREE.SphereGeometry(1.6, 16, 16);
  const mat = new THREE.MeshBasicMaterial({ color: 0xffee58, depthTest: false });
  const dot = new THREE.Mesh(geom, mat);
  dot.renderOrder = 80;
  group.add(dot);
  group.visible = false;
  return group;
}

function wireSlicePanel(volume, clipState, sliceView, marker, onChange) {
  const panel = document.querySelector("#slices-panel");
  if (!panel) return;
  const slider = document.querySelector("#slice-index");
  const readout = document.querySelector("#slice-index-readout");
  const overlay = document.querySelector("#overlay-opacity");
  const length = volume.shape[sliceView.axis];
  slider.min = "0";
  slider.max = String(length - 1);
  slider.value = String(sliceView.index);
  readout.textContent = String(sliceView.index);
  document.querySelectorAll("[data-orient]").forEach((button) => {
    button.addEventListener("click", () => {
      sliceView.axis = Number(button.dataset.orient);
      sliceView.index = Math.min(sliceView.index, volume.shape[sliceView.axis] - 1);
      document.querySelectorAll("[data-orient]").forEach((item) => {
        item.classList.toggle("is-active", item === button);
      });
      slider.max = String(volume.shape[sliceView.axis] - 1);
      slider.value = String(sliceView.index);
      readout.textContent = String(sliceView.index);
      applySliceToClip(clipState, sliceView, true);
      onChange();
      updateOrientLetters(volume, sliceView);
      redrawSlices(volume, sliceView);
    });
  });
  slider.addEventListener("input", () => {
    sliceView.index = Number(slider.value);
    readout.textContent = slider.value;
    applySliceToClip(clipState, sliceView, true);
    onChange();
    redrawSlices(volume, sliceView);
  });
  overlay.addEventListener("input", () => {
    sliceView.overlayOpacity = Number(overlay.value);
    onChange();
    redrawSlices(volume, sliceView);
  });
  document.querySelector("#slice-grid").addEventListener("click", (event) => {
    const figure = event.target.closest("figure");
    const canvas = figure && figure.querySelector("canvas");
    if (!canvas) return;
    const voxel = canvasToVoxel(canvas, event, volume, sliceView);
    if (!voxel) return;
    sliceView.crosshair = voxel;
    placeMarker(marker, volume, voxel);
    redrawSlices(volume, sliceView);
  });
  wireWindowLevel(volume, clipState, sliceView, onChange);
  applySliceToClip(clipState, sliceView, false);
  updateOrientLetters(volume, sliceView);
}

function applySliceToClip(clipState, sliceView, enable) {
  clipState.forEach((state, axis) => {
    if (axis === sliceView.axis) {
      state.index = sliceView.index;
      if (enable) state.enabled = true;
    }
  });
  const host = document.querySelector("#clip-controls");
  if (!host) return;
  host.querySelectorAll(".clip").forEach((wrap, axis) => {
    const slider = wrap.querySelector("input[type=range]");
    const toggle = wrap.querySelector("input[type=checkbox]");
    const readout = wrap.querySelector("span");
    if (slider) slider.value = String(clipState[axis].index);
    if (readout) readout.textContent = String(clipState[axis].index);
    if (toggle) toggle.checked = clipState[axis].enabled;
  });
}

function wireWindowLevel(volume, clipState, sliceView, onChange) {
  const select = document.querySelector("#wl-seq");
  const width = document.querySelector("#wl-width");
  const level = document.querySelector("#wl-level");
  const apply = () => {
    const seq = select.value;
    sliceView.selectedSeq = seq;
    const win = sliceView.windows[seq];
    if (!win) return;
    const w = Number(width.value);
    const c = Number(level.value);
    win.lo = c - w / 2;
    win.hi = c + w / 2;
    if (win.hi <= win.lo) win.hi = win.lo + 1;
    onChange();
    redrawSlices(volume, sliceView);
  };
  const syncSliders = () => {
    const win = sliceView.windows[select.value];
    if (!win) return;
    const span = Math.max(win.hi - win.lo, 1);
    const max = Math.max(span * 4, 1);
    width.min = "1";
    width.max = String(Math.round(max));
    width.value = String(Math.round(span));
    level.min = String(Math.round(win.lo - span));
    level.max = String(Math.round(win.hi + span));
    level.value = String(Math.round((win.lo + win.hi) / 2));
  };
  select.addEventListener("change", () => {
    syncSliders();
  });
  width.addEventListener("input", apply);
  level.addEventListener("input", apply);
  syncSliders();
}

const sliceImages = {};
let overlayImage = null;
let overlayKey = "";

function redrawSlices(volume, sliceView) {
  const axis = sliceView.axis;
  const index = sliceView.index;
  const key = `${axis}:${index}`;
  updateOrientLetters(volume, sliceView);
  SLICE_SEQS.forEach((seq) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    const token = `${seq}:${key}:${sliceView.windows[seq]?.lo}:${sliceView.windows[seq]?.hi}`;
    img.onload = () => {
      sliceImages[seq] = { img, token };
      paintSliceCanvas(seq, volume, sliceView);
    };
    img.src = sliceUrl(seq, axis, index, sliceView);
  });
  if (overlayKey !== key) {
    overlayKey = key;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      overlayImage = img;
      SLICE_SEQS.forEach((seq) => paintSliceCanvas(seq, volume, sliceView));
    };
    img.src = overlayUrl(axis, index);
  } else {
    SLICE_SEQS.forEach((seq) => paintSliceCanvas(seq, volume, sliceView));
  }
}

function paintSliceCanvas(seq, volume, sliceView) {
  const figure = document.querySelector(`#slice-grid figure[data-seq="${seq}"]`);
  const canvas = figure && figure.querySelector("canvas");
  if (!canvas) return;
  const packed = sliceImages[seq];
  if (!packed) return;
  const img = packed.img;
  canvas.width = img.width;
  canvas.height = img.height;
  layoutSliceFrame(figure, volume, sliceView.axis);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  if (overlayImage && sliceView.overlayOpacity > 0) {
    ctx.save();
    ctx.globalAlpha = sliceView.overlayOpacity;
    ctx.drawImage(overlayImage, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  }
  if (sliceView.crosshair) {
    const pt = voxelToCanvas(sliceView.crosshair, volume, sliceView, canvas);
    if (pt) {
      ctx.strokeStyle = "#ffee58";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pt.x, 0);
      ctx.lineTo(pt.x, canvas.height);
      ctx.moveTo(0, pt.y);
      ctx.lineTo(canvas.width, pt.y);
      ctx.stroke();
    }
  }
}

function layoutSliceFrame(figure, volume, axis) {
  const frame = figure.querySelector(".slice-frame");
  if (!frame) return;
  const extent = (volume.extent_mm && volume.extent_mm[String(axis)]) || [1, 1];
  const aspect = Number(extent[0]) / Math.max(Number(extent[1]), 1e-6);
  const box = figure.getBoundingClientRect();
  let width = box.width;
  let height = width / aspect;
  if (height > box.height) {
    height = box.height;
    width = height * aspect;
  }
  frame.style.width = `${Math.max(width, 1)}px`;
  frame.style.height = `${Math.max(height, 1)}px`;
  const canvas = frame.querySelector("canvas");
  if (canvas) {
    canvas.style.width = "100%";
    canvas.style.height = "100%";
  }
}

function updateOrientLetters(volume, sliceView) {
  const letters = (volume.letters && volume.letters[String(sliceView.axis)]) || ORIENT_LETTERS[sliceView.axis];
  document.querySelectorAll("#slice-grid figure").forEach((figure) => {
    layoutSliceFrame(figure, volume, sliceView.axis);
    figure.querySelectorAll("[data-edge]").forEach((node) => {
      node.textContent = letters[node.dataset.edge] || "";
    });
  });
}

function canvasToVoxel(canvas, event, volume, sliceView) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const col = Math.round(((event.clientX - rect.left) / rect.width) * (canvas.width - 1));
  const row = Math.round(((event.clientY - rect.top) / rect.height) * (canvas.height - 1));
  const ijk = orientedPixelToIjk(sliceView.axis, sliceView.index, row, col, volume.shape);
  return ijk.map((value, axis) => Math.min(Math.max(value, 0), volume.shape[axis] - 1));
}

function voxelToCanvas(ijk, volume, sliceView, canvas) {
  const axis = sliceView.axis;
  const nx = volume.shape[0];
  const ny = volume.shape[1];
  const nz = volume.shape[2];
  let col;
  let row;
  if (axis === 2) {
    col = nx - 1 - ijk[0];
    row = ny - 1 - ijk[1];
  } else if (axis === 1) {
    col = nx - 1 - ijk[0];
    row = nz - 1 - ijk[2];
  } else {
    col = ny - 1 - ijk[1];
    row = nz - 1 - ijk[2];
  }
  return { x: col + 0.5, y: row + 0.5 };
}

function makeOrientationGizmo() {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 20);
  scene.add(new THREE.AxesHelper(1));
  [
    ["R", 0xff5555, [1.25, 0, 0]],
    ["A", 0x55cc77, [0, 1.25, 0]],
    ["S", 0x5599ff, [0, 0, 1.25]],
  ].forEach(([text, color, pos]) => {
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: gizmoLabelTexture(text, color), depthTest: false }));
    sprite.position.set(...pos);
    sprite.scale.set(0.45, 0.45, 1);
    scene.add(sprite);
  });
  return { scene, camera };
}

function gizmoLabelTexture(text, hex) {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = `#${hex.toString(16).padStart(6, "0")}`;
  ctx.font = "700 42px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 32, 34);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function renderOrientationGizmo(renderer, mainCamera, controls, gizmo) {
  const dpr = renderer.getPixelRatio();
  const size = 88 * dpr;
  const width = renderer.domElement.width;
  const height = renderer.domElement.height;
  const offset = 12 * dpr;
  renderer.clearDepth();
  renderer.setScissorTest(true);
  renderer.setViewport(offset, height - size - offset, size, size);
  renderer.setScissor(offset, height - size - offset, size, size);
  const target = controls.target;
  gizmo.camera.aspect = 1;
  gizmo.camera.updateProjectionMatrix();
  gizmo.camera.position.copy(mainCamera.position).sub(target).setLength(3.4);
  gizmo.camera.up.copy(mainCamera.up);
  gizmo.camera.lookAt(0, 0, 0);
  renderer.render(gizmo.scene, gizmo.camera);
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, width, height);
}

function placeMarker(marker, volume, ijk) {
  const world = voxelToWorld(volume.affine, ijk);
  marker.position.copy(world);
  marker.visible = true;
}

async function readJson(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}
