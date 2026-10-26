import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

const AXIS_NAME = ["Sagittal", "Coronal", "Axial"];
const TUMOR_ORDER = ["enhancing", "necrosis", "edema"];
const GROUP_ORDER = [
  "Cortex",
  "Subcortical",
  "Ventricles",
  "Brainstem & Cerebellum",
  "Tumor",
  "Tracts",
  "Brain",
  "Anatomy",
];
const PRESETS = [
  { id: "tumor-focus", label: "Tumor focus" },
  { id: "anatomy", label: "Anatomy" },
  { id: "tumor-tracts", label: "Tumor + Tracts" },
];

const params = new URLSearchParams(location.search);
const caseId = params.get("case");
const uploadScreen = document.querySelector("#upload-screen");
const viewerRoot = document.querySelector("#viewer");
const tooltip = document.querySelector("#tooltip");
const tractNote = document.querySelector("#tract-note");

document.querySelector("#upload-form").addEventListener("submit", onUpload);

if (caseId) {
  uploadScreen.hidden = true;
  viewerRoot.hidden = false;
  startViewer(caseId);
}

async function onUpload(event) {
  event.preventDefault();
  const status = document.querySelector("#upload-status");
  const form = event.currentTarget;
  const body = new FormData(form);
  if (!form.anatomy.checked) body.delete("anatomy");
  else body.set("anatomy", "1");
  if (!form.tracts.checked) body.delete("tracts");
  else body.set("tracts", "1");
  status.textContent = "Reconstructing…";
  form.querySelector("button").disabled = true;
  try {
    const response = await fetch("/api/cases", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    location.href = payload.viewer;
  } catch (error) {
    status.textContent = error.message;
    form.querySelector("button").disabled = false;
  }
}

async function startViewer(id) {
  const manifest = await fetch(`/api/cases/${encodeURIComponent(id)}/manifest.json`).then(readJson);
  document.querySelector("#case-title").textContent = manifest.case_id || id;
  buildPanel(manifest);
  buildProximity(manifest);
  buildTumorCard(manifest);
  buildPresetButtons();

  const canvas = document.querySelector("#view");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setClearColor(0x14171a);
  renderer.localClippingEnabled = true;
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = "label-layer";
  viewerRoot.appendChild(labelRenderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(1, -1, 2);
  scene.add(key);

  const planes = [0, 1, 2].map(() => new THREE.Plane());
  const sliceMeshes = planes.map(() => makeSliceMesh());
  sliceMeshes.forEach((mesh) => scene.add(mesh));

  const gltf = await new GLTFLoader().loadAsync(`/api/cases/${encodeURIComponent(id)}/scene.glb`);
  scene.add(gltf.scene);
  const pickable = [];
  const byName = new Map();
  gltf.scene.traverse((object) => {
    if (!object.isMesh) return;
    const name = object.name || object.parent?.name;
    if (!name || !manifest.structures[name]) return;
    object.name = name;
    const entry = manifest.structures[name];
    const opacity = entry.color[3] / 255;
    const material = Array.isArray(object.material) ? object.material[0].clone() : object.material.clone();
    material.color.setRGB(entry.color[0] / 255, entry.color[1] / 255, entry.color[2] / 255);
    material.vertexColors = false;
    material.side = THREE.DoubleSide;
    material.depthTest = true;
    material.clippingPlanes = planes.filter((_plane, axis) => clipEnabled(axis));
    applyOpacity(material, opacity);
    object.material = material;
    object.renderOrder = renderOrderFor(name, entry.source);
    object.userData.entry = entry;
    object.userData.group = structureGroup(entry);
    object.userData.baseOpacity = opacity;
    pickable.push(object);
    byName.set(name, object);
  });
  const labels = attachLabels(scene, byName, manifest);
  frameCamera(camera, controls, gltf.scene);

  let volume = await fetch(`/api/cases/${encodeURIComponent(id)}/volume?modality=flair`).then(readJson);
  const clipState = volume.shape.map((length, axis) => ({
    enabled: true,
    index: Math.floor((length - 1) / 2),
  }));
  buildClipControls(volume, clipState, () => refreshClips());

  document.querySelector("#modality").addEventListener("change", async (event) => {
    volume = await fetch(
      `/api/cases/${encodeURIComponent(id)}/volume?modality=${event.target.value}`,
    ).then(readJson);
    clipState.forEach((state, axis) => {
      state.index = Math.min(state.index, volume.shape[axis] - 1);
    });
    buildClipControls(volume, clipState, () => refreshClips());
    refreshClips();
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

  function clipEnabled(axis) {
    return clipState[axis].enabled;
  }

  function refreshClips() {
    const active = [];
    clipState.forEach((state, axis) => {
      const normal = axisDirection(volume.affine, axis).multiplyScalar(-1);
      const center = voxelToWorld(volume.affine, centerIndex(volume.shape, axis, state.index));
      planes[axis].setFromNormalAndCoplanarPoint(normal, center);
      if (state.enabled) active.push(planes[axis]);
      updateSliceMesh(sliceMeshes[axis], volume, axis, state.index, state.enabled);
      sliceMeshes[axis].material.clippingPlanes = planes.filter((_plane, other) => other !== axis && clipState[other].enabled);
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
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
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

function legendItem(name, entry) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "legend-item";
  button.dataset.name = name;
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = `rgb(${entry.color[0]}, ${entry.color[1]}, ${entry.color[2]})`;
  const label = document.createElement("span");
  label.className = "name";
  label.textContent = name;
  const volume = document.createElement("span");
  volume.className = "volume";
  volume.textContent = `${Number(entry.volume_cm3).toFixed(2)} cm³`;
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
}

function structureRank(name, manifest) {
  const source = manifest.structures[name].source;
  const tumor = TUMOR_ORDER.indexOf(name);
  if (source === "model") return (tumor === -1 ? 9 : tumor);
  if (source === "atlas-estimated") return 100;
  if (name === "brain") return 80;
  return 40;
}

function renderOrderFor(name, source) {
  if (name === "enhancing") return 0;
  if (name === "necrosis") return 1;
  if (name === "edema") return 20;
  if (source === "atlas-estimated") return 30;
  if (name === "brain") return 40;
  return 10;
}

function applyOpacity(material, opacity) {
  material.opacity = opacity;
  material.transparent = opacity < 0.999;
  material.depthWrite = !material.transparent;
}

function wireLegend(byName, camera, controls) {
  document.querySelectorAll(".legend-item").forEach((row) => {
    const name = row.dataset.name;
    row.addEventListener("click", () => {
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
  if (id === "anatomy") {
    names.forEach((name) => setVisible(byName, name, true));
    syncGroupSliders(byName);
    frameCamera(camera, controls, root);
  } else if (id === "tumor-focus") {
    names.forEach((name) => setVisible(byName, name, true));
    byName.forEach((mesh) => {
      if (mesh.userData.group === "Cortex") setBaseOpacity(mesh, 0.15);
      else if (mesh.userData.group === "Tumor") setBaseOpacity(mesh, 1);
      else restoreOpacity(mesh);
    });
    setSlider("Cortex", 1);
    setSlider("Tumor", 1);
    frameObjects(camera, controls, meshesInGroup(byName, "Tumor"));
  } else if (id === "tumor-tracts") {
    names.forEach((name) => {
      const mesh = byName.get(name);
      const show = mesh.userData.group === "Tumor" || mesh.userData.group === "Tracts";
      setVisible(byName, name, show);
      if (!show) return;
      if (mesh.userData.group === "Tumor") setBaseOpacity(mesh, 1);
      else restoreOpacity(mesh);
    });
    setSlider("Tumor", 1);
    setSlider("Tracts", 1);
    frameObjects(camera, controls, [
      ...meshesInGroup(byName, "Tumor"),
      ...meshesInGroup(byName, "Tracts"),
    ]);
  }
  syncLegend(byName);
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
    const centroid = new THREE.Vector3(...entry.centroid_mm);
    const surface = closestSurfacePoint(mesh, centroid);
    const direction = centroid.clone().sub(center);
    if (direction.lengthSq() < 1e-6) direction.set(1, 0, 0);
    direction.normalize();
    const labelPos = centroid.clone().addScaledVector(direction, offset);
    const element = document.createElement("div");
    element.className = "structure-label";
    element.textContent = name;
    element.style.color = `rgb(${entry.color[0]}, ${entry.color[1]}, ${entry.color[2]})`;
    const css = new CSS2DObject(element);
    css.position.copy(labelPos);
    scene.add(css);
    const geometry = new THREE.BufferGeometry().setFromPoints([surface, labelPos]);
    const material = new THREE.LineBasicMaterial({
      color: new THREE.Color(entry.color[0] / 255, entry.color[1] / 255, entry.color[2] / 255),
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
  ctx.fillStyle = "#14171a";
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
    ctx.fillStyle = object.element.style.color || "#fff";
    ctx.fillText(name, x + 6, y);
  });
  ctx.restore();
}

function drawLegendOnCanvas(ctx, left, width, height, byName, manifest) {
  const scale = window.devicePixelRatio || 1;
  ctx.save();
  ctx.fillStyle = "#121417";
  ctx.fillRect(left, 0, width, height);
  ctx.fillStyle = "#e7e4df";
  ctx.font = `${Math.round(16 * scale)}px Segoe UI, sans-serif`;
  let y = 36 * scale;
  ctx.fillText(document.querySelector("#case-title").textContent || "Case", left + 16 * scale, y);
  y += 28 * scale;
  const cardLines = tumorCardLines();
  if (cardLines.length) {
    ctx.fillStyle = "#e7e4df";
    ctx.font = `${Math.round(13 * scale)}px Segoe UI, sans-serif`;
    cardLines.forEach((line) => {
      ctx.fillText(line, left + 16 * scale, y);
      y += 18 * scale;
    });
    y += 10 * scale;
  }
  if (!tractNote.hidden) {
    ctx.fillStyle = "#f0e2b6";
    ctx.font = `${Math.round(13 * scale)}px Segoe UI, sans-serif`;
    ctx.fillText("Tracts are atlas-estimated", left + 16 * scale, y);
    y += 28 * scale;
  }
  const sections = document.querySelectorAll("#groups .group");
  sections.forEach((section) => {
    ctx.fillStyle = "#e7e4df";
    ctx.font = `${Math.round(14 * scale)}px Segoe UI, sans-serif`;
    ctx.fillText(section.dataset.group, left + 16 * scale, y);
    y += 22 * scale;
    section.querySelectorAll(".legend-item").forEach((row) => {
      const entry = manifest.structures[row.dataset.name];
      const mesh = byName.get(row.dataset.name);
      ctx.globalAlpha = mesh && mesh.visible ? 1 : 0.35;
      ctx.fillStyle = `rgb(${entry.color[0]}, ${entry.color[1]}, ${entry.color[2]})`;
      ctx.fillRect(left + 16 * scale, y - 10 * scale, 12 * scale, 12 * scale);
      ctx.fillStyle = "#e7e4df";
      ctx.font = `${Math.round(12 * scale)}px Segoe UI, sans-serif`;
      ctx.fillText(row.dataset.name, left + 36 * scale, y);
      const volume = `${Number(entry.volume_cm3).toFixed(2)} cm³`;
      ctx.fillStyle = "#b7b1a8";
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
  const material = new THREE.MeshBasicMaterial({
    side: THREE.DoubleSide,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -2,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData.isSlice = true;
  mesh.renderOrder = 50;
  mesh.visible = false;
  return mesh;
}

const sliceTokens = [0, 0, 0];

function updateSliceMesh(mesh, volume, axis, index, enabled) {
  mesh.visible = enabled;
  if (!enabled) return;
  const corners = sliceCorners(volume.affine, volume.shape, axis, index);
  const positions = mesh.geometry.attributes.position;
  corners.forEach((corner, vertex) => {
    positions.setXYZ(vertex, corner.x, corner.y, corner.z);
  });
  positions.needsUpdate = true;
  mesh.geometry.computeBoundingSphere();
  const token = ++sliceTokens[axis];
  const modality = document.querySelector("#modality").value;
  const url = `/api/cases/${encodeURIComponent(caseId)}/slice?modality=${modality}&axis=${axis}&index=${index}`;
  new THREE.TextureLoader().load(url, (texture) => {
    if (sliceTokens[axis] !== token) return;
    texture.flipY = false;
    texture.colorSpace = THREE.NoColorSpace;
    mesh.material.map = texture;
    mesh.material.needsUpdate = true;
  });
}

function sliceCorners(affine, shape, axis, index) {
  const free = [0, 1, 2].filter((item) => item !== axis);
  return [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ].map(([alongB, alongC]) => {
    const ijk = [0, 0, 0];
    ijk[axis] = index;
    ijk[free[0]] = alongB ? shape[free[0]] - 1 : 0;
    ijk[free[1]] = alongC ? shape[free[1]] - 1 : 0;
    return voxelToWorld(affine, ijk);
  });
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

async function readJson(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}
