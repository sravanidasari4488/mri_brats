import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const AXIS_NAME = ["Sagittal", "Coronal", "Axial"];
const TUMOR_ORDER = ["enhancing", "necrosis", "edema"];

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

  const canvas = document.querySelector("#view");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setClearColor(0x14171a);
  renderer.localClippingEnabled = true;

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
    pickable.push(object);
    byName.set(name, object);
  });
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
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(pickable, false).find((item) => item.object.visible);
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
    tooltip.hidden = true;
  });

  wireToggles(byName);
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

  function frame() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  window.addEventListener("resize", () => renderer.setSize(window.innerWidth, window.innerHeight, false));
  requestAnimationFrame(frame);
}

function buildPanel(manifest) {
  const groups = { Tumor: [], Anatomy: [], Tracts: [] };
  const names = Object.keys(manifest.structures).sort((a, b) => structureRank(a, manifest) - structureRank(b, manifest));
  for (const name of names) {
    const entry = manifest.structures[name];
    const group = entry.source === "atlas-estimated" ? "Tracts" : entry.source === "model" ? "Tumor" : "Anatomy";
    groups[group].push([name, entry]);
  }
  const root = document.querySelector("#groups");
  root.replaceChildren();
  for (const [title, rows] of Object.entries(groups)) {
    if (!rows.length) continue;
    const section = document.createElement("section");
    section.className = "group";
    section.dataset.group = title;
    const heading = document.createElement("h2");
    heading.textContent = title;
    section.appendChild(heading);
    for (const [name, entry] of rows) {
      section.appendChild(structureRow(name, entry));
    }
    root.appendChild(section);
  }
}

function structureRow(name, entry) {
  const row = document.createElement("div");
  row.className = "row";
  row.dataset.name = name;
  const label = document.createElement("label");
  const toggle = document.createElement("input");
  toggle.type = "checkbox";
  toggle.checked = true;
  toggle.dataset.role = "visible";
  label.append(toggle, document.createTextNode(name));
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = "1";
  slider.step = "0.01";
  slider.value = String(entry.color[3] / 255);
  slider.dataset.role = "opacity";
  row.append(label, slider);
  return row;
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

function wireToggles(byName) {
  document.querySelectorAll(".row").forEach((row) => {
    const mesh = byName.get(row.dataset.name);
    if (!mesh) return;
    row.querySelector("[data-role=visible]").addEventListener("change", (event) => {
      mesh.visible = event.target.checked;
      updateTractNote(byName);
    });
    row.querySelector("[data-role=opacity]").addEventListener("input", (event) => {
      applyOpacity(mesh.material, Number(event.target.value));
    });
  });
  updateTractNote(byName);
}

function updateTractNote(byName) {
  let visible = false;
  byName.forEach((mesh, name) => {
    const row = document.querySelector(`.row[data-name="${CSS.escape(name)}"]`);
    const tract = row && row.closest("[data-group=Tracts]");
    if (tract && mesh.visible) visible = true;
  });
  tractNote.hidden = !visible;
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
