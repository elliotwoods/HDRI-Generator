import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const fileInput = document.getElementById("heicFile");
const sceneType = document.getElementById("sceneType");
const submitBtn = document.getElementById("submitBtn");
const progressBar = document.getElementById("progressBar");
const statusText = document.getElementById("statusText");
const statusLog = document.getElementById("statusLog");
const downloadLinks = document.getElementById("downloadLinks");
const downloadHdr = document.getElementById("downloadHdr");
const downloadPreview = document.getElementById("downloadPreview");
const dropZone = document.getElementById("dropZone");
const viewerEl = document.getElementById("viewer");
const exposureSlider = document.getElementById("exposureSlider");
const exposureValue = document.getElementById("exposureValue");

let currentFile = null;
let activeJobId = null;
let activeSocket = null;

function makePlaceholder() {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  const sky = ctx.createLinearGradient(0, 0, 0, canvas.height * 0.6);
  sky.addColorStop(0, "#7dd3fc");
  sky.addColorStop(1, "#38bdf8");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, canvas.width, canvas.height * 0.6);
  const ground = ctx.createLinearGradient(0, canvas.height * 0.6, 0, canvas.height);
  ground.addColorStop(0, "#86efac");
  ground.addColorStop(1, "#22c55e");
  ctx.fillStyle = ground;
  ctx.fillRect(0, canvas.height * 0.6, canvas.width, canvas.height * 0.4);
  return canvas.toDataURL("image/jpeg", 0.8);
}

let viewer = null;
let pollTimer = null;
let activeTexture = null;
let textureLoader = null;
let loadToken = 0;

function createGridTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 2048;
  canvas.height = 1024;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
  const stepX = 128;
  const stepY = 128;
  for (let x = 0; x <= canvas.width; x += stepX) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= canvas.height; y += stepY) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(248, 250, 252, 0.6)";
  ctx.beginPath();
  ctx.moveTo(0, canvas.height * 0.5);
  ctx.lineTo(canvas.width, canvas.height * 0.5);
  ctx.stroke();

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function initViewer() {
  const width = viewerEl.clientWidth;
  const height = viewerEl.clientHeight;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = parseFloat(exposureSlider.value);
  viewerEl.insertBefore(renderer.domElement, viewerEl.firstChild);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 200);
  camera.position.set(0, 0, 0.1);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.enableZoom = true;
  controls.rotateSpeed = 0.45;
  controls.zoomSpeed = 0.8;
  controls.minDistance = 0.1;
  controls.maxDistance = 30;

  const geometry = new THREE.SphereGeometry(50, 64, 32);
  geometry.scale(-1, 1, 1);
  const material = new THREE.MeshBasicMaterial({ map: createGridTexture() });
  const sphere = new THREE.Mesh(geometry, material);
  sphere.rotation.y = Math.PI;
  scene.add(sphere);

  textureLoader = new THREE.TextureLoader();

  viewer = { renderer, scene, camera, controls, sphere };

  function resize() {
    const nextWidth = viewerEl.clientWidth;
    const nextHeight = viewerEl.clientHeight;
    renderer.setSize(nextWidth, nextHeight);
    camera.aspect = nextWidth / nextHeight;
    camera.updateProjectionMatrix();
  }

  window.addEventListener("resize", resize);

  function render() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(render);
  }
  render();
}

function applyTexture(texture) {
  if (!viewer) return;
  if (activeTexture) {
    activeTexture.dispose();
  }
  activeTexture = texture;
  activeTexture.flipY = false;
  activeTexture.colorSpace = THREE.SRGBColorSpace;
  viewer.sphere.material.map = activeTexture;
  viewer.sphere.material.needsUpdate = true;
}

function updatePanorama(url) {
  if (!viewer || !textureLoader) return;
  const token = ++loadToken;
  textureLoader.load(
    url,
    (texture) => {
      if (token !== loadToken) {
        texture.dispose();
        return;
      }
      applyTexture(texture);
    },
    undefined,
    (err) => {
      console.warn("Failed to load panorama texture:", err);
    }
  );
}

function logStatus(message) {
  const line = document.createElement("div");
  line.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
  statusLog.prepend(line);
}

function fileToEquirectPreview(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const w = 2048;
      const h = 1024;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, w, h);

      const scale = Math.min(w * 0.9 / img.width, h * 0.9 / img.height);
      const drawW = img.width * scale;
      const drawH = img.height * scale;
      const x = (w - drawW) * 0.5;
      const y = (h - drawH) * 0.5;
      ctx.drawImage(img, x, y, drawW, drawH);
      resolve(canvas.toDataURL("image/jpeg", 0.8));
    };
    img.onerror = () => {
      console.warn("Browser cannot preview this file type for viewer; using placeholder.");
      resolve(makePlaceholder());
    };
    img.src = URL.createObjectURL(file);
  });
}

async function uploadFile(file) {
  if (activeSocket) {
    activeSocket.close();
    activeSocket = null;
  }
  currentFile = file;
  if (!file) {
    statusText.textContent = "Please choose a HEIC file.";
    return;
  }

  submitBtn.disabled = true;
  statusText.textContent = "Uploading...";
  progressBar.style.width = "0%";
  downloadLinks.style.display = "none";
  statusLog.textContent = "";
  logStatus("Upload started.");

  const previewUrl = await fileToEquirectPreview(file);
  updatePanorama(previewUrl);

  const formData = new FormData();
  formData.append("file", file);
  formData.append("scene_type", sceneType.value);

  let response;
  try {
    response = await fetch("/api/upload", { method: "POST", body: formData });
  } catch (err) {
    console.error("Upload request failed:", err);
    statusText.textContent = "Upload failed.";
    logStatus("Upload failed.");
    alert("Upload failed. Check console for details.");
    submitBtn.disabled = false;
    return;
  }
  if (!response.ok) {
    statusText.textContent = "Upload failed.";
    logStatus("Upload failed.");
    console.error("Upload failed with status", response.status);
    alert("Upload failed. Check console for details.");
    submitBtn.disabled = false;
    return;
  }

  let payload;
  try {
    payload = await response.json();
  } catch (err) {
    console.error("Failed to parse upload response:", err);
    alert("Upload response error. Check console for details.");
    submitBtn.disabled = false;
    return;
  }
  const { job_id } = payload;
  activeJobId = job_id;
  statusText.textContent = "Processing...";
  logStatus(`Job ${job_id} queued.`);
  connectWebSocket(job_id);
  startStatusPolling(job_id);
}

function connectWebSocket(jobId) {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws/jobs/${jobId}`);
  activeSocket = socket;

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.event === "status") {
      const pct = Math.round(data.progress * 100);
      progressBar.style.width = `${pct}%`;
      statusText.textContent = `Stage: ${data.stage} (${pct}%)`;
      logStatus(`Stage: ${data.stage} (${pct}%)`);
    }
    if (data.event === "preview") {
      updatePanorama(`${data.preview_url}?t=${Date.now()}`);
      logStatus("Preview updated.");
    }
    if (data.event === "done") {
      progressBar.style.width = "100%";
      statusText.textContent = "Done.";
      updatePanorama(`${data.preview_url}?t=${Date.now()}`);
      downloadLinks.style.display = "flex";
      downloadHdr.href = data.hdr_url;
      downloadPreview.href = data.preview_url;
      submitBtn.disabled = false;
      logStatus("Job complete.");
    }
    if (data.event === "error") {
      statusText.textContent = `Error: ${data.message}`;
      submitBtn.disabled = false;
      logStatus(`Error: ${data.message}`);
      console.error("Job error:", data.message);
      alert(`Generation error: ${data.message}`);
    }
  };

  socket.onerror = (err) => {
    console.error("WebSocket error:", err);
    alert("WebSocket error. Check console for details.");
  };

  socket.onclose = () => {
    submitBtn.disabled = false;
  };
}

async function pollStatus(jobId) {
  try {
    const resp = await fetch(`/api/status/${jobId}`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.status === "error") {
      statusText.textContent = `Error: ${data.error_message || "Unknown error"}`;
      logStatus(`Error: ${data.error_message || "Unknown error"}`);
      alert(`Generation error: ${data.error_message || "Unknown error"}`);
      stopStatusPolling();
      submitBtn.disabled = false;
      return;
    }
    if (typeof data.progress === "number") {
      const pct = Math.round(data.progress * 100);
      progressBar.style.width = `${pct}%`;
      statusText.textContent = `Stage: ${data.current_stage} (${pct}%)`;
    }
    if (data.preview_url) {
      updatePanorama(`${data.preview_url}?t=${Date.now()}`);
    }
    if (data.status === "done") {
      progressBar.style.width = "100%";
      statusText.textContent = "Done.";
      downloadLinks.style.display = "flex";
      downloadHdr.href = `/api/result/${jobId}/hdr`;
      downloadPreview.href = `/api/result/${jobId}/preview`;
      stopStatusPolling();
      submitBtn.disabled = false;
    }
  } catch (err) {
    console.error("Status polling failed:", err);
  }
}

function startStatusPolling(jobId) {
  stopStatusPolling();
  pollTimer = setInterval(() => pollStatus(jobId), 2000);
  pollStatus(jobId);
}

function stopStatusPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function handleFile(file) {
  if (!file) return;
  console.log("Selected file:", file.name, file.type, file.size);
  fileInput.value = "";
  uploadFile(file);
}

submitBtn.addEventListener("click", () => {
  const file = fileInput.files[0] || currentFile;
  if (file) {
    uploadFile(file);
  } else {
    statusText.textContent = "Please choose a HEIC file.";
  }
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#d97706";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "rgba(0,0,0,0.2)";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "rgba(0,0,0,0.2)";
  const file = e.dataTransfer.files[0];
  if (!file) {
    console.error("No file dropped.");
    alert("No file detected in drop.");
    return;
  }
  handleFile(file);
});

dropZone.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) {
    console.error("No file selected.");
    alert("No file selected.");
    return;
  }
  handleFile(file);
});

window.addEventListener("dragover", (e) => {
  e.preventDefault();
});

window.addEventListener("drop", (e) => {
  e.preventDefault();
});

window.addEventListener("error", (event) => {
  console.error("Unhandled error:", event.error || event.message);
});

exposureSlider.addEventListener("input", () => {
  const value = parseFloat(exposureSlider.value);
  exposureValue.textContent = value.toFixed(2);
  if (viewer) {
    viewer.renderer.toneMappingExposure = value;
  }
});

initViewer();
