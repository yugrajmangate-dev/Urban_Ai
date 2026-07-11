/**
 * app.js — Urban AI Solar Panel Dust Detection
 * Handles: tab switching, image upload, webcam capture,
 *          demo sample requests, /predict API calls,
 *          result animation, prediction history, particle canvas.
 */

"use strict";

// ══════════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════════
const state = {
  activeTab:       "upload",
  uploadedBase64:  null,
  webcamStream:    null,
  webcamActive:    false,
  predictionCount: 0,
  history:         [],         // [{label, confidence, time}]
};

// ══════════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  initParticles();
  checkModelReady();
});

// ══════════════════════════════════════════════════════════════════
// MODEL READINESS CHECK
// ══════════════════════════════════════════════════════════════════
let tfjsModel = null;
let useClientSideAI = false;

async function loadTFJSModel() {
  if (!tfjsModel && typeof tf !== "undefined") {
    try {
      console.log("Loading TensorFlow.js web graph model...");
      tfjsModel = await tf.loadGraphModel("static/tfjs_model/model.json");
      console.log("TFJS Model loaded locally in browser!");
    } catch (e) {
      console.warn("Could not load TFJS model:", e);
    }
  }
  return tfjsModel;
}

async function checkModelReady() {
  const badge = document.getElementById("modelStatusBadge");
  badge.classList.add("loading");
  badge.textContent = "⬤ Checking…";

  // Check if we can load TFJS model first (Static/Vercel/Client-side)
  if (typeof tf !== "undefined") {
    try {
      await loadTFJSModel();
      if (tfjsModel) {
        useClientSideAI = true;
        badge.textContent = "⬤ Model Ready (Browser AI)";
        badge.classList.remove("loading");
        badge.style.color = "#4ade80";
        return;
      }
    } catch (e) {
      console.warn("TFJS load error, checking Flask backend...", e);
    }
  }

  try {
    const resp = await fetch("/predict", { method: "OPTIONS" });
    if (resp.ok || resp.status === 405 || resp.status === 200) {
      badge.textContent = "⬤ Model Ready (Flask API)";
      badge.classList.remove("loading");
      badge.style.color = "#4ade80";
    } else {
      throw new Error("not ready");
    }
  } catch {
    badge.textContent = "⬤ Offline";
    badge.classList.add("loading");
    badge.style.color = "#ef4444";
  }
}


// ══════════════════════════════════════════════════════════════════
// TAB SWITCHING
// ══════════════════════════════════════════════════════════════════
function switchTab(tabName) {
  // Deactivate all tabs & panels
  document.querySelectorAll(".tab").forEach(t => {
    t.classList.remove("tab--active");
    t.setAttribute("aria-selected", "false");
  });
  document.querySelectorAll(".tab-panel").forEach(p => {
    p.classList.remove("tab-panel--active");
  });

  // Activate selected
  const tabBtn   = document.getElementById("tab" + capitalise(tabName));
  const tabPanel = document.getElementById("panel" + capitalise(tabName));
  if (tabBtn)   { tabBtn.classList.add("tab--active"); tabBtn.setAttribute("aria-selected", "true"); }
  if (tabPanel) tabPanel.classList.add("tab-panel--active");

  // Stop webcam if leaving webcam tab
  if (tabName !== "webcam" && state.webcamStream) {
    stopWebcam();
  }

  state.activeTab = tabName;
}

function capitalise(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// ══════════════════════════════════════════════════════════════════
// UPLOAD TAB — drag & drop / file input
// ══════════════════════════════════════════════════════════════════
function onDragOver(e) {
  e.preventDefault();
  document.getElementById("dropZone").classList.add("drag-over");
}

function onDragLeave(e) {
  document.getElementById("dropZone").classList.remove("drag-over");
}

function onDrop(e) {
  e.preventDefault();
  document.getElementById("dropZone").classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) loadFile(file);
}

function onFileSelected(e) {
  const file = e.target.files[0];
  if (file) loadFile(file);
}

function loadFile(file) {
  const reader = new FileReader();
  reader.onload = (ev) => {
    state.uploadedBase64 = ev.target.result;  // full data-URL
    showPreview(ev.target.result);
    document.getElementById("analyseUploadBtn").disabled = false;
  };
  reader.readAsDataURL(file);
}

function showPreview(dataUrl) {
  const dropZone   = document.getElementById("dropZone");
  const previewCon = document.getElementById("previewContainer");
  const previewImg = document.getElementById("previewImg");

  previewImg.src = dataUrl;
  dropZone.classList.add("hidden");
  previewCon.classList.remove("hidden");
}

function clearPreview() {
  state.uploadedBase64 = null;
  document.getElementById("dropZone").classList.remove("hidden");
  document.getElementById("previewContainer").classList.add("hidden");
  document.getElementById("analyseUploadBtn").disabled = true;
  document.getElementById("fileInput").value = "";
}

function analyseUpload() {
  if (!state.uploadedBase64) return;
  callPredict({ image: state.uploadedBase64 });
}

// ══════════════════════════════════════════════════════════════════
// WEBCAM TAB
// ══════════════════════════════════════════════════════════════════
async function startWebcam() {
  const video   = document.getElementById("webcamVideo");
  const overlay = document.getElementById("webcamOverlay");
  const startBtn  = document.getElementById("startWebcamBtn");
  const captureBtn = document.getElementById("captureBtn");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    state.webcamStream = stream;
    state.webcamActive = true;
    video.srcObject = stream;
    overlay.classList.add("hidden");
    startBtn.textContent = "⏹ Stop Camera";
    startBtn.onclick = stopWebcam;
    captureBtn.disabled = false;
  } catch (err) {
    showError("Webcam access denied or not available: " + err.message);
  }
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(t => t.stop());
    state.webcamStream = null;
  }
  state.webcamActive = false;

  const video    = document.getElementById("webcamVideo");
  const overlay  = document.getElementById("webcamOverlay");
  const startBtn = document.getElementById("startWebcamBtn");
  const captureBtn = document.getElementById("captureBtn");

  video.srcObject = null;
  overlay.classList.remove("hidden");
  startBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg> Start Camera`;
  startBtn.onclick = startWebcam;
  captureBtn.disabled = true;
}

function captureWebcam() {
  const video  = document.getElementById("webcamVideo");
  const canvas = document.getElementById("webcamCanvas");

  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const base64 = canvas.toDataURL("image/jpeg", 0.92);
  callPredict({ image: base64 });
}

// ══════════════════════════════════════════════════════════════════
// DEMO TAB
// ══════════════════════════════════════════════════════════════════
function analyseDemo(source) {
  callPredict({ source });
}

// ══════════════════════════════════════════════════════════════════
// CORE: CALL /predict API
// ══════════════════════════════════════════════════════════════════
async function runClientSidePrediction(payload) {
  showLoading();
  setButtonsLoading(true);
  try {
    let imgElement = new Image();
    imgElement.crossOrigin = "anonymous";

    let srcUrl = null;
    if (payload.image) {
      srcUrl = payload.image.includes(",") ? payload.image : "data:image/jpeg;base64," + payload.image;
    } else if (payload.source) {
      if (payload.source === "clean") srcUrl = "sample_clean.jpg";
      else if (payload.source === "dusty") srcUrl = "sample_dusty.jpg";
    }

    if (!srcUrl) throw new Error("No image data found for analysis.");

    await new Promise((resolve, reject) => {
      imgElement.onload = resolve;
      imgElement.onerror = () => reject(new Error("Failed to load image element for TFJS."));
      imgElement.src = srcUrl;
    });

    const prediction = tf.tidy(() => {
      const tensor = tf.browser.fromPixels(imgElement);
      const resized = tf.image.resizeBilinear(tensor, [224, 224]);
      const prepared = resized.toFloat().div(127.5).sub(1.0);
      const batched = prepared.expandDims(0);
      return tfjsModel.predict(batched);
    });

    const scores = await prediction.data();
    prediction.dispose();

    let idx = scores[1] > scores[0] ? 1 : 0;
    let label = idx === 1 ? "Dusty" : "Clean";
    let confidence = Math.round(scores[idx] * 10000) / 100;

    showResult(label, confidence);
    addToHistory(label, confidence);
  } catch (err) {
    console.error("TFJS Prediction Error:", err);
    showError("Client-side AI inference failed: " + err.message);
  } finally {
    setButtonsLoading(false);
  }
}

async function callPredict(payload) {
  if (useClientSideAI && tfjsModel) {
    return runClientSidePrediction(payload);
  }

  showLoading();
  setButtonsLoading(true);

  try {
    const resp = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const data = await resp.json();

    if (!resp.ok || !data.success) {
      showError(data.error || "Server returned an error.");
      return;
    }

    showResult(data.status, data.confidence);
    addToHistory(data.status, data.confidence);

  } catch (err) {
    if (typeof tf !== "undefined") {
      await loadTFJSModel();
      if (tfjsModel) {
        useClientSideAI = true;
        setButtonsLoading(false);
        return runClientSidePrediction(payload);
      }
    }
    showError("Network error: could not reach the server. Is Flask running?");
  } finally {
    setButtonsLoading(false);
  }
}


// ══════════════════════════════════════════════════════════════════
// RESULT DISPLAY
// ══════════════════════════════════════════════════════════════════
function showLoading() {
  document.getElementById("resultIdle").classList.add("hidden");
  document.getElementById("resultDisplay").classList.add("hidden");
  document.getElementById("resultError").classList.add("hidden");
  document.getElementById("resultLoading").classList.remove("hidden");
}

function showResult(label, confidence) {
  document.getElementById("resultLoading").classList.add("hidden");
  document.getElementById("resultIdle").classList.add("hidden");
  document.getElementById("resultError").classList.add("hidden");

  const display = document.getElementById("resultDisplay");
  display.classList.remove("hidden");

  const isClean      = label.toLowerCase() === "clean";
  const ringColor    = isClean ? "#22c55e" : "#f59e0b";
  const confPercent  = Math.round(confidence);
  const circumference = 427.26;  // 2π × 68

  // Animate ring fill
  const ringFill = document.getElementById("ringFill");
  ringFill.style.stroke = ringColor;
  // Use rAF to trigger CSS transition
  requestAnimationFrame(() => {
    const offset = circumference - (confPercent / 100) * circumference;
    ringFill.style.strokeDashoffset = offset;
  });

  // Confidence value
  const confVal = document.getElementById("confidenceValue");
  confVal.style.color = ringColor;
  animateNumber(confVal, 0, confPercent, 1200, "%");

  // Status badge
  const badge = document.getElementById("statusBadge");
  badge.className = "status-badge " + (isClean ? "clean" : "dusty");
  document.getElementById("statusIcon").textContent = isClean ? "✅" : "⚠️";
  document.getElementById("statusText").textContent = label.toUpperCase();

  // Detail cards
  document.getElementById("detailClass").textContent = label;
  document.getElementById("detailConf").textContent  = confPercent + "%";
  document.getElementById("detailTime").textContent  = new Date().toLocaleTimeString();

  // Recommendation
  const rec = document.getElementById("recommendation");
  rec.className = "recommendation " + (isClean ? "clean" : "dusty");
  rec.textContent = isClean
    ? "✅ The solar panel appears clean. No immediate maintenance required. Continue regular monitoring."
    : "⚠️ Dust accumulation detected on the solar panel. Cleaning is recommended to restore optimal energy output. Dust can reduce efficiency by 15–40%.";
}

function showError(msg) {
  document.getElementById("resultLoading").classList.add("hidden");
  document.getElementById("resultDisplay").classList.add("hidden");
  document.getElementById("resultIdle").classList.add("hidden");
  document.getElementById("resultError").classList.remove("hidden");
  document.getElementById("errorMsg").textContent = msg;
}

function resetResult() {
  document.getElementById("resultError").classList.add("hidden");
  document.getElementById("resultIdle").classList.remove("hidden");
}

// Animate a number from start → end over duration ms
function animateNumber(el, start, end, duration, suffix = "") {
  const startTime = performance.now();
  function step(now) {
    const elapsed  = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);  // ease-out cubic
    el.textContent = Math.round(start + (end - start) * eased) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ══════════════════════════════════════════════════════════════════
// PREDICTION HISTORY
// ══════════════════════════════════════════════════════════════════
function addToHistory(label, confidence) {
  state.predictionCount++;
  const entry = {
    n:    state.predictionCount,
    label,
    conf: Math.round(confidence),
    time: new Date().toLocaleTimeString(),
  };
  state.history.unshift(entry);   // newest first
  if (state.history.length > 10) state.history.pop();

  renderHistory();
}

function renderHistory() {
  const list    = document.getElementById("historyList");
  const empty   = document.getElementById("historyEmpty");
  const clearBtn = document.getElementById("clearHistoryBtn");

  list.innerHTML = "";

  if (state.history.length === 0) {
    empty.style.display = "";
    clearBtn.style.display = "none";
    return;
  }

  empty.style.display = "none";
  clearBtn.style.display = "";

  state.history.forEach(entry => {
    const isClean = entry.label.toLowerCase() === "clean";
    const li = document.createElement("li");
    li.className = "history-item";
    li.setAttribute("role", "listitem");
    li.innerHTML = `
      <span class="history-item__num">#${entry.n}</span>
      <span class="history-item__dot ${isClean ? "clean" : "dusty"}"></span>
      <span class="history-item__label" style="color:${isClean ? "#4ade80" : "#fbbf24"}">${entry.label}</span>
      <span class="history-item__conf">${entry.conf}%</span>
      <span class="history-item__time">${entry.time}</span>
    `;
    list.appendChild(li);
  });
}

function clearHistory() {
  state.history = [];
  state.predictionCount = 0;
  renderHistory();
}

// ══════════════════════════════════════════════════════════════════
// BUTTON LOADING STATE
// ══════════════════════════════════════════════════════════════════
function setButtonsLoading(loading) {
  const ids = [
    "analyseUploadBtn", "captureBtn",
    "btnDemoClean", "btnDemoDusty",
  ];
  ids.forEach(id => {
    const btn = document.getElementById(id);
    if (!btn) return;
    if (loading) {
      btn.dataset.origText = btn.innerHTML;
      btn.innerHTML = `<div style="width:14px;height:14px;border:2px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.8s linear infinite;display:inline-block"></div> Analysing…`;
      btn.disabled = true;
      btn.classList.add("loading");
    } else {
      if (btn.dataset.origText) btn.innerHTML = btn.dataset.origText;
      btn.classList.remove("loading");
      // Only re-enable upload btn if image is loaded
      if (id === "analyseUploadBtn") {
        btn.disabled = !state.uploadedBase64;
      } else if (id === "captureBtn") {
        btn.disabled = !state.webcamActive;
      } else {
        btn.disabled = false;
      }
    }
  });
}

// ══════════════════════════════════════════════════════════════════
// PARTICLE CANVAS — solar energy theme
// ══════════════════════════════════════════════════════════════════
function initParticles() {
  const canvas = document.getElementById("particleCanvas");
  const ctx    = canvas.getContext("2d");

  let W = 0, H = 0;
  const PARTICLE_COUNT = 80;
  const particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener("resize", resize);

  // Particle factory
  function makeParticle() {
    return {
      x:     Math.random() * W,
      y:     Math.random() * H,
      r:     Math.random() * 1.8 + 0.5,
      vx:    (Math.random() - 0.5) * 0.35,
      vy:    (Math.random() - 0.5) * 0.35 - 0.15,
      alpha: Math.random() * 0.6 + 0.15,
      hue:   Math.random() > 0.6 ? 42 : 218,  // gold or blue
    };
  }

  for (let i = 0; i < PARTICLE_COUNT; i++) particles.push(makeParticle());

  // Connection lines between nearby particles
  function drawConnections() {
    const MAX_DIST = 110;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MAX_DIST) {
          const alpha = (1 - dist / MAX_DIST) * 0.12;
          ctx.beginPath();
          ctx.strokeStyle = `rgba(100, 140, 255, ${alpha})`;
          ctx.lineWidth = 0.8;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    drawConnections();

    particles.forEach(p => {
      // Move
      p.x += p.vx;
      p.y += p.vy;

      // Wrap around edges
      if (p.x < -10) p.x = W + 10;
      if (p.x > W + 10) p.x = -10;
      if (p.y < -10) p.y = H + 10;
      if (p.y > H + 10) p.y = -10;

      // Draw
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue}, 80%, 65%, ${p.alpha})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  draw();
}
