/* =====================================================
   RoadGuard v2 — script.js
   Features: Upload, Webcam, Batch, History,
             Top-3, Grad-CAM, Dark mode,
             Drag-drop, Clipboard paste,
             Sound alerts, CSV export, Toast errors,
             German / India dataset toggle
   ===================================================== */

const BACKEND = window.location.origin.includes("localhost")
    ? "http://localhost:5000"
    : window.location.origin;

let currentFile     = null;
let webcamStream    = null;
let history         = [];
let soundEnabled    = true;
let batchApiResults = [];
let currentDataset  = "german";   // 'german' | 'india'

/* YOLOv8 DOM refs */
const detectBtn      = document.getElementById("detectBtn");
const detectCard     = document.getElementById("detectCard");
const detectImg      = document.getElementById("detectImg");
const detectCount    = document.getElementById("detectCount");
const detectList     = document.getElementById("detectList");
const detectResetBtn = document.getElementById("detectResetBtn");

/* Dangerous class IDs (German GTSRB) — will trigger sound alert */
const DANGER_IDS = new Set([9,10,13,14,15,16,17,18,19,20,21,22,23,25,27,28,29,30,31]);

/* ══ DOM refs — Upload ══ */
const dropZone    = document.getElementById("dropZone");
const imgInput    = document.getElementById("imgInput");
const previewWrap = document.getElementById("previewWrap");
const imgPreview  = document.getElementById("imgPreview");
const clearBtn    = document.getElementById("clearBtn");
const analyzeBtn  = document.getElementById("analyzeBtn");
const loadingDiv  = document.getElementById("loadingDiv");
const resultCard  = document.getElementById("resultCard");
const rName       = document.getElementById("r-name");
const rBadge      = document.getElementById("r-badge");
const rDesc       = document.getElementById("r-desc");
const rConf       = document.getElementById("r-conf");
const barFill     = document.getElementById("barFill");
const top3List    = document.getElementById("top3List");
const confWarning = document.getElementById("confWarning");
const resetBtn    = document.getElementById("resetBtn");
const gradcamBtn  = document.getElementById("gradcamBtn");
const gradcamWrap = document.getElementById("gradcamWrap");
const gradcamImg  = document.getElementById("gradcamImg");
const rDsBadge    = document.getElementById("r-dsbadge");

/* ══ DOM refs — Webcam ══ */
const webcamVideo    = document.getElementById("webcamVideo");
const webcamCanvas   = document.getElementById("webcamCanvas");
const startCamBtn    = document.getElementById("startCamBtn");
const captureBtn     = document.getElementById("captureBtn");
const stopCamBtn     = document.getElementById("stopCamBtn");
const camLoadingDiv  = document.getElementById("camLoadingDiv");
const camResultCard  = document.getElementById("camResultCard");
const camName        = document.getElementById("cam-name");
const camBadge       = document.getElementById("cam-badge");
const camDesc        = document.getElementById("cam-desc");
const camConf        = document.getElementById("cam-conf");
const camBarFill     = document.getElementById("camBarFill");
const camTop3List    = document.getElementById("camTop3List");
const camDsBadge     = document.getElementById("cam-dsbadge");

/* ══ DOM refs — Batch ══ */
const batchDropZone    = document.getElementById("batchDropZone");
const batchInput       = document.getElementById("batchInput");
const batchFileList    = document.getElementById("batchFileList");
const batchAnalyzeBtn  = document.getElementById("batchAnalyzeBtn");
const batchLoadingDiv  = document.getElementById("batchLoadingDiv");
const batchResults     = document.getElementById("batchResults");
const csvDownloadBtn   = document.getElementById("csvDownloadBtn");

/* ══ DOM refs — Misc ══ */
const themeToggle  = document.getElementById("themeToggle");
const soundToggle  = document.getElementById("soundToggle");
const historyList  = document.getElementById("historyList");
const clearHistBtn = document.getElementById("clearHistBtn");
const toast        = document.getElementById("toast");
const accuracyPill = document.getElementById("accuracyPill");

/* ═══════════════════════════════════════════════
   DATASET TOGGLE
   ═══════════════════════════════════════════════ */
window.setDataset = function(ds) {
    currentDataset = ds;
    document.getElementById("dsGerman").classList.toggle("active", ds === "german");
    document.getElementById("dsIndia").classList.toggle("active",  ds === "india");
    accuracyPill.textContent = ds === "india" ? "🇮🇳 90.7% accuracy (India)" : "97.7% accuracy";
    showToast(ds === "india"
        ? "🇮🇳 Switched to Indian Traffic Signs (58 classes)"
        : "🌍 Switched to German Traffic Signs (43 classes)", "success");
};

/* ═══════════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════════ */
document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
        btn.classList.add("active");
        document.getElementById("tab-" + btn.dataset.tab).classList.remove("hidden");
    });
});

/* ═══════════════════════════════════════════════
   DARK / LIGHT MODE
   ═══════════════════════════════════════════════ */
const html = document.documentElement;
themeToggle.addEventListener("click", () => {
    const isDark = html.getAttribute("data-theme") === "dark";
    html.setAttribute("data-theme", isDark ? "light" : "dark");
    themeToggle.textContent = isDark ? "🌙" : "☀️";
});

/* ═══════════════════════════════════════════════
   SOUND ALERT
   ═══════════════════════════════════════════════ */
soundToggle.addEventListener("click", () => {
    soundEnabled = !soundEnabled;
    soundToggle.textContent = soundEnabled ? "🔔" : "🔕";
    showToast(soundEnabled ? "Sound alerts ON" : "Sound alerts OFF", "success");
});

function playDangerBeep() {
    if (!soundEnabled) return;
    try {
        const ctx  = new (window.AudioContext || window.webkitAudioContext)();
        const osc  = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = "sine";
        osc.frequency.setValueAtTime(880, ctx.currentTime);
        osc.frequency.setValueAtTime(660, ctx.currentTime + 0.15);
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.5);
    } catch {}
}

/* ═══════════════════════════════════════════════
   TOAST NOTIFICATION
   ═══════════════════════════════════════════════ */
let toastTimer;
function showToast(msg, type = "") {
    clearTimeout(toastTimer);
    toast.textContent = msg;
    toast.className = "toast" + (type ? " " + type : "");
    toast.classList.remove("hidden");
    toastTimer = setTimeout(() => toast.classList.add("hidden"), 3000);
}

/* ═══════════════════════════════════════════════
   UPLOAD — DRAG / DROP / CLICK / PASTE
   ═══════════════════════════════════════════════ */
dropZone.addEventListener("click", () => imgInput.click());

dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) loadFile(file);
});

imgInput.addEventListener("change", e => {
    if (e.target.files[0]) loadFile(e.target.files[0]);
});

/* Paste from clipboard (Ctrl+V) */
document.addEventListener("paste", e => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
        if (item.type.startsWith("image/")) {
            const file = item.getAsFile();
            if (file) {
                document.querySelector('[data-tab="upload"]').click();
                loadFile(file);
            }
            break;
        }
    }
});

function loadFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
        imgPreview.src = ev.target.result;
        show(previewWrap);
        // Show action row (both buttons)
        document.getElementById("actionRow").classList.remove("hidden");
        hide(resultCard);
        hide(detectCard);
        barFill.style.width = "0%";
        hide(gradcamWrap);
    };
    reader.readAsDataURL(file);
}

clearBtn.addEventListener("click", resetUpload);
resetBtn.addEventListener("click", resetUpload);

function resetUpload() {
    currentFile = null;
    imgInput.value = "";
    hide(previewWrap);
    document.getElementById("actionRow").classList.add("hidden");
    hide(resultCard);
    hide(detectCard);
    barFill.style.width = "0%";
    hide(gradcamWrap);
}

/* ═══════════════════════════════════════════════
   YOLOV8 DETECT
   ═══════════════════════════════════════════════ */
detectBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    detectBtn.textContent = "⏳ Detecting…";
    detectBtn.disabled    = true;
    hide(resultCard);
    hide(detectCard);
    show(loadingDiv);

    const result = await postImage(BACKEND + "/detect", "file", currentFile);
    hide(loadingDiv);
    detectBtn.textContent = "🎯 YOLOv8 Detect";
    detectBtn.disabled    = false;

    if (!result || result.error) {
        const msg = result?.error || "Detection failed!";
        showToast("⚠️ " + msg, "error");
        return;
    }

    // Show annotated image
    detectImg.src = "data:image/jpeg;base64," + result.image;

    // Show count badge
    const n = result.count || 0;
    detectCount.textContent = n + (n === 1 ? " object" : " objects");

    // Build detection list
    detectList.innerHTML = "";
    (result.detections || []).forEach((d, i) => {
        const item = document.createElement("div");
        item.className = "detect-item";
        item.innerHTML = `
            <div class="detect-num">${i+1}</div>
            <div class="detect-info">
                <div class="detect-label">🚦 ${d.sign_label}</div>
                <div class="detect-sub">YOLO: ${d.yolo_class} (${d.yolo_conf}%)</div>
            </div>
            <div class="detect-pct">${d.confidence}%</div>`;
        detectList.appendChild(item);
        addHistory({ label: d.sign_label, confidence: d.confidence, description: "" }, "yolo", currentDataset);
    });

    if (n === 0) {
        detectList.innerHTML = '<p style="text-align:center;color:var(--text2);padding:12px">No objects detected in this image.</p>';
    }

    show(detectCard);
    showToast(`🎯 Found ${n} object${n !== 1 ? "s" : ""}!`, n > 0 ? "success" : "");
});

detectResetBtn.addEventListener("click", resetUpload);

/* ═══════════════════════════════════════════════
   ANALYZE — SINGLE IMAGE
   ═══════════════════════════════════════════════ */
analyzeBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    hide(analyzeBtn);
    show(loadingDiv);
    hide(resultCard);
    hide(gradcamWrap);

    const result = await postImage(BACKEND + "/predict", "file", currentFile);
    hide(loadingDiv);

    if (!result || result.error) {
        showToast(result?.error || "Cannot reach Flask backend!", "error");
        show(analyzeBtn);
        return;
    }

    const top = result.top[0];
    const ds  = result.dataset || currentDataset;

    /* Update dataset badge on result card */
    if (rDsBadge) {
        rDsBadge.textContent = ds === "india" ? "🇮🇳 India" : "🌍 German";
        rDsBadge.className   = "ds-badge " + ds;
    }

    displayResult(top, result.top,
        rName, rBadge, rDesc, rConf, barFill, top3List, confWarning);
    show(resultCard);
    addHistory(top, "upload", ds);

    if (ds === "german" && DANGER_IDS.has(top.class_id)) {
        playDangerBeep();
        showToast("⚠️ Warning sign detected!", "warning");
    }
});

/* Grad-CAM */
gradcamBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    gradcamBtn.textContent = "⏳ Generating heatmap…";
    gradcamBtn.disabled = true;

    const result = await postImage(BACKEND + "/gradcam", "file", currentFile);
    gradcamBtn.textContent = "🔥 Show Heatmap (Grad-CAM)";
    gradcamBtn.disabled = false;

    if (!result || result.error) {
        showToast("Grad-CAM failed: " + (result?.error || "unknown"), "error");
        return;
    }

    gradcamImg.src = "data:image/png;base64," + result.image;
    show(gradcamWrap);
});

/* ═══════════════════════════════════════════════
   WEBCAM
   ═══════════════════════════════════════════════ */
startCamBtn.addEventListener("click", async () => {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamVideo.srcObject = webcamStream;
        hide(startCamBtn);
        show(captureBtn);
        show(stopCamBtn);
        hide(camResultCard);
    } catch (err) {
        showToast("Could not access camera: " + err.message, "error");
    }
});

captureBtn.addEventListener("click", async () => {
    webcamCanvas.width  = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    webcamCanvas.getContext("2d").drawImage(webcamVideo, 0, 0);

    webcamCanvas.toBlob(async blob => {
        hide(captureBtn);
        show(camLoadingDiv);
        hide(camResultCard);

        const result = await postImage(BACKEND + "/predict", "file", blob, "capture.png");
        hide(camLoadingDiv);
        show(captureBtn);

        if (!result || result.error) {
            showToast(result?.error || "Backend not reachable.", "error");
            return;
        }

        const top = result.top[0];
        const ds  = result.dataset || currentDataset;

        if (camDsBadge) {
            camDsBadge.textContent = ds === "india" ? "🇮🇳 India" : "🌍 German";
            camDsBadge.className   = "ds-badge " + ds;
        }

        displayResult(top, result.top,
            camName, camBadge, camDesc, camConf, camBarFill, camTop3List, null);
        show(camResultCard);
        addHistory(top, "webcam", ds);
        if (ds === "german" && DANGER_IDS.has(top.class_id)) {
            playDangerBeep(); showToast("⚠️ Warning sign!", "warning");
        }
    }, "image/png");
});

stopCamBtn.addEventListener("click", () => {
    if (webcamStream) webcamStream.getTracks().forEach(t => t.stop());
    webcamVideo.srcObject = null;
    webcamStream = null;
    show(startCamBtn);
    hide(captureBtn);
    hide(stopCamBtn);
});

/* ═══════════════════════════════════════════════
   BATCH UPLOAD
   ═══════════════════════════════════════════════ */
batchDropZone.addEventListener("click", () => batchInput.click());

batchDropZone.addEventListener("dragover", e => {
    e.preventDefault();
    batchDropZone.classList.add("drag-over");
});
batchDropZone.addEventListener("dragleave", () => batchDropZone.classList.remove("drag-over"));
batchDropZone.addEventListener("drop", e => {
    e.preventDefault();
    batchDropZone.classList.remove("drag-over");
    loadBatchFiles(e.dataTransfer.files);
});

batchInput.addEventListener("change", e => loadBatchFiles(e.target.files));

let batchFiles = [];

function loadBatchFiles(files) {
    batchFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
    if (!batchFiles.length) return;

    batchFileList.innerHTML = "";
    batchFiles.forEach(f => {
        const chip = document.createElement("div");
        chip.className = "batch-chip";
        chip.innerHTML = `<span>🖼 ${f.name}</span>`;
        batchFileList.appendChild(chip);
    });

    show(batchFileList);
    show(batchAnalyzeBtn);
    hide(batchResults);
}

batchAnalyzeBtn.addEventListener("click", async () => {
    if (!batchFiles.length) return;
    hide(batchAnalyzeBtn);
    show(batchLoadingDiv);
    hide(batchResults);

    const form = new FormData();
    batchFiles.forEach(f => form.append("files", f));
    form.append("dataset", currentDataset);

    let result;
    try {
        const res = await fetch(BACKEND + "/predict_batch", { method: "POST", body: form });
        result = await res.json();
    } catch {
        hide(batchLoadingDiv);
        show(batchAnalyzeBtn);
        showToast("Backend not reachable. Is Flask running?", "error");
        return;
    }

    hide(batchLoadingDiv);
    show(batchAnalyzeBtn);

    if (!result.results) { showToast("Batch failed.", "error"); return; }

    batchResults.innerHTML = "";
    result.results.forEach((r, i) => {
        const row = document.createElement("div");
        row.className = "batch-row";
        if (r.error) {
            row.innerHTML = `
                <div class="batch-num">${i+1}</div>
                <div class="batch-info">
                    <div class="batch-fname">${r.filename}</div>
                    <div class="batch-error">❌ ${r.error}</div>
                </div>`;
        } else {
            row.innerHTML = `
                <div class="batch-num">${i+1}</div>
                <div class="batch-info">
                    <div class="batch-label">🚦 ${r.label}</div>
                    <div class="batch-fname">${r.filename}</div>
                </div>
                <div class="batch-conf">${r.confidence}%</div>`;
            addHistory({ label: r.label, confidence: r.confidence, description: r.description }, "batch", currentDataset);
        }
        batchResults.appendChild(row);
    });
    batchApiResults = result.results;
    show(batchResults);
    show(csvDownloadBtn);
});

/* CSV Download */
csvDownloadBtn.addEventListener("click", () => {
    if (!batchApiResults.length) return;
    const rows = [["#", "Filename", "Sign", "Confidence (%)", "Description"]];
    batchApiResults.forEach((r, i) => {
        if (!r.error) rows.push([i+1, r.filename, r.label, r.confidence, r.description || ""]);
        else          rows.push([i+1, r.filename, "ERROR", "", r.error]);
    });
    const csv  = rows.map(r => r.map(c => `"${String(c).replace(/"/g,'""')}"`).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = "roadguard_results.csv"; a.click();
    URL.revokeObjectURL(url);
    showToast("✅ CSV downloaded!", "success");
});

/* ═══════════════════════════════════════════════
   HISTORY
   ═══════════════════════════════════════════════ */
function addHistory(top, source, ds) {
    const time = new Date().toLocaleTimeString();
    history.unshift({ ...top, time, source, dataset: ds });
    renderHistory();
}

function renderHistory() {
    if (!history.length) {
        historyList.innerHTML = '<p class="history-empty">No detections yet.</p>';
        return;
    }
    historyList.innerHTML = "";
    history.slice(0, 20).forEach(h => {
        const item = document.createElement("div");
        item.className = "history-item";
        const srcIcon = h.source === "webcam" ? "🎥" : h.source === "batch" ? "📁" : "📷";
        const dsTag   = h.dataset === "india" ? " 🇮🇳" : "";
        item.innerHTML = `
            <div class="hist-label">🚦 ${h.label}${dsTag}</div>
            <div class="hist-meta">
                <span class="hist-conf">${h.confidence}%</span>
                <span>${srcIcon} ${h.source}</span>
                <span>${h.time}</span>
            </div>`;
        historyList.appendChild(item);
    });
}

clearHistBtn.addEventListener("click", () => {
    history = [];
    renderHistory();
});

/* ═══════════════════════════════════════════════
   VIDEO PROCESSING
   ═══════════════════════════════════════════════ */
const videoDropZone   = document.getElementById("videoDropZone");
const videoInput      = document.getElementById("videoInput");
const videoFileInfo   = document.getElementById("videoFileInfo");
const videoFileName   = document.getElementById("videoFileName");
const videoClearBtn   = document.getElementById("videoClearBtn");
const videoAnalyzeBtn = document.getElementById("videoAnalyzeBtn");
const videoLoadingDiv = document.getElementById("videoLoadingDiv");
const videoResult     = document.getElementById("videoResult");
const videoStats      = document.getElementById("videoStats");
const videoCollage    = document.getElementById("videoCollage");
const videoSummary    = document.getElementById("videoSummary");
const videoTimeline   = document.getElementById("videoTimeline");
const videoResetBtn   = document.getElementById("videoResetBtn");

let currentVideo = null;

videoDropZone.addEventListener("click", () => videoInput.click());
videoDropZone.addEventListener("dragover", e => { e.preventDefault(); videoDropZone.classList.add("drag-over"); });
videoDropZone.addEventListener("dragleave", () => videoDropZone.classList.remove("drag-over"));
videoDropZone.addEventListener("drop", e => {
    e.preventDefault();
    videoDropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("video/")) loadVideo(file);
});
videoInput.addEventListener("change", e => { if (e.target.files[0]) loadVideo(e.target.files[0]); });

function loadVideo(file) {
    if (file.size > 50 * 1024 * 1024) {
        showToast("Video too large! Max 50 MB.", "error"); return;
    }
    currentVideo = file;
    videoFileName.textContent = file.name;
    show(videoFileInfo);
    show(videoAnalyzeBtn);
    hide(videoDropZone);
    hide(videoResult);
}

function resetVideo() {
    currentVideo = null;
    videoInput.value = "";
    show(videoDropZone);
    hide(videoFileInfo);
    hide(videoAnalyzeBtn);
    hide(videoResult);
    hide(videoLoadingDiv);
}
videoClearBtn.addEventListener("click", resetVideo);
videoResetBtn.addEventListener("click", resetVideo);

videoAnalyzeBtn.addEventListener("click", async () => {
    if (!currentVideo) return;
    hide(videoAnalyzeBtn);
    show(videoLoadingDiv);
    hide(videoResult);

    const form = new FormData();
    form.append("file", currentVideo);
    form.append("dataset", currentDataset);

    let result;
    try {
        const res = await fetch(BACKEND + "/process_video", { method: "POST", body: form });
        result = await res.json();
    } catch {
        hide(videoLoadingDiv);
        show(videoAnalyzeBtn);
        showToast("Video processing failed — backend not reachable.", "error");
        return;
    }

    hide(videoLoadingDiv);

    if (result.error) {
        showToast("Error: " + result.error, "error");
        show(videoAnalyzeBtn);
        return;
    }

    // ── Stats pills ──
    videoStats.innerHTML = `
        <div class="video-stat-pill">⏱ ${result.duration}s</div>
        <div class="video-stat-pill">🎞 ${result.total_frames} frames</div>
        <div class="video-stat-pill">🔍 ${result.sampled} sampled</div>
        <div class="video-stat-pill">🚦 ${Object.keys(result.summary).length} unique signs</div>`;

    // ── Collage ──
    if (result.thumbnail) {
        videoCollage.src = "data:image/jpeg;base64," + result.thumbnail;
        show(videoCollage);
    }

    // ── Summary chips ──
    videoSummary.innerHTML = "";
    const sorted = Object.entries(result.summary).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([label, count]) => {
        const chip = document.createElement("div");
        chip.className = "video-sign-chip";
        chip.innerHTML = `🚦 ${label} <span class="video-sign-count">${count}×</span>`;
        videoSummary.appendChild(chip);
        addHistory({ label, confidence: 0, description: "" }, "video", currentDataset);
    });

    // ── Timeline ──
    videoTimeline.innerHTML = "";
    result.detections.forEach(d => {
        const item = document.createElement("div");
        item.className = "video-tl-item";
        item.innerHTML = `
            <div class="video-tl-time">${d.time_str}</div>
            <div class="video-tl-label">🚦 ${d.label}</div>
            <div class="video-tl-conf">${d.confidence}%</div>`;
        videoTimeline.appendChild(item);
    });

    show(videoResult);
    showToast(`🎬 Found ${Object.keys(result.summary).length} unique sign type(s)!`, "success");
});

/* ═══════════════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════════════ */
function show(el) { el?.classList.remove("hidden"); }
function hide(el) { el?.classList.add("hidden"); }

async function postImage(url, fieldName, file, filename) {
    const form = new FormData();
    form.append(fieldName, file, filename || file.name || "image.png");
    form.append("dataset", currentDataset);    // always send current dataset
    try {
        const res = await fetch(url, { method: "POST", body: form });
        return await res.json();
    } catch {
        return null;
    }
}

function displayResult(top, top3, nameEl, badgeEl, descEl, confEl, barEl, top3El, warnEl) {
    nameEl.textContent  = "🚦 " + top.label;
    badgeEl.textContent = top.confidence + "%";
    descEl.textContent  = top.description || "";
    confEl.textContent  = "Confidence: " + top.confidence + "%";

    barEl.style.width = "0%";
    setTimeout(() => { barEl.style.width = top.confidence + "%"; }, 80);

    /* Low confidence warning */
    if (warnEl) {
        if (top.confidence < 60) show(warnEl);
        else hide(warnEl);
    }

    /* Top-3 list */
    if (top3El) {
        top3El.innerHTML = "";
        top3.forEach((p, i) => {
            const item = document.createElement("div");
            item.className = "top3-item";
            item.innerHTML = `
                <div class="top3-rank rank-${i+1}">${i+1}</div>
                <div class="top3-info">
                    <div class="top3-label">${p.label}</div>
                    <div class="top3-bar-track">
                        <div class="top3-bar-fill" style="width:0%" data-pct="${p.confidence}"></div>
                    </div>
                </div>
                <div class="top3-pct">${p.confidence}%</div>`;
            top3El.appendChild(item);
        });
        /* Animate top-3 bars */
        setTimeout(() => {
            top3El.querySelectorAll(".top3-bar-fill").forEach(b => {
                b.style.width = b.dataset.pct + "%";
            });
        }, 100);
    }
}
