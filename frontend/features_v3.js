/* =====================================================
   RoadGuard v3 — ADDITIONS to script.js
   New features: Live Webcam, Voice Alerts, Speed Alert,
                 Analytics Dashboard, A/B Compare, Geocoding
   ===================================================== */

// ── Voice Alert System ────────────────────────────────────────────────────────
const synth = window.speechSynthesis;
let lastSpoken = "";
let lastSpokeTime = 0;

function speakAlert(text) {
    if (!document.getElementById("voiceEnabled")?.checked) return;
    const now = Date.now();
    if (text === lastSpoken && now - lastSpokeTime < 5000) return; // debounce
    lastSpoken = text;
    lastSpokeTime = now;
    synth.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 1.0; utt.pitch = 1.1; utt.volume = 1.0;
    synth.speak(utt);
}

// ── Live Webcam Real-Time Detection ──────────────────────────────────────────
let liveStream      = null;
let liveInterval    = null;
let livePredicting  = false;
let liveDetCount    = 0;
let liveConfSum     = 0;
let liveDangerCount = 0;
let liveFrameCount  = 0;
let liveFpsTimer    = Date.now();

const liveVideo      = document.getElementById("liveVideo");
const liveCanvas     = document.getElementById("liveCanvas");
const liveOverlay    = document.getElementById("liveOverlay");
const liveOverlayLbl = document.getElementById("liveOverlayLabel");
const liveOverlayCnf = document.getElementById("liveOverlayConf");
const liveName       = document.getElementById("liveName");
const liveBar        = document.getElementById("liveBar");
const liveConf       = document.getElementById("liveConf");
const liveDesc       = document.getElementById("liveDesc");
const liveTop3       = document.getElementById("liveTop3");
const liveDetEl      = document.getElementById("liveDetCount");
const liveAvgEl      = document.getElementById("liveAvgConf");
const liveDanEl      = document.getElementById("liveDangers");
const liveDot        = document.getElementById("liveDot");
const liveStatusText = document.getElementById("liveStatusText");
const liveFpsEl      = document.getElementById("liveFps");
const speedAlert     = document.getElementById("speedAlert");
const startLiveBtn   = document.getElementById("startLiveBtn");
const stopLiveBtn    = document.getElementById("stopLiveBtn");

if (startLiveBtn) {
    startLiveBtn.addEventListener("click", startLiveDetection);
    stopLiveBtn.addEventListener("click", stopLiveDetection);
}

async function startLiveDetection() {
    try {
        liveStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        liveVideo.srcObject = liveStream;
        liveVideo.play();
        liveDetCount = liveConfSum = liveDangerCount = liveFrameCount = 0;

        startLiveBtn.classList.add("hidden");
        stopLiveBtn.classList.remove("hidden");
        liveDot.classList.add("live-active");
        liveStatusText.textContent = "Live detecting…";

        const intervalMs = parseInt(document.getElementById("liveIntervalSel").value);
        liveInterval = setInterval(captureAndPredict, intervalMs);

    } catch (e) {
        showToast("Camera access denied: " + e.message, "error");
    }
}

function stopLiveDetection() {
    clearInterval(liveInterval);
    if (liveStream) liveStream.getTracks().forEach(t => t.stop());
    liveStream = null;
    startLiveBtn.classList.remove("hidden");
    stopLiveBtn.classList.add("hidden");
    liveDot.classList.remove("live-active");
    liveStatusText.textContent = "Camera off";
    if (liveOverlay) liveOverlay.classList.add("hidden");
}

async function captureAndPredict() {
    if (livePredicting || !liveStream) return;
    livePredicting = true;

    // Capture frame from video
    const ctx = liveCanvas.getContext("2d");
    liveCanvas.width  = liveVideo.videoWidth  || 320;
    liveCanvas.height = liveVideo.videoHeight || 240;
    ctx.drawImage(liveVideo, 0, 0, liveCanvas.width, liveCanvas.height);

    liveCanvas.toBlob(async (blob) => {
        try {
            const fd = new FormData();
            fd.append("file", blob, "frame.jpg");
            fd.append("dataset", currentDataset);

            const resp = await fetch(`${BACKEND}/predict_live`, { method: "POST", body: fd });
            const data = await resp.json();

            if (data.confidence > 0) {
                // Update UI
                liveName.textContent = data.label;
                liveConf.textContent = `Confidence: ${data.confidence}%`;
                liveBar.style.width  = data.confidence + "%";
                liveBar.style.background = data.confidence > 70 ? "#10b981" : data.confidence > 40 ? "#f59e0b" : "#ef4444";
                liveDesc.textContent = data.description || "";

                // Overlay
                liveOverlayLbl.textContent = data.label;
                liveOverlayCnf.textContent = data.confidence + "%";
                liveOverlay.classList.remove("hidden");

                // Top 3
                if (data.top3) {
                    liveTop3.innerHTML = data.top3.map(p =>
                        `<div class="top3-item">
                            <span>${p.label}</span>
                            <div class="t3-bar-wrap"><div class="t3-bar" style="width:${p.confidence}%"></div></div>
                            <span class="t3-pct">${p.confidence}%</span>
                        </div>`
                    ).join("");
                }

                // Stats
                liveDetCount++;
                liveConfSum += data.confidence;
                liveDetEl.textContent  = liveDetCount;
                liveAvgEl.textContent  = Math.round(liveConfSum / liveDetCount) + "%";

                // Danger check
                const isDanger = ["stop","no entry","no vehicles","danger","warning","curve","slippery"].some(w => data.label.toLowerCase().includes(w));
                if (isDanger) {
                    liveDangerCount++;
                    liveDanEl.textContent = liveDangerCount;
                    speakAlert("Warning! " + data.label + " detected.");
                } else {
                    speakAlert(data.label + " sign detected.");
                }

                // Speed limit check
                checkSpeedLimit(data.label);

                // FPS
                liveFrameCount++;
                const now = Date.now();
                if (now - liveFpsTimer > 2000) {
                    const fps = (liveFrameCount / ((now - liveFpsTimer) / 1000)).toFixed(1);
                    liveFpsEl.textContent = `${fps} fps`;
                    liveFrameCount = 0;
                    liveFpsTimer = now;
                }
            }
        } catch (e) {
            console.warn("Live predict error:", e);
        }
        livePredicting = false;
    }, "image/jpeg", 0.7);
}

// ── Speed Limit Alert ─────────────────────────────────────────────────────────
function checkSpeedLimit(label) {
    const mySpeed = parseFloat(document.getElementById("mySpeedInput")?.value || 0);
    if (!mySpeed) return;

    // Extract limit from label
    const match = label.match(/(\d+)\s*km/i);
    if (!match) { speedAlert.classList.add("hidden"); return; }
    const limit = parseInt(match[1]);

    if (mySpeed > limit) {
        speedAlert.textContent = `⚠️ Over limit! ${limit} km/h limit — you're doing ${mySpeed} km/h`;
        speedAlert.style.background = "#ef4444";
        speedAlert.classList.remove("hidden");
        speakAlert(`Speed alert! Limit is ${limit} kilometres per hour. Slow down!`);
    } else {
        speedAlert.textContent = `✅ Within limit — ${limit} km/h`;
        speedAlert.style.background = "#10b981";
        speedAlert.classList.remove("hidden");
    }
}

// ── Analytics Dashboard ───────────────────────────────────────────────────────
let chartTimeline   = null;
let chartTopSigns   = null;
let chartDataset    = null;
let chartConfidence = null;

async function loadAnalytics() {
    const days = document.getElementById("analyticsDays")?.value || 30;
    try {
        const resp = await fetch(`${BACKEND}/api/analytics?days=${days}`);
        const data = await resp.json();
        renderAnalytics(data);
    } catch (e) {
        showToast("Analytics load failed", "error");
    }
}

function renderAnalytics(data) {
    // Summary cards
    document.getElementById("aTotalDet").textContent  = data.total;
    document.getElementById("aAvgConf").textContent   = data.avg_confidence + "%";
    document.getElementById("aUniqueSign").textContent = data.top_signs.length;
    document.getElementById("aTopSign").textContent   = data.top_signs[0]?.label || "—";

    const C = Chart.defaults;
    C.color = getComputedStyle(document.documentElement).getPropertyValue("--text").trim() || "#333";

    // ── Timeline chart ────────────────────────────────────────────────────────
    if (chartTimeline) chartTimeline.destroy();
    chartTimeline = new Chart(document.getElementById("chartTimeline"), {
        type: "line",
        data: {
            labels: data.detections_by_day.map(d => d.day),
            datasets: [{
                label: "Detections",
                data:  data.detections_by_day.map(d => d.count),
                borderColor: "#0ea5e9", backgroundColor: "rgba(14,165,233,0.15)",
                tension: 0.4, fill: true, pointRadius: 4
            }]
        },
        options: { responsive: true, plugins: { legend: { display: false } } }
    });

    // ── Top signs bar chart ───────────────────────────────────────────────────
    if (chartTopSigns) chartTopSigns.destroy();
    chartTopSigns = new Chart(document.getElementById("chartTopSigns"), {
        type: "bar",
        data: {
            labels: data.top_signs.map(s => s.label.substring(0, 18)),
            datasets: [{
                label: "Count",
                data:  data.top_signs.map(s => s.count),
                backgroundColor: [
                    "#0ea5e9","#f59e0b","#10b981","#ef4444","#8b5cf6",
                    "#f97316","#06b6d4","#84cc16","#ec4899","#6366f1"
                ]
            }]
        },
        options: {
            responsive: true, indexAxis: "y",
            plugins: { legend: { display: false } },
            scales: { x: { beginAtZero: true } }
        }
    });

    // ── Dataset doughnut ─────────────────────────────────────────────────────
    if (chartDataset) chartDataset.destroy();
    const dsLabels = Object.keys(data.dataset_split);
    const dsData   = Object.values(data.dataset_split);
    chartDataset = new Chart(document.getElementById("chartDataset"), {
        type: "doughnut",
        data: {
            labels: dsLabels.map(l => l === "german" ? "🇩🇪 German" : "🇮🇳 Indian"),
            datasets: [{ data: dsData, backgroundColor: ["#3b82f6","#f97316"], borderWidth: 2 }]
        },
        options: { responsive: true, plugins: { legend: { position: "bottom" } } }
    });

    // ── Confidence distribution ───────────────────────────────────────────────
    if (chartConfidence) chartConfidence.destroy();
    const cb = data.confidence_buckets;
    chartConfidence = new Chart(document.getElementById("chartConfidence"), {
        type: "bar",
        data: {
            labels: Object.keys(cb).map(k => k + "%"),
            datasets: [{
                label: "Detections",
                data:  Object.values(cb),
                backgroundColor: ["#ef4444","#f97316","#f59e0b","#84cc16","#10b981"]
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

// ── A/B Model Comparison ──────────────────────────────────────────────────────
let compareFile = null;

const compareDropZone = document.getElementById("compareDropZone");
const compareInput    = document.getElementById("compareInput");
const comparePreview  = document.getElementById("comparePreview");
const comparePrevWrap = document.getElementById("comparePreviewWrap");
const compareBtn      = document.getElementById("compareBtn");
const compareLoading  = document.getElementById("compareLoading");
const compareResults  = document.getElementById("compareResults");

if (compareDropZone) {
    compareDropZone.addEventListener("click", () => compareInput?.click());
    compareDropZone.addEventListener("dragover", e => { e.preventDefault(); compareDropZone.classList.add("drag-over"); });
    compareDropZone.addEventListener("dragleave", () => compareDropZone.classList.remove("drag-over"));
    compareDropZone.addEventListener("drop", e => {
        e.preventDefault(); compareDropZone.classList.remove("drag-over");
        const f = e.dataTransfer.files[0];
        if (f) loadCompareFile(f);
    });
    compareInput?.addEventListener("change", e => { if (e.target.files[0]) loadCompareFile(e.target.files[0]); });
    document.getElementById("compareClearBtn")?.addEventListener("click", () => {
        compareFile = null;
        comparePrevWrap?.classList.add("hidden");
        compareBtn?.classList.add("hidden");
        compareResults?.classList.add("hidden");
    });
    compareBtn?.addEventListener("click", runCompare);
}

function loadCompareFile(file) {
    compareFile = file;
    const url = URL.createObjectURL(file);
    comparePreview.src = url;
    comparePrevWrap.classList.remove("hidden");
    compareBtn.classList.remove("hidden");
    compareResults.classList.add("hidden");
}

async function runCompare() {
    if (!compareFile) return;
    compareLoading.classList.remove("hidden");
    compareResults.classList.add("hidden");

    const fd = new FormData();
    fd.append("file", compareFile);

    try {
        const resp = await fetch(`${BACKEND}/compare`, { method: "POST", body: fd });
        const data = await resp.json();
        renderCompareResult("compareGermanResult", data.german);
        renderCompareResult("compareIndiaResult",  data.india);
        compareResults.classList.remove("hidden");
    } catch (e) {
        showToast("Comparison failed: " + e.message, "error");
    }
    compareLoading.classList.add("hidden");
}

function renderCompareResult(elId, predictions) {
    const el = document.getElementById(elId);
    if (!predictions || !predictions.length) {
        el.innerHTML = `<p style="opacity:0.6;padding:12px;">Model not loaded or no prediction.</p>`;
        return;
    }
    el.innerHTML = predictions.map((p, i) =>
        `<div class="compare-pred ${i === 0 ? 'compare-top' : ''}">
            <span class="compare-rank">#${i+1}</span>
            <div class="compare-pred-info">
                <strong>${p.label}</strong>
                <div class="bar-track" style="margin:4px 0;">
                    <div class="bar-fill" style="width:${p.confidence}%;background:${i===0?'#10b981':'#94a3b8'}"></div>
                </div>
                <small>${p.confidence}% confidence</small>
            </div>
        </div>`
    ).join("");
}

// ── Tab switch hook: load analytics when analytics tab opens ──────────────────
document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
        if (btn.dataset.tab === "analytics") {
            setTimeout(loadAnalytics, 100);
        }
    });
});

// ── Add refresh analytics button listener ─────────────────────────────────────
document.getElementById("refreshAnalytics")?.addEventListener("click", loadAnalytics);
