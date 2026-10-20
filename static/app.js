/**
 * LinesDraw Studio - Interactive Client-Side Stroke Animator
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Elements ---
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const samplePills = document.querySelectorAll(".sample-pill-btn");
  const styleCards = document.querySelectorAll(".style-card");

  // Sliders
  const gammaSlider = document.getElementById("gammaSlider");
  const gammaVal = document.getElementById("gammaVal");
  const densitySlider = document.getElementById("densitySlider");
  const densityVal = document.getElementById("densityVal");
  const spacingSlider = document.getElementById("spacingSlider");
  const spacingVal = document.getElementById("spacingVal");
  const widthSlider = document.getElementById("widthSlider");
  const widthVal = document.getElementById("widthVal");
  const budgetSlider = document.getElementById("budgetSlider");
  const budgetVal = document.getElementById("budgetVal");
  const recomputeBtn = document.getElementById("recomputeBtn");

  // Telemetry
  const statLinesCount = document.getElementById("statLinesCount");
  const statDarkness = document.getElementById("statDarkness");
  const statEdgeCoverage = document.getElementById("statEdgeCoverage");
  const statCalcTime = document.getElementById("statCalcTime");

  // Stage & Canvas
  const paperSheet = document.getElementById("paperSheet");
  const drawingCanvas = document.getElementById("drawingCanvas");
  const ctx = drawingCanvas.getContext("2d");
  const refImage = document.getElementById("refImage");
  const penCursor = document.getElementById("penCursor");
  const emptyPlaceholder = document.getElementById("emptyPlaceholder");
  const statusIndicator = document.getElementById("statusIndicator");
  const statusMessage = document.getElementById("statusMessage");

  // Transport Controls
  const btnPlayPause = document.getElementById("btnPlayPause");
  const playIcon = document.getElementById("playIcon");
  const pauseIcon = document.getElementById("pauseIcon");
  const btnRestart = document.getElementById("btnRestart");
  const btnStep = document.getElementById("btnStep");
  const btnFastForward = document.getElementById("btnFastForward");
  const speedSlider = document.getElementById("speedSlider");
  const speedValueLabel = document.getElementById("speedValueLabel");
  const progressScrubber = document.getElementById("progressScrubber");
  const currentStrokeNum = document.getElementById("currentStrokeNum");
  const totalStrokeNum = document.getElementById("totalStrokeNum");
  const progressPercent = document.getElementById("progressPercent");

  // View Toggles & Header Buttons
  const viewBtns = document.querySelectorAll(".view-btn");
  const soundToggleBtn = document.getElementById("soundToggleBtn");
  const soundOnIcon = document.getElementById("soundOnIcon");
  const soundOffIcon = document.getElementById("soundOffIcon");
  const downloadHighResBtn = document.getElementById("downloadHighResBtn");

  // --- State ---
  let lines = [];
  let currentStrokeIndex = 0;
  let isPlaying = false;
  let animFrameId = null;
  let sheetWidth = 600;
  let sheetHeight = 700;
  let currentImageFile = null;
  let currentSampleId = null;
  let currentStyle = "adaptive";
  let strokeWidth = 0.7;
  let soundEnabled = true;

  // Stroke width stepper & preset buttons
  const btnDecreaseWidth = document.getElementById("btnDecreaseWidth");
  const btnIncreaseWidth = document.getElementById("btnIncreaseWidth");
  const presetBtns = document.querySelectorAll(".btn-preset");

  function setStrokeWidth(newVal) {
    newVal = Math.max(0.3, Math.min(2.0, Math.round(newVal * 100) / 100));
    strokeWidth = newVal;
    widthSlider.value = newVal;
    widthVal.textContent = `${newVal.toFixed(2)} px`;
    presetBtns.forEach((btn) => {
      const w = parseFloat(btn.dataset.width);
      btn.classList.toggle("selected", Math.abs(w - newVal) < 0.04);
    });
    redrawUpToIndex(currentStrokeIndex);
    if (currentImageFile || currentSampleId) {
      recomputeBtn.removeAttribute("disabled");
    }
  }

  if (btnDecreaseWidth) {
    btnDecreaseWidth.addEventListener("click", () => setStrokeWidth(strokeWidth - 0.1));
  }
  if (btnIncreaseWidth) {
    btnIncreaseWidth.addEventListener("click", () => setStrokeWidth(strokeWidth + 0.1));
  }
  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => setStrokeWidth(parseFloat(btn.dataset.width)));
  });

  // --- Web Audio Plotter Sound Synthesizer ---
  let audioCtx = null;
  function playPlotterSound() {
    if (!soundEnabled) return;
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === "suspended") {
        audioCtx.resume();
      }
      const bufferSize = audioCtx.sampleRate * 0.015; // 15ms short click/friction
      const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.3));
      }

      const noise = audioCtx.createBufferSource();
      noise.buffer = buffer;

      const filter = audioCtx.createBiquadFilter();
      filter.type = "bandpass";
      filter.frequency.value = 2800 + Math.random() * 800;
      filter.Q.value = 3.0;

      const gain = audioCtx.createGain();
      gain.gain.setValueAtTime(0.04, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.014);

      noise.connect(filter);
      filter.connect(gain);
      gain.connect(audioCtx.destination);
      noise.start();
    } catch (e) {
      // Audio not supported or blocked
    }
  }

  // --- Parameter Sliders Event Listeners ---
  gammaSlider.addEventListener("input", (e) => (gammaVal.textContent = e.target.value));
  densitySlider.addEventListener("input", (e) => (densityVal.textContent = e.target.value));
  spacingSlider.addEventListener("input", (e) => (spacingVal.textContent = `${e.target.value} px`));
  budgetSlider.addEventListener("input", (e) => (budgetVal.textContent = `${parseInt(e.target.value, 10).toLocaleString()} lines`));
  widthSlider.addEventListener("input", (e) => {
    setStrokeWidth(parseFloat(e.target.value));
  });

  speedSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    speedValueLabel.textContent = `${val} line${val > 1 ? "s" : ""}/frame`;
  });


  // Style cards selection
  styleCards.forEach((card) => {
    card.addEventListener("click", () => {
      styleCards.forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      currentStyle = card.dataset.style;
      if (currentImageFile || currentSampleId) {
        recomputeBtn.removeAttribute("disabled");
      }
    });
  });

  // Sound Toggle
  soundToggleBtn.addEventListener("click", () => {
    soundEnabled = !soundEnabled;
    soundToggleBtn.classList.toggle("active", soundEnabled);
    soundOnIcon.style.display = soundEnabled ? "block" : "none";
    soundOffIcon.style.display = soundEnabled ? "none" : "block";
  });

  // View mode toggles
  viewBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      viewBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.dataset.view;
      paperSheet.classList.toggle("view-ghost", mode === "overlay");
      refImage.style.display = mode === "split" || mode === "overlay" ? "block" : "none";
    });
  });

  // --- Drag and Drop File Upload ---
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      handleImageSelection(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleImageSelection(e.target.files[0]);
    }
  });

  // Benchmark sample pills
  samplePills.forEach((pill) => {
    pill.addEventListener("click", () => {
      const sampleId = pill.dataset.sample;
      handleSampleSelection(sampleId);
    });
  });

  recomputeBtn.addEventListener("click", () => {
    if (currentImageFile) {
      submitImageForProcessing(currentImageFile, null);
    } else if (currentSampleId) {
      submitImageForProcessing(null, currentSampleId);
    }
  });

  function handleImageSelection(file) {
    currentImageFile = file;
    currentSampleId = null;
    submitImageForProcessing(file, null);
  }

  function handleSampleSelection(sampleId) {
    currentSampleId = sampleId;
    currentImageFile = null;
    submitImageForProcessing(null, sampleId);
  }

  // --- API Image Processing ---
  async function submitImageForProcessing(file, sampleId) {
    pauseDrawing();
    setStatus("Analyzing image & computing mathematical lines...", true);

    const formData = new FormData();
    if (file) {
      formData.append("file", file);
    } else if (sampleId) {
      formData.append("sample_id", sampleId);
    }
    formData.append("style", currentStyle);
    formData.append("gamma", gammaSlider.value);
    formData.append("rho_max", densitySlider.value);
    formData.append("base_spacing", spacingSlider.value);
    formData.append("line_width", widthSlider.value);
    formData.append("line_budget", budgetSlider.value);
    formData.append("iterations", 1);

    try {
      const response = await fetch("/api/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned error ${response.status}`);
      }

      const data = await response.json();
      onLinesGenerated(data);
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`, false);
    }
  }

  function onLinesGenerated(data) {
    lines = data.lines || [];
    sheetWidth = data.width;
    sheetHeight = data.height;

    // Setup canvas resolution
    drawingCanvas.width = sheetWidth;
    drawingCanvas.height = sheetHeight;
    drawingCanvas.style.width = `${sheetWidth}px`;
    drawingCanvas.style.height = `${sheetHeight}px`;

    paperSheet.style.width = `${sheetWidth}px`;
    paperSheet.style.height = `${sheetHeight}px`;

    // Reference image
    if (data.preview_original) {
      refImage.src = data.preview_original;
    }

    // Telemetry
    statLinesCount.textContent = `${lines.length.toLocaleString()}`;
    statDarkness.textContent = `${data.metadata.mean_darkness}`;
    statEdgeCoverage.textContent = `${data.metadata.edge_coverage_pct}%`;
    statCalcTime.textContent = `${data.metadata.computation_time_s}s`;

    // Face-Aware Telemetry Badge
    const faceCard = document.getElementById("faceAwareCard");
    const faceText = document.getElementById("statFaceAware");
    if (faceCard && faceText) {
      if (data.metadata.face_aware && data.metadata.face_aware.detected) {
        faceCard.style.display = "block";
        const n = data.metadata.face_aware.num_faces;
        faceText.textContent = `Active (${n} face): Eyes, Eyebrows, Nose, Lips, Jawline Prioritized`;
      } else {
        faceCard.style.display = "none";
      }
    }

    // Enable buttons

    emptyPlaceholder.style.display = "none";
    recomputeBtn.removeAttribute("disabled");
    downloadHighResBtn.removeAttribute("disabled");
    btnPlayPause.removeAttribute("disabled");
    btnRestart.removeAttribute("disabled");
    btnStep.removeAttribute("disabled");
    btnFastForward.removeAttribute("disabled");
    progressScrubber.removeAttribute("disabled");

    progressScrubber.max = lines.length;
    totalStrokeNum.textContent = lines.length.toLocaleString();

    // Reset canvas to pure white sheet and start drawing
    clearCanvas();
    currentStrokeIndex = 0;
    updateProgressUI();
    setStatus(`Ready to draw ${lines.length.toLocaleString()} lines on paper sheet`, false);

    // Auto-play drawing
    playDrawing();
  }

  function clearCanvas() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, sheetWidth, sheetHeight);
  }

  // --- Real-time Line-by-Line Drawing Engine ---
  function drawSingleLine(seg) {
    // seg: [x1, y1, x2, y2, family, length]
    const x1 = seg[0];
    const y1 = seg[1];
    const x2 = seg[2];
    const y2 = seg[3];

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);

    // Near-black ink with sub-pixel antialiasing
    ctx.strokeStyle = "rgba(18, 18, 22, 0.95)";
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = "round";
    ctx.stroke();
    ctx.restore();

    // Move glowing pen tip cursor
    penCursor.style.display = "block";
    penCursor.style.left = `${x2}px`;
    penCursor.style.top = `${y2}px`;
  }

  function animationLoop() {
    if (!isPlaying) return;

    const speed = parseInt(speedSlider.value, 10);
    const endIndex = Math.min(lines.length, currentStrokeIndex + speed);

    for (let i = currentStrokeIndex; i < endIndex; i++) {
      drawSingleLine(lines[i]);
    }

    currentStrokeIndex = endIndex;
    updateProgressUI();
    playPlotterSound();

    if (currentStrokeIndex >= lines.length) {
      pauseDrawing();
      penCursor.style.display = "none";
      setStatus("Completed! All mathematical lines placed.", false);
    } else {
      animFrameId = requestAnimationFrame(animationLoop);
    }
  }

  function playDrawing() {
    if (lines.length === 0) return;
    if (currentStrokeIndex >= lines.length) {
      clearCanvas();
      currentStrokeIndex = 0;
    }
    isPlaying = true;
    playIcon.style.display = "none";
    pauseIcon.style.display = "block";
    btnPlayPause.title = "Pause Drawing (Space)";
    statusIndicator.classList.add("drawing");
    setStatus("Drawing lines stroke by stroke...", true);
    animFrameId = requestAnimationFrame(animationLoop);
  }

  function pauseDrawing() {
    isPlaying = false;
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
    playIcon.style.display = "block";
    pauseIcon.style.display = "none";
    btnPlayPause.title = "Play Drawing (Space)";
    statusIndicator.classList.remove("drawing");
  }

  function redrawUpToIndex(targetIndex) {
    clearCanvas();
    const clampedIndex = Math.max(0, Math.min(lines.length, targetIndex));
    for (let i = 0; i < clampedIndex; i++) {
      const seg = lines[i];
      ctx.beginPath();
      ctx.moveTo(seg[0], seg[1]);
      ctx.lineTo(seg[2], seg[3]);
      ctx.strokeStyle = "rgba(18, 18, 22, 0.95)";
      ctx.lineWidth = strokeWidth;
      ctx.lineCap = "round";
      ctx.stroke();
    }
    currentStrokeIndex = clampedIndex;
    updateProgressUI();
    if (clampedIndex > 0 && clampedIndex <= lines.length) {
      const lastSeg = lines[clampedIndex - 1];
      penCursor.style.display = "block";
      penCursor.style.left = `${lastSeg[2]}px`;
      penCursor.style.top = `${lastSeg[3]}px`;
    } else {
      penCursor.style.display = "none";
    }
  }

  function updateProgressUI() {
    currentStrokeNum.textContent = currentStrokeIndex.toLocaleString();
    progressScrubber.value = currentStrokeIndex;
    const pct = lines.length > 0 ? ((currentStrokeIndex / lines.length) * 100).toFixed(1) : "0.0";
    progressPercent.textContent = `${pct}%`;
  }

  function setStatus(msg, isBusy) {
    statusMessage.textContent = msg;
    if (isBusy) {
      statusIndicator.classList.add("drawing");
    } else {
      statusIndicator.classList.remove("drawing");
    }
  }

  // --- Transport Button Handlers ---
  btnPlayPause.addEventListener("click", () => {
    if (isPlaying) {
      pauseDrawing();
      setStatus("Paused", false);
    } else {
      playDrawing();
    }
  });

  btnRestart.addEventListener("click", () => {
    pauseDrawing();
    clearCanvas();
    currentStrokeIndex = 0;
    penCursor.style.display = "none";
    updateProgressUI();
    playDrawing();
  });

  btnStep.addEventListener("click", () => {
    pauseDrawing();
    if (currentStrokeIndex < lines.length) {
      drawSingleLine(lines[currentStrokeIndex]);
      currentStrokeIndex++;
      updateProgressUI();
      playPlotterSound();
    }
  });

  btnFastForward.addEventListener("click", () => {
    pauseDrawing();
    redrawUpToIndex(lines.length);
    penCursor.style.display = "none";
    setStatus("Completed! All lines rendered.", false);
  });

  progressScrubber.addEventListener("input", (e) => {
    pauseDrawing();
    const targetIdx = parseInt(e.target.value, 10);
    redrawUpToIndex(targetIdx);
  });

  // Keyboard shortcut: Spacebar to toggle Play/Pause
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !btnPlayPause.disabled && e.target.tagName !== "INPUT") {
      e.preventDefault();
      btnPlayPause.click();
    } else if (e.code === "KeyR" && !btnRestart.disabled && e.target.tagName !== "INPUT") {
      btnRestart.click();
    }
  });

  // --- High-Res Export Download ---
  downloadHighResBtn.addEventListener("click", async () => {
    if (lines.length === 0) return;
    setStatus("Rendering supersampled high-res PNG...", true);

    try {
      const response = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lines: lines,
          width: sheetWidth,
          height: sheetHeight,
          scale: 2.0,
          line_width: strokeWidth,
          supersampling: 2,
        }),
      });

      if (!response.ok) throw new Error("Export failed");

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = downloadUrl;
      a.download = `mathematical_line_art_${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(downloadUrl);
      setStatus("High-res PNG downloaded successfully!", false);
    } catch (err) {
      console.error(err);
      setStatus("Export failed.", false);
    }
  });

  // Load default portrait benchmark on startup
  handleSampleSelection("portrait");
});
