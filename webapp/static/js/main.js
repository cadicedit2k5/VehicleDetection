/**
 * YOLOv3 Vehicle Detection — Frontend Logic
 * Handles: image upload, API calls, canvas rendering, stats animation
 */

// ─── Constants ──────────────────────────────────────────────────────────────
const CLASS_COLORS = {
  car:        '#00d2ff',
  bus:        '#ff8c00',
  truck:      '#32dc64',
  motorcycle: '#dc32b4',
  bicycle:    '#ffdc1e',
};

const CLASS_ICONS = {
  car: '🚗', bus: '🚌', truck: '🚛', motorcycle: '🏍️', bicycle: '🚲',
};

// ─── DOM Refs ────────────────────────────────────────────────────────────────
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const confSlider     = document.getElementById('conf-slider');
const iouSlider      = document.getElementById('iou-slider');
const confValue      = document.getElementById('conf-value');
const iouValue       = document.getElementById('iou-value');
const confFill       = document.getElementById('conf-fill');
const iouFill        = document.getElementById('iou-fill');
const detectBtn      = document.getElementById('detect-btn');
const btnLabel       = document.getElementById('btn-label');
const btnPulse       = document.getElementById('btn-pulse');
const statusDot      = document.getElementById('status-dot');
const statusText     = document.getElementById('status-text');
const deviceBadge    = document.getElementById('device-badge');
const deviceLabel    = document.getElementById('device-label');
const emptyState     = document.getElementById('empty-state');
const canvasWrapper  = document.getElementById('canvas-wrapper');
const canvas         = document.getElementById('result-canvas');
const ctx            = canvas.getContext('2d');
const detectOverlay  = document.getElementById('detecting-overlay');
const statsSection   = document.getElementById('stats-section');
const classStats     = document.getElementById('class-stats');
const totalCount     = document.getElementById('total-count');
const inferenceTime  = document.getElementById('inference-time-val');
const boxLegend      = document.getElementById('box-legend');
const downloadBtn    = document.getElementById('download-btn');
const toggleBoxesBtn = document.getElementById('toggle-boxes-btn');
const toastContainer = document.getElementById('toast-container');
const infoCheckpoint = document.getElementById('info-checkpoint');
const infoDevice     = document.getElementById('info-device');
const infoClasses    = document.getElementById('info-classes');

// ─── State ──────────────────────────────────────────────────────────────────
let state = {
  imageFile:      null,
  imageObj:       null,        // HTMLImageElement
  currentResult:  null,        // Last API response
  showBoxes:      true,
  modelLoaded:    false,
};

// ─── Particles Background ────────────────────────────────────────────────────
(function initParticles() {
  const container = document.getElementById('particles-container');
  const COLORS = ['#00d2ff', '#7b2ff7', '#32dc64', '#ff8c00'];
  for (let i = 0; i < 28; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 3 + 1;
    p.style.cssText = `
      width: ${size}px; height: ${size}px;
      left: ${Math.random() * 100}%;
      bottom: -${size}px;
      background: ${COLORS[Math.floor(Math.random() * COLORS.length)]};
      animation-duration: ${Math.random() * 20 + 15}s;
      animation-delay: ${Math.random() * 20}s;
      opacity: 0;
    `;
    container.appendChild(p);
  }
})();

// ─── Slider logic ────────────────────────────────────────────────────────────
function updateSlider(slider, fillEl, valueEl) {
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  const val = parseFloat(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  fillEl.style.width = pct + '%';
  valueEl.textContent = val.toFixed(2);
}

confSlider.addEventListener('input', () => updateSlider(confSlider, confFill, confValue));
iouSlider .addEventListener('input', () => updateSlider(iouSlider,  iouFill,  iouValue));

updateSlider(confSlider, confFill, confValue);
updateSlider(iouSlider,  iouFill,  iouValue);

// ─── Model Info ──────────────────────────────────────────────────────────────
async function fetchModelInfo() {
  try {
    const res = await fetch('/model-info');
    const info = await res.json();

    if (info.loaded) {
      statusDot.className  = 'badge-dot connected';
      statusText.textContent = 'Model sẵn sàng';
      state.modelLoaded = true;
    } else {
      statusDot.className  = 'badge-dot error';
      statusText.textContent = 'Không tìm thấy checkpoint';
    }

    deviceLabel.textContent = (info.device || 'cpu').toUpperCase();
    deviceBadge.style.display = 'flex';
    infoCheckpoint.textContent = info.checkpoint || '—';
    infoDevice.textContent     = (info.device || '—').toUpperCase();
    infoClasses.textContent    = info.classes ? info.classes.join(', ') : '5';

    buildLegend(info.classes || Object.keys(CLASS_COLORS));

  } catch (err) {
    statusDot.className  = 'badge-dot error';
    statusText.textContent = 'Lỗi kết nối server';
    showToast('Không thể kết nối đến server Flask. Hãy chạy app.py.', 'error');
  }
}

// ─── Legend ──────────────────────────────────────────────────────────────────
function buildLegend(classes) {
  boxLegend.innerHTML = '';
  classes.forEach(cls => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `
      <div class="legend-dot" style="background:${CLASS_COLORS[cls] || '#888'}"></div>
      <span>${CLASS_ICONS[cls] || ''} ${cls}</span>
    `;
    boxLegend.appendChild(item);
  });
}

// ─── Upload ──────────────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
  else showToast('Vui lòng chọn file ảnh hợp lệ.', 'error');
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ─── Paste Image (Ctrl+V) ─────────────────────────────────────────────────────
document.addEventListener('paste', e => {
  const items = (e.clipboardData || e.originalEvent?.clipboardData)?.items;
  if (!items) return;

  for (const item of items) {
    if (item.type.startsWith('image/')) {
      e.preventDefault();
      const blob = item.getAsFile();
      if (!blob) continue;

      // Tạo File với tên gợi nhớ từ timestamp
      const ext = item.type.split('/')[1] || 'png';
      const file = new File([blob], `pasted_image_${Date.now()}.${ext}`, { type: item.type });

      // Flash hiệu ứng trên drop zone
      dropZone.classList.add('drag-over');
      setTimeout(() => dropZone.classList.remove('drag-over'), 400);

      handleFile(file);
      showToast('Đã dán ảnh từ clipboard ✓', 'success');
      return;   // chỉ xử lý ảnh đầu tiên
    }
  }
});

function handleFile(file) {
  state.imageFile = file;
  detectBtn.disabled = false;
  btnLabel.textContent = 'Phát hiện phương tiện';

  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      state.imageObj = img;
      showImageOnCanvas(img);
      showToast(`Ảnh "${file.name}" đã sẵn sàng`, 'info');
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// ─── Canvas ──────────────────────────────────────────────────────────────────
function showImageOnCanvas(img) {
  emptyState.style.display    = 'none';
  canvasWrapper.style.display = 'flex';

  const maxW = canvasWrapper.clientWidth  - 32;
  const maxH = Math.round(window.innerHeight * 0.65);
  const scale = Math.min(maxW / img.width, maxH / img.height, 1);

  canvas.width  = Math.round(img.width  * scale);
  canvas.height = Math.round(img.height * scale);

  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  state.currentResult = null;
}

function drawBoxesOnCanvas(result) {
  if (!state.imageObj) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageObj, 0, 0, canvas.width, canvas.height);

  if (!state.showBoxes || !result || !result.boxes) return;

  const W = canvas.width;
  const H = canvas.height;

  const confThresh = parseFloat(confSlider.value);

  result.boxes.forEach(box => {
    if (box.score < confThresh) return;

    const color = CLASS_COLORS[box.class_name] || '#ffffff';
    const x1 = (box.x - box.w / 2) * W;
    const y1 = (box.y - box.h / 2) * H;
    const bw  = box.w * W;
    const bh  = box.h * H;

    // Shadow glow
    ctx.save();
    ctx.shadowColor   = color;
    ctx.shadowBlur    = 12;

    // Box border — thick accent
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(x1, y1, bw, bh);

    ctx.restore();

    // Corner accents
    const cornerLen = Math.min(bw, bh) * 0.2;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 3;
    ctx.lineCap     = 'round';

    [[x1, y1, 1, 1], [x1 + bw, y1, -1, 1], [x1, y1 + bh, 1, -1], [x1 + bw, y1 + bh, -1, -1]]
      .forEach(([cx, cy, dx, dy]) => {
        ctx.beginPath();
        ctx.moveTo(cx + dx * cornerLen, cy);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx, cy + dy * cornerLen);
        ctx.stroke();
      });

    // Label background
    const label = `${CLASS_ICONS[box.class_name] || ''} ${box.class_name}  ${(box.score * 100).toFixed(0)}%`;
    ctx.font = 'bold 12px Inter, sans-serif';
    const tw = ctx.measureText(label).width;
    const th = 18;
    const lx = Math.max(0, x1);
    const ly = y1 - th - 6 < 0 ? y1 + 4 : y1 - th - 6;

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(lx, ly, tw + 12, th + 4, 4);
    ctx.fill();

    ctx.fillStyle = '#000000';
    ctx.fillText(label, lx + 6, ly + th - 2);
  });
}

// ─── Detect ──────────────────────────────────────────────────────────────────
detectBtn.addEventListener('click', runDetect);

async function runDetect() {
  if (!state.imageFile) return;

  detectBtn.disabled = true;
  btnLabel.textContent = 'Đang phân tích...';
  btnPulse.classList.add('active');
  detectOverlay.classList.add('visible');

  const formData = new FormData();
  formData.append('image', state.imageFile);
  formData.append('conf_thresh', confSlider.value);
  formData.append('iou_thresh',  iouSlider.value);

  try {
    const res = await fetch('/detect', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showToast(data.error || 'Lỗi server', 'error');
      return;
    }

    state.currentResult = data;
    renderResult(data);
    showToast(`Phát hiện ${data.total} phương tiện trong ${data.inference_ms}ms`, 'success');

  } catch (err) {
    showToast('Không thể kết nối server. Hãy chạy app.py.', 'error');
  } finally {
    detectBtn.disabled = false;
    btnLabel.textContent = 'Phát hiện phương tiện';
    btnPulse.classList.remove('active');
    detectOverlay.classList.remove('visible');
  }
}

// ─── Render Result ───────────────────────────────────────────────────────────
function renderResult(data) {
  // Draw boxes
  drawBoxesOnCanvas(data);

  // Stats panel
  statsSection.style.display = 'block';
  inferenceTime.textContent = data.inference_ms;

  // Animate total count
  animateCounter(totalCount, 0, data.total, 500);

  // Class stats rows
  const maxCount = Math.max(...Object.values(data.count_per_class), 1);
  classStats.innerHTML = '';

  const order = ['car', 'bus', 'truck', 'motorcycle', 'bicycle'];
  order.forEach(cls => {
    const count = data.count_per_class[cls] || 0;
    const color = CLASS_COLORS[cls];
    const row = document.createElement('div');
    row.className = 'stat-row' + (count > 0 ? ' active' : '');
    row.innerHTML = `
      <span class="stat-icon">${CLASS_ICONS[cls]}</span>
      <span class="stat-name">${cls}</span>
      <div class="stat-bar-wrap">
        <div class="stat-bar" id="bar-${cls}" style="background:${color}"></div>
      </div>
      <span class="stat-count" style="color:${color}">${count}</span>
    `;
    classStats.appendChild(row);

    // Animate bar width
    requestAnimationFrame(() => {
      setTimeout(() => {
        document.getElementById(`bar-${cls}`).style.width = (count / maxCount * 100) + '%';
      }, 50);
    });
  });
}

// ─── Re-draw when slider changes (client-side filter) ────────────────────────
confSlider.addEventListener('input', () => {
  updateSlider(confSlider, confFill, confValue);
  if (state.currentResult) drawBoxesOnCanvas(state.currentResult);
});

// ─── Toggle boxes ────────────────────────────────────────────────────────────
toggleBoxesBtn.addEventListener('click', () => {
  state.showBoxes = !state.showBoxes;
  toggleBoxesBtn.style.opacity = state.showBoxes ? '1' : '0.4';
  if (state.currentResult) drawBoxesOnCanvas(state.currentResult);
  else if (state.imageObj) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.imageObj, 0, 0, canvas.width, canvas.height);
  }
});

// ─── Download ────────────────────────────────────────────────────────────────
downloadBtn.addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = 'detection_result.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
  showToast('Đã tải ảnh kết quả!', 'success');
});

// ─── Animate Counter ─────────────────────────────────────────────────────────
function animateCounter(el, from, to, duration) {
  const start = performance.now();
  const update = (time) => {
    const t = Math.min((time - start) / duration, 1);
    el.textContent = Math.round(from + (to - from) * easeOut(t));
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

// ─── Toast ───────────────────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

// ─── Canvas roundRect polyfill ───────────────────────────────────────────────
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
    this.beginPath();
    this.moveTo(x + r, y);
    this.lineTo(x + w - r, y);
    this.quadraticCurveTo(x + w, y, x + w, y + r);
    this.lineTo(x + w, y + h - r);
    this.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    this.lineTo(x + r, y + h);
    this.quadraticCurveTo(x, y + h, x, y + h - r);
    this.lineTo(x, y + r);
    this.quadraticCurveTo(x, y, x + r, y);
    this.closePath();
  };
}

// ─── Init ────────────────────────────────────────────────────────────────────
fetchModelInfo();
