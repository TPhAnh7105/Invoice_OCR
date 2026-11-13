const $ = (id) => document.getElementById(id);
const LABELS = ['SELLER', 'ADDRESS', 'TIMESTAMP', 'TOTAL_COST'];
const NAMES = { SELLER: 'Tên cửa hàng', ADDRESS: 'Địa chỉ', TIMESTAMP: 'Ngày / giờ', TOTAL_COST: 'Tổng tiền', NONE: 'Chưa xác định' };
const COLORS = { SELLER: '#288465', ADDRESS: '#528ec4', TIMESTAMP: '#b38638', TOTAL_COST: '#9b68b3', NONE: '#84918b' };
let current = null, busy = false, activeField = null, requestVersion = 0, dirty = false;
let regions = [], objectUrl = null;

function error(message) { $('error').textContent = message; $('error').hidden = false; }
function setBusy(value) {
  busy = value;
  document.body.classList.toggle('busy', value);
  $('sample-button').disabled = value;
  $('change-file').disabled = value || !current;
  $('pick-file').disabled = value;
  $('file-input').disabled = value;
  $('export-button').disabled = value || current?.status !== 'done';
}
function showProgress(stage, percent, message) {
  $('progress-area').hidden = false;
  $('progress-message').textContent = message;
  $('progress-value').textContent = `${percent}%`;
  $('progress-bar').style.width = `${percent}%`;
  document.querySelectorAll('[data-stage]').forEach(el => el.classList.toggle('active', el.dataset.stage === stage));
}
async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Yêu cầu thất bại (${response.status}).`;
    try { const data = await response.json(); message = typeof data.detail === 'string' ? data.detail : message; } catch {}
    throw new Error(message);
  }
  return response;
}
function resetFields() {
  LABELS.forEach(key => {
    $(`field-${key}`).value = '';
    $(`field-${key}`).disabled = true;
    const badge = $(`conf-${key}`);
    if (badge) { badge.textContent = '0%'; badge.className = 'field-conf-badge'; }
  });
  const overallBadge = $('overall-confidence-badge');
  if (overallBadge) { overallBadge.hidden = true; overallBadge.textContent = 'ĐỘ TIN CẬY: 0%'; }
  $('regions-section').hidden = true;
  $('regions-body').replaceChildren();
  $('result-note').lastElementChild.textContent = 'Bạn có thể kiểm tra và sửa kết quả trước khi xuất.';
  $('result-badge').textContent = 'ĐANG ĐỌC ẢNH';
  $('result-badge').classList.add('active');
  $('region-count').textContent = '';
  $('overlay').replaceChildren();
  $('image-stage').classList.remove('expanded');
  activeField = null; regions = []; dirty = false;
  focusField(null);
}

async function upload(file) {
  if (busy || !file) return;
  $('error').hidden = true;
  if (file.size > 15 * 1024 * 1024) { error('Ảnh vượt quá 15 MB. Vui lòng chọn ảnh nhỏ hơn.'); return; }
  const version = ++requestVersion;
  current = null;
  resetFields();
  setBusy(true);
  showProgress('uploading', 0, 'Đang tải ảnh lên…');
  $('drop-zone').hidden = true;
  $('viewer').hidden = false;
  $('filename').textContent = file.name;
  $('image-meta').textContent = 'Ảnh đã chọn';
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  $('receipt-image').src = objectUrl;
  try {
    const body = new FormData(); body.append('file', file);
    current = await (await api('/api/jobs', { method: 'POST', body })).json();
    // The server normalizes EXIF and transparency. Always draw on this exact image.
    $('receipt-image').src = current.image_url;
    $('overlay').setAttribute('viewBox', `0 0 ${current.width} ${current.height}`);
    $('image-meta').textContent = `${current.width} × ${current.height} px`;
    await poll(version);
  } catch (err) {
    error(err.message || 'Không kết nối được máy chủ. Hãy kiểm tra demo đang chạy.');
    $('result-badge').textContent = 'CHƯA HOÀN TẤT';
    showProgress('error', 0, 'Chưa thể hoàn tất. Bạn có thể chọn lại ảnh.');
  } finally {
    if (version === requestVersion) {
      setBusy(false);
      $('change-file').disabled = false;
      $('file-input').value = '';
    }
  }
}

async function poll(version) {
  while (version === requestVersion) {
    if (current.status === 'error') throw new Error(current.message);
    showProgress(current.stage, current.progress, current.message);
    if (current.regions?.length) drawRegions(current.regions);
    if (current.status === 'done') {
      renderResult();
      loadHistory(current.id);
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 750));
    current = await (await api(`/api/jobs/${current.id}`)).json();
  }
}

function drawRegions(items) {
  regions = items;
  $('overlay').replaceChildren();
  const assignments = new Map();
  if (current?.result) {
    for (const key of LABELS) {
      const rawIds = current.result.fields[key].region_ids || [];
      for (const raw of rawIds) {
        const id = typeof raw === 'object' && raw !== null ? raw.id : raw;
        assignments.set(Number(id), key);
      }
    }
  }
  for (const region of items) {
    const selectedLabel = assignments.get(region.id);
    if (current?.status === 'done' && $('box-scope').value === 'fields' && !selectedLabel) continue;
    const displayLabel = selectedLabel || region.label;
    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    polygon.setAttribute('points', region.box.map(p => p.join(',')).join(' '));
    polygon.setAttribute('stroke', COLORS[displayLabel] || COLORS.NONE);
    polygon.setAttribute('fill', COLORS[displayLabel] || COLORS.NONE);
    polygon.classList.add('region-poly');
    polygon.dataset.label = displayLabel;
    polygon.dataset.regionId = region.id;
    polygon.setAttribute('tabindex', '0');
    polygon.setAttribute('role', 'button');
    polygon.setAttribute('aria-label', `${NAMES[displayLabel]}: ${region.text}`);
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = `${NAMES[displayLabel]} · ${region.text}`;
    polygon.append(title);
    polygon.addEventListener('click', () => focusField(displayLabel));
    polygon.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); focusField(displayLabel); } });
    $('overlay').append(polygon);
  }
  focusField(activeField);
}

function focusField(label) {
  activeField = label;
  const rawList = current?.result?.fields?.[label]?.region_ids || [];
  const related = rawList.map(x => typeof x === 'object' && x !== null ? Number(x.id) : Number(x));
  document.querySelectorAll('.field-card').forEach(el => el.classList.toggle('selected', el.dataset.label === label));
  document.querySelectorAll('.region-poly').forEach(el => {
    const match = current?.result ? related.includes(Number(el.dataset.regionId)) : el.dataset.label === label;
    el.classList.toggle('dimmed', Boolean(label) && !match);
    el.classList.toggle('focused', Boolean(label) && match);
  });
  document.querySelectorAll('#regions-body tr').forEach(el => el.classList.toggle('selected', Boolean(label) && el.dataset.label === label));
}

function renderResult() {
  const result = current.result;
  $('progress-area').hidden = true;
  $('result-badge').textContent = result.region_count ? 'ĐÃ NHẬN DẠNG' : 'CHƯA TÌM THẤY CHỮ';
  $('region-count').textContent = `${result.region_count} vùng chữ`;
  for (const key of LABELS) {
    const fieldData = result.fields[key];
    const input = $(`field-${key}`);
    input.value = fieldData.value;
    input.disabled = false;
    input.placeholder = 'Chưa nhận dạng được — bạn có thể bổ sung';
    const badge = $(`conf-${key}`);
    if (badge) {
      const conf = typeof fieldData.confidence === 'number' ? fieldData.confidence : (fieldData.value ? 100 : 0);
      badge.textContent = `${conf.toFixed(1)}%`;
      badge.className = 'field-conf-badge ' + (conf >= 80 ? 'high' : conf >= 50 ? 'medium' : 'low');
    }
  }
  const overallBadge = $('overall-confidence-badge');
  if (overallBadge) {
    const oConf = typeof result.overall_confidence === 'number' ? result.overall_confidence : 0;
    overallBadge.textContent = `ĐỘ TIN CẬY: ${oConf.toFixed(1)}%`;
    overallBadge.className = 'result-badge overall-conf-badge ' + (oConf >= 80 ? 'active' : oConf >= 50 ? 'medium' : 'warning');
    overallBadge.hidden = false;
  }
  $('result-note').lastElementChild.textContent = result.notice || `Hoàn tất trong ${result.elapsed_seconds.toLocaleString('vi-VN')} giây. Kiểm tra và sửa nội dung nếu cần.`;
  drawRegions(result.regions);
  $('regions-body').replaceChildren();
  for (const region of result.regions) {
    const row = document.createElement('tr');
    row.dataset.label = region.label;
    row.addEventListener('click', () => focusField(region.label));
    for (const value of [String(region.id + 1).padStart(2, '0'), region.text, NAMES[region.label], `${Math.round(region.ocr_confidence * 100)}%`]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    const badge = document.createElement('span'); badge.className = 'table-label';
    const dot = document.createElement('span'); dot.className = 'field-dot'; dot.style.background = COLORS[region.label];
    badge.append(dot, document.createTextNode(NAMES[region.label]));
    row.children[2].replaceChildren(badge);
    $('regions-body').append(row);
  }
  $('regions-section').hidden = !result.regions.length;
}

async function exportExcel() {
  if (current?.status !== 'done') return;
  $('export-button').disabled = true;
  $('error').hidden = true;
  try {
    if (dirty) {
      const fields = Object.fromEntries(LABELS.map(key => [key, $(`field-${key}`).value]));
      current = await (await api(`/api/jobs/${current.id}/fields`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ fields }) })).json();
      dirty = false;
      loadHistory(current.id);
    }
    const response = await api(`/api/jobs/${current.id}/excel`);
    const url = URL.createObjectURL(await response.blob());
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `hoa-don-${current.id.slice(0, 8)}.xlsx`; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    $('result-note').lastElementChild.textContent = 'Đã xuất Excel với các nội dung đang hiển thị.';
  } catch (err) { error(err.message); }
  finally { $('export-button').disabled = false; }
}

async function switchJob(jobId) {
  if (busy) return;
  if (current?.id === jobId) return;

  // Autosave any unsaved modifications on current invoice
  if (dirty && current?.id && current.status === 'done') {
    try {
      const fields = Object.fromEntries(LABELS.map(key => [key, $(`field-${key}`).value]));
      await api(`/api/jobs/${current.id}/fields`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fields })
      });
      dirty = false;
    } catch (e) {
      console.warn('Autosave skipped:', e);
    }
  }

  $('error').hidden = true;
  const version = ++requestVersion;
  setBusy(true);
  updateActiveHistory(jobId);

  try {
    const job = await (await api(`/api/jobs/${jobId}`)).json();
    current = job;
    dirty = false;

    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    $('drop-zone').hidden = true;
    $('viewer').hidden = false;
    $('receipt-image').src = current.image_url;
    $('overlay').setAttribute('viewBox', `0 0 ${current.width} ${current.height}`);
    $('image-meta').textContent = `${current.width} × ${current.height} px`;
    $('filename').textContent = current.filename;
    $('change-file').disabled = false;

    if (current.status === 'done' && current.result) {
      renderResult();
      $('export-button').disabled = false;
    } else if (current.status === 'processing' || current.status === 'queued') {
      showProgress(current.stage, current.progress, current.message);
      await poll(version);
    } else {
      resetFields();
      $('result-badge').textContent = 'CHƯA HOÀN TẤT';
      error(current.message || 'Hóa đơn này xử lý chưa hoàn tất.');
    }
  } catch (err) {
    error('Không thể mở hóa đơn: ' + (err.message || err));
  } finally {
    if (version === requestVersion) {
      setBusy(false);
      $('change-file').disabled = false;
    }
  }
}

async function loadHistory(selectedId = null) {
  try {
    const res = await api('/api/jobs');
    const items = await res.json();
    renderHistory(items, selectedId || current?.id);
  } catch (err) {
    console.warn('Could not load history:', err);
  }
}

function renderHistory(items, activeId = null) {
  const section = $('history-section');
  const list = $('history-list');
  const countEl = $('history-count');
  if (!section || !list) return;

  if (!items || !items.length) {
    section.hidden = true;
    return;
  }

  section.hidden = false;
  if (countEl) countEl.textContent = items.length;
  list.replaceChildren();

  for (const item of items) {
    const card = document.createElement('div');
    card.className = 'history-card' + (item.id === activeId ? ' active' : '');
    card.dataset.id = item.id;
    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');
    card.setAttribute('aria-label', `Hóa đơn ${item.seller || item.filename}`);

    const thumbWrap = document.createElement('div');
    thumbWrap.className = 'history-thumb-wrap';
    const thumb = document.createElement('img');
    thumb.className = 'history-thumb';
    thumb.src = item.image_url;
    thumb.alt = item.filename;
    thumb.loading = 'lazy';
    thumb.onerror = () => {
      thumb.remove();
      const ph = document.createElement('span');
      ph.className = 'history-thumb-placeholder';
      ph.textContent = 'Hóa đơn';
      thumbWrap.append(ph);
    };
    thumbWrap.append(thumb);

    const content = document.createElement('div');
    content.className = 'history-card-content';

    const top = document.createElement('div');
    top.className = 'history-card-top';
    const seller = document.createElement('span');
    seller.className = 'history-seller';
    seller.textContent = item.seller || 'Chưa nhận dạng tên';
    seller.title = item.seller || item.filename;

    const confVal = typeof item.overall_confidence === 'number' ? item.overall_confidence : 0;
    const confBadge = document.createElement('span');
    confBadge.className = 'history-conf field-conf-badge ' + (confVal >= 80 ? 'high' : confVal >= 50 ? 'medium' : 'low');
    confBadge.textContent = `${confVal.toFixed(0)}%`;
    top.append(seller, confBadge);

    const mid = document.createElement('div');
    mid.className = 'history-card-mid';
    const total = document.createElement('span');
    total.className = 'history-total';
    total.textContent = item.total_cost || '---';
    mid.append(total);

    const bottom = document.createElement('div');
    bottom.className = 'history-card-bottom';
    const filename = document.createElement('span');
    filename.className = 'history-filename';
    filename.textContent = item.filename;
    filename.title = item.filename;

    const time = document.createElement('span');
    time.className = 'history-time';
    if (item.created_at) {
      try {
        const d = new Date(item.created_at);
        time.textContent = `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')} ${d.getDate()}/${d.getMonth() + 1}`;
      } catch {
        time.textContent = '';
      }
    }
    bottom.append(filename, time);

    content.append(top, mid, bottom);
    card.append(thumbWrap, content);

    const onSelect = () => switchJob(item.id);
    card.addEventListener('click', onSelect);
    card.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onSelect();
      }
    });

    list.append(card);
  }
}

function updateActiveHistory(activeId) {
  document.querySelectorAll('.history-card').forEach(card => {
    card.classList.toggle('active', card.dataset.id === activeId);
  });
}

$('pick-file').addEventListener('click', event => { event.stopPropagation(); if (!busy) $('file-input').click(); });
$('drop-zone').addEventListener('click', () => { if (!busy) $('file-input').click(); });
$('drop-zone').addEventListener('keydown', event => { if (event.target === $('drop-zone') && !busy && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); $('file-input').click(); } });
$('change-file').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', event => upload(event.target.files[0]));
for (const name of ['dragenter', 'dragover']) $('drop-zone').addEventListener(name, event => { event.preventDefault(); if (!busy) $('drop-zone').classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) $('drop-zone').addEventListener(name, event => { event.preventDefault(); $('drop-zone').classList.remove('dragging'); });
$('drop-zone').addEventListener('drop', event => upload(event.dataTransfer.files[0]));
window.addEventListener('dragover', event => event.preventDefault());
window.addEventListener('drop', event => event.preventDefault());
$('sample-button').addEventListener('click', async () => {
  if (busy) return;
  $('sample-button').disabled = true;
  try { const response = await api('/api/sample'); await upload(new File([await response.blob()], 'hoa-don-mau.jpg', { type: 'image/jpeg' })); }
  catch (err) { error(err.message); }
  finally { $('sample-button').disabled = busy; }
});
$('show-boxes').addEventListener('change', event => { $('overlay').style.display = event.target.checked ? '' : 'none'; });
$('box-scope').addEventListener('change', () => drawRegions(regions));
$('expand').addEventListener('click', () => $('image-stage').classList.toggle('expanded'));
$('clear-focus').addEventListener('click', () => focusField(null));
document.querySelectorAll('[data-focus]').forEach(el => el.addEventListener('click', () => focusField(activeField === el.dataset.focus ? null : el.dataset.focus)));
LABELS.forEach(key => $(`field-${key}`).addEventListener('input', () => { dirty = true; $('result-note').lastElementChild.textContent = 'Nội dung đã chỉnh sửa sẽ được lưu khi bạn xuất Excel.'; }));
$('export-button').addEventListener('click', exportExcel);

// Load history on initial startup
loadHistory();
