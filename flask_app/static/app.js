/* ── Config ──────────────────────────────────────────────────────────────── */
const API_BASE = '';   // same origin

/* ── DOM refs ────────────────────────────────────────────────────────────── */
const $symptomSelect     = $('#symptom-select');
const predictForm        = document.getElementById('predict-form');
const predictBtn         = document.getElementById('predict-btn');
const btnText            = document.getElementById('btn-text');
const btnSpinner         = document.getElementById('btn-spinner');
const symptomHint        = document.getElementById('symptom-hint');
const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsError       = document.getElementById('results-error');
const resultsContent     = document.getElementById('results-content');
const errorMsg           = document.getElementById('error-msg');

/* ── Load symptoms and initialise Select2 ───────────────────────────────── */
async function loadSymptoms() {
  try {
    const [catRes, symRes] = await Promise.all([
      fetch(`${API_BASE}/symptoms/categories`),
      fetch(`${API_BASE}/symptoms`),
    ]);
    if (!catRes.ok || !symRes.ok) throw new Error('API returned an error response.');

    const catData = await catRes.json();
    const symData = await symRes.json();

    const grouped = {};
    catData.categories.forEach(c => { grouped[c.id] = []; });
    symData.symptoms.forEach(s => {
      if (grouped[s.category]) grouped[s.category].push({ id: s.id, text: s.label });
    });

    const select2Data = catData.categories
      .filter(c => (grouped[c.id] || []).length > 0)
      .map(c => ({
        text: `${c.label}  (${grouped[c.id].length})`,
        children: grouped[c.id],
      }));

    $symptomSelect.select2({
      data: select2Data,
      theme: 'bootstrap-5',
      placeholder: 'Search and select symptoms…',
      allowClear: true,
      width: '100%',
      closeOnSelect: false,
      dropdownParent: $symptomSelect.closest('.card'),
    });

    $symptomSelect.on('change', () => {
      predictBtn.disabled = ($symptomSelect.val() || []).length === 0;
    });

    symptomHint.textContent =
      `${symData.total} symptoms across ${catData.total} body-system categories`;

  } catch (err) {
    symptomHint.textContent = '⚠ Failed to load symptoms — is the API server running?';
    symptomHint.classList.add('text-danger');
    console.error(err);
  }
}

/* ── Utility helpers ────────────────────────────────────────────────────── */
function confColor(conf) {
  if (conf >= 0.65) return '#16a34a';
  if (conf >= 0.35) return '#ea580c';
  return '#2563eb';
}

function rankClass(n) {
  return ['rank-1', 'rank-2', 'rank-3'][n - 1] || 'rank-n';
}

function humanise(id) {
  return id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/* ── Panel visibility ───────────────────────────────────────────────────── */
function showPanel(panelEl) {
  [resultsPlaceholder, resultsError, resultsContent].forEach(el => el.classList.add('d-none'));
  panelEl.classList.remove('d-none');
}

function showError(msg) {
  errorMsg.innerHTML = `<strong>Error:</strong> ${msg}`;
  showPanel(resultsError);
}

/* ── Render SHAP explanation ────────────────────────────────────────────── */
function renderShap(shap) {
  if (!shap) return '';

  const maxAbs = Math.max(
    ...shap.top_supporting.map(f => Math.abs(f.shap_value)),
    ...shap.top_contradicting.map(f => Math.abs(f.shap_value)),
    0.001,
  );

  function shapRow(f, direction) {
    const pct      = (Math.abs(f.shap_value) / maxAbs * 100).toFixed(1);
    const reported = f.feature_value === 1.0;
    const color    = direction === 'pos' ? '#16a34a' : '#dc2626';
    const sign     = direction === 'pos' ? '+' : '−';
    const badge    = reported
      ? '<span class="shap-badge reported" title="You reported this symptom">reported</span>'
      : '<span class="shap-badge absent"   title="This symptom was NOT reported">absent</span>';

    return `
      <div class="shap-row">
        <div class="shap-label">
          <span class="shap-symptom-name">${f.symptom_label}</span>
          ${badge}
        </div>
        <div class="shap-bar-area">
          <div class="shap-bar-wrap">
            <div class="shap-bar"
                 style="width:0%;background:${color}"
                 data-target="${pct}"></div>
          </div>
          <span class="shap-val" style="color:${color}">${sign}${Math.abs(f.shap_value).toFixed(4)}</span>
        </div>
      </div>`;
  }

  const supportRows = shap.top_supporting.length
    ? shap.top_supporting.map(f => shapRow(f, 'pos')).join('')
    : '<p class="text-muted small mb-0">No positive factors found.</p>';

  const contradictRows = shap.top_contradicting.length
    ? shap.top_contradicting.map(f => shapRow(f, 'neg')).join('')
    : '<p class="text-muted small mb-0">No contradicting factors found.</p>';

  return `
    <hr class="text-muted opacity-25 my-3" />

    <div class="shap-section">
      <div class="shap-header">
        <span class="shap-title">Why this diagnosis?</span>
        <span class="shap-subtitle">SHAP — SHapley Additive exPlanations</span>
      </div>

      <div class="shap-explainer-note">
        Each bar shows how much a symptom <em>shifted</em> the model's prediction
        away from its baseline (average across all training cases). Longer bar = stronger
        influence. Green = pushed the model <strong>toward</strong> this disease;
        red = pushed <strong>away</strong>.
      </div>

      <div class="shap-block">
        <div class="shap-block-label supporting">
          <span class="shap-dot" style="background:#16a34a"></span>
          Supporting factors &mdash; evidence FOR this disease
        </div>
        ${supportRows}
      </div>

      <div class="shap-block mt-3">
        <div class="shap-block-label contradicting">
          <span class="shap-dot" style="background:#dc2626"></span>
          Contradicting factors &mdash; evidence AGAINST this disease
        </div>
        ${contradictRows}
      </div>

      <div class="shap-footnote">
        Baseline (model average): <strong>${shap.base_value.toFixed(4)}</strong> &nbsp;·&nbsp;
        Sum of all SHAP values + baseline = model's raw output for this prediction.
      </div>
    </div>`;
}

/* ── Render full results ────────────────────────────────────────────────── */
function renderResults(data) {
  const topConf = data.predictions.length ? data.predictions[0].confidence : 1;

  const banner = `
    <div class="top-banner">
      <div class="model-badge">${data.model_used.replace(/_/g, ' ')}</div>
      <div class="disease-name">${data.top_disease}</div>
      <div class="confidence-text">Confidence &nbsp;·&nbsp; <strong>${data.top_confidence_pct}</strong></div>
    </div>`;

  const rows = data.predictions.map((p, i) => {
    const rank = i + 1;
    const barW = topConf > 0 ? ((p.confidence / topConf) * 100).toFixed(1) : 0;
    return `
      <div class="prediction-row">
        <div class="rank-badge ${rankClass(rank)}">${rank}</div>
        <div class="prediction-info">
          <div class="disease-label" title="${p.disease}">${p.disease}</div>
          <div class="confidence-bar-wrap">
            <div class="confidence-bar"
                 style="width:0%;background:${confColor(p.confidence)}"
                 data-target="${barW}"></div>
          </div>
        </div>
        <div class="conf-label">${p.confidence_pct}</div>
      </div>`;
  }).join('');

  const matched = data.symptoms_matched
    .map(s => `<span class="symptom-chip">${humanise(s)}</span>`).join('');

  const unrecSection = data.symptoms_unrecognised.length ? `
    <div class="mt-2">
      <div class="section-label">Unrecognised</div>
      ${data.symptoms_unrecognised.map(s => `<span class="symptom-chip warn">${s}</span>`).join('')}
    </div>` : '';

  resultsContent.innerHTML = `
    ${banner}
    <div class="section-label">All Predictions</div>
    <div class="mb-3">${rows}</div>
    <hr class="text-muted opacity-25 my-3" />
    <div class="section-label">Matched Symptoms (${data.symptom_count_matched})</div>
    <div>${matched}</div>
    ${unrecSection}
    ${renderShap(data.shap_explanation)}`;

  showPanel(resultsContent);

  // Animate all bars after paint
  requestAnimationFrame(() => {
    resultsContent.querySelectorAll('[data-target]').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  });
}

/* ── Form submit ────────────────────────────────────────────────────────── */
predictForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const selected = $symptomSelect.val() || [];
  if (!selected.length) { showError('Please select at least one symptom.'); return; }

  predictBtn.disabled = true;
  btnText.textContent  = 'Predicting…';
  btnSpinner.classList.remove('d-none');

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symptoms: selected,
        model:    document.getElementById('model-select').value,
        top_n:    parseInt(document.getElementById('top-n').value, 10) || 5,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      const detail = data.detail;
      const msg = typeof detail === 'object'
        ? (detail.message || JSON.stringify(detail))
        : (detail || `HTTP ${res.status}`);
      showError(msg);
      return;
    }

    renderResults(data);

  } catch (err) {
    showError('Could not reach the API. Make sure <code>uvicorn api.main:app</code> is running.');
    console.error(err);
  } finally {
    predictBtn.disabled = ($symptomSelect.val() || []).length === 0;
    btnText.textContent  = 'Predict Disease';
    btnSpinner.classList.add('d-none');
  }
});

/* ── Boot ───────────────────────────────────────────────────────────────── */
loadSymptoms();
