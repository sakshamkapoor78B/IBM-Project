
const API_BASE = (window.location.protocol === "file:")
  ? "http://localhost:5000/api"
  : "/api";

const els = {
  statusDot: document.getElementById("statusDot"),
  statusLabel: document.getElementById("statusLabel"),
  form: document.getElementById("houseForm"),
  submitBtn: document.getElementById("submitBtn"),
  formError: document.getElementById("formError"),
  results: document.getElementById("results"),
  verdictFigure: document.getElementById("verdictFigure"),
  verdictModel: document.getElementById("verdictModel"),
  verdictSpread: document.getElementById("verdictSpread"),
  appraiserPanel: document.getElementById("appraiserPanel"),
  importanceChart: document.getElementById("importanceChart"),
  metricsTableBody: document.querySelector("#metricsTable tbody"),
  locationRange: document.getElementById("locationRange"),
  locationOutput: document.getElementById("locationOutput"),
};

const currency = (value) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);

// ---------------------------------------------------------------------
// Live range output
// ---------------------------------------------------------------------
els.locationRange.addEventListener("input", () => {
  els.locationOutput.textContent = els.locationRange.value;
});

// ---------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.model_loaded) {
      els.statusDot.classList.add("online");
      els.statusLabel.textContent = "Bureau online — model loaded";
    } else {
      els.statusDot.classList.add("offline");
      els.statusLabel.textContent = "Model not trained yet — run train_model.py";
    }
  } catch (err) {
    els.statusDot.classList.add("offline");
    els.statusLabel.textContent = "Bureau offline — start the backend (python app.py)";
  }
}

// ---------------------------------------------------------------------
// Model info (feature importances + metrics table), loaded once
// ---------------------------------------------------------------------
async function loadModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/model-info`);
    if (!res.ok) return;
    const data = await res.json();
    renderImportance(data.feature_importances);
    renderMetricsTable(data.metrics, data.best_model);
  } catch (err) {
    // Silently skip; the performance section just stays empty.
  }
}

function renderImportance(importances) {
  const entries = Object.entries(importances).slice(0, 8); // top 8 features
  const max = Math.max(...entries.map(([, v]) => v));
  els.importanceChart.innerHTML = entries.map(([name, value]) => {
    const pct = (value / max) * 100;
    const label = name.replace(/_/g, " ");
    return `
      <div class="importance-row">
        <span class="importance-label">${label}</span>
        <div class="importance-track"><div class="importance-fill" style="width:${pct}%"></div></div>
        <span class="importance-pct">${(value * 100).toFixed(1)}%</span>
      </div>`;
  }).join("");
}

function renderMetricsTable(metrics, bestModel) {
  const rows = Object.entries(metrics)
    .sort((a, b) => b[1].r2 - a[1].r2)
    .map(([name, m]) => {
      const isBest = name === bestModel;
      return `
        <tr class="${isBest ? "best-row" : ""}">
          <td>${name}${isBest ? " &#9733;" : ""}</td>
          <td>${m.r2.toFixed(4)}</td>
          <td>${currency(m.mae)}</td>
          <td>${currency(m.rmse)}</td>
        </tr>`;
    }).join("");
  els.metricsTableBody.innerHTML = rows;
}

// ---------------------------------------------------------------------
// Form submission -> prediction
// ---------------------------------------------------------------------
els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  els.formError.textContent = "";
  els.submitBtn.disabled = true;
  els.submitBtn.querySelector("span").textContent = "Filing case…";

  const formData = new FormData(els.form);
  const payload = {
    area_sqft: Number(formData.get("area_sqft")),
    bedrooms: Number(formData.get("bedrooms")),
    bathrooms: Number(formData.get("bathrooms")),
    stories: Number(formData.get("stories")),
    garage_spaces: Number(formData.get("garage_spaces")),
    age_years: Number(formData.get("age_years")),
    location_score: Number(formData.get("location_score")),
    has_pool: formData.get("has_pool") ? 1 : 0,
    has_basement: formData.get("has_basement") ? 1 : 0,
    near_main_road: formData.get("near_main_road") ? 1 : 0,
    furnishing: formData.get("furnishing"),
    condition: formData.get("condition"),
  };

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      els.formError.textContent = data.error || "Something went wrong. Please check the fields and try again.";
      return;
    }

    renderVerdict(data);
    els.results.hidden = false;
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    els.formError.textContent = "Could not reach the appraisal bureau. Is the backend running on port 5000?";
  } finally {
    els.submitBtn.disabled = false;
    els.submitBtn.querySelector("span").textContent = "File for Appraisal";
  }
});

function renderVerdict(data) {
  els.verdictFigure.textContent = currency(data.predicted_price);
  els.verdictModel.textContent = data.best_model_used;
  els.verdictSpread.textContent = currency(data.spread);

  const entries = Object.entries(data.appraiser_estimates);
  const max = Math.max(...entries.map(([, v]) => v), data.predicted_price);
  const bestModelNormalized = data.best_model_used.toLowerCase().replace(/[^a-z]/g, "");

  els.appraiserPanel.innerHTML = entries.map(([name, value]) => {
    const nameNormalized = name.toLowerCase().replace(/[^a-z]/g, "");
    const isFinal = nameNormalized === bestModelNormalized;
    const pct = (value / max) * 100;
    return `
      <div class="appraiser-row ${isFinal ? "is-final" : ""}">
        <div class="appraiser-row-top">
          <span class="appraiser-name">${name}</span>
          <span class="appraiser-value">${currency(value)}</span>
        </div>
        <div class="appraiser-bar-track">
          <div class="appraiser-bar-fill" style="width:${pct}%"></div>
        </div>
      </div>`;
  }).join("");
}

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
checkHealth();
loadModelInfo();
