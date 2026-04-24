// app.js — Terra frontend
// Vanilla JS, no build step. Talks to the FastAPI backend at BASE_URL.

// When the frontend is served by FastAPI at /ui/ (the recommended setup),
// same-origin relative URLs work and sidestep CORS entirely. Override via
// ?api=http://host:port for cross-origin development.
const BASE_URL = window.location.search.includes("api=")
  ? new URLSearchParams(window.location.search).get("api")
  : window.location.origin;

// ---- class palette, mirroring backend/inference.py PALETTE exactly ----
const CLASS_ORDER = [
  "no_data", "background", "building", "road",
  "water", "barren", "forest", "agriculture",
];
const CLASS_COLORS = {
  no_data:      "rgb(0,0,0)",
  background:   "rgb(210,210,210)",
  building:     "rgb(220,20,60)",
  road:         "rgb(70,70,70)",
  water:        "rgb(30,144,255)",
  barren:       "rgb(210,180,140)",
  forest:       "rgb(34,139,34)",
  agriculture:  "rgb(255,215,0)",
};

// ---- shared state ----
const state = {
  sourceDataUrl: null,     // what we show in the "Source" frame
  sourceBlob: null,        // what we POST to /classify
  classify: null,          // full ClassifyResponse once we have it
  // Monotonic token for classify calls. Each loadFile bumps it; only the
  // latest call's result is allowed to write to state.classify or paint
  // the UI. This kills the race that appeared when a user clicks demo
  // buttons faster than /classify can complete (~300ms each).
  gen: 0,
  busy: false,             // true while a classify or agent request is in flight
};

// ===================================================================
// boot: probe /health and paint the status pills
// ===================================================================
async function probeHealth() {
  const dots = {
    backend:  document.getElementById("dot-backend"),
    device:   document.getElementById("dot-device"),
    agent:    document.getElementById("dot-agent"),
  };
  const lbls = {
    backend:  document.getElementById("lbl-backend"),
    device:   document.getElementById("lbl-device"),
    agent:    document.getElementById("lbl-agent"),
  };
  try {
    const res = await fetch(`${BASE_URL}/health`, { cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status}`);
    const h = await res.json();
    dots.backend.classList.add("ok");
    lbls.backend.textContent = "Backend ok";

    if (h.device) {
      const d = String(h.device).toLowerCase();
      dots.device.classList.add(d.includes("cuda") ? "ok" : "warn");
      lbls.device.textContent = `Device ${h.device}`;
    } else {
      dots.device.classList.add("warn");
      lbls.device.textContent = "Device loading";
    }

    if (h.agent_available) {
      dots.agent.classList.add("ok");
      lbls.agent.textContent = "Agent ready";
    } else {
      dots.agent.classList.add("warn");
      lbls.agent.textContent = "Agent offline";
    }
  } catch (e) {
    dots.backend.classList.add("err");
    lbls.backend.textContent = "Backend unreachable";
    dots.device.classList.add("err");
    lbls.device.textContent = "—";
    dots.agent.classList.add("err");
    lbls.agent.textContent = "—";
  }
}

// ===================================================================
// file input
// ===================================================================
const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");

fileInput.addEventListener("change", e => {
  const f = e.target.files && e.target.files[0];
  if (f) loadFile(f);
});

["dragenter", "dragover"].forEach(ev => {
  drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("hot"); });
});
["dragleave", "drop"].forEach(ev => {
  drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("hot"); });
});
drop.addEventListener("drop", e => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) loadFile(f);
});

async function loadFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("File must be an image (PNG or JPG).");
    return;
  }
  // New load — invalidate any in-flight work from prior clicks.
  const gen = ++state.gen;
  state.sourceBlob = file;
  state.sourceDataUrl = await blobToDataUrl(file);
  // Guard: user may have clicked again while we were reading the blob.
  if (gen !== state.gen) return;
  renderSourcePreview(state.sourceDataUrl);
  resetReport();
  await runClassify(gen);
}

function blobToDataUrl(blob) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(blob);
  });
}

function renderSourcePreview(dataUrl) {
  const frame = document.getElementById("frame-source");
  frame.innerHTML = "";
  const img = new Image();
  img.src = dataUrl;
  img.alt = "Source tile";
  frame.appendChild(img);
}

// ===================================================================
// demo shortcuts
// ===================================================================
document.querySelectorAll(".demos .demo").forEach(btn => {
  btn.addEventListener("click", async () => {
    const id = btn.dataset.demo;
    // Assume demos are served from /frontend/demos/ alongside app.js.
    // README explains how to populate this directory.
    const url = `demos/${id}.png`;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${res.status}`);
      const blob = await res.blob();
      const file = new File([blob], `${id}.png`, { type: "image/png" });
      await loadFile(file);
    } catch (e) {
      showError(`Demo image not found at ${url}. See README.md "Demo images".`);
    }
  });
});

// ===================================================================
// /classify
// ===================================================================
async function runClassify(gen) {
  const frame = document.getElementById("frame-mask");
  frame.innerHTML = '<span class="empty">Classifying…</span>';
  document.getElementById("panel-composition").style.display = "none";

  const form = new FormData();
  form.append("file", state.sourceBlob);
  form.append("tta", "true");
  form.append("pixel_size_m", "0.3");

  let data;
  try {
    const res = await fetch(`${BASE_URL}/classify`, { method: "POST", body: form });
    // Bail silently if a newer load has started — don't paint stale results.
    if (gen !== state.gen) return;
    if (!res.ok) {
      const t = await res.text();
      throw new Error(`${res.status} ${t.slice(0, 200)}`);
    }
    data = await res.json();
    if (gen !== state.gen) return;  // re-check after the await
  } catch (e) {
    if (gen !== state.gen) return;
    frame.innerHTML = '<span class="empty">Error</span>';
    showError(`/classify failed: ${e.message}`);
    return;
  }

  state.classify = data;
  renderMask(data.mask_png_base64);
  renderComposition(data);
  document.getElementById("panel-composition").style.display = "block";
  document.getElementById("btn-analyze").disabled = false;
}

function renderMask(b64) {
  const frame = document.getElementById("frame-mask");
  frame.innerHTML = "";
  const img = new Image();
  img.src = `data:image/png;base64,${b64}`;
  img.alt = "Predicted class mask";
  frame.appendChild(img);
}

function renderComposition(data) {
  const pcts = data.percentages || {};
  const em = data.emissions || {};

  // composition bar
  const bar = document.getElementById("comp-bar");
  bar.innerHTML = "";
  // Render in fixed order so demo-to-demo transitions animate cleanly.
  for (const cls of CLASS_ORDER) {
    const p = pcts[cls] || 0;
    if (p <= 0) continue;
    const seg = document.createElement("div");
    seg.className = "seg";
    seg.style.flex = `${p.toFixed(4)} 0 0`;
    seg.style.background = CLASS_COLORS[cls];
    seg.title = `${cls}: ${p.toFixed(2)}%`;
    bar.appendChild(seg);
  }

  // legend
  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  const ranked = CLASS_ORDER
    .map(c => [c, pcts[c] || 0])
    .filter(([_, p]) => p > 0.05)
    .sort((a, b) => b[1] - a[1]);
  for (const [cls, p] of ranked) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `
      <span class="swatch" style="background:${CLASS_COLORS[cls]}"></span>
      <span class="name">${cls}</span>
      <span class="pct">${p.toFixed(2)}%</span>
    `;
    legend.appendChild(row);
  }

  // totals
  const fmtSigned = x => (x >= 0 ? "+" : "") + x.toFixed(2);
  const annualEl = document.getElementById("kv-annual");
  annualEl.innerHTML = `${fmtSigned(em.total_annual_tco2e_per_yr || 0)}<span class="unit">tCO₂e/yr</span>`;
  annualEl.classList.remove("pos", "neg");
  annualEl.classList.add((em.total_annual_tco2e_per_yr || 0) > 0 ? "pos" : "neg");

  document.getElementById("kv-area").innerHTML =
    `${(em.total_area_ha || 0).toFixed(2)}<span class="unit">ha</span>`;
  document.getElementById("kv-embodied").innerHTML =
    `${fmtSigned(em.total_embodied_tco2e || 0)}<span class="unit">tCO₂e</span>`;
  document.getElementById("kv-infer").innerHTML =
    `${data.inference_ms}<span class="unit">ms</span>`;

  // warnings
  const whost = document.getElementById("warnings-host");
  whost.innerHTML = "";
  if (data.warnings && data.warnings.length) {
    const div = document.createElement("div");
    div.className = "warnings";
    div.textContent = data.warnings.join(" · ");
    whost.appendChild(div);
  }
}

// ===================================================================
// /agent/report
// ===================================================================
document.getElementById("btn-analyze").addEventListener("click", runAgent);

const PROGRESS_STEPS = [
  "Composing query",
  "Surveying composition",
  "Estimating emissions",
  "Considering interventions",
  "Writing report",
];

async function runAgent() {
  if (!state.classify) return;
  const gen = state.gen;        // capture: if user swaps image, we abort
  const btn = document.getElementById("btn-analyze");
  btn.disabled = true;
  btn.textContent = "Analyzing…";

  document.getElementById("report-empty").style.display = "none";
  document.getElementById("report-content").style.display = "none";
  document.getElementById("report-meta").style.display = "none";

  const prog = document.getElementById("report-progress");
  prog.style.display = "flex";
  prog.innerHTML = "";
  const stepEls = PROGRESS_STEPS.map((label, i) => {
    const el = document.createElement("div");
    el.className = "step";
    el.style.animationDelay = `${i * 60}ms`;
    el.innerHTML = `
      <span class="num">${String(i + 1).padStart(2, "0")}</span>
      <span class="label">${label}</span>
      <span class="state"></span>
    `;
    prog.appendChild(el);
    return el;
  });

  // Rotate through "active" markers on a timer so the user has something
  // to watch during the ~10–25s agent wall-clock.
  let activeIdx = 0;
  const markActive = i => {
    stepEls.forEach((el, j) => {
      el.classList.remove("active");
      const s = el.querySelector(".state");
      if (j < i) { el.classList.add("done"); s.innerHTML = `<span class="check">✓</span>`; }
      else if (j === i) { el.classList.add("active"); s.innerHTML = `<span class="spinner"></span>`; }
      else { s.innerHTML = ""; }
    });
  };
  markActive(0);
  // Advance one step every ~3s; the last step ("Writing report") is where
  // we park until the response lands.
  const ticker = setInterval(() => {
    if (activeIdx < PROGRESS_STEPS.length - 1) {
      activeIdx += 1;
      markActive(activeIdx);
    }
  }, 3000);

  const body = {
    percentages: state.classify.percentages,
    emissions: state.classify.emissions,
    total_area_ha: state.classify.emissions.total_area_ha,
    image_label: (state.sourceBlob && state.sourceBlob.name) || "",
    query: DEFAULT_QUERY,
  };

  let report;
  const t0 = performance.now();
  try {
    const res = await fetch(`${BASE_URL}/agent/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    // If the user swapped images mid-report, pretend this never happened.
    if (gen !== state.gen) { clearInterval(ticker); return; }
    if (!res.ok) {
      const t = await res.text();
      throw new Error(`${res.status} ${t.slice(0, 300)}`);
    }
    report = await res.json();
    if (gen !== state.gen) { clearInterval(ticker); return; }
  } catch (e) {
    if (gen !== state.gen) { clearInterval(ticker); return; }
    clearInterval(ticker);
    prog.style.display = "none";
    document.getElementById("report-empty").style.display = "flex";
    showError(`/agent/report failed: ${e.message}`);
    btn.disabled = false;
    btn.textContent = "Generate Sustainability Report";
    return;
  }
  const wallMs = Math.round(performance.now() - t0);

  clearInterval(ticker);
  // Mark every step done; then hide progress and paint report.
  stepEls.forEach(el => {
    el.classList.remove("active");
    el.classList.add("done");
    el.querySelector(".state").innerHTML = `<span class="check">✓</span>`;
  });
  await sleep(250);
  prog.style.display = "none";

  renderReport(report, wallMs);
  btn.disabled = false;
  btn.textContent = "Regenerate";
}

const DEFAULT_QUERY = (
  "Produce a sustainability report for this parcel. Cover the land " +
  "composition, the current emissions footprint (annual flux + embodied " +
  "stock), one or two realistic interventions worth considering, and any " +
  "model-quality caveats the user should know about."
);

// ===================================================================
// report rendering
// ===================================================================
function renderReport(report, wallMs) {
  const content = document.getElementById("report-content");
  content.innerHTML = markdownToHtml(report.final_text || "(no text)");
  content.style.display = "block";

  // meta footer: tool calls, usage, sources
  const meta = document.getElementById("report-meta");
  meta.innerHTML = "";
  meta.style.display = "grid";

  // Group tool calls by turn so duplicate-same-turn calls read as a group.
  const byTurn = {};
  for (const tc of (report.tool_calls || [])) {
    (byTurn[tc.turn] = byTurn[tc.turn] || []).push(tc);
  }
  const trace = document.createElement("div");
  trace.className = "row";
  trace.innerHTML = `<span class="label">Tool trace</span>`;
  const traceBody = document.createElement("div");
  traceBody.style.flex = "1";
  Object.keys(byTurn).sort((a, b) => Number(a) - Number(b)).forEach((turn, i) => {
    for (const tc of byTurn[turn]) {
      const el = document.createElement("span");
      el.className = "tool" + (tc.error ? " err" : "");
      el.style.animationDelay = `${i * 120}ms`;
      el.textContent = `t${turn} · ${tc.name}${tc.error ? " ✗" : ""}`;
      el.title = tc.error ? tc.error : JSON.stringify(tc.input);
      traceBody.appendChild(el);
    }
  });
  trace.appendChild(traceBody);
  meta.appendChild(trace);

  // usage stats
  const u = report.usage || {};
  const stats = document.createElement("div");
  stats.className = "row";
  const cachedPct = u.input_tokens
    ? Math.round(100 * (u.cache_read_input_tokens || 0) / u.input_tokens)
    : 0;
  stats.innerHTML = `
    <span class="label">Turns</span>   <span class="val">${report.turns_used} / stopped: ${report.stop_reason}</span>
    <span class="label">Tokens</span>  <span class="val">${(u.input_tokens||0).toLocaleString()} in · ${(u.output_tokens||0).toLocaleString()} out${cachedPct ? ` · ${cachedPct}% cached` : ""}</span>
    <span class="label">Wall</span>    <span class="val">${(wallMs/1000).toFixed(1)}s</span>
  `;
  meta.appendChild(stats);

  // sources — derived from the emissions payload, not the report text
  const srcs = (state.classify && state.classify.emissions && state.classify.emissions.sources_cited) || {};
  if (Object.keys(srcs).length) {
    const srcBox = document.createElement("div");
    srcBox.className = "sources";
    Object.keys(srcs).sort().forEach(k => {
      const line = document.createElement("div");
      line.className = "src-line";
      line.innerHTML = `<span class="src-key">[${k}]</span><span>${escapeHtml(srcs[k])}</span>`;
      srcBox.appendChild(line);
    });
    meta.appendChild(srcBox);
  }
}

// ===================================================================
// markdown -> html
// Minimal renderer tuned to what Haiku actually emits in our prompts:
// ## h2, ### h3, **bold**, *em*, `code`, -/* lists, [SRC-N] tags.
// Does NOT claim to be a general markdown parser.
// ===================================================================
function markdownToHtml(md) {
  const esc = s => s
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const lines = md.split(/\r?\n/);
  const out = [];
  let inList = false;
  let listType = null;
  const closeList = () => {
    if (inList) {
      out.push(listType === "ol" ? "</ol>" : "</ul>");
      inList = false; listType = null;
    }
  };

  for (let raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { closeList(); continue; }

    let m;
    // Horizontal rule: --- or *** on its own line, or === as setext-style.
    // Haiku sometimes opens reports with a stray `---` separator; render
    // it as a subtle rule (or drop it if it's the first block).
    if (/^(-{3,}|\*{3,}|={3,})\s*$/.test(line)) {
      closeList();
      if (out.length > 0) out.push('<hr />');
      continue;
    }
    if ((m = line.match(/^###\s+(.*)$/))) { closeList(); out.push(`<h3>${inline(m[1])}</h3>`); continue; }
    if ((m = line.match(/^##\s+(.*)$/)))  { closeList(); out.push(`<h2>${inline(m[1])}</h2>`); continue; }
    if ((m = line.match(/^#\s+(.*)$/)))   { closeList(); out.push(`<h1>${inline(m[1])}</h1>`); continue; }

    if ((m = line.match(/^(\s*)[-*]\s+(.*)$/))) {
      if (!inList || listType !== "ul") { closeList(); out.push("<ul>"); inList = true; listType = "ul"; }
      out.push(`<li>${inline(m[2])}</li>`); continue;
    }
    if ((m = line.match(/^(\s*)\d+\.\s+(.*)$/))) {
      if (!inList || listType !== "ol") { closeList(); out.push("<ol>"); inList = true; listType = "ol"; }
      out.push(`<li>${inline(m[2])}</li>`); continue;
    }

    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return out.join("\n");

  function inline(s) {
    s = esc(s);
    // [SRC-1] or [SRC-1, SRC-4] -> styled chips (split on commas inside a single bracket)
    s = s.replace(/\[((?:SRC-\d+)(?:\s*,\s*SRC-\d+)*)\]/g, (_, grp) => {
      return grp.split(/\s*,\s*/)
        .map(k => `<span class="src">[${k}]</span>`)
        .join("");
    });
    // bold, italic, inline code
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)/g, "<em>$1</em>");
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    return s;
  }
}

// ===================================================================
// helpers
// ===================================================================
function resetReport() {
  document.getElementById("report-empty").style.display = "flex";
  document.getElementById("report-progress").style.display = "none";
  document.getElementById("report-content").style.display = "none";
  document.getElementById("report-meta").style.display = "none";
  document.getElementById("btn-analyze").textContent = "Generate Sustainability Report";
  document.getElementById("btn-analyze").disabled = true;
}

function showError(msg) {
  let host = document.getElementById("error-host");
  if (!host) {
    host = document.createElement("div");
    host.id = "error-host";
    document.querySelector("main").prepend(host);
  }
  host.innerHTML = `<div class="error-banner">${escapeHtml(msg)}</div>`;
  setTimeout(() => { host.innerHTML = ""; }, 8000);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ---- go ----
probeHealth();
// re-probe every 30s so a backend restart eventually shows as green again
setInterval(probeHealth, 30000);
