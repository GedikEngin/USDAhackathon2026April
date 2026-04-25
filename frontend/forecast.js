// forecast.js — v2 yield forecast view
// Loaded alongside app.js. Talks to the FastAPI backend at the same BASE_URL.
//
// Race safety: state.forecastGen is a monotonic token incremented at the
// start of every fetch flow. After every await we check `gen !== state.forecastGen`
// and bail out — same pattern v1 land-use uses for state.gen.

(function () {
  "use strict";

  // ---- shared state for the forecast view ----------------------------------
  const fState = {
    gen: 0,                    // monotonic token, bumped per loadForecast()
    states: null,              // /forecast/states response, cached for the session
    activeState: null,         // alpha code, e.g. "IA"
    activeYear: null,          // int
    activeDate: null,          // "08-01"|"09-01"|"10-01"|"EOS"|"all"
    lastResponse: null,        // last /forecast/{state} response, for narrate
  };

  // Same BASE_URL convention as v1 app.js. Read it from the global app.js
  // sets up via window.location, but to stay decoupled, we read it here too.
  const BASE_URL = window.location.search.includes("api=")
    ? new URLSearchParams(window.location.search).get("api")
    : window.location.origin;

  // ---- DOM refs (cached lazily so script order doesn't matter) -------------
  function $(id) { return document.getElementById(id); }

  // ---- helpers -------------------------------------------------------------
  function fmtDate(canonical, year) {
    // "08-01" + 2025 -> "Aug 1, 2025"; "EOS" + 2025 -> "End of season 2025"
    if (canonical === "EOS") return `End of season ${year}`;
    const [m, d] = canonical.split("-").map(Number);
    const mm = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m - 1];
    return `${mm} ${d}, ${year}`;
  }
  function fmtBu(v) {
    if (v == null || !isFinite(v)) return "—";
    return `${v.toFixed(1)} bu/ac`;
  }
  function fmtSigned(v) {
    if (v == null || !isFinite(v)) return "—";
    const sign = v >= 0 ? "+" : "";
    return `${sign}${v.toFixed(1)}`;
  }
  function pretty(feat) {
    // Friendly feature names. Keep raw column names recognizable but soften.
    const map = {
      gdd_cum_f50_c86:    "Growing degree days",
      edd_hours_gt86f:    "Heat stress (>86°F hrs)",
      edd_hours_gt90f:    "Extreme heat (>90°F hrs)",
      vpd_kpa_veg:        "VPD, vegetative",
      vpd_kpa_silk:       "VPD, silking",
      vpd_kpa_grain:      "VPD, grain fill",
      prcp_cum_mm:        "Cumulative precip",
      dry_spell_max_days: "Longest dry spell",
      srad_total_veg:     "Solar, vegetative",
      srad_total_silk:    "Solar, silking",
      srad_total_grain:   "Solar, grain fill",
      d0_pct: "Drought D0+", d1_pct: "Drought D1+",
      d2_pct: "Drought D2+", d3_pct: "Drought D3+", d4_pct: "Drought D4",
      d2plus: "Severe drought (D2+)",
      ndvi_peak: "NDVI peak", ndvi_gs_mean: "NDVI growing-season mean",
      ndvi_gs_integral: "NDVI integral", ndvi_silking_mean: "NDVI at silking",
      ndvi_veg_mean: "NDVI vegetative mean",
      irrigated_share: "Irrigated share", harvest_ratio: "Harvest ratio",
      acres_planted_all: "Planted acres",
      nccpi3corn: "Soil productivity (corn)",
      nccpi3all: "Soil productivity (general)",
      aws0_100: "Avail. water 0-100cm", aws0_150: "Avail. water 0-150cm",
      soc0_30: "Soil organic C, topsoil",
      soc0_100: "Soil organic C, 0-100cm",
      rootznemc: "Root-zone EC", rootznaws: "Root-zone water",
      droughty: "Droughty soil flag", pctearthmc: "Earthen %",
      pwsl1pomu: "Wetland prevalence",
      state: "State (regional bias)",
      year: "Year (long-run trend)",
    };
    return map[feat] || feat;
  }

  // Value formatter: "feature value" gets units where meaningful, and the
  // 'year' feature gets a more honest "long-run trend" label since the raw
  // year integer (e.g. 2025) is not what the user cares about.
  function prettyValue(feat, val) {
    if (val == null || !isFinite(val)) return "—";
    if (feat === "year") return "long-run trend";
    if (feat === "state") return "regional indicator";
    const units = {
      gdd_cum_f50_c86: "°F-days",
      edd_hours_gt86f: "°F-hrs", edd_hours_gt90f: "°F-hrs",
      vpd_kpa_veg: "kPa", vpd_kpa_silk: "kPa", vpd_kpa_grain: "kPa",
      prcp_cum_mm: "mm", dry_spell_max_days: "days",
      srad_total_veg: "MJ/m²", srad_total_silk: "MJ/m²", srad_total_grain: "MJ/m²",
      d0_pct: "%", d1_pct: "%", d2_pct: "%", d3_pct: "%", d4_pct: "%",
      d2plus: "%",
      ndvi_peak: "", ndvi_gs_mean: "", ndvi_gs_integral: "",
      ndvi_silking_mean: "", ndvi_veg_mean: "",
      irrigated_share: "", harvest_ratio: "",
      acres_planted_all: "ac",
      nccpi3corn: "", nccpi3all: "",
      aws0_100: "cm", aws0_150: "cm",
      soc0_30: "g/kg", soc0_100: "g/kg",
    };
    const u = units[feat];
    const numStr = Math.abs(val) >= 100 ? val.toFixed(0)
                 : Math.abs(val) >= 10  ? val.toFixed(1)
                                        : val.toFixed(2);
    return u ? `${numStr} ${u}` : numStr;
  }

  // ===================================================================
  // boot: fetch /forecast/states once, build the picker UI
  // ===================================================================
  async function bootForecast() {
    const root = $("view-forecast");
    if (!root) return;                  // forecast view not in DOM
    setForecastStatus("loading");

    try {
      const r = await fetch(`${BASE_URL}/forecast/states`, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      fState.states = await r.json();
    } catch (e) {
      setForecastStatus("err", String(e));
      $("fc-states").innerHTML =
        `<div class="fc-empty">Forecast endpoints unavailable. ${e}</div>`;
      return;
    }

    setForecastStatus("ok", `Model: ${fState.states.model_version}`);
    renderStatePicker();
    // Default selection: newest state in payload (preserve picker order CO/IA/MO/NE/WI),
    // and the default_year/default_date the backend recommends.
    fState.activeState = fState.states.states[0].alpha;
    fState.activeYear = fState.states.default_year;
    fState.activeDate = "all";          // "all" sentinel == omit ?date= param
    renderYearPicker();
    renderDatePicker();
    loadForecast();
  }

  function setForecastStatus(level, label) {
    const dot = $("dot-forecast");
    const lbl = $("lbl-forecast");
    if (!dot || !lbl) return;
    dot.classList.remove("ok", "warn", "err");
    if (level === "ok")      { dot.classList.add("ok");   lbl.textContent = label || "Forecast ready"; }
    else if (level === "warn"){ dot.classList.add("warn"); lbl.textContent = label || "Forecast partial"; }
    else if (level === "err") { dot.classList.add("err");  lbl.textContent = label || "Forecast offline"; }
    else                      { lbl.textContent = label || "Forecast loading"; }
  }

  // ===================================================================
  // pickers
  // ===================================================================
  function renderStatePicker() {
    const el = $("fc-states");
    el.innerHTML = "";
    fState.states.states.forEach(s => {
      const b = document.createElement("button");
      b.className = "fc-pick";
      b.dataset.alpha = s.alpha;
      b.innerHTML = `<span class="fc-pick-major">${s.alpha}</span>`
                  + `<span class="fc-pick-minor">${s.name}</span>`
                  + `<span class="fc-pick-note">${s.n_counties} counties</span>`;
      b.onclick = () => {
        if (fState.activeState === s.alpha) return;
        fState.activeState = s.alpha;
        // Clamp year if not in this state's list (edge case; rare in practice).
        if (!s.available_years.includes(fState.activeYear)) {
          fState.activeYear = s.available_years[s.available_years.length - 1];
          renderYearPicker();
        }
        markActive("fc-states", "alpha", s.alpha);
        loadForecast();
      };
      el.appendChild(b);
    });
    markActive("fc-states", "alpha", fState.activeState);
  }

  function renderYearPicker() {
    const el = $("fc-years");
    el.innerHTML = "";
    const stateInfo = fState.states.states.find(s => s.alpha === fState.activeState);
    const years = (stateInfo && stateInfo.available_years) || [];
    // Show last 6 years (most recent on the right) for compact UI.
    const visible = years.slice(-6);
    visible.forEach(y => {
      const b = document.createElement("button");
      b.className = "fc-pick fc-pick-year";
      b.dataset.year = String(y);
      b.textContent = String(y);
      b.onclick = () => {
        if (fState.activeYear === y) return;
        fState.activeYear = y;
        markActive("fc-years", "year", String(y));
        loadForecast();
      };
      el.appendChild(b);
    });
    markActive("fc-years", "year", String(fState.activeYear));
  }

  function renderDatePicker() {
    const el = $("fc-dates");
    el.innerHTML = "";
    const items = [
      { code: "all",   label: "All 4 dates" },
      { code: "08-01", label: "Aug 1" },
      { code: "09-01", label: "Sep 1" },
      { code: "10-01", label: "Oct 1" },
      { code: "EOS",   label: "End of season" },
    ];
    items.forEach(it => {
      const b = document.createElement("button");
      b.className = "fc-pick fc-pick-date";
      b.dataset.date = it.code;
      b.textContent = it.label;
      b.onclick = () => {
        if (fState.activeDate === it.code) return;
        fState.activeDate = it.code;
        markActive("fc-dates", "date", it.code);
        loadForecast();
      };
      el.appendChild(b);
    });
    markActive("fc-dates", "date", fState.activeDate);
  }

  function markActive(parentId, attr, value) {
    const parent = $(parentId);
    if (!parent) return;
    Array.from(parent.children).forEach(c => {
      if (c.dataset[attr] === value) c.classList.add("active");
      else c.classList.remove("active");
    });
  }

  // ===================================================================
  // load forecast (handles "all 4" vs single-date with one code path)
  // ===================================================================
  async function loadForecast() {
    if (!fState.activeState || !fState.activeYear) return;

    const gen = ++fState.gen;
    paintLoading();

    // Always fetch the all-4 form. If the user picked a single date we just
    // render one panel out of the by_date map. This keeps the API call cost
    // constant and avoids two code paths.
    const url = `${BASE_URL}/forecast/${fState.activeState}?year=${fState.activeYear}`;

    let resp;
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (gen !== fState.gen) return;          // user picked something else
      if (!r.ok) {
        const detail = await safeJson(r);
        throw new Error(detail.detail || `HTTP ${r.status}`);
      }
      resp = await r.json();
      if (gen !== fState.gen) return;
    } catch (e) {
      if (gen !== fState.gen) return;
      paintError(String(e));
      return;
    }

    fState.lastResponse = resp;
    paintForecast(resp);
  }

  async function safeJson(r) { try { return await r.json(); } catch { return {}; } }

  // ===================================================================
  // paint: loading / error / forecast
  // ===================================================================
  function paintLoading() {
    $("fc-headline").innerHTML =
      `<div class="fc-empty">Loading ${fState.activeState} ${fState.activeYear}…</div>`;
    $("fc-cone-svg").innerHTML = "";
    $("fc-kpis").innerHTML = "";
    $("fc-analogs").innerHTML = "";
    $("fc-drivers").innerHTML = "";
    $("fc-narrative").innerHTML = "";
  }
  function paintError(msg) {
    $("fc-headline").innerHTML = `<div class="fc-empty fc-err">${msg}</div>`;
  }

  function paintForecast(resp) {
    // Pick the dates to render based on activeDate.
    let dateMap;
    if (fState.activeDate === "all") {
      dateMap = resp.by_date;
    } else {
      // Single date: use forecast field if set, else fall back to by_date
      // (server may return either form; we accept both).
      const single = resp.forecast || (resp.by_date && resp.by_date[fState.activeDate]);
      dateMap = single ? { [fState.activeDate]: single } : {};
    }

    paintHeadline(resp);
    paintCone(resp, dateMap);
    paintKpis(resp, dateMap);

    // Analogs and drivers come from the LATEST date in dateMap (e.g. EOS when
    // showing all 4, or the picked date when showing 1). Same convention an
    // operator would use: the most-informed forecast.
    const order = ["08-01", "09-01", "10-01", "EOS"];
    let bestKey = null;
    for (const k of order) if (dateMap[k]) bestKey = k;
    paintAnalogs(bestKey ? dateMap[bestKey] : null);
    paintDrivers(bestKey ? dateMap[bestKey] : null);
    paintNarrativeButton(resp, bestKey);
  }

  function paintHeadline(resp) {
    const truth = resp.truth_state_yield_bu_acre;
    const truthLine = (truth != null)
      ? `<div class="fc-truth">NASS truth: <b>${truth.toFixed(1)} bu/ac</b></div>`
      : `<div class="fc-truth fc-truth-pending">NASS truth: not yet reported (forecast year)</div>`;

    const h5 = resp.history && resp.history.mean_5yr_bu_acre;
    const h10 = resp.history && resp.history.mean_10yr_bu_acre;
    const histLine =
      `<div class="fc-hist">5-yr mean: <b>${h5 != null ? h5.toFixed(1) : "—"}</b>` +
      ` &middot; 10-yr mean: <b>${h10 != null ? h10.toFixed(1) : "—"}</b></div>`;

    $("fc-headline").innerHTML =
      `<div class="fc-headline-state">${resp.state_name}, ${resp.year}</div>` +
      truthLine + histLine;
  }

  // -------------------------------------------------------------------
  // Cone chart, hand-drawn SVG. 4 x-positions (08-01, 09-01, 10-01, EOS).
  // Y-axis: bu/ac, auto-scaled. Plot:
  //   - Filled p10–p90 band (semi-transparent --accent)
  //   - p50 line (--accent)
  //   - Point estimate as orange dot
  //   - Truth (if any) as green dashed line spanning all dates
  //   - State 5-year mean as faint dashed line (reference)
  // For empty cones: draw the point estimate only, with a "cone pending" note.
  // -------------------------------------------------------------------
  function paintCone(resp, dateMap) {
    const svg = $("fc-cone-svg");
    svg.innerHTML = "";
    const order = ["08-01", "09-01", "10-01", "EOS"];
    const present = order.filter(d => dateMap[d]);
    if (present.length === 0) {
      svg.innerHTML = `<text x="50%" y="50%" fill="var(--ink-faint)" text-anchor="middle">No data</text>`;
      return;
    }

    const W = 720, H = 280, padL = 56, padR = 24, padT = 24, padB = 36;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const innerW = W - padL - padR;
    const innerH = H - padT - padB;

    // Gather y-range candidates: points, p10/p90, truth, 5-yr.
    const yvals = [];
    present.forEach(d => {
      const f = dateMap[d];
      if (f.point_estimate_bu_acre != null) yvals.push(f.point_estimate_bu_acre);
      if (f.cone) { yvals.push(f.cone.p10, f.cone.p90); }
    });
    if (resp.truth_state_yield_bu_acre != null) yvals.push(resp.truth_state_yield_bu_acre);
    if (resp.history && resp.history.mean_5yr_bu_acre != null) yvals.push(resp.history.mean_5yr_bu_acre);
    if (yvals.length === 0) {
      svg.innerHTML = `<text x="50%" y="50%" fill="var(--ink-faint)" text-anchor="middle">No values</text>`;
      return;
    }
    let yMin = Math.min(...yvals);
    let yMax = Math.max(...yvals);
    const pad = Math.max(8, (yMax - yMin) * 0.15);
    yMin -= pad; yMax += pad;

    const xPos = i => padL + (innerW * (i + 0.5)) / 4;        // 4 slots, centered
    const yPos = v => padT + innerH * (1 - (v - yMin) / (yMax - yMin));

    const ns = "http://www.w3.org/2000/svg";
    const mk = (tag, attrs, text) => {
      const e = document.createElementNS(ns, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      if (text != null) e.textContent = text;
      return e;
    };

    // y-axis ticks: 5 ticks
    const ticks = 5;
    for (let i = 0; i < ticks; i++) {
      const t = yMin + (yMax - yMin) * (i / (ticks - 1));
      const y = yPos(t);
      svg.appendChild(mk("line", {
        x1: padL, x2: W - padR, y1: y, y2: y,
        stroke: "var(--rule)", "stroke-width": 1,
      }));
      svg.appendChild(mk("text", {
        x: padL - 8, y: y + 4, fill: "var(--ink-faint)",
        "text-anchor": "end", "font-size": "10",
      }, t.toFixed(0)));
    }
    // y-axis label
    svg.appendChild(mk("text", {
      x: padL - 44, y: padT + innerH / 2,
      fill: "var(--ink-faint)", "font-size": "10",
      "text-anchor": "middle",
      transform: `rotate(-90 ${padL - 44} ${padT + innerH / 2})`,
    }, "bu/ac"));

    // 5-year reference line
    if (resp.history && resp.history.mean_5yr_bu_acre != null) {
      const y = yPos(resp.history.mean_5yr_bu_acre);
      svg.appendChild(mk("line", {
        x1: padL, x2: W - padR, y1: y, y2: y,
        stroke: "var(--ink-faint)", "stroke-width": 1, "stroke-dasharray": "2 4",
      }));
      svg.appendChild(mk("text", {
        x: W - padR - 6, y: y - 4, fill: "var(--ink-faint)",
        "text-anchor": "end", "font-size": "10",
      }, "5-yr mean"));
    }

    // Truth line (only when we have it — 2023/2024)
    if (resp.truth_state_yield_bu_acre != null) {
      const y = yPos(resp.truth_state_yield_bu_acre);
      svg.appendChild(mk("line", {
        x1: padL, x2: W - padR, y1: y, y2: y,
        stroke: "var(--ok)", "stroke-width": 1.5, "stroke-dasharray": "5 4",
      }));
      svg.appendChild(mk("text", {
        x: W - padR - 6, y: y - 4, fill: "var(--ok)",
        "text-anchor": "end", "font-size": "10",
      }, "truth"));
    }

    // Cone band: build a polygon p10 along bottom, p90 reversed on top.
    const topPts = [], botPts = [];
    order.forEach((d, i) => {
      const f = dateMap[d];
      if (!f || !f.cone) return;
      botPts.push(`${xPos(i)},${yPos(f.cone.p10)}`);
      topPts.push(`${xPos(i)},${yPos(f.cone.p90)}`);
    });
    if (topPts.length >= 2) {
      const poly = botPts.concat(topPts.reverse()).join(" ");
      svg.appendChild(mk("polygon", {
        points: poly,
        fill: "var(--accent)", "fill-opacity": "0.15",
        stroke: "var(--accent)", "stroke-opacity": "0.4", "stroke-width": 1,
      }));
    }

    // p50 line
    const medPts = [];
    order.forEach((d, i) => {
      const f = dateMap[d];
      if (f && f.cone) medPts.push([xPos(i), yPos(f.cone.p50)]);
    });
    if (medPts.length >= 2) {
      const path = medPts.map((p, i) => (i === 0 ? "M" : "L") + p[0] + "," + p[1]).join(" ");
      svg.appendChild(mk("path", {
        d: path, fill: "none", stroke: "var(--accent)",
        "stroke-width": 1.2, "stroke-opacity": "0.7",
      }));
    }

    // Point estimates
    order.forEach((d, i) => {
      const f = dateMap[d];
      if (!f || f.point_estimate_bu_acre == null) return;
      svg.appendChild(mk("circle", {
        cx: xPos(i), cy: yPos(f.point_estimate_bu_acre),
        r: 5, fill: "var(--accent)",
        stroke: "var(--bg)", "stroke-width": 1.5,
      }));
    });

    // x-axis labels
    order.forEach((d, i) => {
      const lbl = d === "EOS" ? "EOS" : d;
      svg.appendChild(mk("text", {
        x: xPos(i), y: H - 12,
        fill: "var(--ink-dim)", "text-anchor": "middle", "font-size": "10",
        "letter-spacing": "0.1em",
      }, lbl));
    });

    // "cone pending" tag if any cone is missing
    const anyMissing = present.some(d => !dateMap[d].cone);
    if (anyMissing) {
      svg.appendChild(mk("text", {
        x: padL, y: padT - 8, fill: "var(--warn)",
        "font-size": "10", "letter-spacing": "0.1em",
      }, "Cone pending: NDVI 2025 not yet pulled"));
    }
  }

  // -------------------------------------------------------------------
  // KPI tiles. Show whichever set the user has selected.
  // For "all": show the EOS tile (most-informed) plus a trajectory mini-table.
  // For single date: show that date's KPIs in detail.
  // -------------------------------------------------------------------
  function paintKpis(resp, dateMap) {
    const order = ["08-01", "09-01", "10-01", "EOS"];
    const present = order.filter(d => dateMap[d]);
    if (present.length === 0) { $("fc-kpis").innerHTML = ""; return; }

    let html = "";
    present.forEach(d => {
      const f = dateMap[d];
      const point = f.point_estimate_bu_acre;
      const cone = f.cone;
      const cw = cone ? cone.width_80.toFixed(1) : "—";
      const h5 = resp.history && resp.history.mean_5yr_bu_acre;
      const delta5 = (point != null && h5 != null) ? (point - h5) : null;
      html += `
        <div class="fc-kpi">
          <div class="fc-kpi-date">${fmtDate(d, resp.year)}</div>
          <div class="fc-kpi-point">${fmtBu(point)}</div>
          <div class="fc-kpi-row">
            <span class="fc-kpi-key">cone width</span>
            <span class="fc-kpi-val">${cw}</span>
          </div>
          <div class="fc-kpi-row">
            <span class="fc-kpi-key">vs 5-yr mean</span>
            <span class="fc-kpi-val ${delta5 != null && delta5 >= 0 ? 'pos' : 'neg'}">${fmtSigned(delta5)}</span>
          </div>
          <div class="fc-kpi-row fc-kpi-meta">
            <span>n_counties: regressor=${f.n_counties_regressor}, cone=${f.n_counties_cone}</span>
          </div>
          <div class="fc-kpi-row fc-kpi-meta">
            <span>cone status: ${f.cone_status}</span>
          </div>
        </div>`;
    });
    $("fc-kpis").innerHTML = html;
  }

  // -------------------------------------------------------------------
  // Analogs. Show the K analog (geoid, year) records from the anchor county.
  // -------------------------------------------------------------------
  function paintAnalogs(forecastForDate) {
    const el = $("fc-analogs");
    if (!forecastForDate || !forecastForDate.analog_years
        || forecastForDate.analog_years.length === 0) {
      const reason = forecastForDate
        ? `<span class="fc-empty fc-empty-soft">${forecastForDate.cone_status}</span>`
        : `<span class="fc-empty fc-empty-soft">no forecast for this date</span>`;
      el.innerHTML = `<div class="fc-analogs-empty">${reason}</div>`;
      return;
    }
    const anchor = forecastForDate.analog_anchor;
    const lines = forecastForDate.analog_years.map(a => `
      <li class="fc-analog">
        <div class="fc-analog-yr">${a.year}</div>
        <div class="fc-analog-place">${a.county_name}, ${a.state_alpha}</div>
        <div class="fc-analog-yield">obs <b>${a.observed_yield_bu_acre.toFixed(1)}</b> &middot;
                                     detrended <b>${a.detrended_yield_bu_acre.toFixed(1)}</b></div>
        <div class="fc-analog-dist">d=${a.distance.toFixed(2)}</div>
      </li>`).join("");
    el.innerHTML = `
      <div class="fc-analog-anchor">
        anchor: ${anchor ? anchor.county_name + ', ' + anchor.state_alpha : "?"}
        <span class="fc-empty-soft">(${anchor ? anchor.rationale : ""})</span>
      </div>
      <ul class="fc-analog-list">${lines}</ul>`;
  }

  function paintDrivers(forecastForDate) {
    const el = $("fc-drivers");
    if (!forecastForDate || !forecastForDate.top_drivers
        || forecastForDate.top_drivers.length === 0) {
      el.innerHTML = `<div class="fc-empty-soft">no drivers available</div>`;
      return;
    }
    const lines = forecastForDate.top_drivers.map(d => {
      const sign = d.shap_bu_acre >= 0 ? "pos" : "neg";
      return `
        <li class="fc-driver">
          <div class="fc-driver-name">${pretty(d.feature)}</div>
          <div class="fc-driver-shap ${sign}">${fmtSigned(d.shap_bu_acre)} bu/ac</div>
          <div class="fc-driver-val">value: <b>${prettyValue(d.feature, d.feature_value_state_mean)}</b></div>
        </li>`;
    }).join("");
    el.innerHTML = `<ul class="fc-driver-list">${lines}</ul>`;
  }

  function paintNarrativeButton(resp, bestKey) {
    const el = $("fc-narrative");
    if (!bestKey) { el.innerHTML = ""; return; }
    el.innerHTML = `
      <button id="fc-narrate-btn" class="demos-style fc-narrate-btn">
        Generate narrative
      </button>
      <div id="fc-narrate-out" class="fc-narrate-out"></div>`;
    $("fc-narrate-btn").onclick = () => runNarrate(resp, bestKey);
  }

  async function runNarrate(resp, bestKey) {
    const out = $("fc-narrate-out");
    out.innerHTML = `<div class="fc-empty-soft">Calling narrator…</div>`;
    const gen = fState.gen;             // capture; abort if user navigated
    try {
      const r = await fetch(`${BASE_URL}/forecast/narrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          state: resp.state, year: resp.year, forecast_date: bestKey,
        }),
      });
      if (gen !== fState.gen) return;
      if (!r.ok) {
        const detail = await safeJson(r);
        throw new Error(detail.detail || `HTTP ${r.status}`);
      }
      const body = await r.json();
      if (gen !== fState.gen) return;

      const md = body.narrative || "(empty)";
      // Reuse v1's markdownToHtml if it's loaded; fall back to <pre> escape.
      const html = (typeof window.markdownToHtml === "function")
        ? window.markdownToHtml(md)
        : `<pre>${md.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")}</pre>`;
      const stubBadge = body.stub
        ? `<div class="fc-stub-badge">stub — Phase F will replace this</div>`
        : "";
      out.innerHTML = stubBadge + `<div class="fc-narrate-md">${html}</div>`;
    } catch (e) {
      if (gen !== fState.gen) return;
      out.innerHTML = `<div class="fc-empty fc-err">narrate failed: ${e}</div>`;
    }
  }

  // ===================================================================
  // view router: hash-based, called from app.js or on its own
  // ===================================================================
  function applyHashRoute() {
    const hash = (window.location.hash || "").replace(/^#/, "").trim();
    const wantForecast = hash === "forecast";
    const lu = $("view-landuse");
    const fc = $("view-forecast");
    const tabLu = $("tab-landuse");
    const tabFc = $("tab-forecast");
    if (lu) lu.style.display = wantForecast ? "none" : "";
    if (fc) fc.style.display = wantForecast ? "" : "none";
    if (tabLu) tabLu.classList.toggle("active", !wantForecast);
    if (tabFc) tabFc.classList.toggle("active",  wantForecast);
    // Boot forecast lazily on first reveal so v1 isn't blocked by /forecast/states.
    if (wantForecast && fState.states === null) bootForecast();
  }

  function bindTabs() {
    const tabLu = $("tab-landuse");
    const tabFc = $("tab-forecast");
    if (tabLu) tabLu.onclick = () => { window.location.hash = "landuse"; applyHashRoute(); };
    if (tabFc) tabFc.onclick = () => { window.location.hash = "forecast"; applyHashRoute(); };
    window.addEventListener("hashchange", applyHashRoute);
  }

  // ===================================================================
  // public surface
  // ===================================================================
  window.Forecast = {
    boot: bootForecast,
    bindTabs,
    applyHashRoute,
    setStatus: setForecastStatus,
  };

  // Auto-bind tabs on DOMContentLoaded; defer boot until the user navigates.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => { bindTabs(); applyHashRoute(); });
  } else {
    bindTabs(); applyHashRoute();
  }
})();
