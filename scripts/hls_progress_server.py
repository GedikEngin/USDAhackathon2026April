#!/usr/bin/env python
"""
scripts/hls_progress_server.py
Crude live progress page for the 4-shard download_hls.py background run.

Reads:
  - runs/download_hls_*of4_*.log   (latest per shard)
  - data/v2/hls/chip_index_shards/*.parquet
  - data/v2/hls/raw/  (transient cache size)
  - data/v2/hls/chips/  (persisted chip volume)

Serves a single auto-refreshing HTML page on http://localhost:8765/.
Stdlib only.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
SHARD_DIR = REPO_ROOT / "data" / "v2" / "hls" / "chip_index_shards"
RAW_DIR = REPO_ROOT / "data" / "v2" / "hls" / "raw"
CHIP_DIR = REPO_ROOT / "data" / "v2" / "hls" / "chips"

PORT = 8765


def latest_log_per_shard() -> dict[str, Path | None]:
    out: dict[str, Path | None] = {f"{k}of4": None for k in (1, 2, 3, 4)}
    for tag in out.keys():
        files = sorted(RUNS_DIR.glob(f"download_hls_{tag}_*.log"))
        if files:
            out[tag] = files[-1]
    return out


CMR_RE = re.compile(r"CMR returned (\d+) granules for ([A-Z]{2})-(\d{4})")
CELL_RE = re.compile(r"^={3,}$")
CELL_HEADER_RE = re.compile(r"CELL ([A-Z]{2})-(\d{4})")
DOWNLOAD_RE = re.compile(r"\[(\d+)/(\d+)\] downloaded ([\w\.]+) \((\d+) files, ([\d\.]+)s\)")
EXTRACT_RE = re.compile(r"\[(\d+)/(\d+)\] ([\w\.]+) -> (\d+) rows \((\d+) chips written\)")
TODO_RE = re.compile(r"after resume-skip: (\d+) granules to download \((\d+) already done\)")
CELL_DONE_RE = re.compile(r"CELL ([A-Z]{2})-(\d{4}) done in ([\d\.]+)s \((\d+) granules processed\)")
ALL_DONE_RE = re.compile(r"ALL CELLS DONE in ([\d\.]+) minutes")
APPROX_SIZE_RE = re.compile(r"approx download size: ([\d\.]+) GB")

# Total cells in the --all grid: 5 states x 12 years = 60. Used for total-GB estimation.
TOTAL_CELLS_IN_GRID = 60


def parse_log(path: Path | None) -> dict:
    if not path or not path.exists():
        return {"status": "no log", "lines": 0}
    state = {
        "log": str(path.name),
        "current_cell": None,
        "todo": None,
        "skipped": None,
        "downloaded": 0,
        "chips_written": 0,
        "granules_processed": 0,
        "cells_done": [],
        "todos_seen": [],
        "downloaded_gb": 0.0,
        "approx_count": 0,
        "all_done": False,
        "last_line": "",
        "last_download_seconds": None,
        "last_extract_chips": None,
    }
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if not line:
                    continue
                state["last_line"] = line[-300:]
                m = CELL_HEADER_RE.search(line)
                if m and "CELL " in line and "done" not in line:
                    state["current_cell"] = f"{m.group(1)}-{m.group(2)}"
                    state["todo"] = None
                    state["skipped"] = None
                    state["downloaded"] = 0
                    state["chips_written"] = 0
                    continue
                m = TODO_RE.search(line)
                if m:
                    todo_n = int(m.group(1))
                    state["todo"] = todo_n
                    state["skipped"] = int(m.group(2))
                    state["todos_seen"].append(todo_n)
                    continue
                m = APPROX_SIZE_RE.search(line)
                if m:
                    state["downloaded_gb"] += float(m.group(1))
                    state["approx_count"] += 1
                    continue
                m = DOWNLOAD_RE.search(line)
                if m:
                    state["downloaded"] = int(m.group(1))
                    state["last_download_seconds"] = float(m.group(5))
                    continue
                m = EXTRACT_RE.search(line)
                if m:
                    state["chips_written"] += int(m.group(5))
                    state["last_extract_chips"] = int(m.group(5))
                    continue
                m = CELL_DONE_RE.search(line)
                if m:
                    cell = f"{m.group(1)}-{m.group(2)}"
                    state["cells_done"].append({
                        "cell": cell,
                        "elapsed_s": float(m.group(3)),
                        "granules": int(m.group(4)),
                    })
                    state["granules_processed"] += int(m.group(4))
                    continue
                if ALL_DONE_RE.search(line):
                    state["all_done"] = True
    except Exception as e:
        state["error"] = str(e)
    return state


def shard_parquet_summary() -> dict:
    """Sum row counts across all shard parquets — the authoritative chip-row count."""
    total_rows = 0
    total_pos = 0
    cells: list[dict] = []
    if not SHARD_DIR.exists():
        return {"total_rows": 0, "total_pos": 0, "cells": []}
    try:
        import pandas as pd  # local import to allow server to start even without pd
        for p in sorted(SHARD_DIR.glob("chip_index_*.parquet")):
            try:
                df = pd.read_parquet(p)
                rows = len(df)
                pos = int(df["chip_path"].notna().sum()) if "chip_path" in df.columns else 0
                total_rows += rows
                total_pos += pos
                cells.append({
                    "cell": p.stem.replace("chip_index_", ""),
                    "rows": rows,
                    "positive": pos,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "mtime": int(p.stat().st_mtime),
                })
            except Exception as e:
                cells.append({"cell": p.stem, "error": str(e)})
    except ImportError:
        pass
    return {"total_rows": total_rows, "total_pos": total_pos, "cells": cells}


_NET_LOCK = threading.Lock()
_NET_PREV: dict = {"t": None, "rx": 0, "tx": 0, "per_iface": {}}


def _read_proc_net_dev() -> dict[str, tuple[int, int]]:
    """Returns {iface: (rx_bytes, tx_bytes)} for all non-loopback interfaces."""
    out: dict[str, tuple[int, int]] = {}
    try:
        with open("/proc/net/dev", "r") as fh:
            for line in fh.readlines()[2:]:
                if ":" not in line:
                    continue
                name, rest = line.split(":", 1)
                name = name.strip()
                if name == "lo":
                    continue
                parts = rest.split()
                if len(parts) < 9:
                    continue
                rx = int(parts[0])
                tx = int(parts[8])
                out[name] = (rx, tx)
    except OSError:
        pass
    return out


def network_summary() -> dict:
    """Current download/upload rate in MB/s, computed as delta since last call.
    First call returns zeros (no baseline yet)."""
    now = time.time()
    cur = _read_proc_net_dev()
    cur_rx = sum(rx for rx, _tx in cur.values())
    cur_tx = sum(tx for _rx, tx in cur.values())
    rate_rx_mbps = 0.0
    rate_tx_mbps = 0.0
    per_iface_rates: dict[str, dict[str, float]] = {}
    with _NET_LOCK:
        prev_t = _NET_PREV["t"]
        if prev_t is not None and now > prev_t:
            dt = now - prev_t
            d_rx = max(0, cur_rx - _NET_PREV["rx"])
            d_tx = max(0, cur_tx - _NET_PREV["tx"])
            rate_rx_mbps = (d_rx / dt) / 1_000_000
            rate_tx_mbps = (d_tx / dt) / 1_000_000
            for name, (rx, tx) in cur.items():
                p = _NET_PREV["per_iface"].get(name)
                if p is None:
                    continue
                d_rx_i = max(0, rx - p[0])
                d_tx_i = max(0, tx - p[1])
                per_iface_rates[name] = {
                    "rx_MBps": (d_rx_i / dt) / 1_000_000,
                    "tx_MBps": (d_tx_i / dt) / 1_000_000,
                }
        _NET_PREV["t"] = now
        _NET_PREV["rx"] = cur_rx
        _NET_PREV["tx"] = cur_tx
        _NET_PREV["per_iface"] = cur
    return {
        "rx_MBps": rate_rx_mbps,
        "tx_MBps": rate_tx_mbps,
        "rx_total_bytes": cur_rx,
        "tx_total_bytes": cur_tx,
        "per_iface": per_iface_rates,
    }


def disk_summary() -> dict:
    def du(path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total
    free = shutil.disk_usage(str(REPO_ROOT)).free
    return {
        "raw_bytes": du(RAW_DIR),
        "chips_bytes": du(CHIP_DIR),
        "free_bytes": free,
    }


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def collect() -> dict:
    logs = latest_log_per_shard()
    shards = {tag: parse_log(p) for tag, p in logs.items()}
    parquet = shard_parquet_summary()
    disk = disk_summary()
    net = network_summary()
    granules_done_total = sum(s.get("granules_processed", 0) for s in shards.values())
    chips_total_in_logs = sum(s.get("chips_written", 0) for s in shards.values())

    # GB-progress estimate: numerator = sum of parsed "approx download size: X GB"
    # across all shards (one per granule download attempt); denominator estimated
    # by extrapolating the average per-seen-cell todo across the full 60-cell grid.
    downloaded_gb = sum(s.get("downloaded_gb", 0.0) for s in shards.values())
    approx_attempts = sum(s.get("approx_count", 0) for s in shards.values())
    cells_seen = sum(len(s.get("todos_seen", [])) for s in shards.values())
    granules_planned_seen = sum(sum(s.get("todos_seen", [])) for s in shards.values())
    avg_grans_per_cell = (granules_planned_seen / cells_seen) if cells_seen > 0 else 0.0
    avg_gb_per_gran = (downloaded_gb / approx_attempts) if approx_attempts > 0 else 0.0
    cells_remaining = max(0, TOTAL_CELLS_IN_GRID - cells_seen)
    granules_remaining_in_seen = sum(
        max(0, (s.get("todo") or 0) - (s.get("downloaded") or 0))
        for s in shards.values() if s.get("current_cell")
    )
    estimated_total_granules = granules_planned_seen + cells_remaining * avg_grans_per_cell
    estimated_total_gb = estimated_total_granules * avg_gb_per_gran
    progress = {
        "downloaded_gb": downloaded_gb,
        "estimated_total_gb": estimated_total_gb,
        "granules_attempted": approx_attempts,
        "estimated_total_granules": estimated_total_granules,
        "cells_seen": cells_seen,
        "cells_in_grid": TOTAL_CELLS_IN_GRID,
        "avg_gb_per_granule": avg_gb_per_gran,
        "avg_granules_per_cell": avg_grans_per_cell,
        "granules_remaining_in_seen": granules_remaining_in_seen,
    }

    return {
        "now": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shards": shards,
        "parquet": parquet,
        "disk": disk,
        "net": net,
        "progress": progress,
        "granules_done_total": granules_done_total,
        "chips_total_in_logs": chips_total_in_logs,
    }


HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>HLS pull progress</title>
<style>
 body { font-family: ui-monospace, Menlo, Consolas, monospace; background:#0d0f12; color:#dde; margin:0; padding:20px; }
 h1 { color:#7fd1ff; margin:0 0 6px 0; font-size:18px; }
 .sub { color:#888; font-size:12px; margin-bottom:18px; }
 .grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:14px; }
 .card { background:#161a1f; border:1px solid #232a31; border-radius:6px; padding:12px 14px; }
 .card h2 { margin:0 0 8px 0; font-size:13px; color:#a8e1ff; }
 .row { display:flex; justify-content:space-between; padding:2px 0; border-bottom:1px dashed #232a31; }
 .row:last-child { border:0; }
 .k { color:#999; }
 .v { color:#dde; }
 .ok { color:#7ce69c; }
 .warn { color:#f5c971; }
 .bar { background:#1f262d; border-radius:4px; overflow:hidden; height:10px; margin:6px 0; }
 .bar > span { display:block; height:100%; background:linear-gradient(90deg,#3a8fe6,#7fd1ff); }
 pre { font-size:11px; color:#bbb; overflow:auto; max-height:60px; margin:6px 0 0 0; white-space:pre-wrap; word-break:break-all; }
 .totals { display:flex; gap:18px; margin:12px 0; flex-wrap:wrap; }
 .totals .pill { background:#161a1f; border:1px solid #232a31; padding:8px 12px; border-radius:6px; }
 .totals .pill b { color:#7fd1ff; }
 .totals .pill.net { border-color:#3a8fe6; }
 .totals .pill.net b { color:#7ce69c; }
 .dlcard { background:#161a1f; border:1px solid #3a8fe6; border-radius:6px; padding:14px 18px; margin:14px 0; }
 .dlcard .head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; }
 .dlcard .head .big { font-size:22px; color:#7fd1ff; font-weight:bold; }
 .dlcard .head .sub2 { color:#9ab; font-size:12px; }
 .dlcard .bigbar { background:#1f262d; border-radius:6px; overflow:hidden; height:18px; margin:6px 0; }
 .dlcard .bigbar > span { display:block; height:100%; background:linear-gradient(90deg,#3a8fe6,#7ce69c); }
 .dlcard .meta { display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:#aab; }
 .dlcard .meta span b { color:#dde; }
 .done { color:#7ce69c; font-weight:bold; }
</style>
</head><body>
<h1>HLS pull progress</h1>
<div class="sub" id="ts">loading…</div>
<div class="dlcard" id="dlcard"></div>
<div class="totals" id="totals"></div>
<div class="grid" id="shards"></div>
<script>
async function tick() {
  try {
    const r = await fetch('/data');
    const d = await r.json();
    document.getElementById('ts').textContent = 'updated ' + d.now;
    const t = document.getElementById('totals');
    const fmt = b => { const u=['B','KB','MB','GB','TB']; let i=0; while(b>=1024&&i<u.length-1){b/=1024;i++} return b.toFixed(1)+' '+u[i] };
    // Big download summary card
    const p = d.progress || {};
    const dlPct = (p.estimated_total_gb && p.estimated_total_gb > 0)
                ? Math.min(100, 100 * p.downloaded_gb / p.estimated_total_gb) : 0;
    document.getElementById('dlcard').innerHTML = `
      <div class="head">
        <div><span class="big">${(p.downloaded_gb||0).toFixed(2)} GB</span> downloaded
             of <span class="big">${(p.estimated_total_gb||0).toFixed(2)} GB</span> est. total</div>
        <div class="sub2">${dlPct.toFixed(1)}%  ·  cells seen ${p.cells_seen||0} / ${p.cells_in_grid||60}  ·  net ↓ ${(d.net.rx_MBps||0).toFixed(2)} MB/s</div>
      </div>
      <div class="bigbar"><span style="width:${dlPct.toFixed(1)}%"></span></div>
      <div class="meta">
        <span>granule attempts: <b>${(p.granules_attempted||0).toLocaleString()}</b></span>
        <span>est. total granules: <b>${Math.round(p.estimated_total_granules||0).toLocaleString()}</b></span>
        <span>avg GB/granule: <b>${(p.avg_gb_per_granule||0).toFixed(3)}</b></span>
        <span>avg granules/cell: <b>${(p.avg_granules_per_cell||0).toFixed(0)}</b></span>
      </div>`;
    const ifacePills = Object.keys(d.net.per_iface||{}).map(n => {
      const r = d.net.per_iface[n];
      return `<div class="pill"><span class="k">iface ${n}</span> &nbsp;&darr; <b>${r.rx_MBps.toFixed(2)} MB/s</b> &nbsp; &uarr; ${r.tx_MBps.toFixed(2)} MB/s</div>`;
    }).join('');
    t.innerHTML = `
      <div class="pill net">network &darr; download: <b>${d.net.rx_MBps.toFixed(2)} MB/s</b></div>
      <div class="pill net">network &uarr; upload: <b>${d.net.tx_MBps.toFixed(2)} MB/s</b></div>
      <div class="pill">total chip rows in shards: <b>${d.parquet.total_rows.toLocaleString()}</b></div>
      <div class="pill">positive chips on disk: <b>${d.parquet.total_pos.toLocaleString()}</b></div>
      <div class="pill">granules processed (logs): <b>${d.granules_done_total.toLocaleString()}</b></div>
      <div class="pill">chips written (logs): <b>${d.chips_total_in_logs.toLocaleString()}</b></div>
      <div class="pill">raw cache: <b>${fmt(d.disk.raw_bytes)}</b></div>
      <div class="pill">chips on disk: <b>${fmt(d.disk.chips_bytes)}</b></div>
      <div class="pill">free disk: <b>${fmt(d.disk.free_bytes)}</b></div>
      ${ifacePills}
    `;
    const g = document.getElementById('shards');
    g.innerHTML = '';
    for (const tag of Object.keys(d.shards)) {
      const s = d.shards[tag];
      const pct = (s.todo && s.todo>0) ? Math.min(100, 100*s.downloaded/s.todo) : 0;
      const status = s.all_done ? '<span class="done">DONE</span>'
                    : (s.current_cell ? ('cell <b>'+s.current_cell+'</b>') : '<span class="warn">starting…</span>');
      g.innerHTML += `
        <div class="card">
          <h2>shard ${tag} — ${status}</h2>
          <div class="row"><span class="k">log</span><span class="v">${s.log||'-'}</span></div>
          <div class="row"><span class="k">cell granules todo</span><span class="v">${s.todo??'-'} (skipped ${s.skipped??'-'})</span></div>
          <div class="row"><span class="k">downloaded (current cell)</span><span class="v">${s.downloaded||0} / ${s.todo||'-'}</span></div>
          <div class="bar"><span style="width:${pct.toFixed(1)}%"></span></div>
          <div class="row"><span class="k">chips written (this run)</span><span class="v ok">${s.chips_written||0}</span></div>
          <div class="row"><span class="k">granules processed (this run)</span><span class="v">${s.granules_processed||0}</span></div>
          <div class="row"><span class="k">last download time</span><span class="v">${s.last_download_seconds??'-'} s</span></div>
          <div class="row"><span class="k">cells finished</span><span class="v">${(s.cells_done||[]).map(c=>c.cell).join(', ')||'-'}</span></div>
          <pre>${(s.last_line||'').replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]))}</pre>
        </div>`;
    }
  } catch (e) {
    document.getElementById('ts').textContent = 'error: '+e;
  }
}
tick();
setInterval(tick, 3000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        return

    def do_GET(self):
        if self.path == "/data":
            payload = json.dumps(collect()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path in ("/", "/index.html"):
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)


def main():
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"hls_progress_server listening on http://localhost:{PORT}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
