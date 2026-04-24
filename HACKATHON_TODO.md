# HACKATHON_TODO

> Pre-built is infrastructure. This is the menu for the hackathon itself.
>
> Tiers are strict. **Do all P0 before any P1. Do all P1 before any P2.**
> If a higher-tier task is blocked, move on to a sibling within the same
> tier before jumping ahead. The temptation to skip to a shiny P2 or P3
> item first is real; resist it. The demo judges will look at P0.

Each task has a time estimate (me-alone baseline; halve it for two people
pair-working well), the files most likely to be touched, and a "definition
of done" so nobody has to interpret vagueness mid-sprint.

---

## P0 — demo-critical. Do these first.

These make the system feel like it does what the pitch says. Without them
the project looks like a backend exercise with a nice report panel.

### P0.1 — Map picker + live tile fetching (4–6h)

**Why:** "Upload a tile" is not a compelling demo verb. "Click anywhere
on the map" is. This is the single highest-leverage hackathon feature.

**What:** Replace (or sit alongside) the upload panel with a Leaflet map.
User clicks a point; we fetch a ~300m×300m tile at ~0.3m GSD from a provider,
hand it to `POST /classify`, and run the rest of the flow as normal.

**Files:** `frontend/index.html`, `frontend/app.js`, possibly a new
`backend/tile_proxy.py` to hide the provider API key.

**Provider options** (ordered by resolution-fit):
1. **NAIP via USGS / Esri** — 0.6m aerial CONUS, free-tier friendly, closest
   match to LoveDA's 0.3m GSD. First choice.
2. **Mapbox Satellite** — ~0.3m in many urban areas, requires an access token.
3. **Esri World Imagery** — variable 0.3–1m, no token for low volume.
4. **Google Static Maps** — 0.15m in places, but ToS is strict about
   saving/processing. Only if you can pre-clear it.

**🚫 Do NOT use Sentinel-2 at 10m/pixel.** The model was fine-tuned on
0.3m imagery. Sentinel tiles will silently tank accuracy and nobody will
know why. If you get stuck on resolution, use a static NAIP tile and
document the choice.

**Definition of done:**
- Click on map → screenshot-style tile loads in the Source panel
- `/classify` runs on it and the full pipeline works end-to-end
- The tile's bounding box is displayed somewhere in the UI
- A visible warning fires if the tile's inferred resolution is coarser
  than 1m/pixel

---

### P0.2 — Demo-image shortlist (1–2h)

**Why:** Right now the demo buttons point at two images. You want 4–6 so
you can pick live based on the room or the question.

**What:** Pre-run `scripts/infer.py` on 15–20 LoveDA val tiles. Rank by
(a) visual appeal of the aerial image, (b) pixel accuracy vs ground truth,
(c) class diversity that would produce an interesting report.

Shortlist 4–6 and add demo buttons. Target coverage:
- 1× forest-dominant (we have 3546)
- 1× "model limits" case (we have 2523)
- 1× urban / building-heavy — exercises the building-emissions caveat
- 1× mixed-use — for a balanced report
- 1× water-dominant — exercises the wetlands assumption

**Files:** `frontend/demos/`, `frontend/index.html` (add buttons),
optionally a `scripts/rank_demos.py` to automate the triage.

**Definition of done:** a `docs/demo_tiles.md` file describing each
shortlisted tile and what report pattern it elicits.

---

### P0.3 — Error-state polish (1h)

**Why:** Live demos fail. When they do, the failure should be legible.

**What:** Audit every path in `frontend/app.js` that could error and make
sure the UI shows something other than "Error" or a stalled spinner.
Specifically:

- Backend unreachable on load → show a blocking modal, not a silently-red
  dot the user might miss.
- `/classify` times out → explicit message, offer retry.
- `/agent/report` fails mid-run → recover gracefully, don't strand the
  progress stepper on step 5.
- Any upload larger than 4 MB → warn before POST.

**Files:** `frontend/app.js`, small CSS additions.

**Definition of done:** kill the backend mid-demo; UI tells the truth about
what happened instead of hanging.

---

## P1 — substantial additions. Do after P0 is green.

Things that would make technical reviewers say "interesting" rather than
just nod.

### P1.1 — Bbox region selection (3–4h)

**Why:** "Analyze this whole tile" is a reasonable first-pass semantic, but
real users would want to draw a rectangle and ask about just that sub-region.

**What:** Add a bbox draw tool on top of either the uploaded image preview
or the map picker. Backend already aggregates whole-image; a bbox means
crop the mask before aggregation. This changes the semantics of
`total_area_ha` for one endpoint path.

**Files:** `frontend/app.js` (canvas/SVG bbox tool), `backend/inference.py`
(accept optional bbox), `backend/models.py` (add `bbox: Optional[...]`),
`scripts/emissions.py` may need a `crop_bbox` helper.

**Definition of done:** draw a box over water in a lake-heavy tile, run the
agent, report correctly talks about only the selected area.

---

### P1.2 — More agent tools (2–4h, grows with number added)

**Why:** The four existing tools cover the base case. More tools → richer
reports. Candidates in rough priority order:

1. **`compare_to_region(benchmark)`** — compare the current parcel's
   emissions intensity against a reference (U.S. avg, state avg, similar
   land-use mix). Requires adding reference data to `emissions.py` but
   that's cheap.
2. **`estimate_payback(from_class, to_class, fraction)`** — for a given
   intervention, compute the break-even year where annual gain cancels
   embodied cost. Pure math over existing data.
3. **`list_available_classes()`** — lets the agent see what emissions
   classes exist without calling `get_emissions_estimate` first. Would
   fix a small inefficiency where the agent sometimes asks for `forest`
   interventions on a parcel with zero forest.
4. **`get_class_confidence()`** — return per-pixel confidence from the
   segmentation softmax so the agent can weight its caveats by actual
   uncertainty. Requires surfacing logits from `InferenceEngine`.

**Files:** `agent/tools.py` (schema + impl for each), minor updates to the
system prompt for each new tool's usage guidance.

**Definition of done:** each new tool has offline smoke-test coverage in
`smoke_tools.py` before any live calls.

---

### P1.3 — Follow-up query chat mode (2h)

**Why:** Right now every "Generate" click is a fresh `agent.run()` — no
memory of prior turns. For a demo, this is fine. For actual use, a user
wants to ask "what if we halved that instead?" as a follow-up.

**What:** Add a text input below the report. Each submission appends to
the conversation. The agent's `messages` list persists across queries
within the same image.

The backend needs a new endpoint `POST /agent/continue` that takes a
conversation handle (or the full message history) and a new query. Keeps
the existing `/agent/report` intact as the session-start endpoint.

**Files:** `agent/claude.py` (add `continue_run` method or mutate `run` to
accept prior messages), `backend/main.py` (new endpoint + in-memory
session dict keyed by uuid), `frontend/app.js` (chat input + history
rendering).

**Definition of done:** click Generate, read report, type "what about the
barren areas?" and see the agent continue the conversation without
re-surveying the whole parcel.

---

### P1.4 — "What-if" sliders tied to /simulate (2h)

**Why:** The narrative report talks about interventions abstractly. A
slider lets the demo audience *feel* the tradeoff.

**What:** A new panel below the composition bar with:
- `from_class` dropdown (populated from classes present)
- `to_class` dropdown
- Fraction slider 0–100%
- Live-updated KPIs showing `delta_annual` and `delta_embodied`

Each slider change debounces and fires `POST /simulate`.

**Files:** `frontend/index.html`, `frontend/app.js`.

**Definition of done:** demo audience sees real-time numbers respond to
slider drags; no agent involvement needed (the `/simulate` endpoint is
pure math and runs in <10ms).

---

## P2 — polish. Do these if P0 and P1 are solid.

Things that make the demo prettier, not things that change what it does.

### P2.1 — Streaming SSE for agent responses (3h)

Replace the fake-progress stepper with real streaming. The Anthropic SDK
supports `client.messages.stream()`; pipe those events to the frontend via
Server-Sent Events. Adds ~100 lines to `agent/claude.py` (a `run_streaming`
method that yields events instead of returning an `AgentReport`) and
requires FastAPI `StreamingResponse` plus frontend `EventSource`.

High visible impact; straightforward but fiddly.

### P2.2 — React migration (4–6h)

The current vanilla JS is ~500 lines and rides the line between "compact"
and "about to get unwieldy." React + Vite would give the hackathon team
a better long-term scaffold. Don't migrate unless you're adding P1 features
that would benefit from component isolation (P1.3 chat mode especially).

### P2.3 — Report export (1h)

"Download as PDF" button on the report panel. Use browser print-to-PDF
with a `@media print` stylesheet. Users like saving things.

### P2.4 — Side-by-side image comparison mode (2h)

Two image frames next to each other, both with their own reports, so a
user can see "this parcel vs that one" without flipping tabs. Ties into
P1.1 (bbox) nicely — compare two regions of the same tile.

### P2.5 — Dark/light theme toggle (1h)

The current dark theme is intentional and good. But a light theme for
screenshots / printing / accessibility is polish worth having. All colors
are already CSS variables — a `data-theme="light"` attribute with
override variables would do it.

### P2.6 — Mobile layout audit (1h)

The layout already stacks on <900px via a media query, but nothing has
been tested on a real phone. Almost certainly there are tap-target issues
and the composition bar is probably too thin at mobile scale.

---

## P3 — ambitious. Probably won't get to these. Document instead.

These are reasonable extensions to *talk about in the pitch* even if they
don't ship.

### P3.1 — Region-specific emissions factors

The current table is global averages. Real factors vary by climate zone,
country, crop type, energy mix. A v2 would accept a `lat, lon` and look up
factors from a regional dataset (e.g., EDGAR gridded, eGRID subregion for
building operational emissions). Non-trivial data engineering.

### P3.2 — Multi-tile / whole-county aggregation

Currently one tile at a time. A county-scale analysis would mean tiling
up a large area, classifying each tile, and aggregating. Backend work is
straightforward (loop over tiles); visualization is the hard part.

### P3.3 — Time-series / change detection

Given two tiles of the same area at different dates, what changed? This is
a classic remote-sensing problem, solvable with our pixelwise output. The
emissions delta interpretation is interesting: was forest lost? was a
building constructed? The agent would write a "what changed here" narrative.

### P3.4 — Fallback LLM provider

The `ReasoningAgent` Protocol is already in place exactly for this. An
`agent/openrouter.py` implementing the same interface would cost ~80 lines,
half of which is error-handling. Worth doing if only to demonstrate that
the abstraction isn't just decoration.

### P3.5 — Training-data feedback loop

Expose a "this classification looks wrong" button on each demo. Collected
corrections become fine-tuning data for a v2 model. Interesting PM story,
real-but-boring MLE work.

---

## Non-goals

Don't do these. They look appealing but either don't add demo value or
open failure modes.

- **Authentication / user accounts.** Single-session demo app. Not
  relevant.
- **Real-time collaboration.** Scope trap.
- **Deploying to a public URL.** Budget has to cover every demo run; public
  exposure means uncapped cost. Run it on a laptop + tailscale / ngrok if
  you need remote access during the hackathon.
- **Swapping the vision model for a bigger one.** SegFormer-B1 is a
  conscious speed-vs-accuracy tradeoff. The weak classes (forest, barren)
  are *surfaced as caveats*, not covered up. Replacing B1 with a bigger
  model would invalidate every measured number in `CURRENT_STATE.md`.

---

## Triage order for day-of

If you get there at 9am and the hackathon ends at 5pm:

- **09:00–12:00** — P0.1 (map picker). This is the win.
- **12:00–13:00** — P0.2 (demo shortlist), lunch.
- **13:00–14:00** — P0.3 (error polish).
- **14:00–16:00** — pick **one** P1 item based on energy + team strengths.
  P1.4 (sliders) is the easiest win; P1.3 (chat mode) is the most
  impressive; P1.1 (bbox) is the cleanest technically.
- **16:00–16:45** — dry-run the demo 3×. Fix what breaks.
- **16:45–17:00** — final polish. Do NOT start anything new.

If a P0 blows up, cut scope within the same P0 task. Don't jump tiers to
escape the problem.
