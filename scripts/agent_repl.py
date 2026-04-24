"""
agent_repl.py — Phase 7 deliverable.

Pre-classifies an image once, then runs the sustainability agent against it.
Supports a one-shot mode (`--query "..."`) and an interactive REPL mode
(default) for iterating on follow-up questions against the same image.

Usage:
    python scripts/agent_repl.py <image_path>
    python scripts/agent_repl.py <image_path> --query "Should we plant more forest?"
    python scripts/agent_repl.py <image_path> --dry-run      # show AgentState, no API call
    python scripts/agent_repl.py <image_path> --verbose      # print per-tool-call detail

Pre-classification happens exactly once at startup. Every agent invocation
in a session reuses the same AgentState; tools never re-run inference.

Env:
    ANTHROPIC_API_KEY  — required for live agent calls (not needed for --dry-run)
    LANDUSE_CHECKPOINT — checkpoint path override (default: model/segformer-b1-run1/best.pt)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from agent.base import AgentReport
from agent.tools import AgentState  # noqa: F401 — re-export for convenience


DEFAULT_CHECKPOINT = PROJECT_ROOT / "model" / "segformer-b1-run1" / "best.pt"
DEFAULT_QUERY = (
    "Produce a sustainability report for this parcel. Cover the land "
    "composition, the current emissions footprint (annual flux + embodied "
    "stock), one or two realistic interventions worth considering, and any "
    "model-quality caveats the user should know about."
)


# ---------------------------------------------------------------------------
# Pre-classification
# ---------------------------------------------------------------------------

def preclassify(image_path: Path, checkpoint: Path) -> AgentState:
    """
    Run inference once and build the AgentState the agent will read from.

    Imports torch/transformers lazily so --dry-run on a machine without a
    GPU-less box can still be partially useful... wait, dry-run needs the
    classifier too. Never mind the lazy-import optimization; just import.
    """
    # Local import so import-time side effects don't hit unless we actually
    # run the command.
    from backend.inference import InferenceEngine

    print(f"[preclassify] loading model from {checkpoint}")
    engine = InferenceEngine(checkpoint_path=checkpoint)

    print(f"[preclassify] classifying {image_path}")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    result = engine.classify(image_bytes, tta=True)

    # total_area_ha from the emissions result (already computed at pixel_size_m=0.3).
    state = AgentState(
        percentages=result.percentages,
        emissions=result.emissions,
        total_area_ha=result.emissions.total_area_ha,
        image_label=str(image_path),
    )

    # One-line summary.
    nonzero = {k: round(v, 2) for k, v in state.percentages.items() if v > 0.5}
    print(
        f"[preclassify] done in {result.inference_ms} ms — "
        f"{state.total_area_ha:.2f} ha — "
        f"composition: {nonzero} — "
        f"net annual: {state.emissions.total_annual_tco2e_per_yr:+.2f} tCO2e/yr"
    )
    if result.warnings:
        for w in result.warnings:
            print(f"[preclassify] warning: {w}")
    return state


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_report(report: AgentReport, verbose: bool) -> None:
    if verbose and report.tool_calls:
        print("\n--- tool calls ---")
        for tc in report.tool_calls:
            status = "ERR" if tc.error else "OK "
            print(f"  [{status}] turn {tc.turn}: {tc.name}({_short_input(tc.input)})")
            if tc.error:
                print(f"           error: {tc.error}")
            elif tc.result is not None:
                preview = json.dumps(tc.result, default=str)
                if len(preview) > 200:
                    preview = preview[:197] + "..."
                print(f"           -> {preview}")

    print("\n--- report ---")
    print(textwrap.fill(report.final_text, width=88, replace_whitespace=False)
          if "\n" not in report.final_text
          else report.final_text)

    u = report.usage
    cache_hit = u.get("cache_read_input_tokens", 0)
    cache_new = u.get("cache_creation_input_tokens", 0)
    in_tok = u.get("input_tokens", 0)
    out_tok = u.get("output_tokens", 0)
    print(
        f"\n[stats] turns={report.turns_used}/{_max_turns_from(report)} "
        f"stop={report.stop_reason} "
        f"tools_called={len(report.tool_calls)} "
        f"tokens: in={in_tok} out={out_tok} cached_read={cache_hit} cached_new={cache_new}"
    )


def _short_input(inp: dict) -> str:
    if not inp:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in inp.items())


def _max_turns_from(report: AgentReport) -> str:
    # Not currently tracked on the report; stringly "?". Cosmetic only.
    return "?"


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def repl_loop(agent, state: AgentState, verbose: bool) -> None:
    print("\n" + "=" * 72)
    print("Interactive REPL. Type a query, blank line + Enter to submit.")
    print("Commands: /quit, /state, /reset")
    print("=" * 72)
    while True:
        try:
            print("\n>>> ", end="", flush=True)
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            query = "\n".join(lines).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[repl] bye")
            return

        if not query:
            continue
        if query in ("/quit", "/exit"):
            print("[repl] bye")
            return
        if query == "/state":
            _dump_state(state)
            continue
        if query == "/reset":
            print("[repl] (AgentState is immutable in this session; nothing to reset)")
            continue

        try:
            report = agent.run(state, query)
        except Exception as exc:  # noqa: BLE001
            print(f"[repl] agent error: {type(exc).__name__}: {exc}")
            continue
        print_report(report, verbose=verbose)


def _dump_state(state: AgentState) -> None:
    print(f"image: {state.image_label}")
    print(f"total_area_ha: {state.total_area_ha:.3f}")
    print(f"percentages (nonzero):")
    for k, v in sorted(state.percentages.items(), key=lambda kv: -kv[1]):
        if v > 0.0:
            print(f"  {k:12s} {v:6.2f}%")
    em = state.emissions
    print(f"total_annual_tco2e_per_yr: {em.total_annual_tco2e_per_yr:+.2f}")
    print(f"total_embodied_tco2e:      {em.total_embodied_tco2e:+.2f}")
    print(f"excluded_fraction:         {em.excluded_fraction:.2%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-classify an image, then run the sustainability agent."
    )
    parser.add_argument("image", type=Path, help="Path to the image to analyze.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("LANDUSE_CHECKPOINT", str(DEFAULT_CHECKPOINT))),
        help="SegFormer checkpoint path.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="One-shot query. If omitted, drops into interactive REPL mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify the image and dump AgentState, but don't call the API.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-tool-call detail in addition to the final report.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Hard cap on agent turns. Default 5 per PHASE_PLAN.md.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model id. Defaults to claude-haiku-4-5.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        sys.exit(2)
    if not args.checkpoint.exists():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(2)

    state = preclassify(args.image, args.checkpoint)

    if args.dry_run:
        print("\n[dry-run] skipping agent; dumping state and exiting.")
        _dump_state(state)
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "error: ANTHROPIC_API_KEY not set. "
            "Set it or pass --dry-run to inspect state without calling the API.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Defer the import so --dry-run doesn't need anthropic installed.
    from agent.claude import ClaudeAgent

    agent_kwargs = {"max_turns": args.max_turns}
    if args.model:
        agent_kwargs["model"] = args.model
    agent = ClaudeAgent(**agent_kwargs)

    if args.query is not None:
        report = agent.run(state, args.query)
        print_report(report, verbose=args.verbose)
    else:
        # If user didn't ask anything, do a default one-shot sustainability
        # report first, then drop into the REPL for follow-ups.
        print("\n[repl] running default sustainability report first; "
              "then interactive.")
        report = agent.run(state, DEFAULT_QUERY)
        print_report(report, verbose=args.verbose)
        repl_loop(agent, state, verbose=args.verbose)


if __name__ == "__main__":
    main()
