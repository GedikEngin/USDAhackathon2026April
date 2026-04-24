"""
ClaudeAgent: ReasoningAgent implementation backed by Claude Haiku 4.5
via the Anthropic API.

Locked decisions (PHASE_PLAN.md, Phase 7):
  - Model: claude-haiku-4-5
  - max_turns = 5 hard cap on the agent loop
  - Prompt caching on system prompt + tools (they're fixed across turns)
  - Persona: sustainability-first blend (data-grounded, practical about
    tradeoffs, honest about model uncertainty)
  - Tool outputs surface [SRC-n] tags AND full citations so the model can
    weave references into prose without fabricating them

Agent loop shape (standard Anthropic tool-use pattern):
  1. POST messages with system + tools
  2. If stop_reason == "end_turn": done, return the text
  3. If stop_reason == "tool_use": execute every tool_use block, append a
     user message with all tool_result blocks, go to 1
  4. If we're about to exceed max_turns: send one final turn WITHOUT tools
     to force the model to synthesize whatever it has into a final text
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.base import AgentReport, ReasoningAgent, ToolCallRecord
from agent.tools import AgentState, TOOL_SCHEMAS, dispatch_tool

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model + config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TURNS = 5
DEFAULT_MAX_TOKENS = 2048  # plenty for a structured sustainability report


SYSTEM_PROMPT = """\
You are a sustainability-focused land-use analyst. A remote-sensing segmentation \
model has classified an aerial image into land-cover classes, and you have \
tools that surface the measurements, emissions estimates, and candidate \
interventions for that image.

Your job is to produce a grounded, honest sustainability report that helps \
the user understand what's on the parcel, its carbon footprint, and how it \
might be improved.

## How to work

1. **Start by calling `get_land_breakdown`** to see what's in the image. Every \
   analysis depends on knowing the composition first.
2. **Call `get_emissions_estimate`** to get the annual flux and embodied stock, \
   with per-class breakdown and source citations.
3. **Use `simulate_intervention` to test specific 'what-if' scenarios** the \
   user asks about, or that your analysis suggests are worth quantifying.
4. **Use `recommend_mitigation` when you want a ranked menu of candidate \
   interventions**. The menu is mechanical — it does not weigh tradeoffs. \
   You must weigh them yourself in prose (see next section).

Be efficient: you have a limited number of tool-calling turns. Chain multiple \
tool calls in a single turn when they're independent. Do not call the same \
tool twice with identical inputs.

## How to weigh interventions (important)

`recommend_mitigation` returns candidates ranked purely by a single axis. \
**Do not blindly recommend the top result.** A ranking that says "convert all \
forest to water" because it lowers embodied stock is mathematically correct \
but obviously absurd. Your job as an analyst is to:

- Rule out absurd conversions (destroying existing sinks, converting usable \
  land to non-functional states, etc.).
- Prefer interventions that *add* sinks or reduce ongoing emissions without \
  destroying valuable existing assets.
- Be explicit about tradeoffs — forest planting removes CO2 annually but \
  costs ~800 tCO2e/ha in embodied stock; sometimes that's worth it, \
  sometimes not.
- Consider the area available. A 0.05 ha intervention may be too small to \
  matter even if it ranks high.

## How to cite

Every number you quote from a tool output should carry its [SRC-n] tag \
inline. When you first mention a source in your response, add the full \
citation (from `sources_cited`) so the reader can verify. Do not fabricate \
citations or invent facts not present in tool outputs.

## How to handle model uncertainty

The tools surface `model_caveats` (image-specific model-quality concerns) \
and `assumptions` (per-class modeling simplifications). Surface these \
honestly in your report — do not bury them. A report that ignores its own \
model's weaknesses is not trustworthy. If a caveat could meaningfully \
change the conclusion, say so.

## Output format

Write your final report as clear prose. Use short section headings if it \
helps readability. Lead with the headline (net sink or source, total \
annual flux, total embodied stock), then the composition, then the \
recommended intervention with rationale, then caveats. Keep it concise — \
a good report is two or three paragraphs, not a wall of text.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_tool_error(exc: Exception) -> str:
    """Compact error string for tool_result content."""
    return f"{type(exc).__name__}: {exc}"


def _collect_text_blocks(content_blocks: List[Any]) -> str:
    """Concatenate text from a response's content blocks, in order."""
    parts: List[str] = []
    for block in content_blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(p for p in parts if p).strip()


def _collect_tool_use_blocks(content_blocks: List[Any]) -> List[Any]:
    return [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]


# ---------------------------------------------------------------------------
# ClaudeAgent
# ---------------------------------------------------------------------------

@dataclass
class ClaudeAgent(ReasoningAgent):
    """
    Anthropic-API-backed sustainability reasoning agent.

    Parameters
    ----------
    client
        An `anthropic.Anthropic` client. Pass one in so the caller owns
        auth (API key loading, timeouts, etc.). If None, the agent will
        lazy-construct one from `ANTHROPIC_API_KEY`.
    model
        Model id. Defaults to Claude Haiku 4.5.
    max_turns
        Hard cap on model calls per session. Locked to 5 by phase plan;
        keep this as a safety rail even if you pass it explicitly.
    max_tokens
        Max output tokens per model call.
    """
    client: Any = None
    model: str = DEFAULT_MODEL
    max_turns: int = DEFAULT_MAX_TURNS
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.client is None:
            # Lazy import so the module can be imported in environments
            # that don't have anthropic installed (e.g., offline dev boxes).
            import anthropic  # noqa: F401 — local import intentional
            self.client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, state: AgentState, user_query: str) -> AgentReport:
        report = AgentReport(final_text="", stop_reason="")
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": user_query},
        ]

        total_input_tokens = 0
        total_output_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0

        for turn in range(1, self.max_turns + 1):
            report.turns_used = turn

            # On the final turn, disable tools so the model is forced to
            # synthesize a text response instead of requesting more work.
            # This is how we guarantee a coherent output when the cap hits.
            is_final = turn == self.max_turns
            call_kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": messages,
            }
            if not is_final:
                call_kwargs["tools"] = self._tools_with_cache()

            log.debug("turn %d: calling model (tools=%s)", turn, not is_final)
            response = self.client.messages.create(**call_kwargs)

            # Usage accounting (best-effort; fields present on real API).
            usage = getattr(response, "usage", None)
            if usage is not None:
                total_input_tokens += getattr(usage, "input_tokens", 0) or 0
                total_output_tokens += getattr(usage, "output_tokens", 0) or 0
                cache_read_tokens += getattr(
                    usage, "cache_read_input_tokens", 0
                ) or 0
                cache_creation_tokens += getattr(
                    usage, "cache_creation_input_tokens", 0
                ) or 0

            content_blocks = list(getattr(response, "content", []) or [])
            stop_reason = getattr(response, "stop_reason", "") or ""
            report.stop_reason = stop_reason

            # Always append the assistant turn to messages so a following
            # tool_result message references valid tool_use ids.
            messages.append({"role": "assistant", "content": content_blocks})

            if stop_reason == "end_turn" or is_final:
                # Final text response. We're done.
                report.final_text = _collect_text_blocks(content_blocks)
                if is_final and stop_reason != "end_turn":
                    # We forced termination; note it.
                    report.stop_reason = f"max_turns ({stop_reason})"
                break

            if stop_reason != "tool_use":
                # Unexpected — surface it and stop.
                report.final_text = _collect_text_blocks(content_blocks)
                report.stop_reason = f"unexpected:{stop_reason}"
                break

            # Execute all tool_use blocks and build one user message of
            # tool_results in the same order.
            tool_use_blocks = _collect_tool_use_blocks(content_blocks)
            tool_results: List[Dict[str, Any]] = []
            for block in tool_use_blocks:
                tool_name = getattr(block, "name", "")
                tool_input = getattr(block, "input", {}) or {}
                tool_id = getattr(block, "id", "")

                record = ToolCallRecord(
                    turn=turn,
                    name=tool_name,
                    input=dict(tool_input),
                )
                try:
                    result = dispatch_tool(tool_name, dict(tool_input), state)
                    record.result = result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": _json_stringify(result),
                    })
                except Exception as exc:  # noqa: BLE001 — want to catch tool bugs too
                    err_str = _format_tool_error(exc)
                    record.error = err_str
                    log.warning("tool %s raised: %s", tool_name, err_str)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": err_str,
                        "is_error": True,
                    })
                report.tool_calls.append(record)

            messages.append({"role": "user", "content": tool_results})
            # Loop back for the next turn.

        report.usage = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_read_input_tokens": cache_read_tokens,
            "cache_creation_input_tokens": cache_creation_tokens,
        }
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tools_with_cache(self) -> List[Dict[str, Any]]:
        """
        Return TOOL_SCHEMAS with a cache_control breakpoint on the last
        tool. Anthropic's prompt caching treats the cache_control on a
        tools entry as caching the entire tools array up through that
        point. Putting it on the last entry caches all four tool defs.
        """
        tools = [dict(s) for s in TOOL_SCHEMAS]
        if tools:
            tools[-1] = dict(tools[-1])
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        return tools


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _json_stringify(obj: Any) -> str:
    """
    Serialize a tool result dict to a compact JSON string for the
    tool_result `content` field. The API also accepts list-of-blocks,
    but a single JSON string is simpler and Haiku parses it fine.
    """
    import json
    return json.dumps(obj, default=str, ensure_ascii=False)
