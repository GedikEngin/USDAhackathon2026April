"""
Offline smoke test for ClaudeAgent's loop logic. Uses a scripted fake
Anthropic client so we can verify:
  - tool_use blocks get dispatched
  - tool_result blocks get appended with correct tool_use_id
  - multiple tool_use blocks in one response all run
  - is_error path works when a tool raises
  - max_turns cap triggers and disables tools on the final call
  - usage accounting accumulates correctly
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.claude import ClaudeAgent
from scripts.smoke_tools import build_3546_state


# ---------- fake API types (mimic anthropic SDK shapes just enough) ----------

@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    name: str
    input: Dict[str, Any]
    id: str
    type: str = "tool_use"


@dataclass
class FakeUsage:
    input_tokens: int = 50
    output_tokens: int = 30
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeResponse:
    content: List[Any]
    stop_reason: str
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    def __init__(self, scripted: List[FakeResponse]):
        self._scripted = list(scripted)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs) -> FakeResponse:
        # Deep-copy so post-hoc inspection sees kwargs as they were AT CALL TIME.
        # The real SDK doesn't mutate its inputs, but the agent mutates its own
        # `messages` list across turns, which would otherwise alias into our log.
        import copy
        self.calls.append(copy.deepcopy(kwargs))
        if not self._scripted:
            raise RuntimeError("fake client ran out of scripted responses")
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted: List[FakeResponse]):
        self.messages = FakeMessages(scripted)


# ---------- scenarios ----------

def scenario_happy_path() -> None:
    """Agent calls 2 tools in turn 1, then returns a text response in turn 2."""
    print("\n--- scenario: happy path ---")
    state = build_3546_state()

    scripted = [
        FakeResponse(
            content=[
                FakeTextBlock("Let me gather the data."),
                FakeToolUseBlock(
                    name="get_land_breakdown", input={}, id="toolu_1"
                ),
                FakeToolUseBlock(
                    name="get_emissions_estimate", input={}, id="toolu_2"
                ),
            ],
            stop_reason="tool_use",
            usage=FakeUsage(input_tokens=1200, output_tokens=80),
        ),
        FakeResponse(
            content=[
                FakeTextBlock(
                    "This parcel is a net carbon sink of -53.66 tCO2e/yr, "
                    "dominated by 72% forest cover [SRC-1]..."
                ),
            ],
            stop_reason="end_turn",
            usage=FakeUsage(
                input_tokens=2500, output_tokens=200,
                cache_read_input_tokens=1800,
            ),
        ),
    ]

    fake = FakeClient(scripted)
    agent = ClaudeAgent(client=fake, max_turns=5)
    report = agent.run(state, "Give me a sustainability report for this parcel.")

    print(f"turns_used: {report.turns_used}")
    print(f"stop_reason: {report.stop_reason}")
    print(f"tool_calls: {len(report.tool_calls)}")
    for tc in report.tool_calls:
        print(f"  turn={tc.turn} name={tc.name} "
              f"ok={tc.error is None} input_keys={list(tc.input)}")
    print(f"final_text (first 100): {report.final_text[:100]!r}")
    print(f"usage: {report.usage}")

    # Assertions
    assert report.turns_used == 2
    assert report.stop_reason == "end_turn"
    assert len(report.tool_calls) == 2
    assert all(tc.error is None for tc in report.tool_calls)
    assert report.usage["cache_read_input_tokens"] == 1800
    assert report.final_text.startswith("This parcel")

    # Verify second API call's messages include tool_results with correct ids
    second_call_messages = fake.messages.calls[1]["messages"]
    # messages = [user query, assistant(turn1 content), user(tool_results)]
    assert len(second_call_messages) == 3
    tool_result_msg = second_call_messages[2]
    assert tool_result_msg["role"] == "user"
    ids = [b["tool_use_id"] for b in tool_result_msg["content"]]
    assert ids == ["toolu_1", "toolu_2"], f"bad ids: {ids}"
    # Tool results should be JSON strings containing the expected content
    first_result_content = tool_result_msg["content"][0]["content"]
    assert "dominant_class" in first_result_content
    print("  ✓ all assertions passed")


def scenario_tool_error_recovery() -> None:
    """Agent tries a bogus simulate (building not present), gets an error, recovers."""
    print("\n--- scenario: tool error recovery ---")
    state = build_3546_state()

    scripted = [
        FakeResponse(
            content=[
                FakeToolUseBlock(
                    name="simulate_intervention",
                    input={"from_class": "building", "to_class": "forest", "fraction_pct": 100},
                    id="toolu_err",
                ),
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(
            content=[
                FakeTextBlock(
                    "I'll check what's actually present first."
                ),
                FakeToolUseBlock(
                    name="get_land_breakdown", input={}, id="toolu_recover"
                ),
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(
            content=[FakeTextBlock("Got it — no buildings present, so...")],
            stop_reason="end_turn",
        ),
    ]

    fake = FakeClient(scripted)
    agent = ClaudeAgent(client=fake, max_turns=5)
    report = agent.run(state, "Simulate tearing down all buildings.")

    print(f"turns_used: {report.turns_used}")
    print(f"tool_calls: {len(report.tool_calls)}")
    for tc in report.tool_calls:
        print(f"  name={tc.name} error={tc.error!r}")

    assert report.turns_used == 3
    assert len(report.tool_calls) == 2
    assert report.tool_calls[0].error is not None
    assert "No building present" in report.tool_calls[0].error
    assert report.tool_calls[1].error is None

    # Verify the is_error flag was set on the tool_result
    second_call_messages = fake.messages.calls[1]["messages"]
    tool_result_msg = second_call_messages[2]
    err_block = tool_result_msg["content"][0]
    assert err_block.get("is_error") is True
    print("  ✓ tool error surfaced with is_error=True, agent recovered")


def scenario_max_turns_cap() -> None:
    """
    Agent never calls end_turn; it just keeps requesting tools. max_turns=3
    should force turn 3 to be called without tools so the model is forced
    to produce text.
    """
    print("\n--- scenario: max_turns cap ---")
    state = build_3546_state()

    scripted = [
        FakeResponse(
            content=[FakeToolUseBlock(name="get_land_breakdown", input={}, id="t1")],
            stop_reason="tool_use",
        ),
        FakeResponse(
            content=[FakeToolUseBlock(name="get_emissions_estimate", input={}, id="t2")],
            stop_reason="tool_use",
        ),
        FakeResponse(
            content=[FakeTextBlock("Forced to wrap up. Summary: net sink.")],
            stop_reason="end_turn",
        ),
    ]

    fake = FakeClient(scripted)
    agent = ClaudeAgent(client=fake, max_turns=3)
    report = agent.run(state, "Take your time.")

    print(f"turns_used: {report.turns_used}")
    print(f"stop_reason: {report.stop_reason}")
    print(f"final_text: {report.final_text!r}")

    # Verify: the 3rd (final) API call had NO tools in it
    third_call_kwargs = fake.messages.calls[2]
    assert "tools" not in third_call_kwargs, \
        "Final turn should be called without tools to force synthesis"
    # Cache_control on system should still be present on every call
    for i, call in enumerate(fake.messages.calls):
        assert call["system"][0].get("cache_control") == {"type": "ephemeral"}, \
            f"call {i} missing system cache_control"
    # First two calls should have tools with cache_control on the last one
    for i in (0, 1):
        tools = fake.messages.calls[i]["tools"]
        assert tools[-1].get("cache_control") == {"type": "ephemeral"}, \
            f"call {i} missing tools cache_control"
        # Other tools should NOT have cache_control (only the last one)
        for t in tools[:-1]:
            assert "cache_control" not in t, \
                f"call {i}: non-last tool has cache_control"

    assert report.turns_used == 3
    assert report.final_text == "Forced to wrap up. Summary: net sink."
    print("  ✓ final turn ran without tools; cache_control placement correct")


def scenario_max_turns_with_no_end_turn() -> None:
    """
    Edge case: model keeps emitting tool_use even on the last (no-tools)
    turn. Since tools aren't provided, it physically can't — but let's
    verify the agent handles it if somehow stop_reason is still tool_use.
    """
    print("\n--- scenario: max_turns cap, model emits tool_use on final turn (shouldn't happen but) ---")
    state = build_3546_state()

    scripted = [
        FakeResponse(
            content=[FakeToolUseBlock(name="get_land_breakdown", input={}, id="t1")],
            stop_reason="tool_use",
        ),
        FakeResponse(
            # On the final turn, even though we don't provide tools, pretend
            # the model STILL tries to emit tool_use + some text.
            content=[
                FakeTextBlock("Here's what I have so far."),
                FakeToolUseBlock(name="get_emissions_estimate", input={}, id="t2"),
            ],
            stop_reason="tool_use",
        ),
    ]

    fake = FakeClient(scripted)
    agent = ClaudeAgent(client=fake, max_turns=2)
    report = agent.run(state, "Hi")

    print(f"turns_used: {report.turns_used}")
    print(f"stop_reason: {report.stop_reason}")
    print(f"final_text: {report.final_text!r}")

    # Our code short-circuits on is_final=True regardless of stop_reason,
    # so we should still get the text block extracted and stop.
    assert report.turns_used == 2
    assert "max_turns" in report.stop_reason
    assert report.final_text == "Here's what I have so far."
    print("  ✓ agent halted cleanly and extracted available text")


def main() -> None:
    scenario_happy_path()
    scenario_tool_error_recovery()
    scenario_max_turns_cap()
    scenario_max_turns_with_no_end_turn()
    print("\n=========================")
    print("All agent-loop smoke tests passed.")


if __name__ == "__main__":
    main()
