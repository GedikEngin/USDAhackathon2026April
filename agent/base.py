"""
ReasoningAgent protocol and supporting types.

Why a protocol when we only ship one implementation?
  Per PHASE_PLAN.md locked decisions: "Agent abstraction: ReasoningAgent
  protocol; ClaudeAgent is the only implementation for pre-build. Fallback
  is a hackathon-day task." Keeping the protocol now means a hackathon-day
  teammate can add agent/fallback.py implementing ReasoningAgent without
  touching the REPL or tool wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from agent.tools import AgentState


# ---------------------------------------------------------------------------
# Turn log — structured record of what happened during an agent session
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """One tool invocation made by the agent."""
    turn: int
    name: str
    input: Dict[str, Any]
    # Either a result dict or an error string. Never both.
    result: Dict[str, Any] | None = None
    error: str | None = None


@dataclass
class AgentReport:
    """The structured output of one agent session."""
    # The final natural-language response from the model.
    final_text: str
    # All tool calls made during the session, in order.
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    # How many agentic turns we used (one turn = one model call).
    turns_used: int = 0
    # Why we stopped — "end_turn", "max_turns", "error", ...
    stop_reason: str = ""
    # Rough token / cost accounting if the adapter provides it.
    usage: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class ReasoningAgent(Protocol):
    """
    Minimal contract for an agent that can produce a sustainability report
    given pre-classified image state.

    Implementations are free to call any number of tools internally, up to
    their own max-turns cap. The agent is expected to:
      - call tools to gather data it needs
      - synthesize a grounded, cited natural-language report
      - surface model-quality caveats honestly
      - respect the hard turn cap
    """

    def run(self, state: AgentState, user_query: str) -> AgentReport:
        """
        Execute one agent session.

        Args:
            state: Pre-classified image data. Tools read from this.
            user_query: The user's question / instruction in natural language.

        Returns:
            AgentReport with the final text, tool call log, and usage stats.
        """
        ...
