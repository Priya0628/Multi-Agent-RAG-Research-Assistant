# FILE: crew/agents.py
# Purpose: Define Agent class + message struct; stub run() so pipeline runs without API keys.

from dataclasses import dataclass
from typing import List, Dict, Callable, Any

@dataclass
class Msg:
    role: str
    content: str

class Agent:
    def __init__(self, name: str, system: str, tools: Dict[str, Callable[..., Any]] | None = None):
        self.name, self.system, self.tools = name, system, tools or {}

    def run(self, history: List[Msg]) -> Msg:
        """
        Stubbed agent:
        - Echoes a short 'draft' so we can test orchestration.
        - We'll replace this with a real OpenAI chat completion later.
        """
        last = history[-1].content if history else ""
        return Msg("assistant", f"[{self.name}] draft based on: {last[:200]}")
