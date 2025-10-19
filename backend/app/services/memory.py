"""Lightweight conversation memory utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple


@dataclass(slots=True)
class ConversationMemoryState:
    """In-memory representation of a single session."""

    history: Deque[Tuple[str, str]] = field(default_factory=deque)
    summary: str = ""


class ConversationMemoryStore:
    """Manage rolling conversation history and summary by session."""

    def __init__(self, *, max_turns: int, summary_max_chars: int = 2000) -> None:
        self._sessions: Dict[str, ConversationMemoryState] = {}
        self.max_turns = max_turns
        self.summary_max_chars = summary_max_chars

    def _get_state(self, session_id: str) -> ConversationMemoryState:
        return self._sessions.setdefault(session_id, ConversationMemoryState())

    def render(self, session_id: str) -> str:
        """Return formatted memory text for the provided session."""

        state = self._sessions.get(session_id)
        if not state or (not state.summary and not state.history):
            return "None."

        sections: List[str] = []

        summary = state.summary.strip()
        if summary:
            sections.append(f"Summary:\n{summary}")

        if state.history:
            interactions = []
            for user_text, assistant_text in state.history:
                interactions.append(f"User: {user_text}\nAssistant: {assistant_text}")
            sections.append("Recent dialogue:\n" + "\n\n".join(interactions))

        return "\n\n".join(sections)

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Record a turn and maintain rolling summary."""

        user_text = user_text.strip()
        assistant_text = assistant_text.strip()

        if not user_text and not assistant_text:
            return

        state = self._get_state(session_id)

        while len(state.history) >= self.max_turns:
            old_user, old_assistant = state.history.popleft()
            summary_piece = f"User: {old_user}\nAssistant: {old_assistant}".strip()
            if not summary_piece:
                continue
            if state.summary:
                state.summary = f"{state.summary.strip()}\n\n{summary_piece}"
            else:
                state.summary = summary_piece
            if len(state.summary) > self.summary_max_chars:
                state.summary = state.summary[-self.summary_max_chars :]

        state.history.append((user_text, assistant_text))

    def clear(self, session_id: str) -> None:
        """Remove stored memory for a session."""

        self._sessions.pop(session_id, None)
