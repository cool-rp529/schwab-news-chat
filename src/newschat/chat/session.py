"""In-memory conversation history. Bounded in both directions: turns per session and
sessions per process, because session ids are client-supplied."""

import uuid
from collections import OrderedDict, deque

from newschat.models import ChatTurn


class SessionStore:
    def __init__(self, max_turns: int = 6, max_sessions: int = 1000) -> None:
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[ChatTurn]] = OrderedDict()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def get(self, session_id: str) -> deque[ChatTurn]:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        if len(self._sessions) >= self.max_sessions:
            self._sessions.popitem(last=False)
        turns: deque[ChatTurn] = deque(maxlen=self.max_turns)
        self._sessions[session_id] = turns
        return turns

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)
