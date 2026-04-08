"""In-memory stub EventBus for agent tests — appends events to a list."""

from __future__ import annotations

from researcher.events import Event


class StubEventBus:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._seq = 0

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def emit(self, event: Event) -> None:
        event = event.model_copy(update={"seq": self._seq})
        self._seq += 1
        self.events.append(event)

    @property
    def dropped_socket_events(self) -> int:
        return 0

    @property
    def next_seq(self) -> int:
        return self._seq
