from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.schemas import SessionState

from .orchestrator import CmiyglSessionOrchestrator


class SessionManagerError(RuntimeError):
    pass


class SessionLimitExceeded(SessionManagerError):
    pass


class SessionAlreadyAttached(SessionManagerError):
    pass


class SessionRateLimitExceeded(SessionManagerError):
    pass


class SessionNotProvisioned(SessionManagerError):
    pass


@dataclass(slots=True)
class SessionPolicy:
    max_concurrent_sessions: int = 25

    # General client-side events:
    # call start/end, transcript text, confirmations, location updates, etc.
    max_client_events_per_minute: int = 180

    # Twilio Media Streams usually send frequent small audio frames.
    max_audio_frames_per_minute: int = 3600

    # For phone calls, idle TTL can be longer than browser sessions.
    idle_session_ttl_seconds: int = 90

    # Keep completed records briefly for final events/logging.
    completed_session_ttl_seconds: int = 300

    # Optional safety limit for very long calls.
    max_session_age_seconds: int = 60 * 60


@dataclass(slots=True)
class ManagedSession:
    session: CmiyglSessionOrchestrator
    connected_clients: int = 0

    call_sid: str = ""
    stream_sid: str = ""

    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    last_seen_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    event_timestamps: deque[datetime] = field(default_factory=deque)
    audio_frame_timestamps: deque[datetime] = field(default_factory=deque)

    @property
    def session_id(self) -> str:
        return self.session.record.config.session_id

    @property
    def is_attached(self) -> bool:
        return self.connected_clients > 0


class SessionRegistry:
    def __init__(
        self,
        session_factory: Callable[[str], CmiyglSessionOrchestrator],
        *,
        policy: SessionPolicy | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.policy = policy or SessionPolicy()
        self._sessions: dict[str, ManagedSession] = {}
        self._call_sid_to_session_id: dict[str, str] = {}
        self._stream_sid_to_session_id: dict[str, str] = {}

    def get(self, session_id: str) -> ManagedSession | None:
        return self._sessions.get(session_id)

    def get_by_call_sid(self, call_sid: str) -> ManagedSession | None:
        session_id = self._call_sid_to_session_id.get(call_sid)
        if session_id is None:
            return None

        return self._sessions.get(session_id)

    def get_by_stream_sid(self, stream_sid: str) -> ManagedSession | None:
        session_id = self._stream_sid_to_session_id.get(stream_sid)
        if session_id is None:
            return None

        return self._sessions.get(session_id)

    def open_session(
        self,
        session_id: str,
        *,
        call_sid: str = "",
        stream_sid: str = "",
    ) -> ManagedSession:
        managed = self._sessions.get(session_id)

        if managed is None or managed.session.is_closed:
            if self._active_session_count() >= self.policy.max_concurrent_sessions:
                raise SessionLimitExceeded(
                    "cmiygl session manager: concurrent session limit reached"
                )

            managed = ManagedSession(
                session=self.session_factory(session_id),
                call_sid=call_sid,
                stream_sid=stream_sid,
            )
            self._sessions[session_id] = managed

        elif managed.connected_clients > 0:
            raise SessionAlreadyAttached(
                f"cmiygl session manager: session_id {session_id!r} is already attached"
            )

        managed.connected_clients += 1
        managed.last_seen_at = datetime.now().astimezone()

        if call_sid:
            managed.call_sid = call_sid
            self._call_sid_to_session_id[call_sid] = session_id

        if stream_sid:
            managed.stream_sid = stream_sid
            self._stream_sid_to_session_id[stream_sid] = session_id

        return managed

    def attach_stream(
        self,
        session_id: str,
        *,
        stream_sid: str,
    ) -> ManagedSession:
        managed = self._require(session_id)

        if not stream_sid.strip():
            raise SessionManagerError("cmiygl session manager: stream_sid is required")

        managed.stream_sid = stream_sid
        managed.last_seen_at = datetime.now().astimezone()
        self._stream_sid_to_session_id[stream_sid] = session_id

        return managed

    def release_session(self, session_id: str) -> None:
        managed = self._sessions.get(session_id)
        if managed is None:
            return

        managed.connected_clients = max(0, managed.connected_clients - 1)
        managed.last_seen_at = datetime.now().astimezone()

    async def detach_session(
        self,
        session_id: str,
        *,
        reason: str = "client_disconnected",
    ) -> None:
        managed = self._sessions.get(session_id)
        if managed is None:
            return

        self.release_session(session_id)

        managed = self._sessions.get(session_id)
        if managed is None or managed.connected_clients > 0:
            return

        if managed.session.has_active_audio_input:
            await managed.session.finish_input(reason=reason)
            await managed.session.wait_closed()
            self._remove_session(session_id)
            return

        if managed.session.is_closed:
            self._remove_session(session_id)

    async def detach_call_sid(
        self,
        call_sid: str,
        *,
        reason: str = "call_ended",
    ) -> None:
        session_id = self._call_sid_to_session_id.get(call_sid)
        if session_id is None:
            return

        await self.detach_session(session_id, reason=reason)

    async def detach_stream_sid(
        self,
        stream_sid: str,
        *,
        reason: str = "stream_closed",
    ) -> None:
        session_id = self._stream_sid_to_session_id.get(stream_sid)
        if session_id is None:
            return

        await self.detach_session(session_id, reason=reason)

    def record_client_event(
        self,
        session_id: str,
        *,
        event_type: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        managed = self._require(session_id)

        now = timestamp or datetime.now().astimezone()
        managed.last_seen_at = now

        cutoff = now - timedelta(minutes=1)

        if event_type == "client.audio.frame":
            self._record_audio_frame(managed, now=now, cutoff=cutoff)
            return

        self._record_general_event(managed, now=now, cutoff=cutoff)

    def record_client_event_by_stream_sid(
        self,
        stream_sid: str,
        *,
        event_type: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        session_id = self._stream_sid_to_session_id.get(stream_sid)
        if session_id is None:
            raise SessionNotProvisioned(
                f"cmiygl session manager: unknown stream_sid {stream_sid!r}"
            )

        self.record_client_event(
            session_id,
            event_type=event_type,
            timestamp=timestamp,
        )

    async def close_session(
        self,
        session_id: str,
        *,
        reason: str = "server_closed",
    ) -> None:
        managed = self._sessions.get(session_id)
        if managed is None:
            return

        if managed.session.has_active_audio_input:
            await managed.session.finish_input(reason=reason)

        if not managed.session.is_closed:
            await managed.session.close(reason=reason)

        await managed.session.wait_closed()
        self._remove_session(session_id)

    async def cleanup(self, *, now: datetime | None = None) -> list[str]:
        now = now or datetime.now().astimezone()

        removed: list[str] = []

        idle_cutoff = now - timedelta(seconds=self.policy.idle_session_ttl_seconds)
        completed_cutoff = now - timedelta(seconds=self.policy.completed_session_ttl_seconds)
        max_age_cutoff = now - timedelta(seconds=self.policy.max_session_age_seconds)

        completed_states = {
            SessionState.ARRIVED,
            SessionState.CALL_ENDED,
            SessionState.FAILED,
        }

        for session_id, managed in list(self._sessions.items()):
            if managed.created_at <= max_age_cutoff:
                await self._finish_and_remove(
                    session_id,
                    managed,
                    reason="session_max_age_timeout",
                )
                removed.append(session_id)
                continue

            if managed.connected_clients == 0 and managed.last_seen_at <= idle_cutoff:
                await self._finish_and_remove(
                    session_id,
                    managed,
                    reason="session_idle_timeout",
                )
                removed.append(session_id)
                continue

            if (
                managed.connected_clients == 0
                and managed.session.record.updated_at <= completed_cutoff
                and managed.session.record.state in completed_states
            ):
                await self._finish_and_remove(
                    session_id,
                    managed,
                    reason="session_completed_timeout",
                )
                removed.append(session_id)
                continue

        return removed

    def active_session_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sessions))

    def active_call_sids(self) -> tuple[str, ...]:
        return tuple(sorted(self._call_sid_to_session_id))

    def active_stream_sids(self) -> tuple[str, ...]:
        return tuple(sorted(self._stream_sid_to_session_id))

    def _require(self, session_id: str) -> ManagedSession:
        managed = self._sessions.get(session_id)

        if managed is None:
            raise SessionManagerError(
                f"cmiygl session manager: unknown session_id {session_id!r}"
            )

        return managed

    def _record_audio_frame(
        self,
        managed: ManagedSession,
        *,
        now: datetime,
        cutoff: datetime,
    ) -> None:
        while (
            managed.audio_frame_timestamps
            and managed.audio_frame_timestamps[0] < cutoff
        ):
            managed.audio_frame_timestamps.popleft()

        if len(managed.audio_frame_timestamps) >= self.policy.max_audio_frames_per_minute:
            raise SessionRateLimitExceeded(
                "cmiygl session manager: audio frame rate limit exceeded"
            )

        managed.audio_frame_timestamps.append(now)

    def _record_general_event(
        self,
        managed: ManagedSession,
        *,
        now: datetime,
        cutoff: datetime,
    ) -> None:
        while managed.event_timestamps and managed.event_timestamps[0] < cutoff:
            managed.event_timestamps.popleft()

        if len(managed.event_timestamps) >= self.policy.max_client_events_per_minute:
            raise SessionRateLimitExceeded(
                "cmiygl session manager: client event rate limit exceeded"
            )

        managed.event_timestamps.append(now)

    async def _finish_and_remove(
        self,
        session_id: str,
        managed: ManagedSession,
        *,
        reason: str,
    ) -> None:
        if managed.session.has_active_audio_input:
            await managed.session.finish_input(reason=reason)

        if not managed.session.is_closed:
            await managed.session.close(reason=reason)

        await managed.session.wait_closed()
        self._remove_session(session_id)

    def _remove_session(self, session_id: str) -> None:
        managed = self._sessions.pop(session_id, None)

        if managed is None:
            return

        if managed.call_sid:
            self._call_sid_to_session_id.pop(managed.call_sid, None)

        if managed.stream_sid:
            self._stream_sid_to_session_id.pop(managed.stream_sid, None)

    def _active_session_count(self) -> int:
        return sum(
            1
            for managed in self._sessions.values()
            if managed.connected_clients > 0
        )
