from __future__ import annotations

from browser_voice.websocket import (
    collect_server_event_payloads as _collect_server_event_payloads,
    serve_event_session,
    serve_managed_event_session,
)

from .orchestrator import CmiyglSessionOrchestrator
from .session_manager import SessionRegistry


async def serve_websocket_session(
    websocket,
    session: CmiyglSessionOrchestrator,
) -> None:
    await serve_event_session(websocket, session)


async def collect_server_event_payloads(
    session: CmiyglSessionOrchestrator,
) -> list[dict[str, object]]:
    return await _collect_server_event_payloads(session)


async def serve_managed_websocket_session(
    websocket,
    registry: SessionRegistry,
) -> None:
    await serve_managed_event_session(websocket, registry)
