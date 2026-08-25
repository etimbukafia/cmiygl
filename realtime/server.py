from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from env_bootstrap import load_repo_env
from stt import AssemblyAIStreamingSTT, DeepgramStreamingSTT, MistralVoxtralSTT
from tts import CartesiaEngine, FailoverEngine, VoxtralTTSEngine

from ..config import CmiyglConfig, STTProvider, TTSProvider, load_cmiygl_config, require_phone_runtime_config
from ..mapService import build_map_tool_registry
from .assistant import AssistantTTSConfig
from .navigation_agent import NavigationAgent
from .orchestrator import CmiyglSessionOrchestrator
from .session_manager import (
    SessionAlreadyAttached,
    SessionLimitExceeded,
    SessionManagerError,
    SessionPolicy,
    SessionRateLimitExceeded,
    SessionRegistry,
)
from .twilio_media import TwilioMediaAdapter, TwilioStreamContext, default_twilio_voice_path, render_twiml_stream_response

LOGGER = logging.getLogger("cmiygl.server")
APP_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class CmiyglServerSettings:
    host: str = "127.0.0.1"
    port: int = 8770
    cleanup_interval_seconds: float = 5.0
    log_level: str = "INFO"
    voice_webhook_path: str = default_twilio_voice_path()


def load_server_settings(cfg: CmiyglConfig | None = None) -> CmiyglServerSettings:
    resolved_cfg = cfg or load_cmiygl_config()
    return CmiyglServerSettings(
        host=resolved_cfg.app.host,
        port=resolved_cfg.app.port,
        cleanup_interval_seconds=5.0,
        log_level=resolved_cfg.app.log_level,
    )


def build_session_registry(
    *,
    session_factory: Callable[[str], CmiyglSessionOrchestrator] | None = None,
) -> SessionRegistry:
    policy = SessionPolicy(
        max_concurrent_sessions=25,
        max_client_events_per_minute=180,
        max_audio_frames_per_minute=3600,
        idle_session_ttl_seconds=90,
        completed_session_ttl_seconds=300,
    )
    return SessionRegistry(
        session_factory or (lambda session_id: CmiyglSessionOrchestrator(session_id)),
        policy=policy,
    )


def build_session_factory(cfg: CmiyglConfig) -> Callable[[str], CmiyglSessionOrchestrator]:
    def factory(session_id: str) -> CmiyglSessionOrchestrator:
        tts_engine, assistant_tts_config = _build_cmiygl_tts(cfg)
        return CmiyglSessionOrchestrator(
            session_id,
            stt=_build_cmiygl_stt(cfg),
            tts_engine=tts_engine,
            assistant_tts_config=assistant_tts_config,
            navigation_agent=NavigationAgent(map_tools=build_map_tool_registry(cfg)),
        )

    return factory


async def serve_managed_twilio_media_stream(
    websocket: Any,
    registry: SessionRegistry,
    *,
    adapter: TwilioMediaAdapter | None = None,
) -> None:
    resolved_adapter = adapter or TwilioMediaAdapter()
    context = TwilioStreamContext(session_id="cmiygl-twilio")
    sender: asyncio.Task[None] | None = None
    managed_session_id = ""
    try:
        async for raw in websocket:
            payload = _decode_json_message(raw)
            client_events = resolved_adapter.inbound_messages_to_client_events(payload, context=context)
            for client_payload in client_events:
                session_id = str(client_payload.get("session_id", "")).strip()
                if not session_id:
                    raise ValueError("cmiygl twilio stream: session_id is required")
                if not managed_session_id:
                    call_sid = str(client_payload.get("call_sid", "")).strip()
                    stream_sid = str(client_payload.get("stream_sid", "")).strip()
                    managed = registry.open_session(session_id, call_sid=call_sid, stream_sid=stream_sid)
                    managed_session_id = session_id
                    sender = asyncio.create_task(
                        _forward_twilio_server_events(websocket, managed.session, resolved_adapter, context),
                        name=f"{session_id}-cmiygl-twilio-sender",
                    )
                registry.record_client_event(session_id, event_type=str(client_payload.get("type", "")).strip())
                managed = registry.get(session_id)
                assert managed is not None
                await managed.session.handle_client_event(client_payload)
    finally:
        if managed_session_id:
            await registry.detach_session(managed_session_id, reason="twilio_disconnected")
        if sender is not None:
            sender.cancel()
            try:
                await sender
            except BaseException:
                pass


async def handle_twilio_connection(websocket: Any, registry: SessionRegistry) -> None:
    try:
        await serve_managed_twilio_media_stream(websocket, registry)
    except (SessionAlreadyAttached, SessionLimitExceeded, SessionRateLimitExceeded) as exc:
        LOGGER.warning("CMIYGL Twilio websocket rejected: %s", exc)
        await _close_with_reason(websocket, code=1008, reason=str(exc))
    except SessionManagerError as exc:
        LOGGER.warning("CMIYGL Twilio websocket session error: %s", exc)
        await _close_with_reason(websocket, code=1011, reason=str(exc))
    except ValueError as exc:
        LOGGER.warning("CMIYGL Twilio websocket bad payload: %s", exc)
        await _close_with_reason(websocket, code=1003, reason=str(exc))
    except RuntimeError as exc:
        LOGGER.warning("CMIYGL Twilio websocket runtime failure: %s", exc)
        await _close_with_reason(websocket, code=1011, reason=str(exc))
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("CMIYGL Twilio websocket unexpected failure")
        await _close_with_reason(websocket, code=1011, reason=str(exc))


@contextlib.asynccontextmanager
async def open_cmiygl_server(
    *,
    settings: CmiyglServerSettings | None = None,
    cfg: CmiyglConfig | None = None,
    registry: SessionRegistry | None = None,
) -> AsyncIterator[websockets.asyncio.server.Server]:
    resolved_cfg = cfg or load_cmiygl_config()
    resolved_settings = settings or load_server_settings(resolved_cfg)
    resolved_registry = registry or build_session_registry(
        session_factory=build_session_factory(resolved_cfg)
    )
    cleanup_task = asyncio.create_task(
        _run_registry_cleanup(resolved_registry, interval_seconds=resolved_settings.cleanup_interval_seconds),
        name="cmiygl-registry-cleanup",
    )
    async with websockets.serve(
        lambda websocket: handle_twilio_connection(websocket, resolved_registry),
        resolved_settings.host,
        resolved_settings.port,
        process_request=lambda conn, request: process_request(
            request,
            cfg=resolved_cfg,
            settings=resolved_settings,
        ),
    ) as server:
        try:
            yield server
        finally:
            cleanup_task.cancel()
            with contextlib.suppress(BaseException):
                await cleanup_task


async def run_server(
    settings: CmiyglServerSettings | None = None,
    cfg: CmiyglConfig | None = None,
) -> int:
    resolved_cfg = cfg or load_cmiygl_config()
    require_phone_runtime_config(resolved_cfg)
    resolved_settings = settings or load_server_settings(resolved_cfg)
    async with open_cmiygl_server(settings=resolved_settings, cfg=resolved_cfg) as server:
        sockets = _socket_labels(server)
        LOGGER.info("CMIYGL Twilio websocket listening on %s", sockets or f"{resolved_settings.host}:{resolved_settings.port}")
        await asyncio.Future()
    return 0


def process_request(
    request: Request,
    *,
    cfg: CmiyglConfig,
    settings: CmiyglServerSettings,
) -> Response | None:
    path = request.path.split("?", 1)[0]
    if path == cfg.twilio.stream_path and request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    if path == "/health":
        return _json_response(200, {"ok": True, "status": "ready"})
    if path == settings.voice_webhook_path:
        stream_url = _derive_websocket_url(
            request,
            host=settings.host,
            port=settings.port,
            path=cfg.twilio.stream_path,
        )
        body = render_twiml_stream_response(
            stream_url=stream_url,
            session_id=f"cmiygl-{id(request)}",
        ).encode("utf-8")
        return Response(
            200,
            "OK",
            Headers(
                [
                    ("Content-Type", "text/xml; charset=utf-8"),
                    ("Content-Length", str(len(body))),
                ]
            ),
            body,
        )
    return _json_response(404, {"ok": False, "error": "not_found"})


def main() -> int:
    load_repo_env(app_env_file=APP_ROOT / ".env")
    cfg = load_cmiygl_config()
    settings = load_server_settings(cfg)
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return asyncio.run(run_server(settings, cfg))
    except KeyboardInterrupt:
        LOGGER.info("CMIYGL Twilio websocket server stopped.")
        return 0


async def _forward_twilio_server_events(
    websocket: Any,
    session: CmiyglSessionOrchestrator,
    adapter: TwilioMediaAdapter,
    context: TwilioStreamContext,
) -> None:
    async for event in session.iter_server_events():
        for message in adapter.server_event_to_twilio_messages(event, context=context):
            await websocket.send(message)


async def _run_registry_cleanup(
    registry: SessionRegistry,
    *,
    interval_seconds: float,
) -> None:
    while True:
        await asyncio.sleep(max(1.0, interval_seconds))
        await registry.cleanup()


def _build_cmiygl_stt(cfg: CmiyglConfig):
    if cfg.stt.provider is STTProvider.DISABLED:
        return None
    if cfg.stt.provider is STTProvider.MISTRAL_VOXTRAL:
        return MistralVoxtralSTT(
            api_key=cfg.stt.mistral_api_key,
            model=cfg.stt.voxtral_model,
            realtime_url=cfg.stt.voxtral_realtime_url,
        )
    if cfg.stt.provider is STTProvider.DEEPGRAM:
        return DeepgramStreamingSTT(api_key=cfg.stt.deepgram_api_key)
    if cfg.stt.provider is STTProvider.ASSEMBLYAI:
        return AssemblyAIStreamingSTT(api_key=cfg.stt.assemblyai_api_key)
    return None


def _build_cmiygl_tts(cfg: CmiyglConfig):
    if cfg.tts.provider is TTSProvider.DISABLED:
        return None, None

    primary = None
    alternate = None

    if cfg.tts.cartesia_api_key and cfg.tts.cartesia_voice_id:
        primary = CartesiaEngine(cfg.tts.cartesia_api_key)
        primary.version = cfg.tts.cartesia_version
        primary.model_id = cfg.tts.cartesia_model_id
        primary.voice_id = cfg.tts.cartesia_voice_id
        primary.language = cfg.tts.cartesia_language
    if cfg.tts.mistral_api_key and cfg.tts.voxtral_tts_voice_id:
        alternate = VoxtralTTSEngine(cfg.tts.mistral_api_key)
        alternate.model_id = cfg.tts.voxtral_tts_model
        alternate.voice_id = cfg.tts.voxtral_tts_voice_id

    engine = None
    if cfg.tts.provider is TTSProvider.CARTESIA:
        engine = primary
    elif cfg.tts.provider is TTSProvider.VOXTRAL:
        engine = alternate
    elif primary is not None and alternate is not None:
        engine = FailoverEngine(primary, alternate)
    else:
        engine = primary or alternate

    if engine is None:
        return None, None

    return engine, AssistantTTSConfig(
        model_id="",
        voice_id="",
        language="",
    )


def _derive_websocket_url(request: Request, *, host: str, port: int, path: str) -> str:
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    proto = "wss" if forwarded_proto == "https" else "ws"
    authority = request.headers.get("Host", "") or f"{host}:{port}"
    return f"{proto}://{authority}{path}"


async def _close_with_reason(websocket: Any, *, code: int, reason: str) -> None:
    await websocket.close(code=code, reason=reason[:120])


def _json_response(status_code: int, payload: dict[str, object]) -> Response:
    body = json.dumps(payload).encode("utf-8")
    return Response(
        status_code,
        "OK",
        Headers(
            [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ]
        ),
        body,
    )


def _decode_json_message(raw: object) -> dict[str, object]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    if isinstance(raw, dict):
        return dict(raw)
    raise ValueError(f"cmiygl twilio stream: unsupported payload {type(raw)!r}")


def _socket_labels(server: websockets.asyncio.server.Server) -> list[str]:
    labels: list[str] = []
    for socket in getattr(server, "sockets", ()) or ():
        host, port = socket.getsockname()[:2]
        labels.append(f"{host}:{port}")
    return labels


if __name__ == "__main__":
    raise SystemExit(main())
