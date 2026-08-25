from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from ..core.schemas import (
    AssistantIntent,
    ClarificationKind,
    SafetyLevel,
    SessionState,
    ToolName,
    ToolStatus,
)


class EventSchemaError(ValueError):
    pass


class ClientEventType(StrEnum):
    CALL_START = "client.call.start"
    CALL_END = "client.call.end"
    INTERRUPT = "client.interrupt"

    TEXT_INPUT = "client.text.input"
    AUDIO_FRAME = "client.audio.frame"
    AUDIO_END = "client.audio.end"

    LOCATION_UPDATE = "client.location.update"
    DESTINATION_INPUT = "client.destination.input"
    LANDMARKS_INPUT = "client.landmarks.input"

    CONFIRM = "client.confirm"
    DENY = "client.deny"

    NEXT_STEP = "client.navigation.next_step"
    REPEAT_STEP = "client.navigation.repeat_step"
    STOP_NAVIGATION = "client.navigation.stop"


class ServerEventType(StrEnum):
    SESSION_STATE = "session.state"

    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"

    INTENT_DECISION = "intent.decision"

    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_SUCCEEDED = "tool.call.succeeded"
    TOOL_CALL_FAILED = "tool.call.failed"

    LOCATION_CANDIDATES = "location.candidates"
    LOCATION_SELECTED = "location.selected"

    DESTINATION_CANDIDATES = "destination.candidates"
    DESTINATION_SELECTED = "destination.selected"

    NEARBY_LANDMARKS = "nearby_landmarks.result"

    ROUTE_READY = "route.ready"
    ROUTE_STEP = "route.step"
    ARRIVAL = "navigation.arrival"

    CLARIFICATION_REQUEST = "clarification.request"
    SAFETY_ALERT = "safety.alert"

    ASSISTANT_TEXT_DELTA = "assistant.text.delta"
    ASSISTANT_TEXT_FINAL = "assistant.text.final"
    ASSISTANT_AUDIO_CLEAR = "assistant.audio.clear"
    ASSISTANT_AUDIO_CHUNK = "assistant.audio.chunk"


@dataclass(slots=True)
class BaseWireEvent:
    session_id: str
    type: StrEnum
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise EventSchemaError("events: session_id must be a non-empty string")

        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise EventSchemaError("events: event_id must be a non-empty string")

    def base_payload(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
        }

    def as_payload(self) -> dict[str, object]:
        payload = self.base_payload()
        payload.update(self.body_payload())
        return payload

    def body_payload(self) -> dict[str, object]:
        return {}


# =========================
# Client events
# =========================

@dataclass(slots=True)
class ClientCallStartEvent(BaseWireEvent):
    call_sid: str = ""
    stream_sid: str = ""
    from_number: str = ""
    to_number: str = ""
    provider: str = "twilio"
    audio_codec: str = "mulaw"
    sample_rate_hz: int = 8_000
    type: ClientEventType = ClientEventType.CALL_START

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if self.sample_rate_hz <= 0:
            raise EventSchemaError("events: sample_rate_hz must be positive")

    def body_payload(self) -> dict[str, object]:
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "from_number": self.from_number,
            "to_number": self.to_number,
            "provider": self.provider,
            "audio_codec": self.audio_codec,
            "sample_rate_hz": self.sample_rate_hz,
        }


@dataclass(slots=True)
class ClientCallEndEvent(BaseWireEvent):
    reason: str = "call_ended"
    type: ClientEventType = ClientEventType.CALL_END

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.reason.strip():
            raise EventSchemaError("events: reason must be non-empty")

    def body_payload(self) -> dict[str, object]:
        return {"reason": self.reason}


@dataclass(slots=True)
class ClientInterruptEvent(BaseWireEvent):
    reason: str = "barge_in"
    type: ClientEventType = ClientEventType.INTERRUPT

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.reason.strip():
            raise EventSchemaError("events: reason must be non-empty")

    def body_payload(self) -> dict[str, object]:
        return {"reason": self.reason}


@dataclass(slots=True)
class ClientTextInputEvent(BaseWireEvent):
    text: str = ""
    is_final: bool = True
    source: str = "stt"
    type: ClientEventType = ClientEventType.TEXT_INPUT

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.text.strip():
            raise EventSchemaError("events: text input cannot be empty")

    def body_payload(self) -> dict[str, object]:
        return {
            "text": self.text,
            "is_final": self.is_final,
            "source": self.source,
        }


@dataclass(slots=True)
class ClientAudioFrameEvent(BaseWireEvent):
    audio_b64: str = ""
    sequence: int = 0
    frame_duration_ms: int = 20
    sample_rate_hz: int = 8_000
    encoding: str = "mulaw"
    type: ClientEventType = ClientEventType.AUDIO_FRAME

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if self.sequence < 0:
            raise EventSchemaError("events: sequence must be non-negative")

        if self.frame_duration_ms <= 0:
            raise EventSchemaError("events: frame_duration_ms must be positive")

        if self.sample_rate_hz <= 0:
            raise EventSchemaError("events: sample_rate_hz must be positive")

        _require_non_empty_b64(self.audio_b64, field_name="audio_b64")

    def body_payload(self) -> dict[str, object]:
        return {
            "audio_b64": self.audio_b64,
            "sequence": self.sequence,
            "frame_duration_ms": self.frame_duration_ms,
            "sample_rate_hz": self.sample_rate_hz,
            "encoding": self.encoding,
        }


@dataclass(slots=True)
class ClientAudioEndEvent(BaseWireEvent):
    reason: str = "end_of_turn"
    type: ClientEventType = ClientEventType.AUDIO_END

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.reason.strip():
            raise EventSchemaError("events: reason must be non-empty")

    def body_payload(self) -> dict[str, object]:
        return {"reason": self.reason}


@dataclass(slots=True)
class ClientLocationUpdateEvent(BaseWireEvent):
    lat: float = 0.0
    lng: float = 0.0
    accuracy_meters: float | None = None
    source: str = "user"
    type: ClientEventType = ClientEventType.LOCATION_UPDATE

    def body_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "lat": self.lat,
            "lng": self.lng,
            "source": self.source,
        }

        if self.accuracy_meters is not None:
            payload["accuracy_meters"] = self.accuracy_meters

        return payload


@dataclass(slots=True)
class ClientDestinationInputEvent(BaseWireEvent):
    destination_text: str = ""
    type: ClientEventType = ClientEventType.DESTINATION_INPUT

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.destination_text.strip():
            raise EventSchemaError("events: destination_text must be non-empty")

    def body_payload(self) -> dict[str, object]:
        return {"destination_text": self.destination_text}


@dataclass(slots=True)
class ClientLandmarksInputEvent(BaseWireEvent):
    landmarks: tuple[str, ...] = ()
    type: ClientEventType = ClientEventType.LANDMARKS_INPUT

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.landmarks:
            raise EventSchemaError("events: landmarks cannot be empty")

        for landmark in self.landmarks:
            if not landmark.strip():
                raise EventSchemaError("events: landmark entries must be non-empty strings")

    def body_payload(self) -> dict[str, object]:
        return {"landmarks": list(self.landmarks)}


@dataclass(slots=True)
class ClientConfirmEvent(BaseWireEvent):
    context: dict[str, object] = field(default_factory=dict)
    type: ClientEventType = ClientEventType.CONFIRM

    def body_payload(self) -> dict[str, object]:
        return {"context": self.context}


@dataclass(slots=True)
class ClientDenyEvent(BaseWireEvent):
    context: dict[str, object] = field(default_factory=dict)
    type: ClientEventType = ClientEventType.DENY

    def body_payload(self) -> dict[str, object]:
        return {"context": self.context}


@dataclass(slots=True)
class ClientNextStepEvent(BaseWireEvent):
    type: ClientEventType = ClientEventType.NEXT_STEP


@dataclass(slots=True)
class ClientRepeatStepEvent(BaseWireEvent):
    type: ClientEventType = ClientEventType.REPEAT_STEP


@dataclass(slots=True)
class ClientStopNavigationEvent(BaseWireEvent):
    reason: str = "user_requested_stop"
    type: ClientEventType = ClientEventType.STOP_NAVIGATION

    def body_payload(self) -> dict[str, object]:
        return {"reason": self.reason}


# =========================
# Server events
# =========================

@dataclass(slots=True)
class SessionStateEvent(BaseWireEvent):
    state: SessionState = SessionState.IDLE
    pending_clarification: str = ""
    safety_level: SafetyLevel = SafetyLevel.NORMAL
    type: ServerEventType = ServerEventType.SESSION_STATE

    def body_payload(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "pending_clarification": self.pending_clarification,
            "safety_level": self.safety_level.value,
        }


@dataclass(slots=True)
class TranscriptPartialEvent(BaseWireEvent):
    text: str = ""
    source: str = "stt"
    type: ServerEventType = ServerEventType.TRANSCRIPT_PARTIAL

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.text.strip():
            raise EventSchemaError("events: transcript partial text cannot be empty")

    def body_payload(self) -> dict[str, object]:
        return {
            "text": self.text,
            "source": self.source,
        }


@dataclass(slots=True)
class TranscriptFinalEvent(BaseWireEvent):
    text: str = ""
    source: str = "stt"
    type: ServerEventType = ServerEventType.TRANSCRIPT_FINAL

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.text.strip():
            raise EventSchemaError("events: transcript final text cannot be empty")

    def body_payload(self) -> dict[str, object]:
        return {
            "text": self.text,
            "source": self.source,
        }


@dataclass(slots=True)
class IntentDecisionEvent(BaseWireEvent):
    intent: AssistantIntent = AssistantIntent.UNKNOWN
    confidence: float = 0.0
    destination_text: str = ""
    visible_landmarks: tuple[str, ...] = ()
    route_mode: str = ""
    requires_clarification: bool = False
    clarification_kind: ClarificationKind | None = None
    safety_level: SafetyLevel = SafetyLevel.NORMAL
    reason: str = ""
    type: ServerEventType = ServerEventType.INTENT_DECISION

    def body_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "destination_text": self.destination_text,
            "visible_landmarks": list(self.visible_landmarks),
            "route_mode": self.route_mode,
            "requires_clarification": self.requires_clarification,
            "safety_level": self.safety_level.value,
            "reason": self.reason,
        }

        if self.clarification_kind is not None:
            payload["clarification_kind"] = self.clarification_kind.value

        return payload


@dataclass(slots=True)
class ToolCallEvent(BaseWireEvent):
    tool_name: ToolName = ToolName.GEOCODE
    status: ToolStatus = ToolStatus.STARTED
    input: dict[str, object] = field(default_factory=dict)
    output_summary: dict[str, object] = field(default_factory=dict)
    error: str = ""
    latency_ms: int | None = None
    type: ServerEventType = ServerEventType.TOOL_CALL_STARTED

    def body_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "tool_name": self.tool_name.value,
            "status": self.status.value,
            "input": self.input,
            "output_summary": self.output_summary,
            "error": self.error,
        }

        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms

        return payload


@dataclass(slots=True)
class LocationCandidatesEvent(BaseWireEvent):
    candidates: tuple[dict[str, object], ...] = ()
    type: ServerEventType = ServerEventType.LOCATION_CANDIDATES

    def body_payload(self) -> dict[str, object]:
        return {"candidates": list(self.candidates)}


@dataclass(slots=True)
class LocationSelectedEvent(BaseWireEvent):
    location: dict[str, object] = field(default_factory=dict)
    type: ServerEventType = ServerEventType.LOCATION_SELECTED

    def body_payload(self) -> dict[str, object]:
        return {"location": self.location}


@dataclass(slots=True)
class DestinationCandidatesEvent(BaseWireEvent):
    candidates: tuple[dict[str, object], ...] = ()
    type: ServerEventType = ServerEventType.DESTINATION_CANDIDATES

    def body_payload(self) -> dict[str, object]:
        return {"candidates": list(self.candidates)}


@dataclass(slots=True)
class DestinationSelectedEvent(BaseWireEvent):
    destination: dict[str, object] = field(default_factory=dict)
    type: ServerEventType = ServerEventType.DESTINATION_SELECTED

    def body_payload(self) -> dict[str, object]:
        return {"destination": self.destination}


@dataclass(slots=True)
class NearbyLandmarksEvent(BaseWireEvent):
    landmarks: tuple[dict[str, object], ...] = ()
    type: ServerEventType = ServerEventType.NEARBY_LANDMARKS

    def body_payload(self) -> dict[str, object]:
        return {"landmarks": list(self.landmarks)}


@dataclass(slots=True)
class RouteReadyEvent(BaseWireEvent):
    route: dict[str, object] = field(default_factory=dict)
    type: ServerEventType = ServerEventType.ROUTE_READY

    def body_payload(self) -> dict[str, object]:
        return {"route": self.route}


@dataclass(slots=True)
class RouteStepEvent(BaseWireEvent):
    step_index: int = 0
    total_steps: int = 0
    instruction: str = ""
    distance_meters: int = 0
    duration_seconds: int = 0
    type: ServerEventType = ServerEventType.ROUTE_STEP

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if self.step_index < 0:
            raise EventSchemaError("events: step_index cannot be negative")

        if self.total_steps < 0:
            raise EventSchemaError("events: total_steps cannot be negative")

        if not self.instruction.strip():
            raise EventSchemaError("events: route step instruction cannot be empty")

    def body_payload(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "instruction": self.instruction,
            "distance_meters": self.distance_meters,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class ArrivalEvent(BaseWireEvent):
    message: str = "You have arrived."
    type: ServerEventType = ServerEventType.ARRIVAL

    def body_payload(self) -> dict[str, object]:
        return {"message": self.message}


@dataclass(slots=True)
class ClarificationRequestEvent(BaseWireEvent):
    prompt: str = ""
    kind: ClarificationKind = ClarificationKind.CURRENT_LOCATION
    candidates: tuple[str, ...] = ()
    context: dict[str, object] = field(default_factory=dict)
    type: ServerEventType = ServerEventType.CLARIFICATION_REQUEST

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.prompt.strip():
            raise EventSchemaError("events: prompt cannot be empty")

    def body_payload(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "kind": self.kind.value,
            "candidates": list(self.candidates),
            "context": self.context,
        }


@dataclass(slots=True)
class SafetyAlertEvent(BaseWireEvent):
    safety_level: SafetyLevel = SafetyLevel.UNCERTAIN
    message: str = ""
    suggested_action: str = ""
    type: ServerEventType = ServerEventType.SAFETY_ALERT

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.message.strip():
            raise EventSchemaError("events: safety alert message cannot be empty")

    def body_payload(self) -> dict[str, object]:
        return {
            "safety_level": self.safety_level.value,
            "message": self.message,
            "suggested_action": self.suggested_action,
        }


@dataclass(slots=True)
class AssistantTextDeltaEvent(BaseWireEvent):
    delta: str = ""
    turn_id: str = ""
    type: ServerEventType = ServerEventType.ASSISTANT_TEXT_DELTA

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.delta.strip():
            raise EventSchemaError("events: delta cannot be empty")

    def body_payload(self) -> dict[str, object]:
        return {
            "delta": self.delta,
            "turn_id": self.turn_id,
        }


@dataclass(slots=True)
class AssistantTextFinalEvent(BaseWireEvent):
    text: str = ""
    turn_id: str = ""
    type: ServerEventType = ServerEventType.ASSISTANT_TEXT_FINAL

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if not self.text.strip():
            raise EventSchemaError("events: final text cannot be empty")

    def body_payload(self) -> dict[str, object]:
        return {
            "text": self.text,
            "turn_id": self.turn_id,
        }


@dataclass(slots=True)
class AssistantAudioClearEvent(BaseWireEvent):
    turn_id: str = ""
    superseded_by_turn_id: str = ""
    reason: str = "barge_in"
    type: ServerEventType = ServerEventType.ASSISTANT_AUDIO_CLEAR

    def body_payload(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "superseded_by_turn_id": self.superseded_by_turn_id,
            "reason": self.reason,
        }


@dataclass(slots=True)
class AssistantAudioChunkEvent(BaseWireEvent):
    audio_b64: str = ""
    chunk_index: int = 0
    sample_rate_hz: int = 8_000
    encoding: str = "mulaw"
    is_final: bool = False
    type: ServerEventType = ServerEventType.ASSISTANT_AUDIO_CHUNK

    def __post_init__(self) -> None:
        BaseWireEvent.__post_init__(self)

        if self.chunk_index < 0:
            raise EventSchemaError("events: chunk_index cannot be negative")

        if self.sample_rate_hz <= 0:
            raise EventSchemaError("events: sample_rate_hz must be positive")

        _require_non_empty_b64(self.audio_b64, field_name="audio_b64")

    def body_payload(self) -> dict[str, object]:
        return {
            "audio_b64": self.audio_b64,
            "chunk_index": self.chunk_index,
            "sample_rate_hz": self.sample_rate_hz,
            "encoding": self.encoding,
            "is_final": self.is_final,
        }


ClientEvent = (
    ClientCallStartEvent
    | ClientCallEndEvent
    | ClientInterruptEvent
    | ClientTextInputEvent
    | ClientAudioFrameEvent
    | ClientAudioEndEvent
    | ClientLocationUpdateEvent
    | ClientDestinationInputEvent
    | ClientLandmarksInputEvent
    | ClientConfirmEvent
    | ClientDenyEvent
    | ClientNextStepEvent
    | ClientRepeatStepEvent
    | ClientStopNavigationEvent
)

ServerEvent = (
    SessionStateEvent
    | TranscriptPartialEvent
    | TranscriptFinalEvent
    | IntentDecisionEvent
    | ToolCallEvent
    | LocationCandidatesEvent
    | LocationSelectedEvent
    | DestinationCandidatesEvent
    | DestinationSelectedEvent
    | NearbyLandmarksEvent
    | RouteReadyEvent
    | RouteStepEvent
    | ArrivalEvent
    | ClarificationRequestEvent
    | SafetyAlertEvent
    | AssistantTextDeltaEvent
    | AssistantTextFinalEvent
    | AssistantAudioClearEvent
    | AssistantAudioChunkEvent
)


# =========================
# Parsers
# =========================

def parse_client_event(payload: dict[str, object]) -> ClientEvent:
    event_type = ClientEventType(_require_str(payload, "type"))
    common = _common_kwargs(payload)

    if event_type == ClientEventType.CALL_START:
        return ClientCallStartEvent(
            **common,
            call_sid=_optional_str(payload, "call_sid", ""),
            stream_sid=_optional_str(payload, "stream_sid", ""),
            from_number=_optional_str(payload, "from_number", ""),
            to_number=_optional_str(payload, "to_number", ""),
            provider=_optional_str(payload, "provider", "twilio"),
            audio_codec=_optional_str(payload, "audio_codec", "mulaw"),
            sample_rate_hz=_optional_int(payload, "sample_rate_hz", 8_000),
        )

    if event_type == ClientEventType.CALL_END:
        return ClientCallEndEvent(
            **common,
            reason=_optional_str(payload, "reason", "call_ended"),
        )

    if event_type == ClientEventType.INTERRUPT:
        return ClientInterruptEvent(
            **common,
            reason=_optional_str(payload, "reason", "barge_in"),
        )

    if event_type == ClientEventType.TEXT_INPUT:
        return ClientTextInputEvent(
            **common,
            text=_require_str(payload, "text"),
            is_final=bool(payload.get("is_final", True)),
            source=_optional_str(payload, "source", "stt"),
        )

    if event_type == ClientEventType.AUDIO_FRAME:
        return ClientAudioFrameEvent(
            **common,
            audio_b64=_require_str(payload, "audio_b64"),
            sequence=_optional_int(payload, "sequence", 0),
            frame_duration_ms=_optional_int(payload, "frame_duration_ms", 20),
            sample_rate_hz=_optional_int(payload, "sample_rate_hz", 8_000),
            encoding=_optional_str(payload, "encoding", "mulaw"),
        )

    if event_type == ClientEventType.AUDIO_END:
        return ClientAudioEndEvent(
            **common,
            reason=_optional_str(payload, "reason", "end_of_turn"),
        )

    if event_type == ClientEventType.LOCATION_UPDATE:
        return ClientLocationUpdateEvent(
            **common,
            lat=_require_float(payload, "lat"),
            lng=_require_float(payload, "lng"),
            accuracy_meters=_optional_float_or_none(payload, "accuracy_meters"),
            source=_optional_str(payload, "source", "user"),
        )

    if event_type == ClientEventType.DESTINATION_INPUT:
        return ClientDestinationInputEvent(
            **common,
            destination_text=_require_str(payload, "destination_text"),
        )

    if event_type == ClientEventType.LANDMARKS_INPUT:
        return ClientLandmarksInputEvent(
            **common,
            landmarks=tuple(_string_list(payload.get("landmarks", []), field_name="landmarks")),
        )

    if event_type == ClientEventType.CONFIRM:
        return ClientConfirmEvent(
            **common,
            context=_dict_payload(payload.get("context", {}), field_name="context"),
        )

    if event_type == ClientEventType.DENY:
        return ClientDenyEvent(
            **common,
            context=_dict_payload(payload.get("context", {}), field_name="context"),
        )

    if event_type == ClientEventType.NEXT_STEP:
        return ClientNextStepEvent(**common)

    if event_type == ClientEventType.REPEAT_STEP:
        return ClientRepeatStepEvent(**common)

    return ClientStopNavigationEvent(
        **common,
        reason=_optional_str(payload, "reason", "user_requested_stop"),
    )


def parse_server_event(payload: dict[str, object]) -> ServerEvent:
    event_type = ServerEventType(_require_str(payload, "type"))
    common = _common_kwargs(payload)

    if event_type == ServerEventType.SESSION_STATE:
        return SessionStateEvent(
            **common,
            state=SessionState(_require_str(payload, "state")),
            pending_clarification=_optional_str(payload, "pending_clarification", ""),
            safety_level=SafetyLevel(_optional_str(payload, "safety_level", SafetyLevel.NORMAL.value)),
        )

    if event_type == ServerEventType.TRANSCRIPT_PARTIAL:
        return TranscriptPartialEvent(
            **common,
            text=_require_str(payload, "text"),
            source=_optional_str(payload, "source", "stt"),
        )

    if event_type == ServerEventType.TRANSCRIPT_FINAL:
        return TranscriptFinalEvent(
            **common,
            text=_require_str(payload, "text"),
            source=_optional_str(payload, "source", "stt"),
        )

    if event_type == ServerEventType.INTENT_DECISION:
        clarification_value = payload.get("clarification_kind")
        clarification_kind = (
            ClarificationKind(str(clarification_value))
            if isinstance(clarification_value, str) and clarification_value.strip()
            else None
        )

        return IntentDecisionEvent(
            **common,
            intent=AssistantIntent(_optional_str(payload, "intent", AssistantIntent.UNKNOWN.value)),
            confidence=_optional_float(payload, "confidence", 0.0),
            destination_text=_optional_str(payload, "destination_text", ""),
            visible_landmarks=tuple(_string_list(payload.get("visible_landmarks", []), field_name="visible_landmarks")),
            route_mode=_optional_str(payload, "route_mode", ""),
            requires_clarification=bool(payload.get("requires_clarification", False)),
            clarification_kind=clarification_kind,
            safety_level=SafetyLevel(_optional_str(payload, "safety_level", SafetyLevel.NORMAL.value)),
            reason=_optional_str(payload, "reason", ""),
        )

    if event_type in {
        ServerEventType.TOOL_CALL_STARTED,
        ServerEventType.TOOL_CALL_SUCCEEDED,
        ServerEventType.TOOL_CALL_FAILED,
    }:
        status = {
            ServerEventType.TOOL_CALL_STARTED: ToolStatus.STARTED,
            ServerEventType.TOOL_CALL_SUCCEEDED: ToolStatus.SUCCEEDED,
            ServerEventType.TOOL_CALL_FAILED: ToolStatus.FAILED,
        }[event_type]

        return ToolCallEvent(
            **common,
            type=event_type,
            tool_name=ToolName(_require_str(payload, "tool_name")),
            status=status,
            input=_dict_payload(payload.get("input", {}), field_name="input"),
            output_summary=_dict_payload(payload.get("output_summary", {}), field_name="output_summary"),
            error=_optional_str(payload, "error", ""),
            latency_ms=_optional_int_or_none(payload, "latency_ms"),
        )

    if event_type == ServerEventType.LOCATION_CANDIDATES:
        return LocationCandidatesEvent(
            **common,
            candidates=tuple(_dict_list(payload.get("candidates", []), field_name="candidates")),
        )

    if event_type == ServerEventType.LOCATION_SELECTED:
        return LocationSelectedEvent(
            **common,
            location=_dict_payload(payload.get("location", {}), field_name="location"),
        )

    if event_type == ServerEventType.DESTINATION_CANDIDATES:
        return DestinationCandidatesEvent(
            **common,
            candidates=tuple(_dict_list(payload.get("candidates", []), field_name="candidates")),
        )

    if event_type == ServerEventType.DESTINATION_SELECTED:
        return DestinationSelectedEvent(
            **common,
            destination=_dict_payload(payload.get("destination", {}), field_name="destination"),
        )

    if event_type == ServerEventType.NEARBY_LANDMARKS:
        return NearbyLandmarksEvent(
            **common,
            landmarks=tuple(_dict_list(payload.get("landmarks", []), field_name="landmarks")),
        )

    if event_type == ServerEventType.ROUTE_READY:
        return RouteReadyEvent(
            **common,
            route=_dict_payload(payload.get("route", {}), field_name="route"),
        )

    if event_type == ServerEventType.ROUTE_STEP:
        return RouteStepEvent(
            **common,
            step_index=_optional_int(payload, "step_index", 0),
            total_steps=_optional_int(payload, "total_steps", 0),
            instruction=_require_str(payload, "instruction"),
            distance_meters=_optional_int(payload, "distance_meters", 0),
            duration_seconds=_optional_int(payload, "duration_seconds", 0),
        )

    if event_type == ServerEventType.ARRIVAL:
        return ArrivalEvent(
            **common,
            message=_optional_str(payload, "message", "You have arrived."),
        )

    if event_type == ServerEventType.CLARIFICATION_REQUEST:
        return ClarificationRequestEvent(
            **common,
            prompt=_require_str(payload, "prompt"),
            kind=ClarificationKind(_optional_str(payload, "kind", ClarificationKind.CURRENT_LOCATION.value)),
            candidates=tuple(_string_list(payload.get("candidates", []), field_name="candidates")),
            context=_dict_payload(payload.get("context", {}), field_name="context"),
        )

    if event_type == ServerEventType.SAFETY_ALERT:
        return SafetyAlertEvent(
            **common,
            safety_level=SafetyLevel(_optional_str(payload, "safety_level", SafetyLevel.UNCERTAIN.value)),
            message=_require_str(payload, "message"),
            suggested_action=_optional_str(payload, "suggested_action", ""),
        )

    if event_type == ServerEventType.ASSISTANT_TEXT_DELTA:
        return AssistantTextDeltaEvent(
            **common,
            delta=_require_str(payload, "delta"),
            turn_id=_optional_str(payload, "turn_id", ""),
        )

    if event_type == ServerEventType.ASSISTANT_TEXT_FINAL:
        return AssistantTextFinalEvent(
            **common,
            text=_require_str(payload, "text"),
            turn_id=_optional_str(payload, "turn_id", ""),
        )

    if event_type == ServerEventType.ASSISTANT_AUDIO_CLEAR:
        return AssistantAudioClearEvent(
            **common,
            turn_id=_optional_str(payload, "turn_id", ""),
            superseded_by_turn_id=_optional_str(payload, "superseded_by_turn_id", ""),
            reason=_optional_str(payload, "reason", "barge_in"),
        )

    return AssistantAudioChunkEvent(
        **common,
        audio_b64=_require_str(payload, "audio_b64"),
        chunk_index=_optional_int(payload, "chunk_index", 0),
        sample_rate_hz=_optional_int(payload, "sample_rate_hz", 8_000),
        encoding=_optional_str(payload, "encoding", "mulaw"),
        is_final=bool(payload.get("is_final", False)),
    )


# =========================
# Convenience constructors
# =========================

def tool_started_event(
    *,
    session_id: str,
    tool_name: ToolName,
    input: dict[str, object],
) -> ToolCallEvent:
    return ToolCallEvent(
        session_id=session_id,
        type=ServerEventType.TOOL_CALL_STARTED,
        tool_name=tool_name,
        status=ToolStatus.STARTED,
        input=input,
    )


def tool_succeeded_event(
    *,
    session_id: str,
    tool_name: ToolName,
    input: dict[str, object],
    output_summary: dict[str, object],
    latency_ms: int | None = None,
) -> ToolCallEvent:
    return ToolCallEvent(
        session_id=session_id,
        type=ServerEventType.TOOL_CALL_SUCCEEDED,
        tool_name=tool_name,
        status=ToolStatus.SUCCEEDED,
        input=input,
        output_summary=output_summary,
        latency_ms=latency_ms,
    )


def tool_failed_event(
    *,
    session_id: str,
    tool_name: ToolName,
    input: dict[str, object],
    error: str,
    latency_ms: int | None = None,
) -> ToolCallEvent:
    return ToolCallEvent(
        session_id=session_id,
        type=ServerEventType.TOOL_CALL_FAILED,
        tool_name=tool_name,
        status=ToolStatus.FAILED,
        input=input,
        error=error,
        latency_ms=latency_ms,
    )


def route_step_event_from_route_payload(
    *,
    session_id: str,
    route: dict[str, object],
    step_index: int,
) -> RouteStepEvent:
    steps = route.get("steps", [])

    if not isinstance(steps, list):
        raise EventSchemaError("events: route steps must be a list")

    if step_index < 0 or step_index >= len(steps):
        raise EventSchemaError("events: step_index is out of range")

    step = steps[step_index]

    if not isinstance(step, dict):
        raise EventSchemaError("events: route step must be an object")

    return RouteStepEvent(
        session_id=session_id,
        step_index=step_index,
        total_steps=len(steps),
        instruction=_require_str(step, "instruction"),
        distance_meters=_optional_int(step, "distance_meters", 0),
        duration_seconds=_optional_int(step, "duration_seconds", 0),
    )


# =========================
# Helpers
# =========================

def _common_kwargs(payload: dict[str, object]) -> dict[str, object]:
    return {
        "session_id": _require_str(payload, "session_id"),
        "event_id": _optional_str(payload, "event_id", uuid4().hex),
        "timestamp": _parse_timestamp(payload.get("timestamp")),
    }


def _require_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)

    if not isinstance(value, str) or not value.strip():
        raise EventSchemaError(f"events: {key} must be a non-empty string")

    return value.strip()


def _optional_str(payload: dict[str, object], key: str, fallback: str) -> str:
    value = payload.get(key)

    if value is None:
        return fallback

    if not isinstance(value, str):
        raise EventSchemaError(f"events: {key} must be a string")

    return value


def _optional_int(payload: dict[str, object], key: str, fallback: int) -> int:
    value = payload.get(key)

    if value is None:
        return fallback

    if not isinstance(value, int):
        raise EventSchemaError(f"events: {key} must be an integer")

    return value


def _optional_int_or_none(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)

    if value is None:
        return None

    if not isinstance(value, int):
        raise EventSchemaError(f"events: {key} must be an integer")

    return value


def _require_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)

    if isinstance(value, int | float):
        return float(value)

    raise EventSchemaError(f"events: {key} must be a number")


def _optional_float(payload: dict[str, object], key: str, fallback: float) -> float:
    value = payload.get(key)

    if value is None:
        return fallback

    if isinstance(value, int | float):
        return float(value)

    raise EventSchemaError(f"events: {key} must be a number")


def _optional_float_or_none(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)

    if value is None:
        return None

    if isinstance(value, int | float):
        return float(value)

    raise EventSchemaError(f"events: {key} must be a number")


def _parse_timestamp(value: object) -> datetime:
    if value is None:
        return datetime.now().astimezone()

    if isinstance(value, datetime):
        return value

    if isinstance(value, str) and value.strip():
        return datetime.fromisoformat(value)

    raise EventSchemaError("events: timestamp must be an ISO datetime string")


def _string_list(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise EventSchemaError(f"events: {field_name} must be a list")

    result: list[str] = []

    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise EventSchemaError(f"events: {field_name} entries must be non-empty strings")

        result.append(item.strip())

    return result


def _dict_payload(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EventSchemaError(f"events: {field_name} must be an object")

    return dict(value)


def _dict_list(value: object, *, field_name: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise EventSchemaError(f"events: {field_name} must be a list")

    result: list[dict[str, object]] = []

    for item in value:
        if not isinstance(item, dict):
            raise EventSchemaError(f"events: {field_name} entries must be objects")

        result.append(dict(item))

    return result


def _require_non_empty_b64(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EventSchemaError(f"events: {field_name} must be a non-empty base64 string")

    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise EventSchemaError(f"events: {field_name} must be valid base64") from exc

    if not decoded:
        raise EventSchemaError(f"events: {field_name} must decode to non-empty bytes")
