from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class SessionState(StrEnum):
    IDLE = "idle"
    CALL_CONNECTED = "call_connected"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    NEEDS_DESTINATION = "needs_destination"
    NEEDS_LOCATION = "needs_location"
    NEEDS_LANDMARKS = "needs_landmarks"
    NEEDS_CONFIRMATION = "needs_confirmation"
    RECOVERING_LOCATION = "recovering_location"
    ROUTING = "routing"
    NAVIGATING = "navigating"
    ARRIVED = "arrived"
    CALL_ENDED = "call_ended"
    FAILED = "failed"


class TranscriptEventKind(StrEnum):
    PARTIAL = "partial"
    FINAL = "final"


class AssistantIntent(StrEnum):
    UNKNOWN = "unknown"
    GREETING = "greeting"
    USER_IS_LOST = "user_is_lost"
    SET_DESTINATION = "set_destination"
    DESCRIBE_LANDMARKS = "describe_landmarks"
    CONFIRM = "confirm"
    DENY = "deny"
    REQUEST_ROUTE = "request_route"
    REQUEST_REPEAT = "request_repeat"
    REQUEST_NEXT_STEP = "request_next_step"
    REQUEST_NEARBY_LANDMARKS = "request_nearby_landmarks"
    ASK_FOR_HELP = "ask_for_help"
    END_CALL = "end_call"


class ToolName(StrEnum):
    GEOCODE = "geocode"
    REVERSE_GEOCODE = "reverse_geocode"
    NEARBY_LANDMARKS = "nearby_landmarks"
    RECOVER_LOCATION = "recover_location"
    ROUTE = "route"


class ToolStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ClarificationKind(StrEnum):
    DESTINATION = "destination"
    CURRENT_LOCATION = "current_location"
    LANDMARKS = "landmarks"
    LOCATION_CONFIRMATION = "location_confirmation"
    ROUTE_CONFIRMATION = "route_confirmation"
    SAFETY = "safety"


class CallDirection(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class NavigationMode(StrEnum):
    WALKING = "walking"
    DRIVING = "driving"
    BICYCLE = "bicycle"


class SafetyLevel(StrEnum):
    NORMAL = "normal"
    UNCERTAIN = "uncertain"
    POSSIBLY_UNSAFE = "possibly_unsafe"
    EMERGENCY = "emergency"


@dataclass(slots=True)
class SessionConfig:
    session_id: str
    assistant_name: str = "Call Me If You Get Lost"
    language: str = "en"
    default_route_mode: NavigationMode = NavigationMode.WALKING
    enable_tts: bool = True
    enable_barge_in: bool = True
    max_clarification_turns: int = 3
    default_location_recovery_radius_meters: int = 1000
    default_nearby_landmark_radius_meters: int = 250
    target_first_response_latency_ms: int = 900
    target_tool_latency_ms: int = 2500


@dataclass(slots=True)
class CallMetadata:
    call_sid: str = ""
    stream_sid: str = ""
    account_sid: str = ""
    from_number: str = ""
    to_number: str = ""
    direction: CallDirection = CallDirection.INBOUND
    provider: str = "twilio"
    started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    ended_at: datetime | None = None


@dataclass(slots=True)
class SessionMetrics:
    first_partial_latency_ms: int | None = None
    first_final_transcript_latency_ms: int | None = None
    first_assistant_text_latency_ms: int | None = None
    first_tts_audio_latency_ms: int | None = None
    first_tool_call_latency_ms: int | None = None
    first_route_ready_latency_ms: int | None = None

    total_turns: int = 0
    clarification_count: int = 0
    tool_call_count: int = 0
    failed_tool_call_count: int = 0
    location_recovery_attempt_count: int = 0
    route_request_count: int = 0
    repeated_instruction_count: int = 0
    barge_in_count: int = 0


@dataclass(slots=True)
class TurnLatencyRecord:
    turn_index: int
    transcript_text: str = ""
    source: str = "stt"
    input_started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    first_partial_at: datetime | None = None
    final_transcript_at: datetime | None = None
    intent_resolved_at: datetime | None = None
    first_tool_call_at: datetime | None = None
    first_tool_result_at: datetime | None = None
    first_assistant_text_at: datetime | None = None
    first_tts_audio_at: datetime | None = None


@dataclass(slots=True)
class TranscriptUpdate:
    kind: TranscriptEventKind
    text: str
    is_final: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now().astimezone())
    source: str = "stt"


@dataclass(slots=True)
class IntentDecision:
    intent: AssistantIntent
    transcript_text: str
    confidence: float = 0.0

    destination_text: str = ""
    visible_landmarks: list[str] = field(default_factory=list)
    route_mode: str = ""

    requires_clarification: bool = False
    clarification_kind: ClarificationKind | None = None
    reason: str = ""

    safety_level: SafetyLevel = SafetyLevel.NORMAL
    timestamp: datetime = field(default_factory=lambda: datetime.now().astimezone())


@dataclass(slots=True)
class ToolCallRecord:
    tool_name: ToolName
    status: ToolStatus
    input: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    finished_at: datetime | None = None
    latency_ms: int | None = None


@dataclass(slots=True)
class PendingClarification:
    kind: ClarificationKind
    prompt: str
    transcript_text: str = ""
    attempts: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


@dataclass(slots=True)
class PendingAssistantOffer:
    prompt: str
    offer_type: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


@dataclass(slots=True)
class AssistantSpeechEvent:
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now().astimezone())
    audio_started_at: datetime | None = None
    audio_finished_at: datetime | None = None
    interrupted: bool = False


@dataclass(slots=True)
class ConversationHistoryEntry:
    role: str
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now().astimezone())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionLogEvent:
    event: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now().astimezone())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NavigationContext:
    """
    Assistant-level navigation memory.

    This intentionally stores summaries/references rather than duplicating
    map service models.
    """

    user_location: dict[str, Any] | None = None
    last_known_location: dict[str, Any] | None = None

    destination_text: str = ""
    selected_destination: dict[str, Any] | None = None
    destination_candidates: list[dict[str, Any]] = field(default_factory=list)

    user_reported_landmarks: list[str] = field(default_factory=list)
    nearby_landmark_summaries: list[dict[str, Any]] = field(default_factory=list)

    recovered_location: dict[str, Any] | None = None
    recovered_location_candidates: list[dict[str, Any]] = field(default_factory=list)

    active_route: dict[str, Any] | None = None
    current_route_step_index: int = 0


@dataclass(slots=True)
class SessionRecord:
    config: SessionConfig
    call: CallMetadata

    state: SessionState = SessionState.IDLE
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    navigation: NavigationContext = field(default_factory=NavigationContext)

    last_transcript_text: str = ""
    last_assistant_text: str = ""

    pending_clarification: PendingClarification | None = None
    pending_assistant_offer: PendingAssistantOffer | None = None

    transcript_updates: list[TranscriptUpdate] = field(default_factory=list)
    intent_decisions: list[IntentDecision] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    assistant_speech_events: list[AssistantSpeechEvent] = field(default_factory=list)
    turn_latency_records: list[TurnLatencyRecord] = field(default_factory=list)
    conversation_history: list[ConversationHistoryEntry] = field(default_factory=list)
    logs: list[SessionLogEvent] = field(default_factory=list)

    safety_level: SafetyLevel = SafetyLevel.NORMAL
    error: str = ""

    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    updated_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def touch(self) -> None:
        self.updated_at = datetime.now().astimezone()

    def add_log(
        self,
        event: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.logs.append(
            SessionLogEvent(
                event=event,
                message=message,
                metadata=metadata or {},
            )
        )
        self.touch()

    def add_conversation_entry(
        self,
        role: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.conversation_history.append(
            ConversationHistoryEntry(
                role=role,
                text=text,
                metadata=metadata or {},
            )
        )
        self.touch()
