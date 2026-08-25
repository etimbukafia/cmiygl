from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.schemas import (
    AssistantIntent,
    ClarificationKind,
    IntentDecision,
    NavigationMode,
    PendingClarification,
    SafetyLevel,
    SessionRecord,
    SessionState,
    ToolCallRecord,
    ToolName,
    ToolStatus,
    TranscriptEventKind,
    TranscriptUpdate,
    TurnLatencyRecord,
)
from ..mapService import MapToolError, MapToolRegistry, RoutePlan
from ..mapService.mapModel import LatLng, ReverseGeocodeInput, RouteInput, SearchInput
from .assistant import (
    build_current_location_confirmation,
    build_destination_confirmation,
    build_destination_not_found_message,
    build_location_not_found_message,
)
from .events import (
    ClarificationRequestEvent,
    DestinationCandidatesEvent,
    DestinationSelectedEvent,
    IntentDecisionEvent,
    LocationCandidatesEvent,
    LocationSelectedEvent,
    RouteReadyEvent,
    SafetyAlertEvent,
    ServerEvent,
    SessionStateEvent,
    TranscriptFinalEvent,
    route_step_event_from_route_payload,
    tool_failed_event,
    tool_started_event,
    tool_succeeded_event,
)
from .session import can_transition, transition_session

_COORDINATE_RE = re.compile(r"(-?\d{1,3}\.\d+)\s*[, ]\s*(-?\d{1,3}\.\d+)")
_WORD_BOUNDARY_SPLIT_RE = re.compile(r"\s*(?:,| and | then | with | near | by | beside | next to | opposite )\s*")


class ExtractedNavigationIntent(BaseModel):
    transcript_text: str
    intent: AssistantIntent = AssistantIntent.UNKNOWN
    confidence: float = 0.0
    destination_text: str = ""
    location_text: str = ""
    visible_landmarks: list[str] = Field(default_factory=list)
    route_mode: Literal["walking", "driving", "bicycle"] | None = None
    confirmation: bool | None = None
    asks_next_step: bool = False
    asks_repeat: bool = False
    route_requested: bool = False
    end_call: bool = False
    safety_level: SafetyLevel = SafetyLevel.NORMAL
    reason: str = ""
    current_location_coordinates: tuple[float, float] | None = None


@dataclass(slots=True)
class NavigationAgentConfig:
    max_tool_calls_per_turn: int = 3
    tool_budget_timeout_ms: int = 4000
    destination_confirm_threshold: float = 0.88
    location_auto_accept_threshold: float = 0.93
    address_auto_accept_threshold: float = 0.92
    max_candidates_per_prompt: int = 3


@dataclass(slots=True)
class NavigationTurnResult:
    assistant_text: str
    events: list[ServerEvent]
    state: SessionState
    intent: IntentDecision


@dataclass(slots=True)
class _TurnToolContext:
    record: SessionRecord
    events: list[ServerEvent]
    config: NavigationAgentConfig
    seen_signatures: set[str] = field(default_factory=set)
    calls_used: int = 0
    deadline: float = 0.0

    def __post_init__(self) -> None:
        self.deadline = time.perf_counter() + (self.config.tool_budget_timeout_ms / 1000.0)


class NavigationAgent:
    def __init__(
        self,
        map_tools: MapToolRegistry,
        *,
        config: NavigationAgentConfig | None = None,
    ) -> None:
        self.map_tools = map_tools
        self.config = config or NavigationAgentConfig()

    def handle_turn(
        self,
        record: SessionRecord,
        transcript_text: str,
        *,
        source: str = "stt",
    ) -> NavigationTurnResult:
        normalized_text = " ".join(transcript_text.split())
        if not normalized_text:
            raise ValueError("navigation agent: transcript_text cannot be empty")

        turn_record = TurnLatencyRecord(
            turn_index=record.metrics.total_turns + 1,
            transcript_text=normalized_text,
            source=source,
        )
        turn_record.final_transcript_at = datetime.now().astimezone()
        record.turn_latency_records.append(turn_record)

        record.metrics.total_turns += 1
        record.last_transcript_text = normalized_text
        record.transcript_updates.append(
            TranscriptUpdate(
                kind=TranscriptEventKind.FINAL,
                text=normalized_text,
                is_final=True,
                source=source,
            )
        )
        record.add_conversation_entry("user", normalized_text, {"source": source})
        self._set_state(record, SessionState.THINKING)

        events: list[ServerEvent] = [
            TranscriptFinalEvent(
                session_id=record.config.session_id,
                text=normalized_text,
                source=source,
            )
        ]
        tool_ctx = _TurnToolContext(record=record, events=events, config=self.config)

        extracted = self.extract_intent(record, normalized_text)
        turn_record.intent_resolved_at = datetime.now().astimezone()
        record.safety_level = extracted.safety_level

        if extracted.safety_level != SafetyLevel.NORMAL:
            events.append(
                SafetyAlertEvent(
                    session_id=record.config.session_id,
                    safety_level=extracted.safety_level,
                    message=self._safety_message(extracted.safety_level),
                    suggested_action=(
                        "If you are in immediate danger, call local emergency services now."
                        if extracted.safety_level == SafetyLevel.EMERGENCY
                        else "Move to a safer visible spot and keep sharing landmarks with me."
                    ),
                )
            )

        assistant_text = ""
        reason = extracted.reason

        if extracted.end_call:
            assistant_text = "Ending the call now."
            self._set_state(record, SessionState.CALL_ENDED)
        elif extracted.asks_repeat:
            assistant_text = self._handle_repeat(record)
        elif extracted.asks_next_step:
            assistant_text = self._handle_next_step(record)
        elif extracted.confirmation is not None and record.pending_clarification is not None:
            assistant_text = self._resolve_confirmation(
                record,
                events,
                tool_ctx,
                confirmed=extracted.confirmation,
            )
        else:
            if extracted.destination_text:
                assistant_text = self._resolve_destination(
                    record,
                    events,
                    tool_ctx,
                    destination_text=extracted.destination_text,
                )
                reason = reason or "destination_resolved"

            if extracted.current_location_coordinates is not None:
                assistant_text = self._resolve_coordinate_location(
                    record,
                    events,
                    tool_ctx,
                    coordinates=extracted.current_location_coordinates,
                ) or assistant_text
                reason = reason or "coordinate_location_resolved"
            elif extracted.location_text:
                assistant_text = self._resolve_location_text(
                    record,
                    events,
                    tool_ctx,
                    location_text=extracted.location_text,
                ) or assistant_text
                reason = reason or "location_text_resolved"
            elif extracted.visible_landmarks:
                assistant_text = self._resolve_landmark_location(
                    record,
                    events,
                    tool_ctx,
                    landmarks=extracted.visible_landmarks,
                ) or assistant_text
                reason = reason or "landmark_location_resolved"

            if (
                record.navigation.selected_destination
                and record.navigation.user_location
                and record.pending_clarification is None
                and (extracted.route_requested or record.navigation.active_route is None)
            ):
                assistant_text = self._build_route(record, events, tool_ctx) or assistant_text
                reason = reason or "route_generated"

            if not assistant_text:
                assistant_text = self._fallback_prompt(record, extracted)
                reason = reason or "fallback_prompt"

        requires_clarification = record.pending_clarification is not None
        clarification_kind = record.pending_clarification.kind if record.pending_clarification else None
        intent_decision = IntentDecision(
            intent=extracted.intent,
            transcript_text=normalized_text,
            confidence=extracted.confidence,
            destination_text=extracted.destination_text,
            visible_landmarks=extracted.visible_landmarks,
            route_mode=extracted.route_mode or "",
            requires_clarification=requires_clarification,
            clarification_kind=clarification_kind,
            reason=reason,
            safety_level=extracted.safety_level,
        )
        record.intent_decisions.append(intent_decision)
        events.insert(
            1,
            IntentDecisionEvent(
                session_id=record.config.session_id,
                intent=intent_decision.intent,
                confidence=intent_decision.confidence,
                destination_text=intent_decision.destination_text,
                visible_landmarks=tuple(intent_decision.visible_landmarks),
                route_mode=intent_decision.route_mode,
                requires_clarification=intent_decision.requires_clarification,
                clarification_kind=intent_decision.clarification_kind,
                safety_level=intent_decision.safety_level,
                reason=intent_decision.reason,
            ),
        )

        record.last_assistant_text = assistant_text
        record.add_conversation_entry("assistant", assistant_text)
        turn_record.first_assistant_text_at = datetime.now().astimezone()
        if record.metrics.first_assistant_text_latency_ms is None:
            record.metrics.first_assistant_text_latency_ms = self._latency_ms(
                turn_record.input_started_at,
                turn_record.first_assistant_text_at,
            )

        record.touch()
        events.append(
            SessionStateEvent(
                session_id=record.config.session_id,
                state=record.state,
                pending_clarification=clarification_kind.value if clarification_kind else "",
                safety_level=record.safety_level,
            )
        )
        return NavigationTurnResult(
            assistant_text=assistant_text,
            events=events,
            state=record.state,
            intent=intent_decision,
        )

    def extract_intent(
        self,
        record: SessionRecord,
        transcript_text: str,
    ) -> ExtractedNavigationIntent:
        lowered = transcript_text.lower().strip()
        pending = record.pending_clarification
        safety_level = self._classify_safety_level(lowered)

        if any(phrase in lowered for phrase in ("goodbye", "end call", "hang up", "bye")):
            return ExtractedNavigationIntent(
                transcript_text=transcript_text,
                intent=AssistantIntent.END_CALL,
                confidence=0.98,
                end_call=True,
                reason="user_requested_end_call",
                safety_level=safety_level,
            )

        if any(phrase in lowered for phrase in ("repeat", "say that again", "again please")):
            return ExtractedNavigationIntent(
                transcript_text=transcript_text,
                intent=AssistantIntent.REQUEST_REPEAT,
                confidence=0.98,
                asks_repeat=True,
                reason="repeat_requested",
                safety_level=safety_level,
            )

        if any(phrase in lowered for phrase in ("next step", "what next", "what is next", "continue")):
            return ExtractedNavigationIntent(
                transcript_text=transcript_text,
                intent=AssistantIntent.REQUEST_NEXT_STEP,
                confidence=0.98,
                asks_next_step=True,
                reason="next_step_requested",
                safety_level=safety_level,
            )

        confirmation = self._match_confirmation(lowered)
        if confirmation is not None and pending is not None:
            return ExtractedNavigationIntent(
                transcript_text=transcript_text,
                intent=AssistantIntent.CONFIRM if confirmation else AssistantIntent.DENY,
                confidence=0.98,
                confirmation=confirmation,
                reason="clarification_response",
                safety_level=safety_level,
            )

        route_mode = self._extract_route_mode(lowered)
        destination_text = self._extract_destination_text(lowered, transcript_text, pending)
        location_text = self._extract_location_text(lowered, transcript_text, pending)
        visible_landmarks = self._extract_landmarks(lowered, transcript_text, pending)
        if destination_text and not self._has_explicit_landmark_phrase(lowered):
            visible_landmarks = []
        if location_text and not self._has_explicit_landmark_phrase(lowered):
            visible_landmarks = []
        coordinates = self._extract_coordinates(lowered)
        route_requested = any(
            phrase in lowered
            for phrase in (
                "show me the route",
                "give me directions",
                "route me",
                "how do i get there",
                "directions",
                "navigate",
            )
        )

        intent = AssistantIntent.UNKNOWN
        confidence = 0.55
        reason = "freeform_turn"

        if destination_text:
            intent = AssistantIntent.SET_DESTINATION
            confidence = 0.84
            reason = "destination_detected"
        elif coordinates is not None or location_text:
            intent = AssistantIntent.USER_IS_LOST
            confidence = 0.78
            reason = "current_location_detected"
        elif visible_landmarks:
            intent = AssistantIntent.DESCRIBE_LANDMARKS
            confidence = 0.82
            reason = "landmarks_detected"
        elif route_requested:
            intent = AssistantIntent.REQUEST_ROUTE
            confidence = 0.80
            reason = "route_requested"
        elif "lost" in lowered or "help me" in lowered:
            intent = AssistantIntent.ASK_FOR_HELP
            confidence = 0.72
            reason = "help_requested"

        return ExtractedNavigationIntent(
            transcript_text=transcript_text,
            intent=intent,
            confidence=confidence,
            destination_text=destination_text,
            location_text=location_text,
            visible_landmarks=visible_landmarks,
            route_mode=route_mode,
            route_requested=route_requested,
            safety_level=safety_level,
            reason=reason,
            current_location_coordinates=coordinates,
        )

    def _resolve_confirmation(
        self,
        record: SessionRecord,
        events: list[ServerEvent],
        tool_ctx: _TurnToolContext,
        *,
        confirmed: bool,
    ) -> str:
        pending = record.pending_clarification
        if pending is None:
            return "Tell me your destination or where you are now."

        context = dict(pending.context)
        record.pending_clarification = None

        if not confirmed:
            if pending.kind == ClarificationKind.DESTINATION:
                self._set_state(record, SessionState.NEEDS_DESTINATION)
                return self._queue_clarification(
                    record,
                    events,
                    build_destination_not_found_message(
                        session_id=record.config.session_id,
                        destination_text=record.navigation.destination_text,
                    ),
                )

            if pending.kind == ClarificationKind.LOCATION_CONFIRMATION:
                self._set_state(record, SessionState.NEEDS_LANDMARKS)
                return self._queue_clarification(
                    record,
                    events,
                    build_location_not_found_message(
                        session_id=record.config.session_id,
                        user_landmarks=record.navigation.user_reported_landmarks,
                        last_known_location=self._location_payload(record.navigation.last_known_location),
                    ),
                )

            return "Tell me the details again."

        if pending.kind == ClarificationKind.DESTINATION:
            destination = self._coerce_mapping(context.get("destination"))
            if destination is not None:
                record.navigation.selected_destination = destination
                events.append(
                    DestinationSelectedEvent(
                        session_id=record.config.session_id,
                        destination=destination,
                    )
                )
            if record.navigation.user_location:
                return self._build_route(record, events, tool_ctx)
            self._set_state(record, SessionState.NEEDS_LOCATION)
            return "Destination confirmed. Now tell me where you are, or describe what you can see around you."

        if pending.kind == ClarificationKind.LOCATION_CONFIRMATION:
            location = self._coerce_mapping(context.get("location"))
            if location is not None:
                self._commit_location(record, events, location=location)
            if record.navigation.selected_destination:
                return self._build_route(record, events, tool_ctx)
            self._set_state(record, SessionState.NEEDS_DESTINATION)
            return "Location confirmed. Now tell me where you want to go."

        return "Confirmed."

    def _resolve_destination(
        self,
        record: SessionRecord,
        events: list[ServerEvent],
        tool_ctx: _TurnToolContext,
        *,
        destination_text: str,
    ) -> str:
        record.navigation.destination_text = destination_text
        try:
            candidates = self._run_tool(
                tool_ctx,
                ToolName.GEOCODE,
                {"q": destination_text, "kind": "destination"},
                lambda: self.map_tools.geocode(SearchInput(q=destination_text), limit=5),
            )
        except MapToolError:
            self._set_state(record, SessionState.NEEDS_DESTINATION)
            return self._queue_clarification(
                record,
                events,
                build_destination_not_found_message(
                    session_id=record.config.session_id,
                    destination_text=destination_text,
                ),
            )

        dumped = [candidate.model_dump(mode="python") for candidate in candidates]
        record.navigation.destination_candidates = dumped
        events.append(
            DestinationCandidatesEvent(
                session_id=record.config.session_id,
                candidates=tuple(dumped[: self.config.max_candidates_per_prompt]),
            )
        )
        self._set_state(record, SessionState.NEEDS_CONFIRMATION)
        return self._queue_clarification(
            record,
            events,
            build_destination_confirmation(
                session_id=record.config.session_id,
                destination=dumped[0],
            ),
        )

    def _resolve_coordinate_location(
        self,
        record: SessionRecord,
        events: list[ServerEvent],
        tool_ctx: _TurnToolContext,
        *,
        coordinates: tuple[float, float],
    ) -> str:
        lat, lng = coordinates
        try:
            place = self._run_tool(
                tool_ctx,
                ToolName.REVERSE_GEOCODE,
                {"lat": lat, "lng": lng},
                lambda: self.map_tools.reverse_geocode(ReverseGeocodeInput(lat=lat, lng=lng)),
            )
            location = place.model_dump(mode="python")
            location["confidence"] = 1.0
        except MapToolError:
            location = {"lat": lat, "lng": lng, "confidence": 1.0}

        self._commit_location(record, events, location=location)
        if record.navigation.selected_destination:
            return ""
        self._set_state(record, SessionState.NEEDS_DESTINATION)
        return "I have your current location. Now tell me where you want to go."

    def _resolve_location_text(
        self,
        record: SessionRecord,
        events: list[ServerEvent],
        tool_ctx: _TurnToolContext,
        *,
        location_text: str,
    ) -> str:
        try:
            candidates = self._run_tool(
                tool_ctx,
                ToolName.GEOCODE,
                {"q": location_text, "kind": "current_location"},
                lambda: self.map_tools.geocode(SearchInput(q=location_text), limit=5),
            )
        except MapToolError:
            self._set_state(record, SessionState.NEEDS_LANDMARKS)
            return self._queue_clarification(
                record,
                events,
                build_location_not_found_message(
                    session_id=record.config.session_id,
                    last_known_location=self._location_payload(record.navigation.last_known_location),
                ),
            )

        dumped = [candidate.model_dump(mode="python") for candidate in candidates]
        record.navigation.recovered_location_candidates = dumped
        events.append(
            LocationCandidatesEvent(
                session_id=record.config.session_id,
                candidates=tuple(dumped[: self.config.max_candidates_per_prompt]),
            )
        )
        top = dumped[0]
        confidence = float(top.get("confidence") or top.get("rank_score") or 0.0)
        if confidence >= self.config.address_auto_accept_threshold:
            self._commit_location(record, events, location=top)
            if record.navigation.selected_destination:
                return ""
            self._set_state(record, SessionState.NEEDS_DESTINATION)
            return "I found your current location. Now tell me where you want to go."

        self._set_state(record, SessionState.NEEDS_CONFIRMATION)
        return self._queue_clarification(
            record,
            events,
            build_current_location_confirmation(
                session_id=record.config.session_id,
                location=top,
            ),
        )

    def _resolve_landmark_location(
        self,
        record: SessionRecord,
        events: list[ServerEvent],
        tool_ctx: _TurnToolContext,
        *,
        landmarks: list[str],
    ) -> str:
        record.navigation.user_reported_landmarks = landmarks
        last_known = self._latlng_from_payload(record.navigation.last_known_location)
        try:
            candidates = self._run_tool(
                tool_ctx,
                ToolName.RECOVER_LOCATION,
                {
                    "landmarks": landmarks,
                    "last_known_location": self._location_payload(record.navigation.last_known_location),
                    "search_radius": record.config.default_location_recovery_radius_meters,
                },
                lambda: self.map_tools.recover_location(
                    landmarks=landmarks,
                    last_known_location=last_known,
                    search_radius=record.config.default_location_recovery_radius_meters,
                ),
            )
        except MapToolError:
            self._set_state(record, SessionState.NEEDS_LANDMARKS)
            return self._queue_clarification(
                record,
                events,
                build_location_not_found_message(
                    session_id=record.config.session_id,
                    user_landmarks=landmarks,
                    last_known_location=self._location_payload(record.navigation.last_known_location),
                    search_radius_meters=record.config.default_location_recovery_radius_meters,
                ),
            )

        record.metrics.location_recovery_attempt_count += 1
        dumped = [candidate.model_dump(mode="python") for candidate in candidates]
        record.navigation.recovered_location_candidates = dumped
        events.append(
            LocationCandidatesEvent(
                session_id=record.config.session_id,
                candidates=tuple(dumped[: self.config.max_candidates_per_prompt]),
            )
        )
        top = dumped[0]
        confidence = float(top.get("rank_score") or top.get("confidence") or 0.0)
        if confidence >= self.config.location_auto_accept_threshold:
            self._commit_location(record, events, location=top)
            if record.navigation.selected_destination:
                return ""
            self._set_state(record, SessionState.NEEDS_DESTINATION)
            return "I estimated where you are from those landmarks. Now tell me where you want to go."

        self._set_state(record, SessionState.NEEDS_CONFIRMATION)
        return self._queue_clarification(
            record,
            events,
            build_current_location_confirmation(
                session_id=record.config.session_id,
                location=top,
                user_landmarks=landmarks,
            ),
        )

    def _build_route(
        self,
        record: SessionRecord,
        events: list[ServerEvent],
        tool_ctx: _TurnToolContext,
    ) -> str:
        origin = self._latlng_from_payload(record.navigation.user_location)
        destination = self._latlng_from_payload(record.navigation.selected_destination)
        if origin is None or destination is None:
            self._set_state(record, SessionState.NEEDS_LOCATION)
            return "I still need both your current location and your destination before I can route you."

        route_query = RouteInput(
            origin=origin,
            destination=destination,
            mode=self._route_mode_for_record(record),
        )
        try:
            route = self._run_tool(
                tool_ctx,
                ToolName.ROUTE,
                route_query.model_dump(mode="python"),
                lambda: self.map_tools.route(route_query),
            )
        except MapToolError:
            self._set_state(record, SessionState.ROUTING)
            return "I could not get a route right now. Stay where you are and tell me to try again in a moment."

        route_payload = self._route_payload(route)
        record.navigation.active_route = route_payload
        record.navigation.current_route_step_index = 0
        record.metrics.route_request_count += 1
        if record.metrics.first_route_ready_latency_ms is None:
            first_turn = record.turn_latency_records[-1]
            record.metrics.first_route_ready_latency_ms = self._latency_ms(
                first_turn.input_started_at,
                datetime.now().astimezone(),
            )

        events.append(RouteReadyEvent(session_id=record.config.session_id, route=route_payload))
        if route_payload.get("steps"):
            events.append(
                route_step_event_from_route_payload(
                    session_id=record.config.session_id,
                    route=route_payload,
                    step_index=0,
                )
            )
        self._set_state(record, SessionState.NAVIGATING)
        return self._narrate_route(route)

    def _handle_repeat(self, record: SessionRecord) -> str:
        route = record.navigation.active_route
        if not route:
            self._set_state(record, SessionState.NEEDS_DESTINATION)
            return "I do not have an active route yet. Tell me your destination first."

        steps = route.get("steps", [])
        index = min(record.navigation.current_route_step_index, max(0, len(steps) - 1))
        if not steps:
            return "I have the route, but I do not have a spoken step to repeat."

        record.metrics.repeated_instruction_count += 1
        return self._format_single_step(steps[index], prefix="Repeat: ")

    def _handle_next_step(self, record: SessionRecord) -> str:
        route = record.navigation.active_route
        if not route:
            self._set_state(record, SessionState.NEEDS_DESTINATION)
            return "I do not have an active route yet. Tell me your destination first."

        steps = route.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return "I have the route, but I do not have step details yet."

        next_index = min(record.navigation.current_route_step_index + 1, len(steps) - 1)
        record.navigation.current_route_step_index = next_index
        return self._format_single_step(steps[next_index], prefix="Next step: ")

    def _fallback_prompt(self, record: SessionRecord, extracted: ExtractedNavigationIntent) -> str:
        if record.pending_clarification is not None:
            self._set_state(record, SessionState.NEEDS_CONFIRMATION)
            return record.pending_clarification.prompt

        if not record.navigation.selected_destination:
            self._set_state(record, SessionState.NEEDS_DESTINATION)
            if extracted.safety_level == SafetyLevel.EMERGENCY:
                return "If you are in immediate danger, call local emergency services now. If you can continue, tell me where you want to go."
            return "Tell me where you want to go."

        if not record.navigation.user_location:
            self._set_state(record, SessionState.NEEDS_LOCATION)
            return "Tell me where you are now, or describe landmarks you can see."

        if not record.navigation.active_route:
            self._set_state(record, SessionState.ROUTING)
            return "I have your location and destination. Say route me when you want directions."

        self._set_state(record, SessionState.NAVIGATING)
        return "You can ask for the next step, ask me to repeat, or describe a wrong turn if you need a reroute."

    def _queue_clarification(
        self,
        record: SessionRecord,
        events: list[ServerEvent],
        clarification_event: ClarificationRequestEvent,
    ) -> str:
        previous = record.pending_clarification
        record.pending_clarification = PendingClarification(
            kind=clarification_event.kind,
            prompt=clarification_event.prompt,
            transcript_text=record.last_transcript_text,
            attempts=(previous.attempts + 1) if previous else 1,
            context=dict(clarification_event.context),
        )
        record.metrics.clarification_count += 1
        events.append(clarification_event)
        return clarification_event.prompt

    def _commit_location(
        self,
        record: SessionRecord,
        events: list[ServerEvent],
        *,
        location: dict[str, Any],
    ) -> None:
        record.navigation.user_location = dict(location)
        record.navigation.last_known_location = dict(location)
        record.navigation.recovered_location = dict(location)
        record.pending_clarification = None
        events.append(LocationSelectedEvent(session_id=record.config.session_id, location=dict(location)))
        self._set_state(record, SessionState.LISTENING)

    def _run_tool(self, tool_ctx: _TurnToolContext, tool_name: ToolName, payload: dict[str, Any], operation):
        signature = self._tool_signature(tool_name, payload)
        if signature in tool_ctx.seen_signatures:
            raise MapToolError(
                tool_name=tool_name.value,
                provider="cmiygl",
                reason="invalid_input",
                retryable=False,
                message=f"Duplicate {tool_name.value} call blocked in the same turn.",
            )
        if tool_ctx.calls_used >= self.config.max_tool_calls_per_turn:
            raise MapToolError(
                tool_name=tool_name.value,
                provider="cmiygl",
                reason="timeout",
                retryable=False,
                message="Turn tool-call budget exceeded.",
            )
        if time.perf_counter() >= tool_ctx.deadline:
            raise MapToolError(
                tool_name=tool_name.value,
                provider="cmiygl",
                reason="timeout",
                retryable=False,
                message="Turn tool budget timed out before the tool call started.",
            )

        tool_ctx.seen_signatures.add(signature)
        tool_ctx.calls_used += 1
        started_at = datetime.now().astimezone()
        tool_ctx.events.append(
            tool_started_event(
                session_id=tool_ctx.record.config.session_id,
                tool_name=tool_name,
                input=payload,
            )
        )
        started = time.perf_counter()
        try:
            result = operation()
        except MapToolError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000)
            self._append_tool_record(
                tool_ctx.record,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                payload=payload,
                output_summary={},
                error=str(exc),
                started_at=started_at,
                latency_ms=latency_ms,
            )
            tool_ctx.record.metrics.failed_tool_call_count += 1
            tool_ctx.events.append(
                tool_failed_event(
                    session_id=tool_ctx.record.config.session_id,
                    tool_name=tool_name,
                    input=payload,
                    error=str(exc),
                    latency_ms=latency_ms,
                )
            )
            raise

        latency_ms = round((time.perf_counter() - started) * 1000)
        summary = self._summarize_tool_result(result)
        self._append_tool_record(
            tool_ctx.record,
            tool_name=tool_name,
            status=ToolStatus.SUCCEEDED,
            payload=payload,
            output_summary=summary,
            error="",
            started_at=started_at,
            latency_ms=latency_ms,
        )
        tool_ctx.record.metrics.tool_call_count += 1
        if tool_ctx.record.metrics.first_tool_call_latency_ms is None:
            first_turn = tool_ctx.record.turn_latency_records[-1]
            tool_ctx.record.metrics.first_tool_call_latency_ms = self._latency_ms(first_turn.input_started_at, started_at)
        if tool_ctx.record.turn_latency_records[-1].first_tool_call_at is None:
            tool_ctx.record.turn_latency_records[-1].first_tool_call_at = started_at
        tool_ctx.record.turn_latency_records[-1].first_tool_result_at = datetime.now().astimezone()
        tool_ctx.events.append(
            tool_succeeded_event(
                session_id=tool_ctx.record.config.session_id,
                tool_name=tool_name,
                input=payload,
                output_summary=summary,
                latency_ms=latency_ms,
            )
        )
        return result

    def _append_tool_record(
        self,
        record: SessionRecord,
        *,
        tool_name: ToolName,
        status: ToolStatus,
        payload: dict[str, Any],
        output_summary: dict[str, Any],
        error: str,
        started_at: datetime,
        latency_ms: int,
    ) -> None:
        record.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                status=status,
                input=payload,
                output_summary=output_summary,
                error=error,
                started_at=started_at,
                finished_at=datetime.now().astimezone(),
                latency_ms=latency_ms,
            )
        )

    def _summarize_tool_result(self, result: Any) -> dict[str, Any]:
        if isinstance(result, list):
            first = result[0] if result else None
            if hasattr(first, "model_dump"):
                return {"count": len(result), "top": first.model_dump(mode="python")}
            return {"count": len(result)}
        if isinstance(result, RoutePlan):
            return {
                "distance_meters": result.distance_meters,
                "duration_seconds": result.duration_seconds,
                "step_count": result.step_count,
                "first_instruction": result.first_instruction,
            }
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="python")
        return {"result": str(result)}

    def _narrate_route(self, route: RoutePlan) -> str:
        step_messages = [step.instruction.strip() for step in route.steps[:2] if step.instruction.strip()]
        prefix = f"I have a {route.provider} route for about {route.distance_meters} meters, around {route.duration_seconds} seconds."
        if step_messages:
            return f"{prefix} First, {' Then, '.join(step_messages)}."
        return prefix

    def _format_single_step(self, step: dict[str, Any], *, prefix: str = "") -> str:
        instruction = str(step.get("instruction") or "").strip()
        distance = int(step.get("distance_meters") or 0)
        duration = int(step.get("duration_seconds") or 0)
        if distance > 0 or duration > 0:
            return f"{prefix}{instruction} for about {distance} meters and {duration} seconds."
        return f"{prefix}{instruction}"

    def _tool_signature(self, tool_name: ToolName, payload: dict[str, Any]) -> str:
        return f"{tool_name.value}:{json.dumps(payload, sort_keys=True, default=str)}"

    def _route_payload(self, route: RoutePlan) -> dict[str, Any]:
        return {
            "provider": route.provider,
            "distance_meters": route.distance_meters,
            "duration_seconds": route.duration_seconds,
            "step_count": route.step_count,
            "first_instruction": route.first_instruction,
            "steps": [step.model_dump(mode="python") for step in route.steps],
        }

    def _location_payload(self, location: dict[str, Any] | None) -> dict[str, Any] | None:
        return dict(location) if isinstance(location, dict) else None

    def _latlng_from_payload(self, payload: dict[str, Any] | None) -> LatLng | None:
        if not isinstance(payload, dict):
            return None
        lat = payload.get("lat")
        lng = payload.get("lng")
        if isinstance(lat, int | float) and isinstance(lng, int | float):
            return LatLng(lat=float(lat), lng=float(lng))
        return None

    def _extract_route_mode(self, lowered: str) -> Literal["walking", "driving", "bicycle"] | None:
        if any(word in lowered for word in ("walk", "walking", "on foot")):
            return "walking"
        if any(word in lowered for word in ("drive", "driving", "car")):
            return "driving"
        if any(word in lowered for word in ("bike", "bicycle", "cycling")):
            return "bicycle"
        return None

    def _extract_coordinates(self, lowered: str) -> tuple[float, float] | None:
        match = _COORDINATE_RE.search(lowered)
        if not match:
            return None
        return float(match.group(1)), float(match.group(2))

    def _extract_destination_text(
        self,
        lowered: str,
        transcript_text: str,
        pending: PendingClarification | None,
    ) -> str:
        if pending and pending.kind == ClarificationKind.DESTINATION:
            return transcript_text.strip()
        for pattern in (
            "take me to ",
            "go to ",
            "going to ",
            "headed to ",
            "heading to ",
            "destination is ",
            "destination ",
            "i want to go to ",
        ):
            index = lowered.find(pattern)
            if index >= 0:
                return transcript_text[index + len(pattern) :].strip(" .,!?")
        return ""

    def _extract_location_text(
        self,
        lowered: str,
        transcript_text: str,
        pending: PendingClarification | None,
    ) -> str:
        if pending and pending.kind == ClarificationKind.CURRENT_LOCATION:
            return transcript_text.strip()
        for pattern in (
            "i am at ",
            "i'm at ",
            "i am on ",
            "i'm on ",
            "i am near ",
            "i'm near ",
            "i am by ",
            "i'm by ",
            "my location is ",
            "i am around ",
            "i'm around ",
        ):
            index = lowered.find(pattern)
            if index >= 0:
                return transcript_text[index + len(pattern) :].strip(" .,!?")
        return ""

    def _extract_landmarks(
        self,
        lowered: str,
        transcript_text: str,
        pending: PendingClarification | None,
    ) -> list[str]:
        if pending and pending.kind == ClarificationKind.LANDMARKS:
            return self._split_landmark_text(transcript_text)

        for phrase in ("i can see ", "there is ", "i see ", "beside ", "next to ", "opposite "):
            index = lowered.find(phrase)
            if index >= 0:
                return self._split_landmark_text(transcript_text[index + len(phrase) :])

        if any(word in lowered for word in ("church", "bus stop", "junction", "park", "shop", "mall")):
            return self._split_landmark_text(transcript_text)
        return []

    def _has_explicit_landmark_phrase(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in ("i can see ", "there is ", "i see ", "beside ", "next to ", "opposite ")
        )

    def _split_landmark_text(self, text: str) -> list[str]:
        candidates = _WORD_BOUNDARY_SPLIT_RE.split(text.strip())
        cleaned: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip(" .,!?:;").strip()
            if not normalized:
                continue
            if len(normalized.split()) > 8:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(normalized)
        return cleaned

    def _match_confirmation(self, lowered: str) -> bool | None:
        if lowered in {"yes", "yeah", "yep", "correct", "that is right", "that's right"}:
            return True
        if lowered in {"no", "nope", "wrong", "that is wrong", "that's wrong"}:
            return False
        return None

    def _classify_safety_level(self, lowered: str) -> SafetyLevel:
        if any(word in lowered for word in ("attack", "bleeding", "kidnap", "danger", "emergency", "crash")):
            return SafetyLevel.EMERGENCY
        if any(word in lowered for word in ("highway", "dark", "night", "unsafe", "scared", "alone")):
            return SafetyLevel.POSSIBLY_UNSAFE
        if any(word in lowered for word in ("not sure", "confused", "unclear", "lost")):
            return SafetyLevel.UNCERTAIN
        return SafetyLevel.NORMAL

    def _safety_message(self, level: SafetyLevel) -> str:
        if level == SafetyLevel.EMERGENCY:
            return "This may be an emergency."
        if level == SafetyLevel.POSSIBLY_UNSAFE:
            return "Your situation may be unsafe."
        return "I am hearing uncertainty in your location details."

    def _route_mode_for_record(self, record: SessionRecord) -> Literal["walking", "driving", "bicycle"]:
        if record.config.default_route_mode == NavigationMode.DRIVING:
            return "driving"
        if record.config.default_route_mode == NavigationMode.BICYCLE:
            return "bicycle"
        return "walking"

    def _coerce_mapping(self, value: Any) -> dict[str, Any] | None:
        return dict(value) if isinstance(value, dict) else None

    def _set_state(self, record: SessionRecord, next_state: SessionState) -> None:
        if record.state == next_state:
            return
        if can_transition(record.state, next_state):
            transition_session(record, next_state)
            return
        record.state = next_state
        record.touch()

    def _latency_ms(self, started_at: datetime, finished_at: datetime) -> int:
        return max(0, round((finished_at - started_at).total_seconds() * 1000))
