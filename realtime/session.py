from __future__ import annotations

from ..core.schemas import (
    CallDirection,
    CallMetadata,
    NavigationContext,
    NavigationMode,
    SessionConfig,
    SessionRecord,
    SessionState,
)


SESSION_LIFECYCLE: tuple[SessionState, ...] = (
    SessionState.IDLE,
    SessionState.CALL_CONNECTED,
    SessionState.LISTENING,
    SessionState.THINKING,
    SessionState.SPEAKING,
    SessionState.NEEDS_DESTINATION,
    SessionState.NEEDS_LOCATION,
    SessionState.NEEDS_LANDMARKS,
    SessionState.NEEDS_CONFIRMATION,
    SessionState.RECOVERING_LOCATION,
    SessionState.ROUTING,
    SessionState.NAVIGATING,
    SessionState.ARRIVED,
    SessionState.CALL_ENDED,
    SessionState.FAILED,
)


ALLOWED_STATE_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.IDLE: {
        SessionState.CALL_CONNECTED,
        SessionState.FAILED,
    },

    SessionState.CALL_CONNECTED: {
        SessionState.LISTENING,
        SessionState.SPEAKING,
        SessionState.NEEDS_DESTINATION,
        SessionState.NEEDS_LOCATION,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.LISTENING: {
        SessionState.THINKING,
        SessionState.SPEAKING,
        SessionState.NEEDS_DESTINATION,
        SessionState.NEEDS_LOCATION,
        SessionState.NEEDS_LANDMARKS,
        SessionState.NEEDS_CONFIRMATION,
        SessionState.RECOVERING_LOCATION,
        SessionState.ROUTING,
        SessionState.NAVIGATING,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.THINKING: {
        SessionState.SPEAKING,
        SessionState.NEEDS_DESTINATION,
        SessionState.NEEDS_LOCATION,
        SessionState.NEEDS_LANDMARKS,
        SessionState.NEEDS_CONFIRMATION,
        SessionState.RECOVERING_LOCATION,
        SessionState.ROUTING,
        SessionState.NAVIGATING,
        SessionState.ARRIVED,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.SPEAKING: {
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.NEEDS_DESTINATION,
        SessionState.NEEDS_LOCATION,
        SessionState.NEEDS_LANDMARKS,
        SessionState.NEEDS_CONFIRMATION,
        SessionState.NAVIGATING,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.NEEDS_DESTINATION: {
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.SPEAKING,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.NEEDS_LOCATION: {
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.SPEAKING,
        SessionState.NEEDS_LANDMARKS,
        SessionState.RECOVERING_LOCATION,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.NEEDS_LANDMARKS: {
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.SPEAKING,
        SessionState.RECOVERING_LOCATION,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.NEEDS_CONFIRMATION: {
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.SPEAKING,
        SessionState.ROUTING,
        SessionState.RECOVERING_LOCATION,
        SessionState.NEEDS_DESTINATION,
        SessionState.NEEDS_LOCATION,
        SessionState.NEEDS_LANDMARKS,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.RECOVERING_LOCATION: {
        SessionState.SPEAKING,
        SessionState.NEEDS_CONFIRMATION,
        SessionState.NEEDS_LANDMARKS,
        SessionState.ROUTING,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.ROUTING: {
        SessionState.SPEAKING,
        SessionState.NAVIGATING,
        SessionState.NEEDS_CONFIRMATION,
        SessionState.NEEDS_DESTINATION,
        SessionState.NEEDS_LOCATION,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.NAVIGATING: {
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.SPEAKING,
        SessionState.ROUTING,
        SessionState.ARRIVED,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.ARRIVED: {
        SessionState.SPEAKING,
        SessionState.CALL_ENDED,
        SessionState.FAILED,
    },

    SessionState.CALL_ENDED: set(),
    SessionState.FAILED: set(),
}


def build_session_record(
    session_id: str,
    *,
    call_sid: str = "",
    stream_sid: str = "",
    account_sid: str = "",
    from_number: str = "",
    to_number: str = "",
    direction: CallDirection = CallDirection.INBOUND,
    assistant_name: str = "Call Me If You Get Lost",
    language: str = "en",
    default_route_mode: NavigationMode = NavigationMode.WALKING,
    enable_tts: bool = True,
    enable_barge_in: bool = True,
    max_clarification_turns: int = 3,
    default_location_recovery_radius_meters: int = 1000,
    default_nearby_landmark_radius_meters: int = 250,
) -> SessionRecord:
    return SessionRecord(
        config=SessionConfig(
            session_id=session_id,
            assistant_name=assistant_name,
            language=language,
            default_route_mode=default_route_mode,
            enable_tts=enable_tts,
            enable_barge_in=enable_barge_in,
            max_clarification_turns=max_clarification_turns,
            default_location_recovery_radius_meters=default_location_recovery_radius_meters,
            default_nearby_landmark_radius_meters=default_nearby_landmark_radius_meters,
        ),
        call=CallMetadata(
            call_sid=call_sid,
            stream_sid=stream_sid,
            account_sid=account_sid,
            from_number=from_number,
            to_number=to_number,
            direction=direction,
            provider="twilio",
        ),
        navigation=NavigationContext(),
    )


def can_transition(
    current: SessionState,
    next_state: SessionState,
) -> bool:
    return next_state in ALLOWED_STATE_TRANSITIONS[current]


def require_transition(
    current: SessionState,
    next_state: SessionState,
) -> None:
    if not can_transition(current, next_state):
        raise ValueError(
            f"Invalid session state transition: {current.value} -> {next_state.value}"
        )


def transition_session(
    record: SessionRecord,
    next_state: SessionState,
) -> None:
    require_transition(record.state, next_state)
    record.state = next_state
    record.touch()
