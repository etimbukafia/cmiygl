from __future__ import annotations

from typing import Any

import asyncio
import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass
from decimal import Decimal

from tts.tts import Engine, Request

from ..core.schemas import ClarificationKind
from .events import (
    AssistantAudioChunkEvent,
    AssistantTextDeltaEvent,
    AssistantTextFinalEvent,
    ClarificationRequestEvent,
)

@dataclass(frozen=True, slots=True)
class AssistantTTSConfig:
    model_id: str = "sonic-3"
    voice_id: str = "cashier-demo"
    language: str = "en"
    text_chunk_words: int = 3
    tts_chunk_words: int = 4
    request_timeout_seconds: float = 10.0

def build_current_location_confirmation(
    *,
    session_id: str,
    location: dict[str, Any],
    user_landmarks: list[str] | tuple[str, ...] | None = None,
    prompt_prefix: str | None = None,
) -> ClarificationRequestEvent:
    """
    Build a voice-friendly confirmation prompt for the user's current location.

    Supports:
        - reverse geocoded locations
        - recovered locations based on user-provided landmarks

    Expected location shape can be flexible, for example:
        {
            "lat": 52.5200,
            "lng": 13.4050,
            "display_name": "Alexanderplatz, Mitte, Berlin, Germany",
            "name": "Alexanderplatz",
            "confidence": 0.82,
            "matched_landmarks": ["church", "fountain"]
        }
    """

    label = _location_label(location)
    confidence = _location_confidence(location)
    confidence_text = _confidence_text(confidence)

    landmarks = _landmarks_from_args_or_location(
        user_landmarks=user_landmarks,
        location=location,
    )

    evidence_text = _landmark_evidence_text(landmarks)

    if prompt_prefix is None:
        if evidence_text:
            prompt_prefix = f"Based on {evidence_text}, I think you are"
        else:
            prompt_prefix = "I think you are"

    prompt = f"{prompt_prefix} {label}{confidence_text} Is that where you are?"

    return ClarificationRequestEvent(
        session_id=session_id,
        kind=ClarificationKind.LOCATION_CONFIRMATION,
        prompt=prompt,
        candidates=(label,),
        context={
            "location": location,
            "confirmation_target": "current_location",
            "confidence": confidence,
            "user_landmarks": list(landmarks),
        },
    )


def _location_label(location: dict[str, Any]) -> str:
    name = _clean_optional_text(location.get("name"))
    display_name = _clean_optional_text(location.get("display_name"))
    approx_address = _clean_optional_text(location.get("approx_address"))

    if name and display_name:
        if display_name.lower().startswith(name.lower()):
            return f"near {display_name}"
        return f"near {name}, {display_name}"

    if display_name:
        return f"near {display_name}"

    if approx_address:
        return f"near {approx_address}"

    lat = location.get("lat")
    lng = location.get("lng")

    if isinstance(lat, int | float) and isinstance(lng, int | float):
        return f"near latitude {lat:.5f}, longitude {lng:.5f}"

    return "near the location I estimated"


def _landmarks_from_args_or_location(
    *,
    user_landmarks: list[str] | tuple[str, ...] | None,
    location: dict[str, Any],
) -> tuple[str, ...]:
    if user_landmarks:
        return tuple(
            landmark.strip()
            for landmark in user_landmarks
            if isinstance(landmark, str) and landmark.strip()
        )

    matched_landmarks = location.get("matched_landmarks")

    if isinstance(matched_landmarks, list | tuple):
        return tuple(
            landmark.strip()
            for landmark in matched_landmarks
            if isinstance(landmark, str) and landmark.strip()
        )

    return ()


def _landmark_evidence_text(landmarks: tuple[str, ...]) -> str:
    if not landmarks:
        return ""

    if len(landmarks) == 1:
        return f"the {landmarks[0]} you mentioned"

    if len(landmarks) == 2:
        return f"the {landmarks[0]} and {landmarks[1]} you mentioned"

    first_items = ", ".join(landmarks[:-1])
    return f"the {first_items}, and {landmarks[-1]} you mentioned"

def _location_confidence(location: dict[str, Any]) -> float | None:
    confidence = _float_or_none(location.get("confidence"))
    if confidence is None:
        return None

    return _clamp_confidence(confidence)


def _confidence_text(confidence: object) -> str:
    if not isinstance(confidence, int | float):
        return ""

    if confidence >= 0.85:
        return " with high confidence."

    if confidence >= 0.6:
        return " with moderate confidence."

    return ", but I am not fully sure."


def build_destination_confirmation(
    *,
    session_id: str,
    destination: dict[str, Any],
    prompt_prefix: str = "I found this destination.",
) -> ClarificationRequestEvent:
    """
    Build a voice-friendly confirmation prompt for the user's destination.

    The destination dict may include:
        {
            "lat": 52.54274275,
            "lng": 13.36690305710228,
            "display_name": "Ditsch, Lindower Straße, Wedding, Berlin, Germany",
            "name": "Ditsch",
            "confidence": 0.78,
            "importance": 0.42,
            "place_rank": 30
        }
    """

    label = _destination_label(destination)
    confidence = _destination_confidence(destination)
    confidence_text = _confidence_text(confidence)

    prompt = f"{prompt_prefix} {label}{confidence_text} Is that where you want to go?"

    return ClarificationRequestEvent(
        session_id=session_id,
        kind=ClarificationKind.DESTINATION,
        prompt=prompt,
        candidates=(label,),
        context={
            "destination": destination,
            "confirmation_target": "destination",
            "confidence": confidence,
        },
    )


def _destination_label(destination: dict[str, Any]) -> str:
    name = _clean_optional_text(destination.get("name"))
    display_name = _clean_optional_text(destination.get("display_name"))
    approx_address = _clean_optional_text(destination.get("approx_address"))

    if name and display_name:
        if display_name.lower().startswith(name.lower()):
            return display_name
        return f"{name}, {display_name}"

    if display_name:
        return display_name

    if approx_address:
        return approx_address

    lat = destination.get("lat")
    lng = destination.get("lng")

    if isinstance(lat, int | float) and isinstance(lng, int | float):
        return f"latitude {lat:.5f}, longitude {lng:.5f}"

    return "the destination I found"


def _destination_confidence(destination: dict[str, Any]) -> float | None:
    """
    Prefer an explicit app-level confidence score.

    If unavailable, use Nominatim's importance as a rough signal.
    Nominatim importance is not exactly confidence, but it can help decide
    whether to sound certain or cautious.
    """

    explicit_confidence = _float_or_none(destination.get("confidence"))
    if explicit_confidence is not None:
        return _clamp_confidence(explicit_confidence)

    importance = _float_or_none(destination.get("importance"))
    if importance is not None:
        return _clamp_confidence(importance)

    return None

def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None

    return None


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clean_optional_text(value: object) -> str:
    if not isinstance(value, str):
        return ""

    return value.strip()

def build_location_not_found_message(
    *,
    session_id: str,
    user_landmarks: list[str] | tuple[str, ...] | None = None,
    last_known_location: dict[str, Any] | None = None,
    search_radius_meters: int | None = None,
) -> ClarificationRequestEvent:
    """
    Build a voice-friendly message for when the assistant cannot recover
    the user's current location from their clues.

    Returns a clarification request because the assistant needs more
    location evidence from the caller.
    """

    prompt = (
        "I could not confidently find your current location yet. "
        "Tell me more things you can see nearby, like a shop name, "
        "street sign, building name, bus stop, church, park, or junction."
    )

    context: dict[str, Any] = {
        "reason": "location_not_found",
    }

    if last_known_location is not None:
        context["last_known_location"] = last_known_location

    if search_radius_meters is not None:
        context["search_radius_meters"] = search_radius_meters

    return ClarificationRequestEvent(
        session_id=session_id,
        kind=ClarificationKind.LANDMARKS,
        prompt=prompt,
        candidates=(),
        context=context,
    )


def build_destination_not_found_message(
    *,
    session_id: str,
    destination_text: str,
    search_context: dict[str, Any] | None = None,
) -> ClarificationRequestEvent:
    """
    Build a voice-friendly message for when the assistant cannot find
    the requested destination.

    Returns a clarification request because the assistant needs a clearer
    destination name or extra details.
    """

    cleaned_destination = _clean_optional_text(destination_text)

    if cleaned_destination:
        prompt = (
            f"I could not find a clear match for {cleaned_destination}. "
            "Please say the destination again with a little more detail, "
            "like the area, street, nearby landmark, or business type."
        )
    else:
        prompt = (
            "I could not find a clear destination. "
            "Please say where you want to go, including the area, street, "
            "nearby landmark, or business type if you know it."
        )

    context: dict[str, Any] = {
        "reason": "destination_not_found",
        "destination_text": cleaned_destination,
    }

    if search_context:
        context["search_context"] = search_context

    return ClarificationRequestEvent(
        session_id=session_id,
        kind=ClarificationKind.DESTINATION,
        prompt=prompt,
        candidates=(),
        context=context,
    )


def _clean_string_sequence(
    values: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if not values:
        return ()

    return tuple(
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip()
    )


def stream_text_deltas(text: str, *, max_words: int = 4) -> list[str]:
    words = text.split()
    if not words:
        return []
    max_words = max(1, max_words)
    chunks: list[str] = []
    for index in range(0, len(words), max_words):
        chunks.append(" ".join(words[index : index + max_words]))
    return chunks


def stream_tts_inputs(text: str, *, max_words: int = 4) -> list[str]:
    words = text.split()
    if not words:
        return []
    max_words = max(1, max_words)
    chunks: list[str] = []
    for index in range(0, len(words), max_words):
        chunk = " ".join(words[index : index + max_words])
        if index + max_words < len(words):
            chunk += " "
        chunks.append(chunk)
    return chunks

async def emit_assistant_text(
    *,
    session_id: str,
    turn_id: str,
    text: str,
    out: asyncio.Queue[object],
    max_words: int = 4,
) -> int:
    chunk_count = 0
    for chunk in stream_text_deltas(text, max_words=max_words):
        await out.put(AssistantTextDeltaEvent(session_id=session_id, turn_id=turn_id, delta=chunk))
        chunk_count += 1
    await out.put(AssistantTextFinalEvent(session_id=session_id, turn_id=turn_id, text=text))
    return chunk_count

async def emit_assistant_audio(
    *,
    session_id: str,
    turn_id: str,
    text: str,
    out: asyncio.Queue[object],
    engine: Engine,
    config: AssistantTTSConfig,
) -> int:
    chunk_index = 0
    async for chunk in engine.synthesize(
        Request(
            model_id=config.model_id,
            voice_id=config.voice_id,
            language=config.language,
            context_id=turn_id,
        ),
        _single_text_stream(text, chunk_words=config.tts_chunk_words),
    ):
        if chunk.frame is None:
            continue
        await out.put(
            AssistantAudioChunkEvent(
                session_id=session_id,
                audio_b64=base64.b64encode(chunk.frame.data).decode("ascii"),
                chunk_index=chunk_index,
                sample_rate_hz=16_000,
                encoding="pcm_s16le",
                is_final=chunk.done,
            )
        )
        chunk_index += 1
    return chunk_index

async def _single_text_stream(text: str, *, chunk_words: int) -> AsyncIterator[str]:
    del chunk_words
    normalized = " ".join(text.split())
    if normalized:
        yield normalized
