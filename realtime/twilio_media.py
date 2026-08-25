from __future__ import annotations

import base64
import html
import json
import struct
from dataclasses import dataclass
from typing import Any

from .events import (
    AssistantAudioChunkEvent,
    AssistantAudioClearEvent,
    ClientAudioEndEvent,
    ClientAudioFrameEvent,
    ClientCallEndEvent,
    ClientCallStartEvent,
)


DEFAULT_TWILIO_VOICE_PATH = "/twilio/voice"


@dataclass(slots=True)
class TwilioInboundAudioTranscoder:
    def mulaw8000_to_pcm16000_b64(self, payload_b64: str) -> str:
        mulaw = base64.b64decode(payload_b64)
        linear_8k = _mulaw_bytes_to_pcm16le(mulaw)
        linear_16k = _upsample_pcm16le_2x(linear_8k)
        return base64.b64encode(linear_16k).decode("ascii")


@dataclass(slots=True)
class TwilioOutboundAudioTranscoder:
    def pcm16000_to_mulaw8000_b64(self, payload_b64: str) -> str:
        linear_16k = base64.b64decode(payload_b64)
        linear_8k = _downsample_pcm16le_2x(linear_16k)
        mulaw = _pcm16le_to_mulaw_bytes(linear_8k)
        return base64.b64encode(mulaw).decode("ascii")


@dataclass(slots=True)
class TwilioStreamContext:
    session_id: str
    call_sid: str = ""
    stream_sid: str = ""
    account_sid: str = ""
    sequence: int = 0
    started: bool = False
    ended: bool = False


class TwilioMediaAdapter:
    def __init__(self) -> None:
        self.inbound_audio = TwilioInboundAudioTranscoder()
        self.outbound_audio = TwilioOutboundAudioTranscoder()

    def inbound_messages_to_client_events(
        self,
        payload: dict[str, Any],
        *,
        context: TwilioStreamContext,
    ) -> list[dict[str, object]]:
        event_type = str(payload.get("event", "")).strip().lower()
        if event_type == "connected":
            return []

        if event_type == "start":
            start = _mapping(payload.get("start"))
            context.call_sid = str(start.get("callSid", "")).strip()
            context.stream_sid = str(start.get("streamSid", "")).strip()
            context.account_sid = str(start.get("accountSid", "")).strip()
            custom_parameters = _mapping(start.get("customParameters"))
            context.session_id = (
                str(custom_parameters.get("session_id", "")).strip()
                or context.call_sid
                or context.stream_sid
                or context.session_id
            )
            context.started = True
            return [
                ClientCallStartEvent(
                    session_id=context.session_id,
                    call_sid=context.call_sid,
                    stream_sid=context.stream_sid,
                    from_number=str(start.get("from", "")).strip(),
                    to_number=str(start.get("to", "")).strip(),
                    provider="twilio",
                    audio_codec="mulaw",
                    sample_rate_hz=8000,
                ).as_payload()
            ]

        if event_type == "media":
            media = _mapping(payload.get("media"))
            track = str(media.get("track", "inbound")).strip().lower()
            if track and track != "inbound":
                return []
            context.sequence += 1
            return [
                ClientAudioFrameEvent(
                    session_id=context.session_id,
                    audio_b64=self.inbound_audio.mulaw8000_to_pcm16000_b64(
                        str(media.get("payload", ""))
                    ),
                    sequence=context.sequence,
                    frame_duration_ms=20,
                    sample_rate_hz=16000,
                    encoding="pcm_s16le",
                ).as_payload()
            ]

        if event_type == "mark":
            return []

        if event_type == "stop":
            context.ended = True
            return [
                ClientAudioEndEvent(
                    session_id=context.session_id,
                    reason="twilio_stop",
                ).as_payload(),
                ClientCallEndEvent(
                    session_id=context.session_id,
                    reason="twilio_stop",
                ).as_payload(),
            ]

        return []

    def server_event_to_twilio_messages(
        self,
        event: object,
        *,
        context: TwilioStreamContext,
    ) -> list[str]:
        if isinstance(event, AssistantAudioChunkEvent):
            return [
                json.dumps(
                    {
                        "event": "media",
                        "streamSid": context.stream_sid,
                        "media": {
                            "payload": self.outbound_audio.pcm16000_to_mulaw8000_b64(
                                event.audio_b64
                            )
                        },
                    }
                )
            ]
        if isinstance(event, AssistantAudioClearEvent):
            return [
                json.dumps(
                    {
                        "event": "clear",
                        "streamSid": context.stream_sid,
                    }
                )
            ]
        return []


def render_twiml_stream_response(
    *,
    stream_url: str,
    session_id: str,
) -> str:
    escaped_url = html.escape(stream_url, quote=True)
    escaped_session_id = html.escape(session_id, quote=True)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect><Stream "
        f'url="{escaped_url}">'
        f'<Parameter name="session_id" value="{escaped_session_id}" />'
        "</Stream></Connect></Response>"
    )


def default_twilio_voice_path() -> str:
    return DEFAULT_TWILIO_VOICE_PATH


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


_MU_LAW_MAX = 0x1FFF
_MU_LAW_BIAS = 0x84


def _mulaw_bytes_to_pcm16le(data: bytes) -> bytes:
    samples = bytearray()
    for value in data:
        sample = _mulaw_decode(value)
        samples.extend(struct.pack("<h", sample))
    return bytes(samples)


def _pcm16le_to_mulaw_bytes(data: bytes) -> bytes:
    if len(data) % 2 != 0:
        raise ValueError("pcm16le payload must contain an even number of bytes")
    encoded = bytearray()
    for (sample,) in struct.iter_unpack("<h", data):
        encoded.append(_mulaw_encode(sample))
    return bytes(encoded)


def _upsample_pcm16le_2x(data: bytes) -> bytes:
    if len(data) % 2 != 0:
        raise ValueError("pcm16le payload must contain an even number of bytes")
    upsampled = bytearray()
    for sample in struct.iter_unpack("<h", data):
        packed = struct.pack("<h", sample[0])
        upsampled.extend(packed)
        upsampled.extend(packed)
    return bytes(upsampled)


def _downsample_pcm16le_2x(data: bytes) -> bytes:
    if len(data) % 2 != 0:
        raise ValueError("pcm16le payload must contain an even number of bytes")
    samples = list(struct.iter_unpack("<h", data))
    downsampled = bytearray()
    for index in range(0, len(samples), 2):
        downsampled.extend(struct.pack("<h", samples[index][0]))
    return bytes(downsampled)


def _mulaw_decode(value: int) -> int:
    value = ~value & 0xFF
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    sample = ((mantissa << 3) + _MU_LAW_BIAS) << exponent
    sample -= _MU_LAW_BIAS
    return -sample if sign else sample


def _mulaw_encode(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    sample = abs(sample)
    sample = min(sample, 32635)
    sample += _MU_LAW_BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and sample & mask == 0:
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF
