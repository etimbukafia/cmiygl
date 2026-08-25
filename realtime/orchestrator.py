from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from audio.frame import AudioFrame
from stt.stt import STT
from tts.tts import Engine

from ..core.schemas import (
    AssistantIntent,
    ClarificationKind,
    IntentDecision,
    PendingClarification,
    SessionRecord,
    SessionState,
)
from ..mapService import build_map_tool_registry
from .assistant import AssistantTTSConfig, emit_assistant_audio, emit_assistant_text
from .events import (
    AssistantAudioClearEvent,
    BaseWireEvent,
    ClarificationRequestEvent,
    ClientAudioEndEvent,
    ClientAudioFrameEvent,
    ClientCallEndEvent,
    ClientCallStartEvent,
    ClientConfirmEvent,
    ClientDestinationInputEvent,
    ClientDenyEvent,
    ClientEvent,
    ClientInterruptEvent,
    ClientLandmarksInputEvent,
    ClientLocationUpdateEvent,
    ClientNextStepEvent,
    ClientRepeatStepEvent,
    ClientStopNavigationEvent,
    ClientTextInputEvent,
    EventSchemaError,
    SessionStateEvent,
    TranscriptPartialEvent,
    parse_client_event,
)
from .navigation_agent import NavigationAgent, NavigationTurnResult
from .session import build_session_record, can_transition, transition_session

_STT_IDLE_FINALIZE_SECONDS = 1.0
_LOW_CONFIDENCE_THRESHOLD = 0.45


@dataclass(slots=True)
class _AssistantSpeechJob:
    turn_id: str
    text: str


class CmiyglSessionOrchestrator:
    def __init__(
        self,
        session_id: str,
        *,
        navigation_agent: NavigationAgent | None = None,
        stt: STT | None = None,
        tts_engine: Engine | None = None,
        assistant_tts_config: AssistantTTSConfig | None = None,
    ) -> None:
        self.record = build_session_record(session_id)
        self.navigation_agent = navigation_agent or NavigationAgent(build_map_tool_registry())
        self.stt = stt
        self.tts_engine = tts_engine
        self.assistant_tts_config = assistant_tts_config or AssistantTTSConfig()

        self.last_turn_result: NavigationTurnResult | None = None
        self.outbound_events: list[BaseWireEvent] = []

        self._server_events: asyncio.Queue[BaseWireEvent | None] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._assistant_speech_queue: asyncio.Queue[_AssistantSpeechJob | None] = asyncio.Queue()
        self._completion = asyncio.Event()

        self._started = False
        self._closed = False
        self._session_end_requested = False
        self._active_audio_input = False
        self._audio_turn_closing = False
        self._audio_frame_count = 0
        self._assistant_turn_seq = 0

        self._stt_task: asyncio.Task[None] | None = None
        self._assistant_speech_task: asyncio.Task[None] | None = None
        self._assistant_audio_task: asyncio.Task[None] | None = None
        self._stt_partial_finalize_task: asyncio.Task[None] | None = None

        self._latest_partial_text = ""
        self._latest_partial_confidence = 0.0
        self._latest_partial_source = "stt"
        self._assistant_audio_turn_id = ""

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._ensure_assistant_speech_task()
        self._set_state(SessionState.CALL_CONNECTED)
        await self._publish_state(force_publish=True)
        self._set_state(SessionState.LISTENING)
        await self._publish_state(force_publish=True)

    async def handle_client_event(self, event_or_payload: ClientEvent | dict[str, object]) -> None:
        event = event_or_payload if not isinstance(event_or_payload, dict) else parse_client_event(event_or_payload)
        if isinstance(event, ClientCallStartEvent):
            await self.start()
            self.record.call.call_sid = event.call_sid
            self.record.call.stream_sid = event.stream_sid
            self.record.call.from_number = event.from_number
            self.record.call.to_number = event.to_number
            self.record.call.provider = event.provider
            self.record.call.account_sid = self.record.call.account_sid or ""
            self.record.touch()
            await self._publish_state(force_publish=True)
            return
        if isinstance(event, ClientTextInputEvent):
            await self.start()
            if event.is_final:
                await self._handle_final_transcript(
                    text=event.text,
                    source=event.source,
                    confidence=1.0,
                )
            else:
                await self._publish_server_event(
                    TranscriptPartialEvent(
                        session_id=self.record.config.session_id,
                        text=event.text,
                        source=event.source,
                    )
                )
            return
        if isinstance(event, ClientAudioFrameEvent):
            await self.start()
            await self._handle_audio_frame(event)
            return
        if isinstance(event, ClientAudioEndEvent):
            await self.start()
            await self.end_audio_input(reason=event.reason)
            return
        if isinstance(event, ClientInterruptEvent):
            await self.start()
            await self.interrupt(reason=event.reason)
            return
        if isinstance(event, ClientCallEndEvent):
            await self.finish_input(reason=event.reason)
            return
        if isinstance(event, ClientLocationUpdateEvent):
            await self.start()
            payload = ClientTextInputEvent(
                session_id=event.session_id,
                text=f"I am at {event.lat}, {event.lng}",
                source=event.source,
            )
            await self.handle_client_event(payload)
            return
        if isinstance(event, ClientDestinationInputEvent):
            await self.start()
            await self.handle_client_event(
                ClientTextInputEvent(
                    session_id=event.session_id,
                    text=f"Take me to {event.destination_text}",
                )
            )
            return
        if isinstance(event, ClientLandmarksInputEvent):
            await self.start()
            await self.handle_client_event(
                ClientTextInputEvent(
                    session_id=event.session_id,
                    text="I can see " + ", ".join(event.landmarks),
                )
            )
            return
        if isinstance(event, ClientConfirmEvent):
            await self.start()
            await self.handle_client_event(ClientTextInputEvent(session_id=event.session_id, text="yes"))
            return
        if isinstance(event, ClientDenyEvent):
            await self.start()
            await self.handle_client_event(ClientTextInputEvent(session_id=event.session_id, text="no"))
            return
        if isinstance(event, ClientNextStepEvent):
            await self.start()
            await self.handle_client_event(ClientTextInputEvent(session_id=event.session_id, text="next step"))
            return
        if isinstance(event, ClientRepeatStepEvent):
            await self.start()
            await self.handle_client_event(ClientTextInputEvent(session_id=event.session_id, text="repeat that"))
            return
        if isinstance(event, ClientStopNavigationEvent):
            await self.finish_input(reason=event.reason)
            return
        raise EventSchemaError(f"cmiygl: unsupported client event {event!r}")

    async def interrupt(self, *, reason: str = "barge_in") -> None:
        self._cancel_pending_stt_partial_finalize()
        self._latest_partial_text = ""
        if self.record.config.enable_barge_in:
            self.record.metrics.barge_in_count += 1
        if self._assistant_audio_task is not None:
            task = self._assistant_audio_task
            turn_id = self._assistant_audio_turn_id
            self._assistant_audio_task = None
            self._assistant_audio_turn_id = ""
            task.cancel()
            try:
                await task
            except BaseException:
                pass
            await self._publish_server_event(
                AssistantAudioClearEvent(
                    session_id=self.record.config.session_id,
                    turn_id=turn_id,
                    reason=reason,
                )
            )
        self._set_state(SessionState.LISTENING)
        await self._publish_state(force_publish=True)

    async def finish_input(self, *, reason: str = "client_closed") -> None:
        if self._session_end_requested:
            return
        self._session_end_requested = True
        self._cancel_pending_stt_partial_finalize()
        if self._stt_task is not None:
            self._audio_turn_closing = True
            await self._audio_queue.put(None)
            return
        await self.close(reason=reason)

    async def end_audio_input(self, *, reason: str = "end_of_turn") -> None:
        if self._stt_task is None:
            return
        self._cancel_pending_stt_partial_finalize()
        self._audio_turn_closing = True
        await self._audio_queue.put(None)

    async def close(self, *, reason: str = "server_closed") -> None:
        if self._closed:
            return
        self._cancel_pending_stt_partial_finalize()
        if self._assistant_audio_task is not None:
            self._assistant_audio_task.cancel()
            try:
                await self._assistant_audio_task
            except BaseException:
                pass
            self._assistant_audio_task = None
        if self._assistant_speech_task is not None:
            await self._assistant_speech_queue.put(None)
            try:
                await self._assistant_speech_task
            except BaseException:
                pass
            self._assistant_speech_task = None
        if self.record.state not in {SessionState.CALL_ENDED, SessionState.FAILED}:
            if can_transition(self.record.state, SessionState.CALL_ENDED):
                transition_session(self.record, SessionState.CALL_ENDED)
            else:
                self.record.state = SessionState.CALL_ENDED
                self.record.touch()
        self.record.call.ended_at = datetime.now().astimezone()
        self.record.add_log("session.closed", "Session closed", {"reason": reason})
        await self._publish_state(force_publish=True)
        self._closed = True
        await self._server_events.put(None)
        self._completion.set()

    async def wait_closed(self) -> SessionRecord:
        await self._completion.wait()
        if self._stt_task is not None:
            await self._stt_task
        return self.record

    async def iter_server_events(self) -> AsyncIterator[BaseWireEvent]:
        while True:
            event = await self._server_events.get()
            if event is None:
                return
            yield event

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def has_active_audio_input(self) -> bool:
        return self._active_audio_input or self._stt_task is not None

    def handle_text_input(
        self,
        text: str,
        *,
        source: str = "stt",
    ) -> NavigationTurnResult:
        result = self.navigation_agent.handle_turn(self.record, text, source=source)
        self.last_turn_result = result
        self.outbound_events.extend(result.events)
        return result

    async def _handle_audio_frame(self, event: ClientAudioFrameEvent) -> None:
        if self.stt is None:
            raise RuntimeError("cmiygl: STT backend is required for audio input")
        if self.record.state == SessionState.SPEAKING and self.record.config.enable_barge_in:
            await self.interrupt(reason="voice_barge_in")
        self._active_audio_input = True
        self._ensure_stt_task()
        decoded = base64.b64decode(event.audio_b64)
        self._audio_frame_count += 1
        await self._audio_queue.put(AudioFrame(data=decoded, timestamp=event.timestamp))

    def _ensure_stt_task(self) -> None:
        if self._stt_task is not None:
            return
        self._audio_turn_closing = False
        self._stt_task = asyncio.create_task(
            self._run_stt(),
            name=f"{self.record.config.session_id}-cmiygl-stt",
        )

    def _ensure_assistant_speech_task(self) -> None:
        if self._assistant_speech_task is not None:
            return
        self._assistant_speech_task = asyncio.create_task(
            self._run_assistant_speech(),
            name=f"{self.record.config.session_id}-cmiygl-assistant-speech",
        )

    async def _run_stt(self) -> None:
        assert self.stt is not None
        try:
            async for transcript in self.stt.transcribe(self._audio_stream()):
                normalized = " ".join(transcript.text.split())
                if not normalized:
                    continue
                if transcript.is_final:
                    self._cancel_pending_stt_partial_finalize()
                    await self._handle_final_transcript(
                        text=normalized,
                        source="stt",
                        confidence=transcript.confidence,
                    )
                    self._latest_partial_text = ""
                    self._latest_partial_confidence = 0.0
                    continue
                self._latest_partial_text = normalized
                self._latest_partial_confidence = transcript.confidence
                self._latest_partial_source = "stt"
                await self._publish_server_event(
                    TranscriptPartialEvent(
                        session_id=self.record.config.session_id,
                        text=normalized,
                        source="stt",
                    )
                )
                self._schedule_stt_partial_finalize()
        finally:
            self._stt_task = None
            self._active_audio_input = False
            self._audio_frame_count = 0
            if self._session_end_requested and not self._closed:
                await self.close(reason="input_finished")
            else:
                self._set_state(SessionState.LISTENING)
                await self._publish_state(force_publish=True)

    async def _audio_stream(self) -> AsyncIterator[AudioFrame]:
        while True:
            frame = await self._audio_queue.get()
            if frame is None:
                return
            yield frame

    async def _handle_final_transcript(
        self,
        *,
        text: str,
        source: str,
        confidence: float,
    ) -> NavigationTurnResult:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("cmiygl: final transcript cannot be empty")
        if self.record.state == SessionState.SPEAKING and self.record.config.enable_barge_in:
            await self.interrupt(reason="new_turn")

        if self._looks_unclear(normalized, confidence, source):
            await self._record_unclear_transcript(normalized, source)
            return await self._emit_unclear_speech_prompt()

        result = self.navigation_agent.handle_turn(self.record, normalized, source=source)
        self.last_turn_result = result
        for event in result.events:
            await self._publish_server_event(event)
        await self._assistant_speech_queue.put(
            _AssistantSpeechJob(
                turn_id=self._next_turn_id(),
                text=result.assistant_text,
            )
        )
        return result

    async def _record_unclear_transcript(self, text: str, source: str) -> None:
        clarification_kind = self._unclear_clarification_kind()
        self.record.last_transcript_text = text
        self.record.add_conversation_entry("user", text, {"source": source})
        await self._publish_server_event(
            ClarificationRequestEvent(
                session_id=self.record.config.session_id,
                kind=clarification_kind,
                prompt="I did not catch that clearly. Please say it again a little more slowly.",
                context={"reason": "low_confidence_speech"},
            )
        )
        self.record.pending_clarification = PendingClarification(
            kind=clarification_kind,
            prompt="I did not catch that clearly. Please say it again a little more slowly.",
            transcript_text=text,
            attempts=1,
            context={"reason": "low_confidence_speech"},
        )
        self.record.metrics.clarification_count += 1
        self._set_state(SessionState.NEEDS_CONFIRMATION)
        await self._publish_state(force_publish=True)

    async def _emit_unclear_speech_prompt(self) -> NavigationTurnResult:
        assistant_text = "I did not catch that clearly. Please say it again a little more slowly."
        await self._assistant_speech_queue.put(
            _AssistantSpeechJob(
                turn_id=self._next_turn_id(),
                text=assistant_text,
            )
        )
        return NavigationTurnResult(
            assistant_text=assistant_text,
            events=[],
            state=self.record.state,
            intent=self.record.intent_decisions[-1]
            if self.record.intent_decisions
            else IntentDecision(
                intent=AssistantIntent.UNKNOWN,
                transcript_text=self.record.last_transcript_text,
                requires_clarification=True,
                clarification_kind=self.record.pending_clarification.kind if self.record.pending_clarification else None,
                reason="low_confidence_speech",
            ),
        )

    async def _run_assistant_speech(self) -> None:
        while True:
            job = await self._assistant_speech_queue.get()
            if job is None:
                return
            await emit_assistant_text(
                session_id=self.record.config.session_id,
                turn_id=job.turn_id,
                text=job.text,
                out=self._server_events,
                max_words=self.assistant_tts_config.text_chunk_words,
            )
            self.record.add_log(
                "assistant.text",
                "Assistant text emitted.",
                {"turn_id": job.turn_id},
            )
            if self.tts_engine is None or not self.record.config.enable_tts:
                self._set_state(self._resume_state_after_speech())
                await self._publish_state(force_publish=True)
                continue
            if self._assistant_audio_task is not None:
                self._assistant_audio_task.cancel()
                try:
                    await self._assistant_audio_task
                except BaseException:
                    pass
            self._assistant_audio_turn_id = job.turn_id
            self._assistant_audio_task = asyncio.create_task(
                self._emit_assistant_audio_job(job),
                name=f"{self.record.config.session_id}-{job.turn_id}-cmiygl-tts",
            )

    async def _emit_assistant_audio_job(self, job: _AssistantSpeechJob) -> None:
        self._set_state(SessionState.SPEAKING)
        await self._publish_state(force_publish=True)
        try:
            await emit_assistant_audio(
                session_id=self.record.config.session_id,
                turn_id=job.turn_id,
                text=job.text,
                out=self._server_events,
                engine=self.tts_engine,
                config=self.assistant_tts_config,
            )
        except asyncio.CancelledError:
            raise
        finally:
            if self._assistant_audio_task is not None and self._assistant_audio_task.done():
                self._assistant_audio_task = None
            if not self._closed:
                self._set_state(self._resume_state_after_speech())
                await self._publish_state(force_publish=True)

    async def _publish_server_event(self, event: BaseWireEvent) -> None:
        self.outbound_events.append(event)
        await self._server_events.put(event)

    async def _publish_state(self, *, force_publish: bool = False) -> None:
        if not force_publish and self._closed:
            return
        pending = self.record.pending_clarification.kind.value if self.record.pending_clarification else ""
        await self._publish_server_event(
            SessionStateEvent(
                session_id=self.record.config.session_id,
                state=self.record.state,
                pending_clarification=pending,
                safety_level=self.record.safety_level,
            )
        )

    def _set_state(self, next_state: SessionState) -> None:
        if self.record.state == next_state:
            return
        if can_transition(self.record.state, next_state):
            transition_session(self.record, next_state)
        else:
            self.record.state = next_state
            self.record.touch()

    def _schedule_stt_partial_finalize(self) -> None:
        self._cancel_pending_stt_partial_finalize()
        self._stt_partial_finalize_task = asyncio.create_task(
            self._finalize_stt_partial_after_idle(),
            name=f"{self.record.config.session_id}-cmiygl-stt-idle-finalize",
        )

    def _cancel_pending_stt_partial_finalize(self) -> None:
        if self._stt_partial_finalize_task is not None:
            self._stt_partial_finalize_task.cancel()
            self._stt_partial_finalize_task = None

    async def _finalize_stt_partial_after_idle(self) -> None:
        await asyncio.sleep(_STT_IDLE_FINALIZE_SECONDS)
        if not self._latest_partial_text:
            return
        text = self._latest_partial_text
        confidence = self._latest_partial_confidence
        self._latest_partial_text = ""
        self._latest_partial_confidence = 0.0
        await self._handle_final_transcript(
            text=text,
            source="stt_idle_finalize",
            confidence=confidence,
        )

    def _looks_unclear(self, text: str, confidence: float, source: str) -> bool:
        if source == "text":
            return False
        tokens = text.split()
        if confidence > 0.0 and confidence < _LOW_CONFIDENCE_THRESHOLD:
            return True
        return len(tokens) < 2

    def _unclear_clarification_kind(self) -> ClarificationKind:
        if not self.record.navigation.selected_destination:
            return ClarificationKind.DESTINATION
        if not self.record.navigation.user_location:
            return ClarificationKind.CURRENT_LOCATION
        return ClarificationKind.LANDMARKS

    def _next_turn_id(self) -> str:
        self._assistant_turn_seq += 1
        return f"{self.record.config.session_id}-turn-{self._assistant_turn_seq:03d}-{uuid4().hex[:8]}"

    def _resume_state_after_speech(self) -> SessionState:
        if self.record.pending_clarification is not None:
            return SessionState.NEEDS_CONFIRMATION
        if self.record.navigation.active_route is not None and self.record.state == SessionState.NAVIGATING:
            return SessionState.NAVIGATING
        if self.record.navigation.active_route is not None:
            return SessionState.NAVIGATING
        if self.record.navigation.selected_destination and not self.record.navigation.user_location:
            return SessionState.NEEDS_LOCATION
        if self.record.navigation.user_location and not self.record.navigation.selected_destination:
            return SessionState.NEEDS_DESTINATION
        return SessionState.LISTENING
