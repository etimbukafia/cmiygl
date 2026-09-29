from __future__ import annotations

import asyncio
import base64
import json
import unittest
from collections.abc import AsyncIterable, AsyncIterator

from harness.audio import AudioFrame
from harness.stt import STT, Transcript
from harness.tts import Chunk, Engine, Request

from cmiygl.realtime.assistant import AssistantTTSConfig
from cmiygl.realtime.events import AssistantAudioChunkEvent, AssistantAudioClearEvent
from cmiygl.realtime.navigation_agent import NavigationAgent
from cmiygl.realtime.orchestrator import CmiyglSessionOrchestrator
from cmiygl.realtime.test_navigation_agent import FakeMapTools
from cmiygl.realtime.twilio_media import (
    TwilioMediaAdapter,
    TwilioStreamContext,
    render_twiml_stream_response,
)


class ScriptedSTT(STT):
    def __init__(self, transcripts: list[Transcript]) -> None:
        self.transcripts = transcripts

    async def transcribe(self, audio_stream: AsyncIterable[AudioFrame]) -> AsyncIterator[Transcript]:
        async for _ in audio_stream:
            pass
        for transcript in self.transcripts:
            yield transcript


class SlowChunkTTS(Engine):
    async def synthesize(self, req: Request, text_stream: AsyncIterable[str]) -> AsyncIterator[Chunk]:
        async for _ in text_stream:
            break
        for _ in range(3):
            await asyncio.sleep(0.02)
            yield Chunk(frame=AudioFrame(data=b"\x00" * 640), context_id=req.context_id)
        yield Chunk(context_id=req.context_id, done=True)


class RealtimePhase3Test(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.agent = NavigationAgent(FakeMapTools())

    async def _collect_events(self, orchestrator: CmiyglSessionOrchestrator) -> tuple[list[object], asyncio.Task[None]]:
        events: list[object] = []

        async def pump() -> None:
            async for event in orchestrator.iter_server_events():
                events.append(event)

        return events, asyncio.create_task(pump())

    async def test_audio_turn_runs_stt_agent_and_tts(self) -> None:
        orchestrator = CmiyglSessionOrchestrator(
            "phase3-session",
            navigation_agent=self.agent,
            stt=ScriptedSTT(
                [
                    Transcript(text="Take me to City Mall", confidence=0.92, is_final=True),
                ]
            ),
            tts_engine=SlowChunkTTS(),
            assistant_tts_config=AssistantTTSConfig(),
        )
        events, collector = await self._collect_events(orchestrator)

        await orchestrator.handle_client_event({"type": "client.call.start", "session_id": "phase3-session"})
        await orchestrator.handle_client_event(
            {
                "type": "client.audio.frame",
                "session_id": "phase3-session",
                "audio_b64": base64.b64encode(b"\x00" * 640).decode("ascii"),
                "sequence": 0,
                "sample_rate_hz": 16000,
                "encoding": "pcm_s16le",
            }
        )
        await orchestrator.handle_client_event({"type": "client.audio.end", "session_id": "phase3-session"})
        deadline = asyncio.get_running_loop().time() + 1.0
        while orchestrator.record.state.value != "needs_confirmation":
            if asyncio.get_running_loop().time() >= deadline:
                self.fail(f"assistant did not finish speaking; state={orchestrator.record.state.value!r}")
            await asyncio.sleep(0.01)
        self.assertEqual(orchestrator.record.state.value, "needs_confirmation")
        await orchestrator.close(reason="test_complete")
        await collector

        payload_types = [event.type.value for event in events]
        self.assertIn("transcript.final", payload_types)
        self.assertIn("assistant.text.final", payload_types)
        self.assertIn("assistant.audio.chunk", payload_types)

    async def test_low_confidence_speech_prompts_repeat_before_agent_tools(self) -> None:
        orchestrator = CmiyglSessionOrchestrator(
            "low-confidence-session",
            navigation_agent=self.agent,
            stt=ScriptedSTT(
                [
                    Transcript(text="mall", confidence=0.2, is_final=True),
                ]
            ),
            tts_engine=None,
        )
        events, collector = await self._collect_events(orchestrator)

        await orchestrator.handle_client_event({"type": "client.call.start", "session_id": "low-confidence-session"})
        await orchestrator.handle_client_event(
            {
                "type": "client.audio.frame",
                "session_id": "low-confidence-session",
                "audio_b64": base64.b64encode(b"\x00" * 640).decode("ascii"),
                "sequence": 0,
                "sample_rate_hz": 16000,
                "encoding": "pcm_s16le",
            }
        )
        await orchestrator.handle_client_event({"type": "client.audio.end", "session_id": "low-confidence-session"})
        await asyncio.sleep(0.05)
        await orchestrator.close(reason="test_complete")
        await collector

        payload_types = [event.type.value for event in events]
        self.assertIn("clarification.request", payload_types)
        self.assertEqual(orchestrator.record.pending_clarification.kind.value, "destination")
        self.assertEqual(orchestrator.record.metrics.tool_call_count, 0)

    async def test_interrupt_clears_active_audio(self) -> None:
        orchestrator = CmiyglSessionOrchestrator(
            "interrupt-session",
            navigation_agent=self.agent,
            stt=None,
            tts_engine=SlowChunkTTS(),
            assistant_tts_config=AssistantTTSConfig(),
        )
        events, collector = await self._collect_events(orchestrator)

        await orchestrator.start()
        await orchestrator.handle_client_event(
            {
                "type": "client.text.input",
                "session_id": "interrupt-session",
                "text": "Take me to City Mall",
                "is_final": True,
            }
        )
        await asyncio.sleep(0.03)
        await orchestrator.handle_client_event(
            {
                "type": "client.interrupt",
                "session_id": "interrupt-session",
                "reason": "barge_in",
            }
        )
        await asyncio.sleep(0.03)
        await orchestrator.close(reason="test_complete")
        await collector

        self.assertTrue(
            any(isinstance(event, AssistantAudioClearEvent) for event in events)
        )


class TwilioMediaAdapterTest(unittest.TestCase):
    def test_inbound_and_outbound_translation(self) -> None:
        adapter = TwilioMediaAdapter()
        context = TwilioStreamContext(session_id="seed")

        start_messages = adapter.inbound_messages_to_client_events(
            {
                "event": "start",
                "start": {
                    "callSid": "CA123",
                    "streamSid": "MZ123",
                    "accountSid": "AC123",
                    "customParameters": {"session_id": "cmiygl-call-1"},
                },
            },
            context=context,
        )
        self.assertEqual(start_messages[0]["type"], "client.call.start")
        self.assertEqual(context.stream_sid, "MZ123")

        linear_16k = b"\x00\x00" * 320
        mulaw_b64 = adapter.outbound_audio.pcm16000_to_mulaw8000_b64(
            base64.b64encode(linear_16k).decode("ascii")
        )
        media_messages = adapter.inbound_messages_to_client_events(
            {
                "event": "media",
                "media": {"payload": mulaw_b64, "track": "inbound"},
            },
            context=context,
        )
        self.assertEqual(media_messages[0]["type"], "client.audio.frame")
        self.assertEqual(media_messages[0]["sample_rate_hz"], 16000)

        outbound = adapter.server_event_to_twilio_messages(
            AssistantAudioChunkEvent(
                session_id="cmiygl-call-1",
                audio_b64=base64.b64encode(b"\x00" * 640).decode("ascii"),
                sample_rate_hz=16000,
                encoding="pcm_s16le",
            ),
            context=context,
        )
        payload = json.loads(outbound[0])
        self.assertEqual(payload["event"], "media")
        self.assertEqual(payload["streamSid"], "MZ123")

    def test_render_twiml_stream_response(self) -> None:
        xml = render_twiml_stream_response(
            stream_url="wss://example.com/ws/cmiygl",
            session_id="cmiygl-session-1",
        )
        self.assertIn("<Connect><Stream", xml)
        self.assertIn("cmiygl-session-1", xml)


if __name__ == "__main__":
    unittest.main()
