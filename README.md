# Call Me If You Get Lost (CMIYGL)

Voice-first navigation assistant for callers who are lost.  
Caller speaks by phone, assistant uses map tools to recover current location, confirm destination, compute route, and guide turn-by-turn.

## Setup

From the repo root:

```powershell
python -m pip install -e .
Copy-Item .\.env.example .\.env -ErrorAction SilentlyContinue
Copy-Item .\app\cmiygl\.env.example .\app\cmiygl\.env -ErrorAction SilentlyContinue
```

Phase 0 config check:

```powershell
python -m app.cmiygl.realtime
```

That command now:
- loads repo `.env` first, then `app/cmiygl/.env`
- validates `cmiygl` config structure
- prints readiness for Twilio, STT, LLM, TTS, and maps

## Product Goal

Build a reliable inbound-call navigation assistant that can:
- infer or confirm caller location from coordinates, address, or landmarks
- resolve destination from free text
- generate route and deliver stepwise spoken instructions
- recover safely when location confidence is low or tools fail

## Architecture Summary

Call flow:
1. User calls Twilio number.
2. Twilio Media Streams sends audio over WebSocket to realtime pipeline.
3. Pipeline performs STT -> navigation agent reasoning + tool calls -> TTS.
4. TTS audio is streamed back to Twilio in-call.

Core app modules already present:
- `mapService/`: map providers and models (`nominatim`, `overpass`, `valhalla`, `locationRecovery`)
- `realtime/`: assistant streaming, events, session lifecycle, session manager
- `core/`: session/intent/tool schemas

## Agent Pattern Decision (Recommended)

Use a **Hybrid FSM + ReAct tool agent**:
- FSM (finite state machine) controls call/session safety and progression.
- ReAct loop handles tool-use reasoning per turn (think -> choose tool -> observe -> decide next action).

Why this is best here:
- Pure ReAct is flexible but can drift in long voice calls.
- Pure FSM is deterministic but brittle for ambiguous language.
- Hybrid gives deterministic guardrails with adaptive reasoning for map/tool usage.

Execution model:
- Session state remains authoritative (`NEEDS_LOCATION`, `RECOVERING_LOCATION`, `ROUTING`, `NAVIGATING`, etc.).
- Inside `THINKING`, run bounded ReAct loop with max tool steps per turn (e.g. 2-4).
- Every tool result updates navigation memory and may trigger confirmation prompts.

## Tooling Contract for Navigation Agent

Use these `mapService` methods as callable tools:
- `geocode(SearchInput) -> list[PlaceResult]`
- `reverse_geocode(ReverseGeocodeInput) -> PlaceResult`
- `nearby_landmarks(LatLng, radius) -> list[LandmarkResult]`
- `recover_location(landmarks, last_known_location, search_radius) -> list[RecoveredLocationCandidate]`
- `route(RouteInput) -> RouteResult`

Add tool wrappers with:
- strict input validation (Pydantic models)
- timeout/retry policy
- standardized error envelope
- compact result summarization for LLM context

## Phase-by-Phase Build Plan

### Phase 0: Baseline + Alignment

Checklist:
- [x] Align module names/imports (`core.schemas` vs `core.schema`) across `realtime` and `core`.
- [x] Verify state enums used in session files are consistent (added explicit `NavigationMode` enum and typed default route mode).
- [x] Confirm Twilio, STT, LLM, and TTS env vars and secrets loading strategy.
- [x] Add a single start command and local dev bootstrap steps.
- [x] Define latency/error SLOs for first response and route turnaround.

Exit criteria:
- Project imports cleanly.
- Session lifecycle boots without runtime import/type errors.

Phase 0 baseline command:
- `python -m app.cmiygl.realtime`

Current bootstrap expectation:
- Run from repository root: `real-time-voice-pipeline/`
- This is a smoke-test boot path for package/import/session lifecycle validation.
- Env loading strategy:
- repo-wide defaults and shared secrets come from `.env`
- app-specific overrides come from `app/cmiygl/.env`
- `app.cmiygl.config.load_cmiygl_config()` is the single config entrypoint
- Full Twilio/STT/LLM/TTS runtime wiring remains Phase 3 work.

Current SLO targets:
- First assistant response target: `<= 900 ms`
- Tool call target: `<= 2500 ms`
- Session metrics should capture first partial, first final transcript, first assistant text, first audio, and first route readiness.

Phase 0 env contract:
- Twilio: `CMIYGL_TWILIO_ACCOUNT_SID`, `CMIYGL_TWILIO_AUTH_TOKEN`, `CMIYGL_TWILIO_PHONE_NUMBER`, `CMIYGL_TWILIO_WEBHOOK_BASE_URL`
- STT:
- `CMIYGL_STT_PROVIDER=mistral_voxtral` uses `CMIYGL_MISTRAL_API_KEY` or shared `MISTRAL_API_KEY`
- `CMIYGL_STT_PROVIDER=deepgram` uses `CMIYGL_DEEPGRAM_API_KEY` or shared `DEEPGRAM_API_KEY`
- `CMIYGL_STT_PROVIDER=assemblyai` uses `CMIYGL_ASSEMBLYAI_API_KEY` or shared `ASSEMBLYAI_API_KEY`
- LLM:
- `CMIYGL_LLM_PROVIDER=mistral` uses `CMIYGL_MISTRAL_API_KEY` or shared `MISTRAL_API_KEY`
- `CMIYGL_LLM_PROVIDER=gemini` uses `CMIYGL_GEMINI_API_KEY` or shared `GEMINI_API_KEY`
- TTS:
- `CMIYGL_TTS_PROVIDER=cartesia` uses `CMIYGL_CARTESIA_API_KEY`/`CARTESIA_API_KEY` and `CMIYGL_CARTESIA_VOICE_ID`/`CARTESIA_VOICE_ID`
- `CMIYGL_TTS_PROVIDER=voxtral` uses `CMIYGL_MISTRAL_API_KEY`/`MISTRAL_API_KEY` and `CMIYGL_VOXTRAL_TTS_VOICE_ID`/`VOXTRAL_TTS_VOICE_ID`
- Maps: `CMIYGL_NOMINATIM_USER_AGENT`, `CMIYGL_OVERPASS_BASE_URL`, `CMIYGL_VALHALLA_BASE_URL`
- Location recovery provider: `CMIYGL_LOCATION_RECOVERY_PROVIDER=overpass|nominatim` with `overpass` as the default

### Phase 1: Map Tool Layer Hardening

Checklist:
- [x] Create `MapToolRegistry` that wires providers (`NominatimProvider`, `OverpassProvider`, `ValhallaProvider`, `LocationRecoveryService`).
- [x] Add provider health checks (`/search`, `/route` probe, etc.).
- [x] Add retries with bounded backoff for transient network failures.
- [x] Normalize provider errors to app-level tool errors (`tool`, `reason`, `retryable`).
- [x] Add confidence normalization for destination and recovered location candidates.
- [x] Add deterministic ranking for candidate destinations and recovered locations.

Exit criteria:
- Tool calls are robust, typed, and predictable.
- Failures produce actionable fallback prompts (not silent errors).

Phase 1 implementation notes:
- `app.cmiygl.mapService.tool_registry.build_map_tool_registry()` is the single wiring entrypoint.
- `MapToolRegistry.health_check()` performs geocode, nearby-landmark, and route probes.
- `MapToolError` is the normalized failure surface for map/tool operations.
- Ranked outputs are compact summaries intended for navigation-agent context, not raw provider payloads.
- Location recovery is selected via dependency injection from config:
- `overpass` uses `LocationRecoveryService(landmark_provider=OverpassProvider(...))`
- `nominatim` uses `NominatimLocationRecoveryService(geocoder=NominatimProvider(...))`

### Phase 2: Navigation Agent Core (Hybrid FSM + ReAct)

Checklist:
- [x] Implement `NavigationAgent` with turn handler: input transcript + session context -> action plan.
- [x] Add intent extraction schema for: destination, landmarks, confirmation, next-step, repeat, safety.
- [x] Add bounded ReAct planner:
- max tool calls per turn
- no duplicate identical tool call in same turn
- tool call budget timeout
- [x] Define agent memory slices:
- stable memory: confirmed destination, confirmed/estimated location, active route
- ephemeral memory: current clarification target, latest candidate lists
- [x] Add explicit confirmation policy:
- destination confirmation before routing
- location confirmation when confidence below threshold
- [x] Add route narration policy:
- summarize first 1-2 maneuvers
- cache full step list for follow-up ("next step", "repeat")

Exit criteria:
- Agent can complete end-to-end flow: locate user -> confirm destination -> generate route -> start navigation.

Phase 2 implementation notes:
- `app.cmiygl.realtime.navigation_agent.NavigationAgent` is the Phase 2 turn engine.
- The agent uses deterministic state authority from `SessionRecord` plus a bounded per-turn tool executor.
- Landmark-only recovery now allows `last_known_location=None`; provider selection still comes from dependency injection in `mapService`.
- `CmiyglSessionOrchestrator.handle_text_input(...)` is the minimal session-facing entrypoint for text-turn execution ahead of full realtime voice wiring.
- Verified scenarios cover destination confirmation, landmark-only recovery, route generation, repeat, next-step, and orchestrator integration.

### Phase 3: Realtime Voice Orchestration

Checklist:
- [x] Wire Twilio Media Streams inbound/outbound events to session orchestrator.
- [x] Ensure barge-in behavior interrupts TTS and returns to listening state.
- [x] Stream partial assistant text while TTS audio is being generated.
- [x] Add utterance segmentation and end-of-turn detection strategy.
- [x] Add "low-confidence speech" fallback prompt before tool calls when transcript is unclear.
- [x] Ensure every assistant response can be rendered both as text event and audio event.

Exit criteria:
- Natural call interaction loop works with interruption handling and low latency.

Phase 3 implementation notes:
- `app.cmiygl.realtime.orchestrator.CmiyglSessionOrchestrator` now implements the realtime event-session contract: client event intake, STT streaming, transcript finalization, agent dispatch, text streaming, audio streaming, and lifecycle close/wait.
- `app.cmiygl.realtime.twilio_media.TwilioMediaAdapter` handles Twilio Media Streams wire conversion in both directions, including built-in mu-law conversion without relying on `audioop`.
- `app.cmiygl.realtime.server` exposes the Twilio runtime surface:
- websocket media stream path from `CMIYGL_TWILIO_STREAM_PATH`
- voice webhook path `/twilio/voice`
- health path `/health`
- Browser/dev websocket helpers remain available in `app.cmiygl.realtime.websocket_api`.
- Verified scenarios now cover STT->agent->TTS flow, low-confidence fallback, barge-in clear, and Twilio media translation.

Phase 3 runtime command:
- `python -m app.cmiygl.realtime.server`

### Phase 4: Navigation UX and Safety Layer

Checklist:
- [ ] Add safety classifier for risky situations (roads/highways/night/confusion escalation).
- [ ] Add emergency path ("If you are in immediate danger, call local emergency services").
- [ ] Add uncertainty language policy tied to confidence buckets.
- [ ] Add caller-friendly location elicitation prompts:
- street sign + shop + junction pattern
- nearby bus stop/church/park prompts
- [ ] Add reroute behavior when caller says they took a wrong turn.

Exit criteria:
- Assistant remains helpful under uncertainty and has safe escalation behavior.

### Phase 5: Observability, QA, and Load Readiness

Checklist:
- [ ] Add structured logs for transcript, intent, tool calls, state transitions, and latency markers.
- [ ] Add per-call timeline artifact for debugging.
- [ ] Add integration tests with mocked tool responses and voice events.
- [ ] Add scenario tests:
- exact coordinate start
- landmark-only recovery
- ambiguous destination requiring clarification
- route API timeout + recovery
- [ ] Add soak test for concurrent sessions and rate limits in `SessionRegistry`.

Exit criteria:
- Regressions are caught by tests and key latencies are measurable.

### Phase 6: Production Rollout

Checklist:
- [ ] Stage environment with Twilio test number and provider endpoints.
- [ ] Canary release by percentage or limited caller set.
- [ ] Add alerting on tool failures, high clarification loops, high call drop rates.
- [ ] Define runbook for outages (nominatim/down, valhalla/down, Twilio stream drops).
- [ ] Post-launch tuning of prompts, confidence thresholds, and retry policy.

Exit criteria:
- Stable production operation with rollback and on-call runbook.

## Required New Components

Implement these modules next:
- `realtime/orchestrator.py`: central session + agent loop coordinator
- `realtime/navigation_agent.py`: hybrid FSM/ReAct decision engine
- `mapService/tool_registry.py`: unified map tool adapters and execution policy
- `realtime/prompt_policy.py`: voice-safe prompts + uncertainty language
- `tests/cmiygl/`: unit + integration scenarios

## Key Technical Decisions

- Keep the LLM context compact: pass summaries, not raw map payloads.
- Use explicit confidence thresholds for confirmation:
- destination: confirm when top candidate confidence < high threshold
- location recovery: confirm always unless very high confidence
- Limit tool latency budget per turn to protect voice UX.
- Prefer deterministic fallbacks over repeated free-form reasoning loops.

## Risks and Mitigations

- Risk: Landmark matching false positives.
  - Mitigation: stricter matching + distance/ranking + confirmation prompt.
- Risk: Route provider unavailable.
  - Mitigation: retry + failover message + ask user to hold while retrying.
- Risk: High voice latency.
  - Mitigation: stream short text chunks and start TTS early.
- Risk: Session drift across turns.
  - Mitigation: FSM authority + explicit navigation memory updates.

## Definition of Done

- Caller can start with either:
- "I am at <landmark>" or
- "I am at <coordinates/address>"
- Assistant can confirm location and destination verbally.
- Assistant can generate and speak route steps.
- Assistant supports repeat/next-step/reroute in-call.
- Safety and fallback prompts trigger correctly.
- Metrics and logs are sufficient to debug a failed call.
