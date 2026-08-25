from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.cmiygl.config import (
    AppSettings,
    CmiyglConfig,
    LLMProvider,
    LLMSettings,
    LocationRecoveryProviderKind,
    MapSettings,
    STTProvider,
    STTSettings,
    TTSProvider,
    TTSSettings,
    TwilioSettings,
)
from app.cmiygl.core.schemas import NavigationMode

from .mapModel import (
    LandmarkResult,
    LatLng,
    PlaceResult,
    ReverseGeocodeInput,
    RouteInput,
    RouteResult,
    RouteStep,
    SearchInput,
)
from .locationRecoveryService.nominatim_locationRecovery import (
    NominatimLocationRecoveryService,
)
from .tool_registry import MapToolError, MapToolErrorReason, MapToolRegistry
from .tool_registry import build_map_tool_registry


class _FakeGeocoder:
    def __init__(self, results: list[PlaceResult] | None = None, *, exc: Exception | None = None) -> None:
        self.results = results or []
        self.exc = exc

    def search(self, query: SearchInput, limit: int = 5) -> list[PlaceResult]:
        del query, limit
        if self.exc is not None:
            raise self.exc
        return self.results

    def reverse_geocode(self, query: ReverseGeocodeInput) -> PlaceResult:
        del query
        if self.exc is not None:
            raise self.exc
        return PlaceResult(
            provider="nominatim",
            lat=6.5,
            lng=3.3,
            display_name="Recovered Address",
            name="Recovered Address",
        )


class _FakeLandmarkProvider:
    def nearby_landmarks(self, location: LatLng, radius: int = 250) -> list[LandmarkResult]:
        del location, radius
        return [
            LandmarkResult(
                name="Bus Stop",
                lat=6.5245,
                lng=3.3793,
                category="amenity",
                type="bus_station",
            )
        ]


class _FakeRecoveryProvider:
    def recover_location(
        self,
        landmarks: list[str],
        last_known_location: LatLng | None = None,
        search_radius: int = 1000,
    ):
        del landmarks, last_known_location, search_radius
        return []


class _FakeRoutingProvider:
    def route(self, query: RouteInput) -> RouteResult:
        del query
        return RouteResult(
            provider="valhalla",
            distance_meters=250,
            duration_seconds=180,
            steps=[RouteStep(instruction="Head north", distance_meters=250, duration_seconds=180)],
            raw={},
        )


class _RetryableUpstreamError(RuntimeError):
    status_code = 503


class MapToolRegistryTests(unittest.TestCase):
    def test_geocode_ranks_exact_match_first(self) -> None:
        registry = MapToolRegistry(
            geocoder=_FakeGeocoder(
                [
                    PlaceResult(
                        provider="nominatim",
                        lat=1.0,
                        lng=1.0,
                        display_name="Coffee House, Ikeja",
                        name="Coffee House",
                        importance=0.3,
                    ),
                    PlaceResult(
                        provider="nominatim",
                        lat=2.0,
                        lng=2.0,
                        display_name="Coffee House Annex, Ikeja",
                        name="Coffee House Annex",
                        importance=0.9,
                    ),
                ]
            ),
            reverse_geocoder=_FakeGeocoder(),
            landmark_provider=_FakeLandmarkProvider(),
            routing_provider=_FakeRoutingProvider(),
            location_recovery=_FakeRecoveryProvider(),
        )

        ranked = registry.geocode("Coffee House", limit=5)

        self.assertEqual(ranked[0].name, "Coffee House")
        self.assertGreaterEqual(ranked[0].rank_score, ranked[1].rank_score)

    def test_retryable_error_is_normalized(self) -> None:
        registry = MapToolRegistry(
            geocoder=_FakeGeocoder(exc=_RetryableUpstreamError("provider offline")),
            reverse_geocoder=_FakeGeocoder(),
            landmark_provider=_FakeLandmarkProvider(),
            routing_provider=_FakeRoutingProvider(),
            location_recovery=_FakeRecoveryProvider(),
        )

        with patch.dict(
            os.environ,
            {
                "CMIYGL_NOMINATIM_MAX_RETRIES": "2",
                "CMIYGL_NOMINATIM_INITIAL_RETRY_DELAY_SECONDS": "0.01",
                "CMIYGL_NOMINATIM_MAX_RETRY_DELAY_SECONDS": "0.01",
                "CMIYGL_NOMINATIM_RETRY_JITTER_RATIO": "0",
            },
            clear=False,
        ):
            with self.assertRaises(MapToolError) as ctx:
                registry.geocode("Ikeja", limit=1)

        self.assertEqual(ctx.exception.reason, MapToolErrorReason.UPSTREAM_UNAVAILABLE)
        self.assertTrue(ctx.exception.retryable)

    def test_build_registry_uses_nominatim_recovery_when_configured(self) -> None:
        cfg = CmiyglConfig(
            app=AppSettings(
                host="127.0.0.1",
                port=8770,
                log_level="INFO",
                session_id="cmiygl-test",
                language="en",
                default_route_mode=NavigationMode.WALKING,
                max_clarification_turns=3,
                location_recovery_radius_meters=1000,
                nearby_landmark_radius_meters=250,
            ),
            twilio=TwilioSettings(
                account_sid="",
                auth_token="",
                phone_number="",
                webhook_base_url="",
                stream_path="/ws/cmiygl",
            ),
            stt=STTSettings(
                provider=STTProvider.DISABLED,
                mistral_api_key="",
                deepgram_api_key="",
                assemblyai_api_key="",
                voxtral_realtime_url="",
                voxtral_model="voxtral-mini-transcribe-realtime-2602",
            ),
            llm=LLMSettings(
                provider=LLMProvider.DISABLED,
                mistral_api_key="",
                gemini_api_key="",
                mistral_model="mistral-small-latest",
                gemini_model="gemini-2.5-flash",
                max_tokens=220,
                temperature=0.2,
            ),
            tts=TTSSettings(
                provider=TTSProvider.DISABLED,
                mistral_api_key="",
                cartesia_api_key="",
                cartesia_voice_id="",
                cartesia_version="2025-04-16",
                cartesia_language="en",
                cartesia_model_id="sonic-3",
                voxtral_tts_model="voxtral-mini-tts-2603",
                voxtral_tts_voice_id="",
            ),
            maps=MapSettings(
                location_recovery_provider=LocationRecoveryProviderKind.NOMINATIM,
                nominatim_user_agent="cmiygl-test/0.1",
                overpass_user_agent="cmiygl-test/0.1",
                overpass_base_url="https://overpass-api.de/api/interpreter",
                valhalla_base_url="http://localhost:8002",
            ),
        )

        registry = build_map_tool_registry(cfg)

        self.assertEqual(registry.location_recovery_backend, "nominatim")
        self.assertIsInstance(registry.location_recovery, NominatimLocationRecoveryService)


if __name__ == "__main__":
    unittest.main()
