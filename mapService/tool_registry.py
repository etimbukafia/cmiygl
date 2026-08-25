from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field

from provider_resilience import retry_delay_seconds, retry_policy, should_retry_provider_error

from ..config import CmiyglConfig, LocationRecoveryProviderKind, load_cmiygl_config
from ..core.schemas import ToolName
from .locationRecoveryService.nominatim_locationRecovery import NominatimLocationRecoveryService
from .locationRecoveryService.locationRecovery import LocationRecoveryService
from .mapInterface import (
    GeocodingProvider,
    LandmarkProvider,
    LocationRecoveryProvider,
    ReverseGeocodingProvider,
    RoutingProvider,
)
from .mapModel import (
    LandmarkResult,
    LatLng,
    PlaceResult,
    ReverseGeocodeInput,
    RouteInput,
    RouteStep,
    SearchInput,
)
from .nominatimService.nominatim import NominatimProvider
from .overpassService.overpass import OverpassProvider
from .valhallaService.valhalla import ValhallaProvider


_T = TypeVar("_T")

_DEFAULT_HEALTH_PROBE_LOCATION = LatLng(lat=6.5244, lng=3.3792)
_DEFAULT_ROUTE_PROBE = RouteInput(
    origin=LatLng(lat=6.5244, lng=3.3792),
    destination=LatLng(lat=6.5265, lng=3.3799),
    mode="walking",
)


class MapToolErrorReason(StrEnum):
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    UNKNOWN = "unknown"


class HealthStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


@dataclass(slots=True)
class MapToolError(RuntimeError):
    tool_name: str
    provider: str
    reason: MapToolErrorReason
    retryable: bool
    message: str
    attempts: int = 1

    def __str__(self) -> str:
        return self.message


class RankedPlaceCandidate(BaseModel):
    provider: str
    lat: float
    lng: float
    display_name: str
    name: str | None = None
    address: dict[str, Any] | None = None
    category: str | None = None
    type: str | None = None
    addresstype: str | None = None
    osm_type: str | None = None
    osm_id: int | str | None = None
    place_id: int | str | None = None
    importance: float | None = None
    confidence: float
    rank_score: float


class LandmarkSummary(BaseModel):
    provider: str
    name: str
    lat: float
    lng: float
    category: str | None = None
    type: str | None = None
    distance_meters: int


class RankedRecoveredLocationCandidate(BaseModel):
    lat: float
    lng: float
    confidence: float
    rank_score: float
    matched_landmarks: list[str]
    matched_landmark_count: int
    evidence_names: list[str]
    evidence_count: int
    approx_address: str | None = None


class RoutePlan(BaseModel):
    provider: str
    distance_meters: int
    duration_seconds: int
    step_count: int
    first_instruction: str = ""
    steps: list[RouteStep] = Field(default_factory=list)


class ProviderHealthCheck(BaseModel):
    provider: str
    status: HealthStatus
    ok: bool
    latency_ms: int
    details: str


class MapHealthReport(BaseModel):
    checks: list[ProviderHealthCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


class MapToolRegistry:
    def __init__(
        self,
        *,
        geocoder: GeocodingProvider,
        reverse_geocoder: ReverseGeocodingProvider,
        landmark_provider: LandmarkProvider,
        routing_provider: RoutingProvider,
        location_recovery: LocationRecoveryProvider,
        location_recovery_backend: str = "overpass",
    ) -> None:
        self.geocoder = geocoder
        self.reverse_geocoder = reverse_geocoder
        self.landmark_provider = landmark_provider
        self.routing_provider = routing_provider
        self.location_recovery = location_recovery
        self.location_recovery_backend = location_recovery_backend

    def geocode(
        self,
        query: SearchInput | str,
        *,
        limit: int = 5,
    ) -> list[RankedPlaceCandidate]:
        resolved_query = SearchInput(q=query) if isinstance(query, str) else query
        results = self._execute(
            tool_name=ToolName.GEOCODE.value,
            provider="nominatim",
            operation=lambda: self.geocoder.search(resolved_query, limit=limit),
        )
        if not results:
            raise MapToolError(
                tool_name=ToolName.GEOCODE.value,
                provider="nominatim",
                reason=MapToolErrorReason.NOT_FOUND,
                retryable=False,
                message="No destination candidates were found.",
            )
        return self._rank_places(results, resolved_query)

    def reverse_geocode(
        self,
        query: ReverseGeocodeInput | LatLng,
    ) -> RankedPlaceCandidate:
        resolved_query = (
            ReverseGeocodeInput(lat=query.lat, lng=query.lng)
            if isinstance(query, LatLng)
            else query
        )
        result = self._execute(
            tool_name=ToolName.REVERSE_GEOCODE.value,
            provider="nominatim",
            operation=lambda: self.reverse_geocoder.reverse_geocode(resolved_query),
        )
        return self._place_candidate(result, confidence=1.0, rank_score=1.0)

    def nearby_landmarks(
        self,
        location: LatLng,
        *,
        radius: int = 250,
        limit: int = 10,
    ) -> list[LandmarkSummary]:
        if radius <= 0:
            raise MapToolError(
                tool_name=ToolName.NEARBY_LANDMARKS.value,
                provider="overpass",
                reason=MapToolErrorReason.INVALID_INPUT,
                retryable=False,
                message="Nearby landmarks radius must be greater than 0.",
            )
        landmarks = self._execute(
            tool_name=ToolName.NEARBY_LANDMARKS.value,
            provider="overpass",
            operation=lambda: self.landmark_provider.nearby_landmarks(location, radius=radius),
        )
        summarized = [
            LandmarkSummary(
                provider=landmark.provider,
                name=landmark.name or "Unnamed landmark",
                lat=landmark.lat,
                lng=landmark.lng,
                category=landmark.category,
                type=landmark.type,
                distance_meters=round(
                    self._distance_meters(
                        location.lat,
                        location.lng,
                        landmark.lat,
                        landmark.lng,
                    )
                ),
            )
            for landmark in landmarks
            if landmark.name
        ]
        summarized.sort(
            key=lambda item: (
                item.distance_meters,
                item.name.lower(),
                item.lat,
                item.lng,
            )
        )
        return summarized[: max(1, limit)]

    def recover_location(
        self,
        landmarks: list[str],
        *,
        last_known_location: LatLng | None = None,
        search_radius: int = 1000,
    ) -> list[RankedRecoveredLocationCandidate]:
        if not landmarks:
            raise MapToolError(
                tool_name=ToolName.RECOVER_LOCATION.value,
                provider=self.location_recovery_backend,
                reason=MapToolErrorReason.INVALID_INPUT,
                retryable=False,
                message="At least one landmark is required for location recovery.",
            )
        candidates = self._execute(
            tool_name=ToolName.RECOVER_LOCATION.value,
            provider=self.location_recovery_backend,
            operation=lambda: self.location_recovery.recover_location(
                landmarks=landmarks,
                last_known_location=last_known_location,
                search_radius=search_radius,
            ),
        )
        if not candidates:
            raise MapToolError(
                tool_name=ToolName.RECOVER_LOCATION.value,
                provider=self.location_recovery_backend,
                reason=MapToolErrorReason.NOT_FOUND,
                retryable=False,
                message="No location candidates matched the supplied landmarks.",
            )

        ranked: list[RankedRecoveredLocationCandidate] = []
        for candidate in candidates:
            approx_address = candidate.approx_address
            if not approx_address:
                with self._ignore_tool_errors():
                    approx_address = self.reverse_geocode(candidate.location).display_name

            evidence_names = sorted(
                {
                    evidence.name.strip()
                    for evidence in candidate.evidence
                    if evidence.name and evidence.name.strip()
                }
            )
            matched_landmarks = sorted(
                {
                    landmark.strip()
                    for landmark in candidate.matched_landmarks
                    if landmark.strip()
                }
            )
            matched_landmark_count = len(matched_landmarks)
            evidence_count = len(evidence_names)
            evidence_ratio = min(1.0, evidence_count / max(1, matched_landmark_count))
            address_bonus = 0.05 if approx_address else 0.0
            rank_score = round(
                min(1.0, (0.8 * self._clamp(candidate.confidence)) + (0.15 * evidence_ratio) + address_bonus),
                3,
            )
            ranked.append(
                RankedRecoveredLocationCandidate(
                    lat=candidate.lat,
                    lng=candidate.lng,
                    confidence=self._clamp(candidate.confidence),
                    rank_score=rank_score,
                    matched_landmarks=matched_landmarks,
                    matched_landmark_count=matched_landmark_count,
                    evidence_names=evidence_names,
                    evidence_count=evidence_count,
                    approx_address=approx_address,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item.rank_score,
                -item.confidence,
                item.approx_address or "",
                item.lat,
                item.lng,
            )
        )
        return ranked

    def route(self, query: RouteInput) -> RoutePlan:
        route = self._execute(
            tool_name=ToolName.ROUTE.value,
            provider="valhalla",
            operation=lambda: self.routing_provider.route(query),
        )
        return RoutePlan(
            provider=route.provider,
            distance_meters=route.distance_meters,
            duration_seconds=route.duration_seconds,
            step_count=len(route.steps),
            first_instruction=route.steps[0].instruction if route.steps else "",
            steps=route.steps,
        )

    def health_check(
        self,
        *,
        geocode_probe: SearchInput | None = None,
        landmarks_probe: LatLng | None = None,
        route_probe: RouteInput | None = None,
    ) -> MapHealthReport:
        geocode_query = geocode_probe or SearchInput(q="Lagos")
        landmark_location = landmarks_probe or _DEFAULT_HEALTH_PROBE_LOCATION
        route_query = route_probe or _DEFAULT_ROUTE_PROBE

        checks = [
            self._probe(
                provider="nominatim",
                operation=lambda: self.geocoder.search(geocode_query, limit=1),
                success_message="Geocode probe completed.",
            ),
            self._probe(
                provider="overpass",
                operation=lambda: self.landmark_provider.nearby_landmarks(
                    landmark_location,
                    radius=250,
                ),
                success_message="Nearby landmark probe completed.",
            ),
            self._probe(
                provider="valhalla",
                operation=lambda: self.routing_provider.route(route_query),
                success_message="Route probe completed.",
            ),
        ]
        return MapHealthReport(checks=checks)

    def _execute(
        self,
        *,
        tool_name: str,
        provider: str,
        operation: Callable[[], _T],
    ) -> _T:
        attempts = 0
        policy = retry_policy(self._retry_prefix(provider), default_attempts=3)
        last_exc: BaseException | None = None
        while attempts < policy.max_attempts:
            attempts += 1
            try:
                return operation()
            except MapToolError:
                raise
            except ValueError as exc:
                raise MapToolError(
                    tool_name=tool_name,
                    provider=provider,
                    reason=MapToolErrorReason.INVALID_INPUT,
                    retryable=False,
                    attempts=attempts,
                    message=str(exc),
                ) from exc
            except Exception as exc:
                last_exc = exc
                retryable = should_retry_provider_error(exc)
                if not retryable or attempts >= policy.max_attempts:
                    raise MapToolError(
                        tool_name=tool_name,
                        provider=provider,
                        reason=self._normalize_reason(exc),
                        retryable=retryable,
                        attempts=attempts,
                        message=f"{provider} {tool_name} failed: {exc}",
                    ) from exc
                time.sleep(retry_delay_seconds(exc, attempts, policy))
        raise MapToolError(
            tool_name=tool_name,
            provider=provider,
            reason=self._normalize_reason(last_exc),
            retryable=bool(last_exc and should_retry_provider_error(last_exc)),
            attempts=attempts,
            message=f"{provider} {tool_name} failed after {attempts} attempts.",
        ) from last_exc

    def _probe(
        self,
        *,
        provider: str,
        operation: Callable[[], object],
        success_message: str,
    ) -> ProviderHealthCheck:
        started = time.perf_counter()
        try:
            operation()
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000)
            return ProviderHealthCheck(
                provider=provider,
                status=HealthStatus.FAILED,
                ok=False,
                latency_ms=latency_ms,
                details=str(exc),
            )
        latency_ms = round((time.perf_counter() - started) * 1000)
        return ProviderHealthCheck(
            provider=provider,
            status=HealthStatus.OK,
            ok=True,
            latency_ms=latency_ms,
            details=success_message,
        )

    def _rank_places(
        self,
        places: list[PlaceResult],
        query: SearchInput,
    ) -> list[RankedPlaceCandidate]:
        query_text = self._query_text(query)
        ranked = [
            self._place_candidate(
                place,
                confidence=self._place_confidence(place, query_text),
                rank_score=self._place_confidence(place, query_text),
            )
            for place in places
        ]
        ranked.sort(
            key=lambda item: (
                -item.rank_score,
                -item.confidence,
                item.display_name.lower(),
                item.lat,
                item.lng,
            )
        )
        return ranked

    def _place_candidate(
        self,
        place: PlaceResult,
        *,
        confidence: float,
        rank_score: float,
    ) -> RankedPlaceCandidate:
        importance = self._coerce_float(place.importance)
        return RankedPlaceCandidate(
            provider=place.provider,
            lat=place.lat,
            lng=place.lng,
            display_name=place.display_name,
            name=place.name,
            address=place.address,
            category=place.category,
            type=place.type,
            addresstype=place.addresstype,
            osm_type=place.osm_type,
            osm_id=place.osm_id,
            place_id=place.place_id,
            importance=importance,
            confidence=round(self._clamp(confidence), 3),
            rank_score=round(self._clamp(rank_score), 3),
        )

    def _place_confidence(self, place: PlaceResult, query_text: str) -> float:
        normalized_query = self._normalize_text(query_text)
        normalized_name = self._normalize_text(place.name or "")
        normalized_display = self._normalize_text(place.display_name)
        importance = self._clamp(self._coerce_float(place.importance) or 0.0)

        match_score = 0.25
        exact_name_bonus = 0.0
        if normalized_query:
            if normalized_query == normalized_name or normalized_query == normalized_display:
                match_score = 1.0
                if normalized_query == normalized_name:
                    exact_name_bonus = 0.2
            elif normalized_name and normalized_name.startswith(normalized_query):
                match_score = 0.92
            elif normalized_query and normalized_query in normalized_display:
                match_score = 0.78
            else:
                query_tokens = set(normalized_query.split())
                display_tokens = set(normalized_display.split())
                if query_tokens:
                    match_score = max(match_score, len(query_tokens & display_tokens) / len(query_tokens))

        completeness_bonus = 0.05 if place.address else 0.0
        addresstype_bonus = 0.05 if place.addresstype else 0.0
        return min(
            1.0,
            (0.7 * match_score)
            + (0.2 * importance)
            + exact_name_bonus
            + completeness_bonus
            + addresstype_bonus,
        )

    def _query_text(self, query: SearchInput) -> str:
        if query.query:
            return query.query
        return " ".join(
            part.strip()
            for part in [
                query.amenity,
                query.street,
                query.city,
                query.county,
                query.state,
                query.country,
                query.postalcode,
            ]
            if part and part.strip()
        )

    def _normalize_reason(self, exc: BaseException | None) -> MapToolErrorReason:
        if exc is None:
            return MapToolErrorReason.UNKNOWN
        lowered = str(exc).lower()
        status_code = self._status_code(exc)
        if status_code == 429 or "rate limit" in lowered or "too many requests" in lowered:
            return MapToolErrorReason.RATE_LIMITED
        if isinstance(exc, TimeoutError) or "timeout" in lowered or "timed out" in lowered:
            return MapToolErrorReason.TIMEOUT
        if isinstance(exc, (ConnectionError, OSError)) or (status_code is not None and status_code >= 500):
            return MapToolErrorReason.UPSTREAM_UNAVAILABLE
        return MapToolErrorReason.UNKNOWN

    def _retry_prefix(self, provider: str) -> str:
        return f"CMIYGL_{provider.upper()}"

    def _status_code(self, exc: BaseException) -> int | None:
        for attr in ("status_code", "status", "http_status", "code"):
            value = getattr(exc, attr, None)
            if isinstance(value, int) and 100 <= value <= 599:
                return value
        response = getattr(exc, "response", None)
        if response is None:
            return None
        status_code = getattr(response, "status_code", None)
        return status_code if isinstance(status_code, int) else None

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _coerce_float(self, value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _normalize_text(self, value: str) -> str:
        return " ".join(value.strip().lower().split())

    def _distance_meters(
        self,
        lat1: float,
        lng1: float,
        lat2: float,
        lng2: float,
    ) -> float:
        from math import atan2, cos, radians, sin, sqrt

        earth_radius_m = 6_371_000
        phi1 = radians(lat1)
        phi2 = radians(lat2)
        delta_phi = radians(lat2 - lat1)
        delta_lambda = radians(lng2 - lng1)
        a = (
            sin(delta_phi / 2) ** 2
            + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
        )
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return earth_radius_m * c

    class _IgnoreToolErrors:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return exc_type is not None and issubclass(exc_type, MapToolError)

    def _ignore_tool_errors(self) -> _IgnoreToolErrors:
        return self._IgnoreToolErrors()


def build_map_tool_registry(cfg: CmiyglConfig | None = None) -> MapToolRegistry:
    resolved_cfg = cfg or load_cmiygl_config()
    geocoder = NominatimProvider(user_agent=resolved_cfg.maps.nominatim_user_agent)
    landmark_provider = OverpassProvider(
        base_url=resolved_cfg.maps.overpass_base_url,
        user_agent=resolved_cfg.maps.overpass_user_agent,
    )
    routing_provider = ValhallaProvider(base_url=resolved_cfg.maps.valhalla_base_url)
    if resolved_cfg.maps.location_recovery_provider is LocationRecoveryProviderKind.NOMINATIM:
        location_recovery: LocationRecoveryProvider = NominatimLocationRecoveryService(geocoder=geocoder)
    else:
        location_recovery = LocationRecoveryService(landmark_provider=landmark_provider)
    return MapToolRegistry(
        geocoder=geocoder,
        reverse_geocoder=geocoder,
        landmark_provider=landmark_provider,
        routing_provider=routing_provider,
        location_recovery=location_recovery,
        location_recovery_backend=resolved_cfg.maps.location_recovery_provider.value,
    )
