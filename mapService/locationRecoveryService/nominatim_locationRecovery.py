from __future__ import annotations

import math
from collections import defaultdict

from ..mapInterface import LocationRecoveryProvider
from ..mapModel import (
    LandmarkResult,
    LatLng,
    PlaceResult,
    RecoveredLocationCandidate,
    ReverseGeocodeInput,
    SearchInput,
)
from ..nominatimService.nominatim import NominatimProvider


class NominatimLocationRecoveryService(LocationRecoveryProvider):
    """
    Recover location by geocoding spoken clues against locality context.

    Overpass remains the stronger default for nearby landmark scans. This
    adapter keeps the provider choice injectable when the caller wants a
    Nominatim-based strategy instead.
    """

    def __init__(self, geocoder: NominatimProvider) -> None:
        self.geocoder = geocoder

    def recover_location(
        self,
        landmarks: list[str],
        last_known_location: LatLng | None = None,
        search_radius: int = 1000,
    ) -> list[RecoveredLocationCandidate]:
        cleaned_landmarks = [item.strip() for item in landmarks if item.strip()]
        if not cleaned_landmarks:
            return []

        locality_context = self._build_locality_context(last_known_location)
        matches: dict[str, list[PlaceResult]] = defaultdict(list)

        for clue in cleaned_landmarks:
            candidates = self.geocoder.search(
                SearchInput(q=self._build_query_text(clue, locality_context)),
                limit=5,
            )
            filtered = self._filter_candidates(
                candidates,
                last_known_location=last_known_location,
                search_radius=search_radius,
            )
            if filtered:
                matches[clue].extend(filtered)

        if not matches:
            return []

        evidence_places = self._dedupe_places(
            [
                place
                for matched_places in matches.values()
                for place in matched_places
            ]
        )
        if not evidence_places:
            return []

        estimated_lat = sum(item.lat for item in evidence_places) / len(evidence_places)
        estimated_lng = sum(item.lng for item in evidence_places) / len(evidence_places)
        matched_landmarks = list(matches.keys())
        clue_score = len(matched_landmarks) / len(cleaned_landmarks)

        distance_score = 0.5
        distance_from_last_known: float | None = None
        if last_known_location is not None:
            distance_from_last_known = self._distance_meters(
                last_known_location.lat,
                last_known_location.lng,
                estimated_lat,
                estimated_lng,
            )
            distance_score = max(0.0, 1.0 - distance_from_last_known / max(1, search_radius))

        confidence = min(
            1.0,
            round((0.7 * clue_score) + (0.3 * distance_score), 3),
        )

        evidence = [self._place_to_landmark(place) for place in evidence_places]
        approx_address = self._reverse_address(estimated_lat, estimated_lng)

        return [
            RecoveredLocationCandidate(
                lat=estimated_lat,
                lng=estimated_lng,
                confidence=confidence,
                matched_landmarks=matched_landmarks,
                evidence=evidence,
                approx_address=approx_address,
                raw={
                    "strategy": "nominatim_geocode_clues",
                    "locality_context": locality_context,
                    "distance_from_last_known_meters": distance_from_last_known,
                    "search_radius": search_radius,
                },
            )
        ]

    def _build_locality_context(self, last_known_location: LatLng | None) -> str:
        if last_known_location is None:
            return ""

        try:
            place = self.geocoder.reverse_geocode(
                ReverseGeocodeInput(
                    lat=last_known_location.lat,
                    lng=last_known_location.lng,
                )
            )
        except Exception:
            return ""

        address = place.address or {}
        parts = [
            str(address.get("suburb") or "").strip(),
            str(address.get("city") or address.get("town") or address.get("village") or "").strip(),
            str(address.get("state") or "").strip(),
            str(address.get("country") or "").strip(),
        ]
        return ", ".join(part for part in parts if part)

    def _build_query_text(self, clue: str, locality_context: str) -> str:
        if locality_context:
            return f"{clue}, {locality_context}"
        return clue

    def _filter_candidates(
        self,
        candidates: list[PlaceResult],
        *,
        last_known_location: LatLng | None,
        search_radius: int,
    ) -> list[PlaceResult]:
        if last_known_location is None:
            return candidates

        within_radius = [
            place
            for place in candidates
            if self._distance_meters(
                last_known_location.lat,
                last_known_location.lng,
                place.lat,
                place.lng,
            ) <= search_radius
        ]
        return within_radius or candidates[:1]

    def _dedupe_places(self, places: list[PlaceResult]) -> list[PlaceResult]:
        seen: set[tuple[str | None, int | str | None]] = set()
        deduped: list[PlaceResult] = []
        for place in places:
            key = (place.osm_type, place.osm_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(place)
        return deduped

    def _place_to_landmark(self, place: PlaceResult) -> LandmarkResult:
        osm_id: int | None = None
        if isinstance(place.osm_id, int):
            osm_id = place.osm_id

        return LandmarkResult(
            provider=place.provider,
            name=place.name or place.display_name,
            lat=place.lat,
            lng=place.lng,
            osm_type=place.osm_type,
            osm_id=osm_id,
            category=place.category,
            type=place.type,
            tags=place.address or {},
            raw=place.raw,
        )

    def _reverse_address(self, lat: float, lng: float) -> str | None:
        try:
            return self.geocoder.reverse_geocode(
                ReverseGeocodeInput(lat=lat, lng=lng)
            ).display_name
        except Exception:
            return None

    def _distance_meters(
        self,
        lat1: float,
        lng1: float,
        lat2: float,
        lng2: float,
    ) -> float:
        earth_radius_m = 6_371_000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lng2 - lng1)
        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return earth_radius_m * c
