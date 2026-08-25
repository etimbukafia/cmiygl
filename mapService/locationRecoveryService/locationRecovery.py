from __future__ import annotations

import math
from collections import defaultdict

from ..mapInterface import LandmarkProvider, LocationRecoveryProvider
from ..mapModel import (
    LatLng,
    LandmarkResult,
    RecoveredLocationCandidate,
)


class LocationRecoveryService(LocationRecoveryProvider):
    def __init__(
        self,
        landmark_provider: LandmarkProvider,
    ) -> None:
        self.landmark_provider = landmark_provider

    def recover_location(
        self,
        landmarks: list[str],
        last_known_location: LatLng | None = None,
        search_radius: int = 1000,
    ) -> list[RecoveredLocationCandidate]:
        if not landmarks:
            return []

        if last_known_location is None:
            raise ValueError(
                "last_known_location is required for this recovery strategy."
            )

        nearby = self.landmark_provider.nearby_landmarks(
            location=last_known_location,
            radius=search_radius,
        )

        matches = self._match_landmarks(
            user_landmarks=landmarks,
            nearby_landmarks=nearby,
        )

        if not matches:
            return []

        candidates = self._build_candidates(
            user_location=last_known_location,
            user_landmarks=landmarks,
            matches=matches,
            search_radius=search_radius,
        )

        return sorted(
            candidates,
            key=lambda candidate: candidate.confidence,
            reverse=True,
        )

    def _match_landmarks(
        self,
        user_landmarks: list[str],
        nearby_landmarks: list[LandmarkResult],
    ) -> dict[str, list[LandmarkResult]]:
        matches: dict[str, list[LandmarkResult]] = defaultdict(list)

        for clue in user_landmarks:
            normalized_clue = self._normalize(clue)

            for landmark in nearby_landmarks:
                if self._landmark_matches(normalized_clue, landmark):
                    matches[clue].append(landmark)

        return dict(matches)

    def _landmark_matches(
        self,
        clue: str,
        landmark: LandmarkResult,
    ) -> bool:
        searchable_parts = [
            landmark.name,
            landmark.category,
            landmark.type,
            landmark.tags.get("amenity"),
            landmark.tags.get("tourism"),
            landmark.tags.get("historic"),
            landmark.tags.get("shop"),
            landmark.tags.get("brand"),
            landmark.tags.get("building"),
            landmark.tags.get("leisure"),
            landmark.tags.get("man_made"),
        ]

        searchable = " ".join(
            self._normalize(str(part))
            for part in searchable_parts
            if part
        )

        return clue in searchable or searchable in clue

    def _build_candidates(
        self,
        user_location: LatLng,
        user_landmarks: list[str],
        matches: dict[str, list[LandmarkResult]],
        search_radius: int,
    ) -> list[RecoveredLocationCandidate]:
        evidence: list[LandmarkResult] = []

        for matched_items in matches.values():
            evidence.extend(matched_items)

        deduped_evidence = self._dedupe_landmarks(evidence)

        if not deduped_evidence:
            return []

        estimated_lat = sum(item.lat for item in deduped_evidence) / len(deduped_evidence)
        estimated_lng = sum(item.lng for item in deduped_evidence) / len(deduped_evidence)

        matched_landmarks = list(matches.keys())

        distance_from_last_known = self._distance_meters(
            user_location.lat,
            user_location.lng,
            estimated_lat,
            estimated_lng,
        )

        clue_score = len(matched_landmarks) / len(user_landmarks)
        distance_score = max(0.0, 1.0 - distance_from_last_known / search_radius)

        confidence = min(
            1.0,
            round((0.75 * clue_score) + (0.25 * distance_score), 3),
        )

        return [
            RecoveredLocationCandidate(
                lat=estimated_lat,
                lng=estimated_lng,
                confidence=confidence,
                matched_landmarks=matched_landmarks,
                evidence=deduped_evidence,
                raw={
                    "strategy": "centroid_of_matched_landmarks",
                    "distance_from_last_known_meters": distance_from_last_known,
                    "search_radius": search_radius,
                },
            )
        ]

    def _dedupe_landmarks(
        self,
        landmarks: list[LandmarkResult],
    ) -> list[LandmarkResult]:
        seen: set[tuple[str | None, int | None]] = set()
        results: list[LandmarkResult] = []

        for landmark in landmarks:
            key = (landmark.osm_type, landmark.osm_id)

            if key in seen:
                continue

            seen.add(key)
            results.append(landmark)

        return results

    def _normalize(self, value: str) -> str:
        return value.strip().lower().replace("_", " ")

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
            + math.cos(phi1)
            * math.cos(phi2)
            * math.sin(delta_lambda / 2) ** 2
        )

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return earth_radius_m * c
