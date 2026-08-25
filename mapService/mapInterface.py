from __future__ import annotations

from typing import Protocol, runtime_checkable

from .mapModel import (
    LatLng,
    SearchInput,
    ReverseGeocodeInput,
    RouteInput,
    PlaceResult,
    LandmarkResult,
    RecoveredLocationCandidate,
    RouteResult,
)


@runtime_checkable
class GeocodingProvider(Protocol):
    def search(
        self,
        query: SearchInput,
        limit: int = 5,
    ) -> list[PlaceResult]:
        """Search for a place/address and return coordinate candidates."""
        ...


@runtime_checkable
class ReverseGeocodingProvider(Protocol):
    def reverse_geocode(
        self,
        query: ReverseGeocodeInput,
    ) -> PlaceResult:
        """Convert coordinates into an approximate address/place."""
        ...


@runtime_checkable
class LandmarkProvider(Protocol):
    def nearby_landmarks(
        self,
        location: LatLng,
        radius: int = 250,
    ) -> list[LandmarkResult]:
        """
        Find recognizable landmarks near a known coordinate.

        Example:
            nearby_landmarks(LatLng(lat=52.52, lng=13.405), radius=500)
        """
        ...


@runtime_checkable
class RoutingProvider(Protocol):
    def route(
        self,
        query: RouteInput,
    ) -> RouteResult:
        """Route from origin to destination."""
        ...


@runtime_checkable
class LocationRecoveryProvider(Protocol):
    def recover_location(
        self,
        landmarks: list[str],
        last_known_location: LatLng | None = None,
        search_radius: int = 1000,
    ) -> list[RecoveredLocationCandidate]:
        """
        Estimate possible user locations from visible landmark clues.

        Example:
            recover_location(
                landmarks=["church", "fountain", "Starbucks"],
                last_known_location=LatLng(lat=52.52, lng=13.405),
            )
        """
        ...
