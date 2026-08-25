from __future__ import annotations

from typing import Any, Optional, Literal

from pydantic import BaseModel, Field, model_validator


class LatLng(BaseModel):
    lat: float
    lng: float

    @classmethod
    def from_place(cls, place: "PlaceResult") -> "LatLng":
        return cls(lat=place.lat, lng=place.lng)


class PlaceResult(BaseModel):
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

    boundingbox: list[str] | None = None
    importance: float | str | None = None
    raw: dict[str, Any] | None = None


class LandmarkResult(BaseModel):
    provider: str = "overpass"

    name: str | None = None
    lat: float
    lng: float

    osm_type: str | None = None
    osm_id: int | None = None

    category: str | None = None
    type: str | None = None

    tags: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] | None = None


class RecoveredLocationCandidate(BaseModel):
    lat: float
    lng: float

    confidence: float

    matched_landmarks: list[str]
    evidence: list[LandmarkResult]

    approx_address: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def location(self) -> LatLng:
        return LatLng(lat=self.lat, lng=self.lng)


class RouteStep(BaseModel):
    instruction: str
    distance_meters: int
    duration_seconds: int
    maneuver_type: int | str | None = None


class RouteResult(BaseModel):
    provider: str
    distance_meters: int
    duration_seconds: int
    steps: list[RouteStep]
    raw: dict[str, Any]


class SearchInput(BaseModel):
    query: str | None = Field(default=None, alias="q")

    amenity: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postalcode: Optional[str] = None

    @model_validator(mode="after")
    def validate_query_type(self):
        structured_fields = [
            self.amenity,
            self.street,
            self.city,
            self.county,
            self.state,
            self.country,
            self.postalcode,
        ]

        has_freeform = self.query is not None
        has_structured = any(value is not None for value in structured_fields)

        if has_freeform and has_structured:
            raise ValueError(
                "Use either free-form query or structured search fields, not both."
            )

        if not has_freeform and not has_structured:
            raise ValueError(
                "Provide either a free-form query or at least one structured search field."
            )

        return self


class ReverseGeocodeInput(BaseModel):
    lat: float
    lng: float


class RouteInput(BaseModel):
    origin: LatLng
    destination: LatLng
    mode: Literal["walking", "driving", "bicycle"] = "walking"
    units: Literal["miles", "kilometers"] = "kilometers"
    language: str = "en-US"
    costing_options: dict[str, Any] | None = None

    @classmethod
    def from_places(
        cls,
        origin: PlaceResult,
        destination: PlaceResult,
        mode: Literal["walking", "driving", "bicycle"] = "walking",
        units: Literal["miles", "kilometers"] = "kilometers",
        language: str = "en-US",
        costing_options: dict[str, Any] | None = None,
    ) -> "RouteInput":
        return cls(
            origin=LatLng.from_place(origin),
            destination=LatLng.from_place(destination),
            mode=mode,
            units=units,
            language=language,
            costing_options=costing_options,
        )

    @classmethod
    def from_latlng_and_place(
        cls,
        origin: LatLng,
        destination: PlaceResult,
        mode: Literal["walking", "driving", "bicycle"] = "walking",
        units: Literal["miles", "kilometers"] = "kilometers",
        language: str = "en-US",
        costing_options: dict[str, Any] | None = None,
    ) -> "RouteInput":
        return cls(
            origin=origin,
            destination=LatLng.from_place(destination),
            mode=mode,
            units=units,
            language=language,
            costing_options=costing_options,
        )

    @classmethod
    def from_place_and_latlng(
        cls,
        origin: PlaceResult,
        destination: LatLng,
        mode: Literal["walking", "driving", "bicycle"] = "walking",
        units: Literal["miles", "kilometers"] = "kilometers",
        language: str = "en-US",
        costing_options: dict[str, Any] | None = None,
    ) -> "RouteInput":
        return cls(
            origin=LatLng.from_place(origin),
            destination=destination,
            mode=mode,
            units=units,
            language=language,
            costing_options=costing_options,
        )