from __future__ import annotations

import requests
from typing import Any

from ..mapInterface import GeocodingProvider, ReverseGeocodingProvider
from ..mapModel import SearchInput, ReverseGeocodeInput, PlaceResult
from .nominatimModel import NominatimJsonV2Place


def nominatim_to_place_result(place: NominatimJsonV2Place) -> PlaceResult:
    return PlaceResult(
        provider="nominatim",
        lat=place.lat,
        lng=place.lon,
        display_name=place.display_name,
        name=place.name,
        address=place.address.model_dump() if place.address else None,
        category=place.category,
        type=place.type,
        addresstype=place.addresstype,
        osm_type=place.osm_type,
        osm_id=place.osm_id,
        place_id=place.place_id,
        boundingbox=place.boundingbox,
        importance=place.importance,
        raw=place.model_dump(),
    )


class NominatimProvider(GeocodingProvider, ReverseGeocodingProvider):
    BASE_URL = "https://nominatim.openstreetmap.org"

    def __init__(self, user_agent: str):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
            }
        )

    def search(
        self,
        query: SearchInput,
        limit: int = 5,
        addressdetails: int = 1,
    ) -> list[PlaceResult]:
        """Search for a place/address and return coordinate candidates."""

        params: dict[str, Any] = query.model_dump(by_alias=True, exclude_none=True)

        params.update(
            {
                "format": "jsonv2",
                "limit": limit,
                "addressdetails": addressdetails,
            }
        )

        try:
            response = self.session.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=5,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Nominatim search failed: {e}") from e

        places = [
            NominatimJsonV2Place.model_validate(item)
            for item in response.json()
        ]

        return [nominatim_to_place_result(place) for place in places]

    def reverse_geocode(
        self,
        query: ReverseGeocodeInput,
    ) -> PlaceResult:
        """Convert coordinates into an approximate address/place."""

        params = {
            "format": "jsonv2",
            "lat": query.lat,
            "lon": query.lng,
            "addressdetails": 1,
        }

        try:
            response = self.session.get(
                f"{self.BASE_URL}/reverse",
                params=params,
                timeout=5,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Nominatim reverse geocode failed: {e}") from e

        place = NominatimJsonV2Place.model_validate(response.json())

        return nominatim_to_place_result(place)
