from __future__ import annotations

from typing import Any

import requests

from ..mapInterface import LandmarkProvider
from ..mapModel import LatLng, LandmarkResult


class OverpassProvider(LandmarkProvider):
    def __init__(
        self,
        base_url: str = "https://overpass-api.de/api/interpreter",
        user_agent: str = "your-app-name/1.0",
    ) -> None:
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def nearby_landmarks(
        self,
        location: LatLng,
        radius: int = 250,
    ) -> list[LandmarkResult]:
        query = self._build_nearby_landmarks_query(
            lat=location.lat,
            lng=location.lng,
            radius=radius,
        )

        data = self._post(query)

        results: list[LandmarkResult] = []

        for element in data.get("elements", []):
            landmark = self._element_to_landmark(element)
            if landmark:
                results.append(landmark)

        return results

    def _build_nearby_landmarks_query(
        self,
        lat: float,
        lng: float,
        radius: int,
    ) -> str:
        return f"""
        [out:json][timeout:25];
        (
          nwr(around:{radius},{lat},{lng})["tourism"];
          nwr(around:{radius},{lat},{lng})["historic"];
          nwr(around:{radius},{lat},{lng})["amenity"~"^(place_of_worship|theatre|arts_centre|fountain|marketplace)$"];
          nwr(around:{radius},{lat},{lng})["leisure"~"^(park|garden)$"];
          nwr(around:{radius},{lat},{lng})["natural"~"^(peak|water|wood)$"];
          nwr(around:{radius},{lat},{lng})["man_made"~"^(tower|lighthouse|obelisk|bridge)$"];
          nwr(around:{radius},{lat},{lng})["shop"];
          nwr(around:{radius},{lat},{lng})["brand"];
        );
        out center tags;
        """

    def _post(self, query: str) -> dict[str, Any]:
        try:
            response = self.session.post(
                self.base_url,
                data={"data": query},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Overpass request failed: {e}") from e

        return response.json()

    def _element_to_landmark(
        self,
        element: dict[str, Any],
    ) -> LandmarkResult | None:
        tags = element.get("tags", {})

        name = (
            tags.get("name")
            or tags.get("brand")
            or tags.get("operator")
        )

        if not name:
            return None

        if "lat" in element and "lon" in element:
            lat = element["lat"]
            lng = element["lon"]
        else:
            center = element.get("center")
            if not center:
                return None

            lat = center.get("lat")
            lng = center.get("lon")

        if lat is None or lng is None:
            return None

        category, item_type = self._extract_category_and_type(tags)

        return LandmarkResult(
            provider="overpass",
            name=name,
            lat=float(lat),
            lng=float(lng),
            osm_type=element.get("type"),
            osm_id=element.get("id"),
            category=category,
            type=item_type,
            tags=tags,
            raw=element,
        )

    def _extract_category_and_type(
        self,
        tags: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        for key in [
            "tourism",
            "historic",
            "amenity",
            "leisure",
            "natural",
            "man_made",
            "shop",
            "brand",
        ]:
            if key in tags:
                return key, str(tags[key])

        return None, None
