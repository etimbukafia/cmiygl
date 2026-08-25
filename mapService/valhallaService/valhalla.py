from __future__ import annotations

from typing import Any

import requests

from ..mapInterface import RoutingProvider
from ..mapModel import RouteInput, RouteResult, RouteStep


class ValhallaProvider(RoutingProvider):
    def __init__(self, base_url: str = "http://localhost:8002") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def route(self, query: RouteInput) -> RouteResult:
        costing = self._costing(query.mode)

        payload: dict[str, Any] = {
            "locations": [
                {
                    "lat": query.origin.lat,
                    "lon": query.origin.lng,
                    "type": "break",
                },
                {
                    "lat": query.destination.lat,
                    "lon": query.destination.lng,
                    "type": "break",
                },
            ],
            "costing": costing,
            "directions_options": {
                "units": query.units,
                "language": query.language,
            },
        }

        if query.costing_options:
            payload["costing_options"] = {
                costing: query.costing_options
            }

        data = self._post("/route", payload)

        trip = data.get("trip", {})
        summary = trip.get("summary", {})

        steps: list[RouteStep] = []

        for leg in trip.get("legs", []):
            for maneuver in leg.get("maneuvers", []):
                steps.append(
                    RouteStep(
                        instruction=maneuver.get("instruction", ""),
                        distance_meters=round(maneuver.get("length", 0) * 1000),
                        duration_seconds=round(maneuver.get("time", 0)),
                        maneuver_type=str(maneuver.get("type"))
                        if maneuver.get("type") is not None
                        else None,
                    )
                )

        return RouteResult(
            provider="valhalla",
            distance_meters=round(summary.get("length", 0) * 1000),
            duration_seconds=round(summary.get("time", 0)),
            steps=steps,
            raw=data,
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Valhalla route request failed: {e}") from e

        return response.json()

    def _costing(self, mode: str) -> str:
        mapping = {
            "walking": "pedestrian",
            "driving": "auto",
            "bicycle": "bicycle",
        }

        try:
            return mapping[mode]
        except KeyError:
            raise ValueError(f"Unsupported travel mode: {mode}") from None
