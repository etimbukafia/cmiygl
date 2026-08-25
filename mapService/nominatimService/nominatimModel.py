from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

class NominatimAddress(BaseModel):
    # Common address fields, but Nominatim can return many more.
    house_number: Optional[str] = None
    road: Optional[str] = None
    neighbourhood: Optional[str] = None
    suburb: Optional[str] = None
    borough: Optional[str] = None
    city: Optional[str] = None
    town: Optional[str] = None
    village: Optional[str] = None
    county: Optional[str] = None
    state_district: Optional[str] = None
    state: Optional[str] = None
    postcode: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None

    # Store extra keys like shop, amenity, ISO3166-2-lvl4, etc.
    model_config = {
        "extra": "allow",
    }


class NominatimJsonV2Place(BaseModel):
    place_id: int | str
    licence: str

    osm_type: Optional[str] = None
    osm_id: Optional[int | str] = None

    lat: float
    lon: float

    place_rank: Optional[int | str] = None
    category: Optional[str] = None

    type: Optional[str] = None
    importance: Optional[float | str] = None
    addresstype: Optional[str] = None

    display_name: str
    name: Optional[str] = None
    address: Optional[NominatimAddress] = None

    boundingbox: Optional[list[str]] = None

    model_config = {
        "extra": "allow",
    }

    @field_validator("lat", "lon", mode="before")
    @classmethod
    def parse_float_string(cls, value: Any) -> float:
        return float(value)
