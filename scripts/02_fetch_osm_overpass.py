from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from cnrisk.paths import DATA_RAW, ensure_dirs


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def build_query(bbox: list[float]) -> str:
    west, south, east, north = bbox
    bbox_q = f"{south},{west},{north},{east}"
    return f"""
    [out:json][timeout:180];
    (
      node["amenity"~"school|college|university|kindergarten|hospital|clinic|doctors"]({bbox_q});
      way["amenity"~"school|college|university|kindergarten|hospital|clinic|doctors"]({bbox_q});
      relation["amenity"~"school|college|university|kindergarten|hospital|clinic|doctors"]({bbox_q});
      node["healthcare"]({bbox_q});
      way["healthcare"]({bbox_q});
      relation["healthcare"]({bbox_q});
    );
    out center tags;
    """


def fetch(bbox: list[float], out_path: str) -> None:
    ensure_dirs()
    payload = None
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = requests.post(endpoint, data={"data": build_query(bbox)}, timeout=240)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as exc:
            last_error = exc
    if payload is None:
        raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_error}") from last_error
    features = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        lon = element.get("lon") or element.get("center", {}).get("lon")
        lat = element.get("lat") or element.get("center", {}).get("lat")
        if lon is None or lat is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "id": f"osm/{element['type']}/{element['id']}",
                    "name": tags.get("name"),
                    "amenity": tags.get("amenity"),
                    "healthcare": tags.get("healthcare"),
                    "source": "OpenStreetMap Overpass",
                },
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--out", default=str(DATA_RAW / "osm" / "osm_facilities.geojson"))
    args = parser.parse_args()
    fetch(args.bbox, args.out)


if __name__ == "__main__":
    main()
