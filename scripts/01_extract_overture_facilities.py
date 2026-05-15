from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from cnrisk.config import load_project_config
from cnrisk.paths import DATA_RAW, ensure_dirs


def sql_list(values: list[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def build_where_clause(config: dict, bbox: list[float] | None, facility_type: str | None) -> str:
    terms: list[str] = []
    if facility_type:
        terms = config["facility_categories"][facility_type]
    else:
        for values in config["facility_categories"].values():
            terms.extend(values)
    category_filter = f"""
        (
            lower(categories.primary) IN ({sql_list(sorted(set(terms)))})
            OR (
                categories.alternate IS NOT NULL
                AND list_has_any(
                list_transform(categories.alternate, x -> lower(x)),
                [{sql_list(sorted(set(terms)))}]
                )
            )
        )
    """
    filters = [category_filter]
    if bbox:
        west, south, east, north = bbox
        filters.append(f"bbox.xmin BETWEEN {west} AND {east}")
        filters.append(f"bbox.ymin BETWEEN {south} AND {north}")
    return " AND ".join(f"({x})" for x in filters)


def extract(config_path: str, out_path: str, bbox: list[float] | None, facility_type: str | None) -> None:
    ensure_dirs()
    config = load_project_config(config_path)
    places_path = config["overture_places_path"]
    where_clause = build_where_clause(config, bbox, facility_type)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_region='{config['overture_s3_region']}';")
    query = f"""
        COPY (
            SELECT
                id,
                names.primary AS name,
                categories,
                confidence,
                CAST(sources AS JSON) AS sources,
                bbox.xmin AS lon,
                bbox.ymin AS lat,
                geometry
            FROM read_parquet('{places_path}', filename=true, hive_partitioning=1)
            WHERE {where_clause}
        ) TO '{out_path.replace("'", "''")}' (FORMAT PARQUET);
    """
    con.execute(query)
    metadata_path = Path(out_path).with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "source": "Overture Maps Places",
                "release": config["overture_release"],
                "bbox": bbox,
                "facility_type": facility_type,
                "query_where": where_clause,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--out", default=str(DATA_RAW / "overture" / "places_facilities.parquet"))
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--facility-type", choices=["school", "health"])
    args = parser.parse_args()
    extract(args.config, args.out, args.bbox, args.facility_type)


if __name__ == "__main__":
    main()
