"""
Template-only: request ERA5-HEAT UTCI from Copernicus CDS (requires ~/.cdsapirc).

Do not run blindly — monthly/hourly UTCI tiles are large. Typical workflow:
1. Define a tight lon/lat bbox per pilot city or small multi-city chip.
2. Download NetCDF for months or years needed.
3. Derive annual counts of days with daily_max_utci >= threshold (project-specific).
4. Raster-to-points join analogous to scripts/04_overlay_raster_hazards.py.

Dataset catalogue name changes; verify on https://cds.climate.copernicus.eu/
"""

from __future__ import annotations

# import cdsapi
#
# c = cdsapi.Client()
# c.retrieve(
#     "derived-utci-historical",
#     {
#         "variable": "universal_thermal_climate_index",
#         "product_type": "consolidated_dataset",
#         "year": "2024",
#         "month": ["06", "07", "08"],
#         "day": [f"{i:02d}" for i in range(1, 32)],
#         "time": [f"{h:02d}:00" for h in range(24)],
#         "area": [north, west, south, east],  # N, W, S, E — CDS convention
#         "format": "netcdf",
#     },
#     "download_utci_chip.nc",
# )


def main() -> None:
    raise SystemExit(
        "Edit this script with a real CDS request and credentials. "
        "See scripts/../docs/26_NATURE_FAMILY_DATA_V2_RUNBOOK.md"
    )


if __name__ == "__main__":
    main()
