#!/usr/bin/env python3
"""
Build a lean combined Citi Bike CSV and a station lookup table using chunked reads.

Typical usage:
    python build_cleaned_combined.py \
        --input-dir "Datasets/Citibike" \
        --pattern "2025*.csv" \
        --output-cleaned "cleaned_combined.csv" \
        --output-stations "station_lookup.csv"

This script:
1. Reads monthly Citi Bike trip CSV files in chunks
2. Keeps only the columns useful for station-level hourly modeling
3. Drops rows missing critical station/timestamp fields
4. Writes a lean combined CSV incrementally
5. Builds a station lookup table with station_id, station_name, lat, lng
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


KEEP_COLS = [
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
    "start_station_name",
    "end_station_name",
    "rideable_type",
    "member_casual",
]

RAW_COLS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]

REQUIRED_FOR_CLEANED = [
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
]

DTYPE_MAP = {
    "ride_id": "string",
    "rideable_type": "string",
    "started_at": "string",
    "ended_at": "string",
    "start_station_name": "string",
    "start_station_id": "string",
    "end_station_name": "string",
    "end_station_id": "string",
    "start_lat": "float64",
    "start_lng": "float64",
    "end_lat": "float64",
    "end_lng": "float64",
    "member_casual": "string",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lean combined Citi Bike CSV in chunks.")
    parser.add_argument("--input-dir", required=True, help="Directory containing monthly Citi Bike CSV files")
    parser.add_argument("--pattern", default="2025*.csv", help="Glob pattern for monthly files inside input-dir")
    parser.add_argument("--output-cleaned", default="cleaned_combined.csv", help="Path to output cleaned combined CSV")
    parser.add_argument("--output-stations", default="station_lookup.csv", help="Path to output station lookup CSV")
    parser.add_argument("--output-summary", default="cleaning_summary.json", help="Path to output JSON summary")
    parser.add_argument("--chunksize", type=int, default=250_000, help="Rows per chunk when reading source CSVs")
    parser.add_argument("--dedupe-within-chunk", action="store_true", help="Drop duplicate ride_id rows within each chunk if ride_id exists")
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def discover_files(input_dir: str, pattern: str) -> List[Path]:
    files = sorted(Path(input_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {input_dir}")
    return files


def standardize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("started_at", "ended_at"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            df[col] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def clean_text_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df


def update_station_lookup(df: pd.DataFrame, station_lookup: Dict[str, Dict[str, object]]) -> None:
    mappings = [
        ("start_station_id", "start_station_name", "start_lat", "start_lng"),
        ("end_station_id", "end_station_name", "end_lat", "end_lng"),
    ]

    for id_col, name_col, lat_col, lng_col in mappings:
        present_cols = [c for c in [id_col, name_col, lat_col, lng_col] if c in df.columns]
        if id_col not in present_cols:
            continue

        station_df = df[present_cols].dropna(subset=[id_col]).copy()
        if station_df.empty:
            continue

        station_df[id_col] = station_df[id_col].astype("string").str.strip()
        station_df = station_df[station_df[id_col].notna()]
        station_df = station_df.drop_duplicates(subset=[id_col])

        for row in station_df.itertuples(index=False):
            record = dict(zip(station_df.columns, row))
            station_id = str(record.get(id_col)).strip()
            if not station_id or station_id == "<NA>":
                continue

            existing = station_lookup.get(station_id, {})
            candidate = {
                "station_id": station_id,
                "station_name": record.get(name_col),
                "lat": record.get(lat_col),
                "lng": record.get(lng_col),
            }

            merged = {
                "station_id": station_id,
                "station_name": existing.get("station_name") if pd.notna(existing.get("station_name")) else candidate.get("station_name"),
                "lat": existing.get("lat") if pd.notna(existing.get("lat")) else candidate.get("lat"),
                "lng": existing.get("lng") if pd.notna(existing.get("lng")) else candidate.get("lng"),
            }
            station_lookup[station_id] = merged


def process_file(
    file_path: Path,
    output_cleaned: Path,
    station_lookup: Dict[str, Dict[str, object]],
    chunksize: int,
    write_header: bool,
    dedupe_within_chunk: bool,
) -> dict:
    stats = {
        "file": str(file_path),
        "rows_read": 0,
        "rows_written": 0,
        "rows_dropped_missing": 0,
        "rows_dropped_duplicate_within_chunk": 0,
        "chunks": 0,
    }

    reader = pd.read_csv(
        file_path,
        chunksize=chunksize,
        dtype=DTYPE_MAP,
        low_memory=False,
        usecols=lambda c: c.strip().lower() in RAW_COLS,
    )

    mode = "w" if write_header else "a"

    for chunk_idx, chunk in enumerate(reader, start=1):
        stats["chunks"] += 1
        stats["rows_read"] += len(chunk)

        chunk = normalize_columns(chunk)

        if dedupe_within_chunk and "ride_id" in chunk.columns:
            before = len(chunk)
            chunk = chunk.drop_duplicates(subset=["ride_id"])
            stats["rows_dropped_duplicate_within_chunk"] += before - len(chunk)

        update_station_lookup(chunk, station_lookup)

        chunk = clean_text_columns(
            chunk,
            [
                "rideable_type",
                "started_at",
                "ended_at",
                "start_station_name",
                "start_station_id",
                "end_station_name",
                "end_station_id",
                "member_casual",
            ],
        )
        chunk = standardize_timestamps(chunk)

        present_required = [c for c in REQUIRED_FOR_CLEANED if c in chunk.columns]
        before_dropna = len(chunk)
        chunk = chunk.dropna(subset=present_required)
        stats["rows_dropped_missing"] += before_dropna - len(chunk)

        cleaned = chunk[[c for c in KEEP_COLS if c in chunk.columns]].copy()
        cleaned.to_csv(output_cleaned, mode=mode, header=write_header, index=False, quoting=csv.QUOTE_MINIMAL)

        mode = "a"
        write_header = False
        stats["rows_written"] += len(cleaned)

        print(
            f"[{file_path.name}] chunk {chunk_idx}: "
            f"read={stats['rows_read']:,} written={stats['rows_written']:,} "
            f"dropped_missing={stats['rows_dropped_missing']:,}"
        )

    return stats


def write_station_lookup(station_lookup: Dict[str, Dict[str, object]], output_stations: Path) -> int:
    if not station_lookup:
        pd.DataFrame(columns=["station_id", "station_name", "lat", "lng"]).to_csv(output_stations, index=False)
        return 0

    station_df = pd.DataFrame(station_lookup.values())
    station_df = station_df.sort_values("station_id").reset_index(drop=True)
    station_df.to_csv(output_stations, index=False)
    return len(station_df)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_cleaned = Path(args.output_cleaned)
    output_stations = Path(args.output_stations)
    output_summary = Path(args.output_summary)

    files = discover_files(str(input_dir), args.pattern)
    print(f"Found {len(files)} input files")
    for f in files:
        print(f" - {f}")

    if output_cleaned.exists():
        output_cleaned.unlink()
    if output_stations.exists():
        output_stations.unlink()

    station_lookup: Dict[str, Dict[str, object]] = {}
    all_stats: List[dict] = []
    write_header = True

    for file_path in files:
        stats = process_file(
            file_path=file_path,
            output_cleaned=output_cleaned,
            station_lookup=station_lookup,
            chunksize=args.chunksize,
            write_header=write_header,
            dedupe_within_chunk=args.dedupe_within_chunk,
        )
        all_stats.append(stats)
        write_header = False

    station_count = write_station_lookup(station_lookup, output_stations)

    summary = {
        "files_processed": len(files),
        "input_files": [str(f) for f in files],
        "output_cleaned": str(output_cleaned),
        "output_stations": str(output_stations),
        "station_count": station_count,
        "chunksize": args.chunksize,
        "per_file": all_stats,
        "totals": {
            "rows_read": int(sum(s["rows_read"] for s in all_stats)),
            "rows_written": int(sum(s["rows_written"] for s in all_stats)),
            "rows_dropped_missing": int(sum(s["rows_dropped_missing"] for s in all_stats)),
            "rows_dropped_duplicate_within_chunk": int(sum(s["rows_dropped_duplicate_within_chunk"] for s in all_stats)),
        },
    }

    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Cleaned CSV:   {output_cleaned.resolve()}")
    print(f"Station table: {output_stations.resolve()}")
    print(f"Summary JSON:  {output_summary.resolve()}")
    print(f"Unique stations: {station_count:,}")


if __name__ == "__main__":
    main()
