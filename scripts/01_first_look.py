"""
AMSE First-Look Flight Data Reduction
Date: 2026-05-06

Run from the AMSE project root:

    python scripts\\01_first_look.py

Purpose:
    1. Read AMSE raw spectrum files from the ZIP archive.
    2. Read NASA P-9 payload flight report from the ZIP archive.
    3. Parse AMSE local timestamps.
    4. Convert AMSE local timestamps to UTC.
    5. Interpolate NASA altitude onto every AMSE spectrum.
    6. Flag saturated spectra.
    7. Write first-look CSV/Parquet products and PNG plots.

This is intentionally a first-look script, not the final science reduction.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_SPECTRA_DIR = PROJECT_ROOT / "data" / "raw_spectra"
NASA_REPORT_DIR = PROJECT_ROOT / "data" / "nasa_flight_report"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
PRODUCT_DIR = PROJECT_ROOT / "outputs" / "reduced_products"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EXPECTED_N_SPECTRA = 138
EXPECTED_N_PIXELS = 3648

# Conservative default for 16-bit detector saturation.
# If the detector scale is different, this can be adjusted later.
SATURATION_LEVEL = 65535
SATURATION_FRACTION = 0.98
SATURATION_THRESHOLD = SATURATION_LEVEL * SATURATION_FRACTION


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def find_one_zip(directory: Path, description: str) -> Path:
    zips = sorted(directory.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No ZIP file found for {description} in: {directory}")
    if len(zips) > 1:
        print(f"WARNING: Multiple ZIP files found for {description}. Using: {zips[0].name}")
    return zips[0]


def spectrum_id_from_name(name: str) -> int:
    match = re.search(r"test_data_(\d+)\.txt$", name.replace("\\", "/"))
    if not match:
        return -1
    return int(match.group(1))


def parse_timestamp_from_text(text: str) -> pd.Timestamp | pd.NaT:
    """
    Try several timestamp patterns commonly found in instrument text files.
    Returns a naive pandas Timestamp interpreted as local AMSE time.
    """

    patterns = [
        # 2019-12-11 14:23:01 or 2019/12/11 14:23:01
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)",

        # 12/11/2019 14:23:01
        r"(\d{1,2}/\d{1,2}/\d{4}[ T]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)",

        # 11-Dec-2019 14:23:01
        r"(\d{1,2}[- ][A-Za-z]{3,9}[- ]\d{4}[ T]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            ts = pd.to_datetime(value, errors="coerce")
            if not pd.isna(ts):
                return ts

    return pd.NaT


def extract_numeric_spectrum(text: str, expected_pixels: int = EXPECTED_N_PIXELS) -> np.ndarray:
    """
    Extract detector samples from a text file.

    The AMSE files are expected to contain 3648 detector samples.
    This function is deliberately tolerant of headers. It extracts numeric
    values line by line and uses the final 3648 numeric values if extra
    numeric metadata are present before the spectrum.
    """

    numeric_values: list[float] = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        # Skip obvious header lines unless they contain a simple numeric list.
        tokens = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", stripped)

        if not tokens:
            continue

        # Avoid treating date/time header lines as detector data.
        if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", stripped):
            continue
        if re.search(r"\d{1,2}/\d{1,2}/\d{4}", stripped):
            continue
        if ":" in stripped and len(tokens) <= 6:
            continue

        for token in tokens:
            try:
                numeric_values.append(float(token))
            except ValueError:
                pass

    arr = np.asarray(numeric_values, dtype=float)

    if arr.size < expected_pixels:
        raise ValueError(f"Only found {arr.size} numeric samples; expected {expected_pixels}")

    if arr.size > expected_pixels:
        arr = arr[-expected_pixels:]

    return arr


def read_amse_spectra_from_zip(zip_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    spectra = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [
            name for name in zf.namelist()
            if re.search(r"test_data_\d+\.txt$", name.replace("\\", "/"))
        ]

        names = sorted(names, key=spectrum_id_from_name)

        print(f"Found {len(names)} AMSE spectrum text files in ZIP.")

        for name in names:
            spectrum_id = spectrum_id_from_name(name)

            raw = zf.read(name)
            text = raw.decode("utf-8", errors="ignore")

            local_timestamp = parse_timestamp_from_text(text)
            spectrum = extract_numeric_spectrum(text)

            spectra.append(spectrum)

            rows.append(
                {
                    "filename": Path(name).name,
                    "zip_member": name,
                    "spectrum_id": spectrum_id,
                    "local_timestamp": local_timestamp,
                    "n_pixels": int(spectrum.size),
                    "min_intensity": float(np.nanmin(spectrum)),
                    "max_intensity": float(np.nanmax(spectrum)),
                    "mean_intensity": float(np.nanmean(spectrum)),
                    "median_intensity": float(np.nanmedian(spectrum)),
                    "saturation_flag": bool(np.nanmax(spectrum) >= SATURATION_THRESHOLD),
                }
            )

    inventory = pd.DataFrame(rows)

    if len(spectra) == 0:
        raise RuntimeError("No AMSE spectra were read.")

    spectrum_array = np.vstack(spectra)

    return inventory, spectrum_array


def find_nasa_csv_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]

    if not csv_names:
        raise FileNotFoundError(f"No CSV file found inside NASA ZIP: {zip_path}")

    if len(csv_names) > 1:
        print(f"WARNING: Multiple CSV files found in NASA ZIP. Using: {csv_names[0]}")

    return csv_names[0]


def find_time_column(columns: list[str]) -> str:
    preferred = []

    for col in columns:
        c = col.lower()
        if "utc" in c and "time" in c:
            preferred.append(col)

    if preferred:
        return preferred[0]

    for col in columns:
        c = col.lower()
        if c == "utc" or "utc" in c:
            return col

    for col in columns:
        c = col.lower()
        if "time" in c:
            return col

    raise KeyError(f"Could not identify NASA UTC/time column. Columns are: {columns}")


def find_altitude_column(columns: list[str]) -> str:
    for col in columns:
        if col == "Altitude_FT":
            return col

    for col in columns:
        c = col.lower()
        if "altitude" in c and ("ft" in c or "feet" in c):
            return col

    for col in columns:
        c = col.lower()
        if "altitude" in c:
            return col

    raise KeyError(f"Could not identify altitude column. Columns are: {columns}")


def read_nasa_flight_report(zip_path: Path) -> pd.DataFrame:
    csv_name = find_nasa_csv_in_zip(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(csv_name) as f:
            nasa = pd.read_csv(f)

    time_col = find_time_column(list(nasa.columns))
    altitude_col = find_altitude_column(list(nasa.columns))

    print(f"NASA time column: {time_col}")
    print(f"NASA altitude column: {altitude_col}")

    nasa["utc_time"] = pd.to_datetime(nasa[time_col], errors="coerce", utc=True)
    nasa["altitude_ft"] = pd.to_numeric(nasa[altitude_col], errors="coerce")

    nasa = nasa.dropna(subset=["utc_time", "altitude_ft"]).copy()
    nasa = nasa.sort_values("utc_time").drop_duplicates(subset=["utc_time"])

    if nasa.empty:
        raise RuntimeError("NASA flight report has no usable UTC/altitude rows.")

    nasa["mission_elapsed_s"] = (
        nasa["utc_time"] - nasa["utc_time"].iloc[0]
    ).dt.total_seconds()

    return nasa


def choose_local_to_utc_offset_hours(
    local_times: pd.Series,
    nasa_utc_times: pd.Series,
) -> int:
    """
    Determine local-to-UTC offset by maximizing overlap with the NASA UTC range.

    For California in December 2019, local PST to UTC should be +8 hours.
    This auto-check helps avoid hard-coding that assumption.
    """

    valid_local = pd.to_datetime(local_times, errors="coerce").dropna()

    if valid_local.empty:
        raise RuntimeError("No valid AMSE local timestamps were parsed from the spectrum files.")

    nasa_min = nasa_utc_times.min()
    nasa_max = nasa_utc_times.max()

    best_offset = None
    best_count = -1

    for offset in range(-12, 15):
        candidate = valid_local + pd.to_timedelta(offset, unit="h")
        candidate = candidate.dt.tz_localize("UTC")

        count = int(((candidate >= nasa_min) & (candidate <= nasa_max)).sum())

        if count > best_count:
            best_count = count
            best_offset = offset

    if best_offset is None:
        raise RuntimeError("Could not determine AMSE local-to-UTC offset.")

    print(f"Selected AMSE local-to-UTC offset: {best_offset:+d} hours")
    print(f"AMSE spectra inside NASA time range after offset: {best_count} / {len(valid_local)}")

    return int(best_offset)


def add_altitude_to_inventory(inventory: pd.DataFrame, nasa: pd.DataFrame) -> pd.DataFrame:
    offset_hours = choose_local_to_utc_offset_hours(inventory["local_timestamp"], nasa["utc_time"])

    out = inventory.copy()

    local_ts = pd.to_datetime(out["local_timestamp"], errors="coerce")
    out["utc_time"] = (local_ts + pd.to_timedelta(offset_hours, unit="h")).dt.tz_localize("UTC")

    amse_seconds = out["utc_time"].astype("int64") / 1e9
    nasa_seconds = nasa["utc_time"].astype("int64") / 1e9

    out["altitude_ft"] = np.interp(
        amse_seconds,
        nasa_seconds,
        nasa["altitude_ft"],
        left=np.nan,
        right=np.nan,
    )

    out["mission_elapsed_s"] = (
        out["utc_time"] - nasa["utc_time"].iloc[0]
    ).dt.total_seconds()

    out["local_to_utc_offset_hours"] = offset_hours

    return out


def save_plots(summary: pd.DataFrame, nasa: pd.DataFrame, spectra: np.ndarray) -> None:
    # NASA altitude versus time
    plt.figure(figsize=(10, 5))
    plt.plot(nasa["utc_time"], nasa["altitude_ft"])
    plt.xlabel("UTC Time")
    plt.ylabel("Altitude [ft]")
    plt.title("NASA P-9 Altitude vs UTC Time")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "altitude_vs_time.png", dpi=150)
    plt.close()

    # AMSE spectrum max versus altitude
    plt.figure(figsize=(8, 5))
    plt.scatter(summary["altitude_ft"], summary["max_intensity"], s=18)
    plt.xlabel("Interpolated Altitude [ft]")
    plt.ylabel("Maximum Spectrum Intensity")
    plt.title("AMSE Spectrum Maximum vs Altitude")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "spectrum_max_vs_altitude.png", dpi=150)
    plt.close()

    # Sample raw spectra
    n = spectra.shape[0]
    sample_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

    plt.figure(figsize=(10, 5))
    for idx in sample_indices:
        plt.plot(np.arange(spectra.shape[1]), spectra[idx], label=f"spectrum {idx}")
    plt.xlabel("Detector Pixel")
    plt.ylabel("Raw Intensity")
    plt.title("Sample Raw AMSE Spectra")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sample_raw_spectra.png", dpi=150)
    plt.close()

    # Saturation timeline
    plt.figure(figsize=(10, 5))
    plt.plot(summary["utc_time"], summary["max_intensity"], marker="o", linestyle="-")
    plt.axhline(SATURATION_THRESHOLD, linestyle="--", label="saturation threshold")
    plt.xlabel("UTC Time")
    plt.ylabel("Maximum Spectrum Intensity")
    plt.title("AMSE Saturation Timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "saturation_timeline.png", dpi=150)
    plt.close()


def main() -> None:
    print("AMSE first-look reduction starting...")
    print(f"Project root: {PROJECT_ROOT}")

    amse_zip = find_one_zip(RAW_SPECTRA_DIR, "AMSE spectra")
    nasa_zip = find_one_zip(NASA_REPORT_DIR, "NASA flight report")

    print(f"AMSE ZIP: {amse_zip}")
    print(f"NASA ZIP: {nasa_zip}")

    inventory, spectra = read_amse_spectra_from_zip(amse_zip)
    nasa = read_nasa_flight_report(nasa_zip)

    summary = add_altitude_to_inventory(inventory, nasa)

    inventory_path = PRODUCT_DIR / "amse_spectrum_inventory.csv"
    summary_csv_path = PRODUCT_DIR / "amse_altitude_tagged_summary.csv"
    summary_parquet_path = PRODUCT_DIR / "amse_altitude_tagged_summary.parquet"

    inventory.to_csv(inventory_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    summary.to_parquet(summary_parquet_path, index=False)

    save_plots(summary, nasa, spectra)

    print()
    print("AMSE FIRST-LOOK SUMMARY")
    print("-----------------------")
    print(f"Number of spectra read: {len(inventory)}")
    print(f"Expected spectra:       {EXPECTED_N_SPECTRA}")
    print(f"Pixels per spectrum:    {spectra.shape[1]}")
    print(f"Expected pixels:        {EXPECTED_N_PIXELS}")
    print(f"Saturated spectra:      {int(summary['saturation_flag'].sum())}")
    print(f"NASA altitude rows:     {len(nasa)}")
    print()
    print("Output products written:")
    print(f"  {inventory_path}")
    print(f"  {summary_csv_path}")
    print(f"  {summary_parquet_path}")
    print()
    print("Figures written:")
    print(f"  {FIGURE_DIR / 'altitude_vs_time.png'}")
    print(f"  {FIGURE_DIR / 'spectrum_max_vs_altitude.png'}")
    print(f"  {FIGURE_DIR / 'sample_raw_spectra.png'}")
    print(f"  {FIGURE_DIR / 'saturation_timeline.png'}")
    print()
    print("AMSE first-look reduction complete.")


if __name__ == "__main__":
    main()
