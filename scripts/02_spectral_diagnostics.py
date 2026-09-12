"""
AMSE Spectral Diagnostics
Date: 2026-05-06

Run from AMSE root:

    python scripts\\02_spectral_diagnostics.py

Purpose:
    Create second-stage first-look diagnostics:
    1. Re-read the AMSE spectra from ZIP.
    2. Save the full raw spectral matrix.
    3. Create a spectral heatmap.
    4. Create altitude-binned median spectra.
    5. Create nominal 600-1000 nm wavelength-axis products.

This still does not perform final wavelength calibration.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_SPECTRA_DIR = PROJECT_ROOT / "data" / "raw_spectra"
PRODUCT_DIR = PROJECT_ROOT / "outputs" / "reduced_products"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

EXPECTED_N_PIXELS = 3648

PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def find_one_zip(directory: Path) -> Path:
    zips = sorted(directory.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No ZIP files found in {directory}")
    return zips[0]


def spectrum_id_from_name(name: str) -> int:
    match = re.search(r"test_data_(\d+)\.txt$", name.replace("\\", "/"))
    if not match:
        return -1
    return int(match.group(1))


def extract_numeric_spectrum(text: str, expected_pixels: int = EXPECTED_N_PIXELS) -> np.ndarray:
    numeric_values = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        tokens = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", stripped)

        if not tokens:
            continue

        # Skip date/time header lines.
        if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", stripped):
            continue
        if re.search(r"\d{1,2}/\d{1,2}/\d{4}", stripped):
            continue
        if ":" in stripped and len(tokens) <= 6:
            continue

        for token in tokens:
            numeric_values.append(float(token))

    arr = np.asarray(numeric_values, dtype=float)

    if arr.size < expected_pixels:
        raise ValueError(f"Only found {arr.size} samples; expected {expected_pixels}")

    if arr.size > expected_pixels:
        arr = arr[-expected_pixels:]

    return arr


def read_spectral_matrix(zip_path: Path) -> tuple[np.ndarray, list[int]]:
    spectra = []
    spectrum_ids = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [
            name for name in zf.namelist()
            if re.search(r"test_data_\d+\.txt$", name.replace("\\", "/"))
        ]

        names = sorted(names, key=spectrum_id_from_name)

        for name in names:
            spectrum_id = spectrum_id_from_name(name)
            text = zf.read(name).decode("utf-8", errors="ignore")
            spectrum = extract_numeric_spectrum(text)

            spectrum_ids.append(spectrum_id)
            spectra.append(spectrum)

    return np.vstack(spectra), spectrum_ids


def robust_scale_matrix(matrix: np.ndarray) -> tuple[float, float]:
    finite = matrix[np.isfinite(matrix)]
    vmin = np.percentile(finite, 1)
    vmax = np.percentile(finite, 99)
    return float(vmin), float(vmax)


def make_spectral_heatmap(matrix: np.ndarray) -> None:
    vmin, vmax = robust_scale_matrix(matrix)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Raw intensity")
    plt.xlabel("Detector Pixel")
    plt.ylabel("Spectrum Number")
    plt.title("AMSE Raw Spectral Image")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "spectral_heatmap_pixel_space.png", dpi=150)
    plt.close()


def make_nominal_wavelength_heatmap(matrix: np.ndarray, wavelength_nm: np.ndarray) -> None:
    vmin, vmax = robust_scale_matrix(matrix)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[wavelength_nm[0], wavelength_nm[-1], 0, matrix.shape[0] - 1],
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Raw intensity")
    plt.xlabel("Nominal Wavelength [nm]")
    plt.ylabel("Spectrum Number")
    plt.title("AMSE Raw Spectral Image, Nominal 600-1000 nm Mapping")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "spectral_heatmap_nominal_wavelength.png", dpi=150)
    plt.close()


def make_altitude_binned_median_spectra(
    matrix: np.ndarray,
    summary: pd.DataFrame,
    wavelength_nm: np.ndarray,
) -> None:
    df = summary.copy()
    df["row_index"] = np.arange(len(df))

    bins = [
        (0, 10000, "0-10 kft"),
        (10000, 50000, "10-50 kft"),
        (50000, 150000, "50-150 kft"),
        (150000, 250000, "150-250 kft"),
        (250000, 400000, "250-400 kft"),
    ]

    plt.figure(figsize=(12, 6))

    rows_written = []

    for low, high, label in bins:
        subset = df[(df["altitude_ft"] >= low) & (df["altitude_ft"] < high)]

        if subset.empty:
            continue

        rows = subset["row_index"].to_numpy(dtype=int)
        median_spectrum = np.nanmedian(matrix[rows, :], axis=0)

        plt.plot(wavelength_nm, median_spectrum, label=f"{label}, n={len(rows)}")

        out = pd.DataFrame(
            {
                "wavelength_nm_nominal": wavelength_nm,
                "median_intensity": median_spectrum,
                "altitude_bin": label,
                "n_spectra": len(rows),
            }
        )
        rows_written.append(out)

    plt.xlabel("Nominal Wavelength [nm]")
    plt.ylabel("Median Raw Intensity")
    plt.title("Altitude-Binned Median AMSE Spectra")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "altitude_binned_median_spectra_nominal_wavelength.png", dpi=150)
    plt.close()

    if rows_written:
        binned = pd.concat(rows_written, ignore_index=True)
        binned.to_csv(
            PRODUCT_DIR / "altitude_binned_median_spectra_nominal_wavelength.csv",
            index=False,
        )


def make_median_ratio_plot(
    matrix: np.ndarray,
    summary: pd.DataFrame,
    wavelength_nm: np.ndarray,
) -> None:
    df = summary.copy()
    df["row_index"] = np.arange(len(df))

    low_alt = df[df["altitude_ft"] < 10000]["row_index"].to_numpy(dtype=int)
    high_alt = df[df["altitude_ft"] > 250000]["row_index"].to_numpy(dtype=int)

    if len(low_alt) == 0 or len(high_alt) == 0:
        print("Skipping high/low altitude ratio plot because one bin is empty.")
        return

    low_med = np.nanmedian(matrix[low_alt, :], axis=0)
    high_med = np.nanmedian(matrix[high_alt, :], axis=0)

    ratio = high_med / np.where(low_med == 0, np.nan, low_med)

    plt.figure(figsize=(12, 5))
    plt.plot(wavelength_nm, ratio)
    plt.xlabel("Nominal Wavelength [nm]")
    plt.ylabel("High-altitude / Low-altitude median ratio")
    plt.title("AMSE High/Low Altitude Spectral Ratio")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "high_low_altitude_spectral_ratio.png", dpi=150)
    plt.close()

    pd.DataFrame(
        {
            "wavelength_nm_nominal": wavelength_nm,
            "high_over_low_ratio": ratio,
            "low_altitude_n": len(low_alt),
            "high_altitude_n": len(high_alt),
        }
    ).to_csv(PRODUCT_DIR / "high_low_altitude_spectral_ratio.csv", index=False)


def main() -> None:
    print("AMSE spectral diagnostics starting...")

    amse_zip = find_one_zip(RAW_SPECTRA_DIR)
    summary_path = PRODUCT_DIR / "amse_altitude_tagged_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            "Missing altitude-tagged summary. Run scripts\\01_first_look.py first."
        )

    summary = pd.read_csv(summary_path)
    matrix, spectrum_ids = read_spectral_matrix(amse_zip)

    wavelength_nm = np.linspace(600.0, 1000.0, matrix.shape[1])

    np.save(PRODUCT_DIR / "amse_raw_spectral_matrix.npy", matrix)

    pd.DataFrame(
        {
            "pixel": np.arange(matrix.shape[1]),
            "wavelength_nm_nominal": wavelength_nm,
        }
    ).to_csv(PRODUCT_DIR / "nominal_pixel_to_wavelength_600_1000nm.csv", index=False)

    make_spectral_heatmap(matrix)
    make_nominal_wavelength_heatmap(matrix, wavelength_nm)
    make_altitude_binned_median_spectra(matrix, summary, wavelength_nm)
    make_median_ratio_plot(matrix, summary, wavelength_nm)

    print()
    print("AMSE SPECTRAL DIAGNOSTICS SUMMARY")
    print("---------------------------------")
    print(f"Spectral matrix shape: {matrix.shape}")
    print(f"Minimum intensity:     {np.nanmin(matrix):.6g}")
    print(f"Maximum intensity:     {np.nanmax(matrix):.6g}")
    print(f"Median intensity:      {np.nanmedian(matrix):.6g}")
    print()
    print("Products written:")
    print(f"  {PRODUCT_DIR / 'amse_raw_spectral_matrix.npy'}")
    print(f"  {PRODUCT_DIR / 'nominal_pixel_to_wavelength_600_1000nm.csv'}")
    print(f"  {PRODUCT_DIR / 'altitude_binned_median_spectra_nominal_wavelength.csv'}")
    print(f"  {PRODUCT_DIR / 'high_low_altitude_spectral_ratio.csv'}")
    print()
    print("Figures written:")
    print(f"  {FIGURE_DIR / 'spectral_heatmap_pixel_space.png'}")
    print(f"  {FIGURE_DIR / 'spectral_heatmap_nominal_wavelength.png'}")
    print(f"  {FIGURE_DIR / 'altitude_binned_median_spectra_nominal_wavelength.png'}")
    print(f"  {FIGURE_DIR / 'high_low_altitude_spectral_ratio.png'}")
    print()
    print("AMSE spectral diagnostics complete.")


if __name__ == "__main__":
    main()