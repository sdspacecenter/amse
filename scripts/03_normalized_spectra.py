"""
AMSE Normalized Spectral Diagnostics
Date: 2026-05-06

Run from AMSE root:

    python scripts\\03_normalized_spectra.py

Purpose:
    Create cleaner normalized spectra for calibration inspection:
    1. Load full spectral matrix from script 02.
    2. Smooth spectra with a median filter.
    3. Estimate continuum with a broad median filter.
    4. Create continuum-normalized spectra.
    5. Plot altitude-binned normalized median spectra.
    6. Plot high-altitude / low-altitude normalized ratio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_DIR = PROJECT_ROOT / "outputs" / "reduced_products"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    matrix_path = PRODUCT_DIR / "amse_raw_spectral_matrix.npy"
    summary_path = PRODUCT_DIR / "amse_altitude_tagged_summary.csv"
    wavelength_path = PRODUCT_DIR / "nominal_pixel_to_wavelength_600_1000nm.csv"

    if not matrix_path.exists():
        raise FileNotFoundError("Missing amse_raw_spectral_matrix.npy. Run script 02 first.")

    if not summary_path.exists():
        raise FileNotFoundError("Missing altitude summary. Run script 01 first.")

    if not wavelength_path.exists():
        raise FileNotFoundError("Missing nominal wavelength file. Run script 02 first.")

    matrix = np.load(matrix_path)
    summary = pd.read_csv(summary_path)
    wavelength = pd.read_csv(wavelength_path)["wavelength_nm_nominal"].to_numpy()

    return matrix, summary, wavelength


def normalize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth and continuum-normalize the spectra.

    The detector intensity scale is about 1-10.
    We use:
      - narrow median filter to suppress pixel spikes
      - broad median filter to estimate continuum
      - normalized = smoothed / continuum
    """

    smoothed = median_filter(matrix, size=(1, 9), mode="nearest")
    continuum = median_filter(smoothed, size=(1, 301), mode="nearest")

    continuum_safe = np.where(continuum <= 0, np.nan, continuum)
    normalized = smoothed / continuum_safe

    return smoothed, continuum, normalized


def plot_normalized_heatmap(normalized: np.ndarray, wavelength: np.ndarray) -> None:
    finite = normalized[np.isfinite(normalized)]
    vmin = np.percentile(finite, 2)
    vmax = np.percentile(finite, 98)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        normalized,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[wavelength[0], wavelength[-1], 0, normalized.shape[0] - 1],
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Continuum-normalized intensity")
    plt.xlabel("Nominal Wavelength [nm]")
    plt.ylabel("Spectrum Number")
    plt.title("AMSE Continuum-Normalized Spectral Image")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "normalized_spectral_heatmap_nominal_wavelength.png", dpi=150)
    plt.close()


def plot_altitude_binned_normalized_spectra(
    normalized: np.ndarray,
    summary: pd.DataFrame,
    wavelength: np.ndarray,
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

    out_rows = []

    plt.figure(figsize=(12, 6))

    for low, high, label in bins:
        subset = df[(df["altitude_ft"] >= low) & (df["altitude_ft"] < high)]

        if subset.empty:
            continue

        rows = subset["row_index"].to_numpy(dtype=int)
        med = np.nanmedian(normalized[rows, :], axis=0)

        # Light smoothing for display only.
        med_display = median_filter(med, size=21, mode="nearest")

        plt.plot(wavelength, med_display, label=f"{label}, n={len(rows)}")

        out_rows.append(
            pd.DataFrame(
                {
                    "wavelength_nm_nominal": wavelength,
                    "normalized_median_intensity": med,
                    "normalized_median_intensity_smoothed": med_display,
                    "altitude_bin": label,
                    "n_spectra": len(rows),
                }
            )
        )

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Nominal Wavelength [nm]")
    plt.ylabel("Continuum-normalized intensity")
    plt.title("Altitude-Binned Continuum-Normalized AMSE Spectra")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "altitude_binned_normalized_spectra.png", dpi=150)
    plt.close()

    if out_rows:
        pd.concat(out_rows, ignore_index=True).to_csv(
            PRODUCT_DIR / "altitude_binned_normalized_spectra.csv",
            index=False,
        )


def plot_candidate_absorption_features(
    normalized: np.ndarray,
    summary: pd.DataFrame,
    wavelength: np.ndarray,
) -> None:
    df = summary.copy()
    df["row_index"] = np.arange(len(df))

    low_rows = df[df["altitude_ft"] < 10000]["row_index"].to_numpy(dtype=int)
    high_rows = df[df["altitude_ft"] > 250000]["row_index"].to_numpy(dtype=int)

    if len(low_rows) == 0 or len(high_rows) == 0:
        print("Skipping candidate absorption plot; low or high altitude bin is empty.")
        return

    low_med = np.nanmedian(normalized[low_rows, :], axis=0)
    high_med = np.nanmedian(normalized[high_rows, :], axis=0)

    low_med_s = median_filter(low_med, size=21, mode="nearest")
    high_med_s = median_filter(high_med, size=21, mode="nearest")

    difference = low_med_s - high_med_s

    plt.figure(figsize=(12, 6))
    plt.plot(wavelength, low_med_s, label=f"Low altitude <10 kft, n={len(low_rows)}")
    plt.plot(wavelength, high_med_s, label=f"High altitude >250 kft, n={len(high_rows)}")
    plt.axhline(1.0, linestyle="--", linewidth=1)

    # Candidate telluric/solar anchors in nominal coordinates.
    for wl, name in [
        (656.28, "H-alpha"),
        (686.7, "O2 B"),
        (720.0, "H2O"),
        (760.5, "O2 A"),
        (825.0, "H2O"),
        (940.0, "H2O"),
    ]:
        plt.axvline(wl, linestyle=":", linewidth=1)
        plt.text(wl, 0.92, name, rotation=90, va="bottom", ha="right", fontsize=8)

    plt.xlabel("Nominal Wavelength [nm]")
    plt.ylabel("Continuum-normalized intensity")
    plt.title("Low vs High Altitude Normalized Spectra with Candidate Anchors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "candidate_absorption_features_nominal_wavelength.png", dpi=150)
    plt.close()

    pd.DataFrame(
        {
            "wavelength_nm_nominal": wavelength,
            "low_altitude_normalized_median": low_med,
            "high_altitude_normalized_median": high_med,
            "low_minus_high_smoothed": difference,
            "low_altitude_n": len(low_rows),
            "high_altitude_n": len(high_rows),
        }
    ).to_csv(PRODUCT_DIR / "candidate_absorption_features_nominal_wavelength.csv", index=False)


def main() -> None:
    print("AMSE normalized spectral diagnostics starting...")

    matrix, summary, wavelength = load_inputs()
    smoothed, continuum, normalized = normalize_matrix(matrix)

    np.save(PRODUCT_DIR / "amse_smoothed_spectral_matrix.npy", smoothed)
    np.save(PRODUCT_DIR / "amse_continuum_matrix.npy", continuum)
    np.save(PRODUCT_DIR / "amse_normalized_spectral_matrix.npy", normalized)

    plot_normalized_heatmap(normalized, wavelength)
    plot_altitude_binned_normalized_spectra(normalized, summary, wavelength)
    plot_candidate_absorption_features(normalized, summary, wavelength)

    print()
    print("AMSE NORMALIZED DIAGNOSTICS SUMMARY")
    print("-----------------------------------")
    print(f"Input matrix shape:       {matrix.shape}")
    print(f"Normalized matrix shape:  {normalized.shape}")
    print(f"Normalized median:        {np.nanmedian(normalized):.6g}")
    print(f"Normalized 2-98 pct:      {np.nanpercentile(normalized, 2):.6g}, {np.nanpercentile(normalized, 98):.6g}")
    print()
    print("Products written:")
    print(f"  {PRODUCT_DIR / 'amse_smoothed_spectral_matrix.npy'}")
    print(f"  {PRODUCT_DIR / 'amse_continuum_matrix.npy'}")
    print(f"  {PRODUCT_DIR / 'amse_normalized_spectral_matrix.npy'}")
    print(f"  {PRODUCT_DIR / 'altitude_binned_normalized_spectra.csv'}")
    print(f"  {PRODUCT_DIR / 'candidate_absorption_features_nominal_wavelength.csv'}")
    print()
    print("Figures written:")
    print(f"  {FIGURE_DIR / 'normalized_spectral_heatmap_nominal_wavelength.png'}")
    print(f"  {FIGURE_DIR / 'altitude_binned_normalized_spectra.png'}")
    print(f"  {FIGURE_DIR / 'candidate_absorption_features_nominal_wavelength.png'}")
    print()
    print("AMSE normalized spectral diagnostics complete.")


if __name__ == "__main__":
    main()
