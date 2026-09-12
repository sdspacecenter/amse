"""
AMSE Pairwise Ascent/Descent Coadd Plots
Date: 2026-05-06

Run from AMSE root:

    python scripts\\05_pairwise_altitude_coadd_plots.py

Purpose:
    Make one clean ascent/descent comparison figure per matched altitude.

Inputs:
    outputs\\reduced_products\\amse_coadded_normalized_spectra_pm5.npy
    outputs\\reduced_products\\ascent_descent_nearest_altitude_pairs.csv
    outputs\\reduced_products\\nominal_pixel_to_wavelength_600_1000nm.csv

Outputs:
    outputs\\figures\\pairwise_ascent_descent_025kft.png
    outputs\\figures\\pairwise_ascent_descent_050kft.png
    ...
    outputs\\reduced_products\\pairwise_ascent_descent_difference_metrics.csv
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

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

# Focus on the more stable central detector/wavelength region first.
WL_MIN = 630.0
WL_MAX = 970.0

DISPLAY_SMOOTH_PIXELS = 21


def load_inputs() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    coadd_path = PRODUCT_DIR / "amse_coadded_normalized_spectra_pm5.npy"
    pairs_path = PRODUCT_DIR / "ascent_descent_nearest_altitude_pairs.csv"
    wavelength_path = PRODUCT_DIR / "nominal_pixel_to_wavelength_600_1000nm.csv"

    if not coadd_path.exists():
        raise FileNotFoundError("Missing normalized coadds. Run scripts\\04_coadd_ascent_descent.py first.")

    if not pairs_path.exists():
        raise FileNotFoundError("Missing ascent/descent pairs. Run scripts\\04_coadd_ascent_descent.py first.")

    if not wavelength_path.exists():
        raise FileNotFoundError("Missing wavelength file. Run scripts\\02_spectral_diagnostics.py first.")

    coadds = np.load(coadd_path)
    pairs = pd.read_csv(pairs_path)
    wavelength = pd.read_csv(wavelength_path)["wavelength_nm_nominal"].to_numpy()

    return coadds, pairs, wavelength


def robust_limits(y1: np.ndarray, y2: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([y1[np.isfinite(y1)], y2[np.isfinite(y2)]])
    lo = np.percentile(values, 1)
    hi = np.percentile(values, 99)
    margin = 0.10 * (hi - lo)
    return float(lo - margin), float(hi + margin)


def make_pair_plot(
    coadds: np.ndarray,
    wavelength: np.ndarray,
    pair: pd.Series,
) -> dict:
    target_ft = float(pair["target_altitude_ft"])

    a_idx = int(pair["ascent_index"])
    d_idx = int(pair["descent_index"])

    ascent = coadds[a_idx, :]
    descent = coadds[d_idx, :]

    ascent_s = median_filter(ascent, size=DISPLAY_SMOOTH_PIXELS, mode="nearest")
    descent_s = median_filter(descent, size=DISPLAY_SMOOTH_PIXELS, mode="nearest")
    diff_s = ascent_s - descent_s

    mask = (
        np.isfinite(wavelength)
        & np.isfinite(ascent_s)
        & np.isfinite(descent_s)
        & (wavelength >= WL_MIN)
        & (wavelength <= WL_MAX)
    )

    wl = wavelength[mask]
    a = ascent_s[mask]
    d = descent_s[mask]
    diff = diff_s[mask]

    ylo, yhi = robust_limits(a, d)

    target_kft = int(round(target_ft / 1000.0))
    filename = FIGURE_DIR / f"pairwise_ascent_descent_{target_kft:03d}kft.png"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[2, 1])

    axes[0].plot(wl, a, label=f"Ascent id {int(pair['ascent_spectrum_id'])}, {pair['ascent_altitude_ft'] / 1000:.1f} kft")
    axes[0].plot(wl, d, label=f"Descent id {int(pair['descent_spectrum_id'])}, {pair['descent_altitude_ft'] / 1000:.1f} kft")
    axes[0].axhline(1.0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Normalized coadd")
    axes[0].set_ylim(ylo, yhi)
    axes[0].legend()
    axes[0].set_title(
        f"AMSE Ascent/Descent Coadd Comparison near {target_kft} kft "
        f"(Δalt={pair['altitude_difference_ft']:.0f} ft)"
    )

    axes[1].plot(wl, diff)
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Nominal Wavelength [nm]")
    axes[1].set_ylabel("Ascent - Descent")
    axes[1].set_title("Difference Spectrum")

    # Candidate reference anchors for visual orientation only.
    anchors = [
        (656.28, "H-alpha"),
        (686.7, "O2 B"),
        (720.0, "H2O"),
        (760.5, "O2 A"),
        (825.0, "H2O"),
        (940.0, "H2O"),
    ]

    for ax in axes:
        for wl0, name in anchors:
            if WL_MIN <= wl0 <= WL_MAX:
                ax.axvline(wl0, linestyle=":", linewidth=1)
                ax.text(wl0, ax.get_ylim()[0], name, rotation=90, va="bottom", ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

    rms_diff = float(np.sqrt(np.nanmean(diff ** 2)))
    median_abs_diff = float(np.nanmedian(np.abs(diff)))
    max_abs_diff = float(np.nanmax(np.abs(diff)))

    return {
        "target_altitude_ft": target_ft,
        "target_altitude_kft": target_kft,
        "ascent_index": a_idx,
        "descent_index": d_idx,
        "ascent_spectrum_id": int(pair["ascent_spectrum_id"]),
        "descent_spectrum_id": int(pair["descent_spectrum_id"]),
        "ascent_altitude_ft": float(pair["ascent_altitude_ft"]),
        "descent_altitude_ft": float(pair["descent_altitude_ft"]),
        "altitude_difference_ft": float(pair["altitude_difference_ft"]),
        "wavelength_min_nm": WL_MIN,
        "wavelength_max_nm": WL_MAX,
        "rms_difference": rms_diff,
        "median_absolute_difference": median_abs_diff,
        "max_absolute_difference": max_abs_diff,
        "figure": str(filename),
    }


def main() -> None:
    print("AMSE pairwise ascent/descent plotting starting...")

    coadds, pairs, wavelength = load_inputs()

    metrics = []

    for _, pair in pairs.iterrows():
        result = make_pair_plot(coadds, wavelength, pair)
        metrics.append(result)

    metrics_df = pd.DataFrame(metrics)
    metrics_path = PRODUCT_DIR / "pairwise_ascent_descent_difference_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print()
    print("PAIRWISE ASCENT/DESCENT METRICS")
    print("-------------------------------")
    print(metrics_df[
        [
            "target_altitude_kft",
            "ascent_spectrum_id",
            "descent_spectrum_id",
            "altitude_difference_ft",
            "rms_difference",
            "median_absolute_difference",
        ]
    ])
    print()
    print(f"Metrics written: {metrics_path}")
    print()
    print("Figures written:")
    for fig in metrics_df["figure"]:
        print(f"  {fig}")

    print()
    print("AMSE pairwise ascent/descent plotting complete.")


if __name__ == "__main__":
    main()