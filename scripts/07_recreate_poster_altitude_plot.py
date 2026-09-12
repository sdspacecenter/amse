"""
AMSE Poster-Style Altitude Spectra Plot
Date: 2026-05-06

Run from AMSE root:

    python scripts\\07_recreate_poster_altitude_plot.py

Purpose:
    Recreate the poster-style plot showing how the spectra change with altitude.

Approach:
    1. Use the +/-5 moving coadded raw spectra.
    2. Select representative coadds nearest chosen target altitudes.
    3. Make stacked/offset spectra plots for ascent and descent separately.
    4. Make a combined altitude-ordered spectral heatmap.

This is meant to reproduce the science content of the poster figure,
not necessarily the final exact visual styling.
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

WL_MIN = 630.0
WL_MAX = 970.0

DISPLAY_SMOOTH_PIXELS = 21

# Representative altitudes to sample for poster-style stacked traces.
TARGET_ALTITUDES_FT = [
    5000,
    25000,
    50000,
    75000,
    100000,
    150000,
    200000,
    250000,
    300000,
    330000,
]


def load_inputs():
    raw_coadds = np.load(PRODUCT_DIR / "amse_coadded_spectra_pm5.npy")
    inventory = pd.read_csv(PRODUCT_DIR / "amse_coadded_spectra_pm5_inventory.csv")
    wavelength = pd.read_csv(PRODUCT_DIR / "nominal_pixel_to_wavelength_600_1000nm.csv")[
        "wavelength_nm_nominal"
    ].to_numpy()

    return raw_coadds, inventory, wavelength


def restrict_wavelength(wavelength: np.ndarray, spectra: np.ndarray):
    mask = (wavelength >= WL_MIN) & (wavelength <= WL_MAX)
    return wavelength[mask], spectra[:, mask], mask


def choose_nearest_indices(
    inventory: pd.DataFrame,
    phase_name: str,
    target_altitudes_ft: list[float],
) -> pd.DataFrame:
    sub = inventory[inventory["flight_phase"] == phase_name].copy()

    rows = []

    if sub.empty:
        return pd.DataFrame()

    for target in target_altitudes_ft:
        idx = int((sub["center_altitude_ft"] - target).abs().idxmin())
        row = inventory.loc[idx]

        rows.append(
            {
                "target_altitude_ft": target,
                "center_index": int(row["center_index"]),
                "center_spectrum_id": int(row["center_spectrum_id"]),
                "center_altitude_ft": float(row["center_altitude_ft"]),
                "center_utc_time": row["center_utc_time"],
                "flight_phase": row["flight_phase"],
            }
        )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["center_index"]).sort_values("center_altitude_ft")
    return out.reset_index(drop=True)


def scale_for_shape_plot(y: np.ndarray) -> np.ndarray:
    """
    Scale each spectrum so shape changes are visible.

    We use a robust local scaling:
      - smooth first
      - subtract minimum
      - divide by (95th percentile - minimum)
    """

    ys = median_filter(y, size=DISPLAY_SMOOTH_PIXELS, mode="nearest").astype(float)

    y_min = np.nanmin(ys)
    y_p95 = np.nanpercentile(ys, 95)

    denom = y_p95 - y_min
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(ys)

    return (ys - y_min) / denom


def make_stacked_plot(
    raw_coadds: np.ndarray,
    selected: pd.DataFrame,
    wavelength: np.ndarray,
    title: str,
    output_name: str,
) -> None:
    if selected.empty:
        print(f"Skipping {output_name}: no spectra selected.")
        return

    plt.figure(figsize=(12, 8))

    offset_step = 1.15

    for i, row in selected.iterrows():
        idx = int(row["center_index"])
        y = raw_coadds[idx, :]
        y_plot = scale_for_shape_plot(y) + i * offset_step

        label = (
            f"{row['center_altitude_ft'] / 1000:.1f} kft "
            f"(id {int(row['center_spectrum_id'])})"
        )

        plt.plot(wavelength, y_plot, label=label, linewidth=1.5)
        plt.text(
            wavelength[-1] + 1.5,
            i * offset_step + 0.35,
            f"{row['center_altitude_ft'] / 1000:.1f} kft",
            va="center",
            fontsize=9,
        )

    for wl0, name in [
        (656.28, "H-alpha"),
        (686.7, "O2 B"),
        (720.0, "H2O"),
        (760.5, "O2 A"),
        (825.0, "H2O"),
        (940.0, "H2O"),
    ]:
        if WL_MIN <= wl0 <= WL_MAX:
            plt.axvline(wl0, linestyle=":", linewidth=1)

    plt.xlabel("Nominal Wavelength [nm]")
    plt.ylabel("Scaled spectrum + vertical offset")
    plt.title(title)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_name, dpi=150, bbox_inches="tight")
    plt.close()


def make_altitude_heatmap(
    raw_coadds: np.ndarray,
    inventory: pd.DataFrame,
    wavelength: np.ndarray,
    output_name: str,
    title: str,
    phase_filter: str | None = None,
) -> None:
    df = inventory.copy()

    if phase_filter is not None:
        df = df[df["flight_phase"] == phase_filter].copy()

    if df.empty:
        print(f"Skipping {output_name}: no rows for phase {phase_filter}")
        return

    df = df.sort_values("center_altitude_ft").reset_index(drop=True)

    matrix_rows = []
    altitudes = []

    for _, row in df.iterrows():
        idx = int(row["center_index"])
        y = raw_coadds[idx, :]
        y = scale_for_shape_plot(y)

        matrix_rows.append(y)
        altitudes.append(float(row["center_altitude_ft"]))

    matrix = np.vstack(matrix_rows)
    altitudes = np.asarray(altitudes)

    finite = matrix[np.isfinite(matrix)]
    vmin = np.percentile(finite, 2)
    vmax = np.percentile(finite, 98)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[wavelength[0], wavelength[-1], altitudes[0] / 1000.0, altitudes[-1] / 1000.0],
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Scaled spectral intensity")
    plt.xlabel("Nominal Wavelength [nm]")
    plt.ylabel("Altitude [kft]")
    plt.title(title)

    for wl0, name in [
        (656.28, "H-alpha"),
        (686.7, "O2 B"),
        (720.0, "H2O"),
        (760.5, "O2 A"),
        (825.0, "H2O"),
        (940.0, "H2O"),
    ]:
        if WL_MIN <= wl0 <= WL_MAX:
            plt.axvline(wl0, linestyle=":", linewidth=1)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_name, dpi=150)
    plt.close()


def main() -> None:
    print("AMSE poster-style altitude plot recreation starting...")

    raw_coadds, inventory, wavelength = load_inputs()
    wavelength_use, raw_coadds_use, _ = restrict_wavelength(wavelength, raw_coadds)

    ascent_selected = choose_nearest_indices(inventory, "ascent", TARGET_ALTITUDES_FT)
    descent_selected = choose_nearest_indices(inventory, "descent", TARGET_ALTITUDES_FT)

    ascent_selected.to_csv(
        PRODUCT_DIR / "poster_style_selected_ascent_spectra.csv",
        index=False,
    )
    descent_selected.to_csv(
        PRODUCT_DIR / "poster_style_selected_descent_spectra.csv",
        index=False,
    )

    make_stacked_plot(
        raw_coadds_use,
        ascent_selected,
        wavelength_use,
        "AMSE Poster-Style Spectra vs Altitude (Ascent)",
        "poster_style_altitude_spectra_ascent.png",
    )

    make_stacked_plot(
        raw_coadds_use,
        descent_selected,
        wavelength_use,
        "AMSE Poster-Style Spectra vs Altitude (Descent)",
        "poster_style_altitude_spectra_descent.png",
    )

    make_altitude_heatmap(
        raw_coadds_use,
        inventory,
        wavelength_use,
        "poster_style_altitude_heatmap_ascent.png",
        "AMSE Spectral Change with Altitude (Ascent)",
        phase_filter="ascent",
    )

    make_altitude_heatmap(
        raw_coadds_use,
        inventory,
        wavelength_use,
        "poster_style_altitude_heatmap_descent.png",
        "AMSE Spectral Change with Altitude (Descent)",
        phase_filter="descent",
    )

    print()
    print("POSTER-STYLE ALTITUDE PLOT PRODUCTS")
    print("-----------------------------------")
    print(f"  {PRODUCT_DIR / 'poster_style_selected_ascent_spectra.csv'}")
    print(f"  {PRODUCT_DIR / 'poster_style_selected_descent_spectra.csv'}")
    print()
    print("Figures written:")
    print(f"  {FIGURE_DIR / 'poster_style_altitude_spectra_ascent.png'}")
    print(f"  {FIGURE_DIR / 'poster_style_altitude_spectra_descent.png'}")
    print(f"  {FIGURE_DIR / 'poster_style_altitude_heatmap_ascent.png'}")
    print(f"  {FIGURE_DIR / 'poster_style_altitude_heatmap_descent.png'}")
    print()
    print("AMSE poster-style altitude plot recreation complete.")


if __name__ == "__main__":
    main()