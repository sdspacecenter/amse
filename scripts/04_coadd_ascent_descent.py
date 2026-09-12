"""
AMSE Coadded Ascent/Descent Spectra
Date: 2026-05-06

Run from AMSE root:

    python scripts\\04_coadd_ascent_descent.py

Purpose:
    Create moving-window coadded spectra around each flight time point.

    For each center spectrum i, coadd spectra i-5 through i+5.
    Then label each coadd by altitude, UTC time, and flight phase.

Outputs:
    outputs\\reduced_products\\amse_coadded_spectra_pm5.npy
    outputs\\reduced_products\\amse_coadded_normalized_spectra_pm5.npy
    outputs\\reduced_products\\amse_coadded_spectra_pm5_inventory.csv

    outputs\\figures\\coadd_altitude_vs_spectrum_id.png
    outputs\\figures\\coadded_selected_timepoints_raw.png
    outputs\\figures\\coadded_selected_timepoints_normalized.png
    outputs\\figures\\coadded_ascent_descent_same_altitude.png
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

WINDOW_HALF_WIDTH = 5
WINDOW_NAME = "pm5"


def load_inputs() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    raw_matrix_path = PRODUCT_DIR / "amse_raw_spectral_matrix.npy"
    norm_matrix_path = PRODUCT_DIR / "amse_normalized_spectral_matrix.npy"
    summary_path = PRODUCT_DIR / "amse_altitude_tagged_summary.csv"
    wavelength_path = PRODUCT_DIR / "nominal_pixel_to_wavelength_600_1000nm.csv"

    if not raw_matrix_path.exists():
        raise FileNotFoundError("Missing raw matrix. Run scripts\\02_spectral_diagnostics.py first.")

    if not norm_matrix_path.exists():
        raise FileNotFoundError("Missing normalized matrix. Run scripts\\03_normalized_spectra.py first.")

    if not summary_path.exists():
        raise FileNotFoundError("Missing altitude summary. Run scripts\\01_first_look.py first.")

    if not wavelength_path.exists():
        raise FileNotFoundError("Missing wavelength file. Run scripts\\02_spectral_diagnostics.py first.")

    raw = np.load(raw_matrix_path)
    normalized = np.load(norm_matrix_path)
    summary = pd.read_csv(summary_path)
    wavelength = pd.read_csv(wavelength_path)["wavelength_nm_nominal"].to_numpy()

    return raw, normalized, summary, wavelength


def classify_flight_phase(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    peak_index = int(out["altitude_ft"].idxmax())
    out["flight_phase"] = "descent"
    out.loc[:peak_index, "flight_phase"] = "ascent"

    # Ground-like spectra are low altitude at beginning/end.
    out.loc[out["altitude_ft"] < 10000, "flight_phase"] = (
        out.loc[out["altitude_ft"] < 10000]
        .index.to_series()
        .apply(lambda idx: "pre-ascent-low" if idx < peak_index else "post-descent-low")
    )

    out["is_peak_region"] = False
    out.loc[max(0, peak_index - 3): min(len(out) - 1, peak_index + 3), "is_peak_region"] = True

    return out


def make_moving_coadds(
    matrix: np.ndarray,
    summary: pd.DataFrame,
    half_width: int = WINDOW_HALF_WIDTH,
    method: str = "median",
) -> tuple[np.ndarray, pd.DataFrame]:
    n_spectra, n_pixels = matrix.shape

    coadds = np.full((n_spectra, n_pixels), np.nan)
    rows = []

    for center in range(n_spectra):
        lo = max(0, center - half_width)
        hi = min(n_spectra, center + half_width + 1)

        block = matrix[lo:hi, :]

        if method == "median":
            coadd = np.nanmedian(block, axis=0)
        elif method == "mean":
            coadd = np.nanmean(block, axis=0)
        else:
            raise ValueError(f"Unknown coadd method: {method}")

        coadds[center, :] = coadd

        center_row = summary.iloc[center]

        rows.append(
            {
                "center_index": center,
                "center_spectrum_id": int(center_row["spectrum_id"]),
                "center_filename": center_row["filename"],
                "center_local_timestamp": center_row["local_timestamp"],
                "center_utc_time": center_row["utc_time"],
                "center_altitude_ft": float(center_row["altitude_ft"]),
                "center_mission_elapsed_s": float(center_row["mission_elapsed_s"]),
                "flight_phase": center_row["flight_phase"],
                "is_peak_region": bool(center_row["is_peak_region"]),
                "window_start_index": lo,
                "window_stop_index_exclusive": hi,
                "n_spectra_coadded": hi - lo,
                "coadd_method": method,
                "window_half_width": half_width,
                "coadd_min": float(np.nanmin(coadd)),
                "coadd_max": float(np.nanmax(coadd)),
                "coadd_median": float(np.nanmedian(coadd)),
            }
        )

    inventory = pd.DataFrame(rows)
    return coadds, inventory


def plot_coadd_altitude_inventory(inventory: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))

    for phase, sub in inventory.groupby("flight_phase"):
        plt.scatter(
            sub["center_spectrum_id"],
            sub["center_altitude_ft"],
            s=25,
            label=phase,
        )

    plt.xlabel("Center Spectrum ID")
    plt.ylabel("Center Altitude [ft]")
    plt.title("AMSE Coadd Centers by Altitude and Flight Phase")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "coadd_altitude_vs_spectrum_id.png", dpi=150)
    plt.close()


def choose_representative_indices(inventory: pd.DataFrame) -> list[int]:
    """
    Select representative time points across the flight.
    """

    targets_ft = [5000, 25000, 75000, 150000, 250000, 330000]

    indices = []

    for target in targets_ft:
        idx = int((inventory["center_altitude_ft"] - target).abs().idxmin())
        indices.append(idx)

    # Include first, peak, and last.
    indices.append(0)
    indices.append(int(inventory["center_altitude_ft"].idxmax()))
    indices.append(len(inventory) - 1)

    return sorted(set(indices))


def plot_selected_coadds(
    coadds: np.ndarray,
    inventory: pd.DataFrame,
    wavelength: np.ndarray,
    output_name: str,
    title: str,
    smooth_for_display: bool = True,
) -> None:
    indices = choose_representative_indices(inventory)

    plt.figure(figsize=(12, 6))

    for idx in indices:
        row = inventory.iloc[idx]
        y = coadds[idx, :]

        if smooth_for_display:
            y = median_filter(y, size=21, mode="nearest")

        label = (
            f"id={int(row['center_spectrum_id'])}, "
            f"{row['center_altitude_ft'] / 1000:.0f} kft, "
            f"{row['flight_phase']}"
        )

        plt.plot(wavelength, y, label=label)

    plt.xlabel("Nominal Wavelength [nm]")
    plt.ylabel("Coadded Intensity")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_name, dpi=150)
    plt.close()


def find_nearest_ascent_descent_pairs(
    inventory: pd.DataFrame,
    target_altitudes_ft: list[float],
) -> pd.DataFrame:
    rows = []

    ascent = inventory[inventory["flight_phase"] == "ascent"].copy()
    descent = inventory[inventory["flight_phase"] == "descent"].copy()

    for target in target_altitudes_ft:
        if ascent.empty or descent.empty:
            continue

        a_idx = int((ascent["center_altitude_ft"] - target).abs().idxmin())
        d_idx = int((descent["center_altitude_ft"] - target).abs().idxmin())

        a = inventory.loc[a_idx]
        d = inventory.loc[d_idx]

        rows.append(
            {
                "target_altitude_ft": target,
                "ascent_index": int(a["center_index"]),
                "ascent_spectrum_id": int(a["center_spectrum_id"]),
                "ascent_altitude_ft": float(a["center_altitude_ft"]),
                "ascent_utc_time": a["center_utc_time"],
                "descent_index": int(d["center_index"]),
                "descent_spectrum_id": int(d["center_spectrum_id"]),
                "descent_altitude_ft": float(d["center_altitude_ft"]),
                "descent_utc_time": d["center_utc_time"],
                "altitude_difference_ft": float(abs(a["center_altitude_ft"] - d["center_altitude_ft"])),
            }
        )

    return pd.DataFrame(rows)


def plot_ascent_descent_pairs(
    normalized_coadds: np.ndarray,
    inventory: pd.DataFrame,
    wavelength: np.ndarray,
) -> None:
    target_altitudes_ft = [25000, 50000, 100000, 150000, 200000, 250000, 300000]
    pairs = find_nearest_ascent_descent_pairs(inventory, target_altitudes_ft)

    pairs.to_csv(PRODUCT_DIR / "ascent_descent_nearest_altitude_pairs.csv", index=False)

    if pairs.empty:
        print("No ascent/descent pairs found.")
        return

    plt.figure(figsize=(12, 7))

    for _, pair in pairs.iterrows():
        a_idx = int(pair["ascent_index"])
        d_idx = int(pair["descent_index"])

        a = median_filter(normalized_coadds[a_idx, :], size=21, mode="nearest")
        d = median_filter(normalized_coadds[d_idx, :], size=21, mode="nearest")

        # Difference is more useful than overplotting too many spectra.
        diff = a - d

        label = (
            f"{pair['target_altitude_ft'] / 1000:.0f} kft "
            f"(Δalt={pair['altitude_difference_ft']:.0f} ft)"
        )

        plt.plot(wavelength, diff, label=label)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Nominal Wavelength [nm]")
    plt.ylabel("Ascent - Descent normalized coadd")
    plt.title("AMSE Ascent/Descent Spectral Difference at Matched Altitudes")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "coadded_ascent_descent_same_altitude.png", dpi=150)
    plt.close()


def main() -> None:
    print("AMSE coadded ascent/descent reduction starting...")

    raw, normalized, summary, wavelength = load_inputs()
    summary = classify_flight_phase(summary)

    raw_coadds, inventory = make_moving_coadds(raw, summary, method="median")
    normalized_coadds, norm_inventory = make_moving_coadds(normalized, summary, method="median")

    np.save(PRODUCT_DIR / f"amse_coadded_spectra_{WINDOW_NAME}.npy", raw_coadds)
    np.save(PRODUCT_DIR / f"amse_coadded_normalized_spectra_{WINDOW_NAME}.npy", normalized_coadds)

    inventory.to_csv(PRODUCT_DIR / f"amse_coadded_spectra_{WINDOW_NAME}_inventory.csv", index=False)

    plot_coadd_altitude_inventory(inventory)

    plot_selected_coadds(
        raw_coadds,
        inventory,
        wavelength,
        "coadded_selected_timepoints_raw.png",
        "AMSE Moving Coadds at Selected Time Points, Raw Spectra",
    )

    plot_selected_coadds(
        normalized_coadds,
        inventory,
        wavelength,
        "coadded_selected_timepoints_normalized.png",
        "AMSE Moving Coadds at Selected Time Points, Normalized Spectra",
    )

    plot_ascent_descent_pairs(normalized_coadds, inventory, wavelength)

    print()
    print("AMSE COADD SUMMARY")
    print("------------------")
    print(f"Raw matrix shape:              {raw.shape}")
    print(f"Raw coadd matrix shape:        {raw_coadds.shape}")
    print(f"Normalized coadd matrix shape: {normalized_coadds.shape}")
    print(f"Window half width:             +/- {WINDOW_HALF_WIDTH} spectra")
    print(f"Nominal full window:           {2 * WINDOW_HALF_WIDTH + 1} spectra")
    print()
    print("Flight phase counts:")
    print(inventory["flight_phase"].value_counts())
    print()
    print("Products written:")
    print(f"  {PRODUCT_DIR / f'amse_coadded_spectra_{WINDOW_NAME}.npy'}")
    print(f"  {PRODUCT_DIR / f'amse_coadded_normalized_spectra_{WINDOW_NAME}.npy'}")
    print(f"  {PRODUCT_DIR / f'amse_coadded_spectra_{WINDOW_NAME}_inventory.csv'}")
    print(f"  {PRODUCT_DIR / 'ascent_descent_nearest_altitude_pairs.csv'}")
    print()
    print("Figures written:")
    print(f"  {FIGURE_DIR / 'coadd_altitude_vs_spectrum_id.png'}")
    print(f"  {FIGURE_DIR / 'coadded_selected_timepoints_raw.png'}")
    print(f"  {FIGURE_DIR / 'coadded_selected_timepoints_normalized.png'}")
    print(f"  {FIGURE_DIR / 'coadded_ascent_descent_same_altitude.png'}")
    print()
    print("AMSE coadded ascent/descent reduction complete.")


if __name__ == "__main__":
    main()