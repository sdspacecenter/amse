
"""
AMSE Apogee Mean Coadd with Bad-Channel Mask
Date: 2026-05-07

Run from AMSE root:

    python scripts\\09_apogee_spectrum_coadd.py

Purpose:
    Replicate the sensible apogee plot using files 39–47:
    - load test_data_39.txt through test_data_47.txt
    - flip detector order for wavelength plotting
    - mask repeatable bad detector channels
    - compute plain mean coadd
    - plot linear and log spectra
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw_spectra"
OUT_DIR = PROJECT_ROOT / "outputs" / "apogee_39_47_bad_channel_masked"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

FILE_IDS = list(range(39, 48))

WAVE_MIN_NM = 600.0
WAVE_MAX_NM = 1000.0

FLIP_PIXEL_ORDER = True

# Native detector channels identified as repeatable bad channels.
BAD_NATIVE_CHANNELS = np.array([512, 700, 760, 1093], dtype=int)

REFERENCE_LINES = [
    (656.28, "H-alpha"),
    (686.7, "O2 B"),
    (720.0, "H2O"),
    (760.5, "O2 A"),
    (820.0, "H2O"),
    (940.0, "H2O"),
]


# ============================================================
# Helpers
# ============================================================

def read_spectrum(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if not lines:
        raise ValueError(f"Empty file: {path}")

    timestamp = pd.to_datetime(lines[0].strip(), errors="coerce")

    values = []
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        try:
            values.append(float(s))
        except ValueError:
            pass

    if not values:
        raise ValueError(f"No numeric spectrum values found in: {path}")

    return timestamp, np.asarray(values, dtype=float)


def find_file(spectrum_id: int) -> Path:
    name = f"test_data_{spectrum_id}.txt"
    matches = list(RAW_DIR.rglob(name))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {name} under {RAW_DIR}. "
            "Place the files in data/raw_spectra/."
        )

    return matches[0]


def rolling_median(y: np.ndarray, window: int = 9) -> np.ndarray:
    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=max(1, window // 2))
        .median()
        .to_numpy()
    )


def add_reference_lines(ax):
    ymin, ymax = ax.get_ylim()

    for wl, label in REFERENCE_LINES:
        ax.axvline(wl, linestyle=":", linewidth=1.0)

        if ax.get_yscale() == "log":
            ymin_safe = max(ymin, 1.0e-12)
            y = 10 ** (
                np.log10(ymin_safe)
                + 0.02 * (np.log10(ymax) - np.log10(ymin_safe))
            )
        else:
            y = ymin + 0.02 * (ymax - ymin)

        ax.text(
            wl + 1.0,
            y,
            label,
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8,
        )


# ============================================================
# Main
# ============================================================

def main():
    print("AMSE apogee 39–47 bad-channel-masked mean coadd")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw directory: {RAW_DIR}")

    spectra = []
    rows = []

    for spectrum_id in FILE_IDS:
        path = find_file(spectrum_id)
        timestamp, spectrum = read_spectrum(path)

        spectra.append(spectrum)

        rows.append(
            {
                "spectrum_id": spectrum_id,
                "filename": path.name,
                "timestamp": timestamp,
                "n_pixels": len(spectrum),
                "min": float(np.nanmin(spectrum)),
                "max": float(np.nanmax(spectrum)),
                "mean": float(np.nanmean(spectrum)),
                "median": float(np.nanmedian(spectrum)),
                "n_nonpositive": int(np.sum(spectrum <= 0)),
            }
        )

    audit = pd.DataFrame(rows)

    lengths = sorted(set(len(s) for s in spectra))
    if len(lengths) != 1:
        raise ValueError(f"Spectra have unequal lengths: {lengths}")

    matrix_native = np.vstack(spectra).astype(float)
    n_pixels = matrix_native.shape[1]

    wavelength_nm = np.linspace(WAVE_MIN_NM, WAVE_MAX_NM, n_pixels)

    if FLIP_PIXEL_ORDER:
        matrix = matrix_native[:, ::-1]
        pixel_order = "flipped"
    else:
        matrix = matrix_native.copy()
        pixel_order = "native"

    # Nonpositive values are invalid. Do not clip them upward.
    matrix[~np.isfinite(matrix)] = np.nan
    matrix[matrix <= 0] = np.nan

    # Convert native bad channels to flipped plotting indices.
    bad_flipped_channels = (n_pixels - 1) - BAD_NATIVE_CHANNELS

    matrix[:, bad_flipped_channels] = np.nan

    mean_coadd = np.nanmean(matrix, axis=0)
    mean_coadd_sm9 = rolling_median(mean_coadd, window=9)
    n_valid_at_pixel = np.sum(np.isfinite(matrix), axis=0)

    # Save audit products.
    audit["pixel_order_used"] = pixel_order
    audit["bad_native_channels_masked"] = ",".join(map(str, BAD_NATIVE_CHANNELS))
    audit.to_csv(OUT_DIR / "apogee_39_47_audit.csv", index=False)

    pd.DataFrame(
        {
            "native_channel": BAD_NATIVE_CHANNELS,
            "flipped_channel": bad_flipped_channels,
            "wavelength_nm_after_flip": wavelength_nm[bad_flipped_channels],
            "mask_reason": "repeatable bad detector channel",
        }
    ).to_csv(OUT_DIR / "bad_channel_mask_table.csv", index=False)

    pd.DataFrame(
        {
            "wavelength_nm": wavelength_nm,
            "mean_coadd_bad_channels_masked": mean_coadd,
            "mean_coadd_9px_rolling_median": mean_coadd_sm9,
            "n_valid_at_pixel": n_valid_at_pixel,
        }
    ).to_csv(OUT_DIR / "apogee_39_47_mean_coadd.csv", index=False)

    # Plot 1: final linear plot.
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        wavelength_nm,
        mean_coadd,
        linewidth=1.4,
        label="Mean coadd, files 39–47, bad channels masked",
    )

    ax.plot(
        wavelength_nm,
        mean_coadd_sm9,
        linewidth=2.0,
        label="9-pixel rolling median guide",
    )

    ax.set_xlim(WAVE_MIN_NM, WAVE_MAX_NM)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Raw counts")
    ax.set_title("AMSE Apogee Mean Coadd, Files 39–47, Bad Channels Masked")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    add_reference_lines(ax)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "apogee_39_47_mean_masked_linear.png", dpi=200)
    plt.close(fig)

    # Plot 2: log plot.
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        wavelength_nm,
        mean_coadd,
        linewidth=1.4,
        label="Mean coadd, files 39–47, bad channels masked",
    )

    ax.plot(
        wavelength_nm,
        mean_coadd_sm9,
        linewidth=2.0,
        label="9-pixel rolling median guide",
    )

    ax.set_yscale("log")
    ax.set_xlim(WAVE_MIN_NM, WAVE_MAX_NM)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Raw counts")
    ax.set_title("AMSE Apogee Mean Coadd, Files 39–47, Bad Channels Masked, Log Scale")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    add_reference_lines(ax)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "apogee_39_47_mean_masked_log.png", dpi=200)
    plt.close(fig)

    # Plot 3: member spectra.
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, row in audit.iterrows():
        ax.plot(
            wavelength_nm,
            matrix[i],
            linewidth=0.7,
            alpha=0.35,
            label=f"id={int(row['spectrum_id'])}, {row['timestamp'].strftime('%H:%M:%S')}",
        )

    ax.plot(
        wavelength_nm,
        mean_coadd,
        color="black",
        linewidth=2.0,
        label="Mean coadd",
    )

    ax.set_xlim(WAVE_MIN_NM, WAVE_MAX_NM)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Raw counts")
    ax.set_title("AMSE Apogee Member Spectra, Files 39–47, Bad Channels Masked")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    add_reference_lines(ax)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "apogee_39_47_members_masked_linear.png", dpi=200)
    plt.close(fig)

    print()
    print("Done.")
    print(f"Output directory: {OUT_DIR}")
    print()
    print("Plots written:")
    print(f"  {OUT_DIR / 'apogee_39_47_mean_masked_linear.png'}")
    print(f"  {OUT_DIR / 'apogee_39_47_mean_masked_log.png'}")
    print(f"  {OUT_DIR / 'apogee_39_47_members_masked_linear.png'}")


if __name__ == "__main__":
    main()