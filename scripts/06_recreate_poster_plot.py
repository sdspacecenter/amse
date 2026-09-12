from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# AMSE poster plot recreation
#
# Purpose:
#   Recreate the poster-style AMSE skybrightness plot:
#   raw/log signal versus wavelength at selected flight times.
#
# Key corrections:
#   1. Do NOT coadd the whole flight.
#   2. Coadd locally around the poster time points only.
#   3. Use timestamp -> altitude tagging.
#   4. Flip the spectral pixel order so identified features land
#      near the expected wavelengths:
#        H-alpha ~656.3 nm
#        O2 B    ~686.7 nm
#        O2 A    ~760.5 nm
#        H2O     ~720, 820, 940 nm
# ============================================================


ROOT = Path.cwd()

RAW_ROOT = ROOT / "data" / "raw_spectra"
NASA_ROOT = ROOT / "data" / "nasa_flight_report"

OUT_PRODUCTS = ROOT / "outputs" / "reduced_products"
OUT_FIGURES = ROOT / "outputs" / "figures"

OUT_PRODUCTS.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)


# AMSE local time appears to be MST/local instrument time.
# NASA table is UTC in the earlier first-look reduction.
LOCAL_TO_UTC_OFFSET_HOURS = 6

# Local coadd window around each selected poster spectrum.
# +/- 5 gives an 11-spectrum local average.
WINDOW_HALF_WIDTH = 5

# Detector nominal wavelength range.
# The detector pixel order will be reversed below.
WAVE_MIN_NM = 600.0
WAVE_MAX_NM = 1000.0

# Poster times from the AMSE preliminary flight-results plot.
# These are local AMSE times shown in the poster table.
POSTER_TARGETS = [
    {"local_time": "11:51:54", "poster_altitude_kft": 0.0,   "color": "blue"},
    {"local_time": "11:53:00", "poster_altitude_kft": None,  "color": "red"},
    {"local_time": "11:53:58", "poster_altitude_kft": 22.0,  "color": "gold"},
    {"local_time": "11:55:05", "poster_altitude_kft": 110.0, "color": "green"},
    {"local_time": "11:56:03", "poster_altitude_kft": 276.0, "color": "orange"},
    {"local_time": "11:57:01", "poster_altitude_kft": 342.0, "color": "cyan"},
]


REFERENCE_LINES = [
    (656.3, "H-alpha"),
    (686.7, "O2 B"),
    (720.0, "H2O"),
    (760.5, "O2 A"),
    (820.0, "H2O"),
    (940.0, "H2O"),
]


def natural_sort_key(path: Path):
    m = re.search(r"test_data_(\d+)\.txt$", path.name)
    return int(m.group(1)) if m else 999999


def find_spectral_files():
    files = sorted(RAW_ROOT.rglob("test_data_*.txt"), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No test_data_*.txt files found under: {RAW_ROOT}")
    return files


def parse_amse_timestamp_from_text(text: str) -> pd.Timestamp:
    """
    Extract HH:MM:SS from the first line of a spectrum file.
    Assumes flight date = 2019-12-11.
    """
    m = re.search(r"(\d{1,2}:\d{2}:\d{2})", str(text))
    if not m:
        raise ValueError(f"Could not find HH:MM:SS timestamp in: {text}")

    hhmmss = m.group(1)
    return pd.to_datetime(f"2019-12-11 {hhmmss}")


def read_spectrum_file(path: Path):
    """
    Read one AMSE spectral text file.

    Expected structure:
      first nonblank line includes timestamp
      remaining lines are spectral values
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 10:
        raise ValueError(f"File appears too short to be a spectrum: {path}")

    header = lines[0]
    local_time = parse_amse_timestamp_from_text(header)

    values = []
    for line in lines[1:]:
        try:
            values.append(float(line))
        except ValueError:
            # Ignore any non-numeric leftover text.
            pass

    spectrum = np.array(values, dtype=float)

    if spectrum.size == 0:
        raise ValueError(f"No numeric spectral values found in: {path}")

    return header, local_time, spectrum


def load_raw_spectra_from_files():
    files = find_spectral_files()

    rows = []
    spectra = []

    for path in files:
        m = re.search(r"test_data_(\d+)\.txt$", path.name)
        spectrum_id = int(m.group(1)) if m else len(rows)

        header, local_time, spectrum = read_spectrum_file(path)
        utc_time = local_time + pd.Timedelta(hours=LOCAL_TO_UTC_OFFSET_HOURS)

        rows.append({
            "spectrum_id": spectrum_id,
            "filename": path.name,
            "path": str(path),
            "timestamp_text": header,
            "local_time": local_time,
            "utc_time": utc_time,
            "n_pixels": spectrum.size,
        })
        spectra.append(spectrum)

    inventory = pd.DataFrame(rows).sort_values("spectrum_id").reset_index(drop=True)

    lengths = sorted(set(len(s) for s in spectra))
    if len(lengths) != 1:
        raise ValueError(f"Spectral files do not all have same length. Found lengths: {lengths}")

    # Sort spectra by actual spectrum_id order.
    spectra_sorted = []
    for sid in inventory["spectrum_id"]:
        idx = [i for i, r in enumerate(rows) if r["spectrum_id"] == sid][0]
        spectra_sorted.append(spectra[idx])

    matrix = np.vstack(spectra_sorted)

    return inventory, matrix


def find_nasa_csv():
    """
    Find or extract the NASA P-9 flight report CSV.
    """
    csvs = list(NASA_ROOT.rglob("*.csv"))
    if csvs:
        return csvs[0]

    zips = list(NASA_ROOT.rglob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No NASA CSV or ZIP found under: {NASA_ROOT}")

    zip_path = zips[0]

    with zipfile.ZipFile(zip_path, "r") as z:
        members = [m for m in z.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise FileNotFoundError(f"No CSV file found inside: {zip_path}")

        member = members[0]
        extracted_path = NASA_ROOT / Path(member).name

        with z.open(member) as src, open(extracted_path, "wb") as dst:
            dst.write(src.read())

    return extracted_path


def load_nasa_altitude_table():
    """
    Load NASA flight report and identify time/altitude columns.

    This tries to be robust because the exact CSV column names may vary.
    """
    csv_path = find_nasa_csv()
    nasa = pd.read_csv(csv_path)

    time_col = None
    altitude_col = None

    for c in nasa.columns:
        cl = str(c).strip().lower()
        if time_col is None and "time" in cl:
            time_col = c

        if altitude_col is None and ("alt" in cl or "altitude" in cl):
            altitude_col = c

    if time_col is None or altitude_col is None:
        raise ValueError(
            "Could not automatically identify NASA time/altitude columns.\n"
            f"Columns found: {list(nasa.columns)}"
        )

    out = nasa[[time_col, altitude_col]].copy()
    out.columns = ["nasa_time_raw", "altitude_raw"]

    out["altitude_ft"] = (
        out["altitude_raw"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0]
        .astype(float)
    )

    parsed = pd.to_datetime(out["nasa_time_raw"], errors="coerce")

    if parsed.isna().all():
        hhmmss = out["nasa_time_raw"].astype(str).str.extract(r"(\d{1,2}:\d{2}:\d{2})")[0]
        parsed = pd.to_datetime("2019-12-11 " + hhmmss, errors="coerce")

    out["utc_time"] = parsed

    out = out.dropna(subset=["utc_time", "altitude_ft"])
    out = out.sort_values("utc_time").reset_index(drop=True)

    if out.empty:
        raise ValueError("NASA altitude table loaded, but no valid time/altitude rows were found.")

    return out


def assign_altitudes(inventory: pd.DataFrame):
    """
    Prefer the existing first-look altitude-tagged summary if present.
    Fall back to interpolating from NASA flight report.
    """
    summary_path = OUT_PRODUCTS / "amse_altitude_tagged_summary.csv"

    inv = inventory.copy()

    if summary_path.exists():
        summary = pd.read_csv(summary_path)

        if "spectrum_id" in summary.columns and "altitude_ft" in summary.columns:
            keep_cols = ["spectrum_id", "altitude_ft"]
            if "mission_elapsed_s" in summary.columns:
                keep_cols.append("mission_elapsed_s")

            merged = inv.merge(summary[keep_cols], on="spectrum_id", how="left")

            if merged["altitude_ft"].notna().sum() > 0:
                merged["altitude_kft"] = merged["altitude_ft"] / 1000.0
                print(f"Using altitude assignments from: {summary_path}")
                return merged

    print("No usable existing altitude-tagged summary found. Interpolating from NASA CSV.")
    nasa = load_nasa_altitude_table()

    nasa_seconds = nasa["utc_time"].astype("int64") / 1e9
    nasa_altitudes = nasa["altitude_ft"].to_numpy()

    spectrum_seconds = inv["utc_time"].astype("int64") / 1e9

    inv["altitude_ft"] = np.interp(spectrum_seconds, nasa_seconds, nasa_altitudes)
    inv["altitude_kft"] = inv["altitude_ft"] / 1000.0

    return inv


def coadd_around_index(matrix: np.ndarray, center_index: int, half_width: int):
    """
    Coadd only locally around a chosen spectrum index.
    """
    lo = max(0, center_index - half_width)
    hi = min(matrix.shape[0], center_index + half_width + 1)

    sub = matrix[lo:hi, :]
    coadd = np.nanmean(sub, axis=0)

    return coadd, lo, hi - 1, sub.shape[0]


def safe_for_log(y: np.ndarray):
    """
    Prevent bad log plotting from zero or negative values.
    """
    y = y.astype(float).copy()
    y[y <= 0] = np.nan
    return y


def main():
    print("AMSE poster recreation starting...")
    print("-----------------------------------")

    inventory, raw_matrix = load_raw_spectra_from_files()
    inventory = assign_altitudes(inventory)

    n_spectra, n_pixels = raw_matrix.shape

    print(f"Loaded spectra: {n_spectra}")
    print(f"Pixels per spectrum: {n_pixels}")

    # Nominal wavelength axis remains left-to-right 600 to 1000 nm.
    wavelength_nm = np.linspace(WAVE_MIN_NM, WAVE_MAX_NM, n_pixels)

    # KEY FIX:
    # Reverse detector pixel order so the spectral features line up physically.
    # This is better than reversing the x-axis.
    flipped_matrix = raw_matrix[:, ::-1]

    coadd_rows = []
    coadd_spectra = []

    for target in POSTER_TARGETS:
        time_string = target["local_time"]
        target_local = pd.to_datetime(f"2019-12-11 {time_string}")

        dt = (inventory["local_time"] - target_local).abs()
        nearest_row_index = int(dt.idxmin())

        center_spectrum_id = int(inventory.loc[nearest_row_index, "spectrum_id"])

        # The matrix is sorted by spectrum_id and reset to row order.
        # Since the files are test_data_0..N, spectrum_id normally equals matrix row.
        center_matrix_index = nearest_row_index

        coadd, lo, hi, n_used = coadd_around_index(
            flipped_matrix,
            center_matrix_index,
            WINDOW_HALF_WIDTH,
        )

        altitude_ft = float(inventory.loc[nearest_row_index, "altitude_ft"])
        altitude_kft = altitude_ft / 1000.0
        nearest_local_time = inventory.loc[nearest_row_index, "local_time"]

        coadd_spectra.append(coadd)

        coadd_rows.append({
            "poster_target_local_time": time_string,
            "poster_table_altitude_kft": target["poster_altitude_kft"],
            "nearest_spectrum_id": center_spectrum_id,
            "nearest_spectrum_local_time": nearest_local_time,
            "nearest_spectrum_utc_time": inventory.loc[nearest_row_index, "utc_time"],
            "altitude_ft": altitude_ft,
            "altitude_kft": altitude_kft,
            "coadd_start_matrix_index": lo,
            "coadd_end_matrix_index": hi,
            "n_spectra_coadded": n_used,
            "plot_color": target["color"],
        })

    coadd_info = pd.DataFrame(coadd_rows)
    coadd_matrix = np.vstack(coadd_spectra)

    inventory_out = OUT_PRODUCTS / "poster_recreated_local_coadd_inventory_FLIPPED.csv"
    spectra_out = OUT_PRODUCTS / "poster_recreated_local_coadded_spectra_FLIPPED.csv"
    figure_out = OUT_FIGURES / "poster_recreated_log_raw_counts_local_pm5_FLIPPED.png"

    coadd_info.to_csv(inventory_out, index=False)

    spectra_table = pd.DataFrame({"wavelength_nm": wavelength_nm})
    for i, row in coadd_info.iterrows():
        col = (
            f"{row['poster_target_local_time']}"
            f"_id{int(row['nearest_spectrum_id'])}"
            f"_{row['altitude_kft']:.1f}kft"
        )
        spectra_table[col] = coadd_matrix[i, :]

    spectra_table.to_csv(spectra_out, index=False)

    # ========================================================
    # Poster-style log plot: no artificial vertical offsets
    # ========================================================
    plt.figure(figsize=(12, 7))

    for i, row in coadd_info.iterrows():
        y = safe_for_log(coadd_matrix[i, :])

        if pd.notna(row["poster_table_altitude_kft"]):
            poster_alt_text = f", poster {row['poster_table_altitude_kft']:.0f} kft"
        else:
            poster_alt_text = ""

        label = (
            f"{row['poster_target_local_time']}  "
            f"{row['altitude_kft']:.1f} kft"
            f"{poster_alt_text}  "
            f"id={int(row['nearest_spectrum_id'])}  "
            f"n={int(row['n_spectra_coadded'])}"
        )

        plt.plot(
            wavelength_nm,
            y,
            color=row["plot_color"],
            linewidth=1.2,
            label=label,
        )

    for x, label in REFERENCE_LINES:
        plt.axvline(x, linestyle=":", linewidth=1.0)
        plt.text(
            x + 1.0,
            12,
            label,
            rotation=90,
            fontsize=9,
            va="bottom",
            ha="left",
        )

    plt.yscale("log")
    plt.xlim(WAVE_MIN_NM, WAVE_MAX_NM)
    plt.ylim(10, 120000)

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Raw counts")
    plt.title("AMSE Skybrightness: Poster Recreation, Local ±5 Coadds, Pixel Order Flipped")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_out, dpi=200)
    plt.close()

    # ========================================================
    # Diagnostic plot: selected altitude/time table
    # ========================================================
    print()
    print("AMSE POSTER RECREATION COMPLETE")
    print("--------------------------------")
    print(f"Raw matrix shape: {raw_matrix.shape}")
    print(f"Flipped matrix shape: {flipped_matrix.shape}")
    print(f"Local coadd window: +/- {WINDOW_HALF_WIDTH} spectra")
    print()
    print("Selected poster spectra:")
    print(
        coadd_info[
            [
                "poster_target_local_time",
                "poster_table_altitude_kft",
                "nearest_spectrum_id",
                "nearest_spectrum_local_time",
                "altitude_kft",
                "coadd_start_matrix_index",
                "coadd_end_matrix_index",
                "n_spectra_coadded",
            ]
        ].to_string(index=False)
    )
    print()
    print("Products written:")
    print(inventory_out)
    print(spectra_out)
    print(figure_out)
    print()
    print("Feature sanity check after flip:")
    print("  H-alpha should be near 656.3 nm")
    print("  O2 B should be near 686.7 nm")
    print("  O2 A should be near 760.5 nm")
    print("  H2O bands should appear near 720, 820, and 940 nm")


if __name__ == "__main__":
    main()
