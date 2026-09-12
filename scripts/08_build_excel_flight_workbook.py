from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\seanc\Documents\amse")

DATA = ROOT / "data"
OUT = ROOT / "outputs"
REDUCED = OUT / "reduced_products"
FIGURES = OUT / "figures"

EXCEL_OUT = OUT / "reduced_products" / "amse_flight_data_workbook.xlsx"

NASA_ZIP = DATA / "nasa_flight_report" / "P9_Payload_Flight_Mission_Report.csv[38].zip"

print("AMSE Excel workbook build starting...")
print(f"Output workbook: {EXCEL_OUT}")

def safe_read_csv(path):
    path = Path(path)
    if path.exists():
        print(f"Reading CSV: {path}")
        return pd.read_csv(path)
    print(f"Missing CSV, skipping: {path}")
    return None

def safe_read_npy(path):
    path = Path(path)
    if path.exists():
        print(f"Reading NPY: {path}")
        return np.load(path)
    print(f"Missing NPY, skipping: {path}")
    return None

def read_nasa_zip(zip_path):
    if not zip_path.exists():
        print(f"Missing NASA ZIP: {zip_path}")
        return None

    print(f"Reading NASA ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_members = [m for m in z.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            print("No CSV found inside NASA ZIP.")
            return None

        member = csv_members[0]
        print(f"Using NASA CSV member: {member}")
        with z.open(member) as f:
            return pd.read_csv(f)

def npy_to_df(matrix, prefix="pixel"):
    if matrix is None:
        return None

    n_rows, n_cols = matrix.shape
    columns = [f"{prefix}_{i:04d}" for i in range(n_cols)]
    df = pd.DataFrame(matrix, columns=columns)
    df.insert(0, "spectrum_id", np.arange(n_rows))
    return df

def write_df(writer, sheet_name, df, index=False):
    if df is None:
        return

    # Excel hard limits.
    max_rows = 1_048_576
    max_cols = 16_384

    rows, cols = df.shape
    if rows > max_rows:
        print(f"WARNING: {sheet_name} has {rows} rows. Truncating to Excel row limit.")
        df = df.iloc[:max_rows, :]
    if cols > max_cols:
        print(f"WARNING: {sheet_name} has {cols} columns. Truncating to Excel column limit.")
        df = df.iloc[:, :max_cols]

    print(f"Writing sheet: {sheet_name}  shape={df.shape}")
    df.to_excel(writer, sheet_name=sheet_name[:31], index=index)

    workbook = writer.book
    worksheet = writer.sheets[sheet_name[:31]]

    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#1F4E78",
        "font_color": "white",
        "border": 1,
        "align": "center",
        "valign": "vcenter"
    })

    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_fmt)

    worksheet.freeze_panes(1, 1)
    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

    for col_num, col in enumerate(df.columns):
        width = min(max(len(str(col)) + 2, 10), 28)
        worksheet.set_column(col_num, col_num, width)

def add_readme(writer):
    workbook = writer.book
    ws = workbook.add_worksheet("README")
    writer.sheets["README"] = ws

    title_fmt = workbook.add_format({
        "bold": True,
        "font_size": 16,
        "font_color": "#1F4E78"
    })

    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#D9EAF7",
        "border": 1
    })

    text_fmt = workbook.add_format({
        "text_wrap": True,
        "valign": "top"
    })

    ws.write("A1", "AMSE Flight Data Workbook", title_fmt)
    ws.write("A3", "Purpose", header_fmt)
    ws.write(
        "B3",
        "Single Excel workbook collecting the AMSE spectral inventory, altitude-tagged spectra, NASA flight report, raw and processed spectral matrices, coadded spectra, ascent/descent comparison products, candidate features, and figure index.",
        text_fmt
    )

    ws.write("A5", "Generated from root directory", header_fmt)
    ws.write("B5", str(ROOT), text_fmt)

    ws.write("A7", "Important notes", header_fmt)
    ws.write(
        "B7",
        "The raw spectral files appear to use scaled detector values rather than the poster's original absolute raw-count display scale. Poster-style plots may therefore use display scaling unless a later absolute calibration is recovered.",
        text_fmt
    )

    ws.write("A9", "Recommended use", header_fmt)
    ws.write(
        "B9",
        "Use the altitude_tagged_summary and coadded/poster-time sheets for science interpretation. Use raw_spectral_matrix and normalized_spectral_matrix for detailed pixel-level analysis. Use nasa_flight_report for the altitude/time reference.",
        text_fmt
    )

    ws.set_column("A:A", 28)
    ws.set_column("B:B", 110)

def add_figure_index(writer):
    image_files = sorted(FIGURES.glob("*.png"))

    rows = []
    for img in image_files:
        rows.append({
            "figure_file": img.name,
            "path": str(img),
            "category": guess_figure_category(img.name)
        })

    df = pd.DataFrame(rows)
    write_df(writer, "figure_index", df)

def guess_figure_category(name):
    n = name.lower()
    if "poster" in n:
        return "poster recreation"
    if "pairwise" in n:
        return "ascent/descent comparison"
    if "coadd" in n:
        return "coadded spectra"
    if "heatmap" in n:
        return "spectral image"
    if "altitude" in n:
        return "altitude analysis"
    if "candidate" in n:
        return "candidate features"
    return "other"

def main():
    EXCEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    spectrum_inventory = safe_read_csv(REDUCED / "amse_spectrum_inventory.csv")
    altitude_summary = safe_read_csv(REDUCED / "amse_altitude_tagged_summary.csv")
    nasa_report = read_nasa_zip(NASA_ZIP)

    ascent_descent_pairs = safe_read_csv(REDUCED / "ascent_descent_nearest_altitude_pairs.csv")
    pairwise_metrics = safe_read_csv(REDUCED / "pairwise_ascent_descent_difference_metrics.csv")
    candidate_features = safe_read_csv(REDUCED / "candidate_absorption_features_nominal_wavelength.csv")
    poster_time_coadds = safe_read_csv(REDUCED / "poster_time_coadded_spectra_pm5.csv")

    raw_matrix = safe_read_npy(REDUCED / "amse_raw_spectral_matrix.npy")
    normalized_matrix = safe_read_npy(REDUCED / "amse_normalized_spectral_matrix.npy")
    coadded_matrix = safe_read_npy(REDUCED / "amse_coadded_normalized_spectra_pm5.npy")

    raw_matrix_df = npy_to_df(raw_matrix, prefix="pixel")
    normalized_matrix_df = npy_to_df(normalized_matrix, prefix="pixel")
    coadded_matrix_df = npy_to_df(coadded_matrix, prefix="pixel")

    with pd.ExcelWriter(EXCEL_OUT, engine="xlsxwriter") as writer:
        add_readme(writer)

        write_df(writer, "spectrum_inventory", spectrum_inventory)
        write_df(writer, "altitude_tagged_summary", altitude_summary)
        write_df(writer, "nasa_flight_report", nasa_report)

        write_df(writer, "raw_spectral_matrix", raw_matrix_df)
        write_df(writer, "normalized_spectral_matrix", normalized_matrix_df)
        write_df(writer, "coadded_pm5_matrix", coadded_matrix_df)

        write_df(writer, "poster_time_coadds", poster_time_coadds)
        write_df(writer, "ascent_descent_pairs", ascent_descent_pairs)
        write_df(writer, "pairwise_difference_metrics", pairwise_metrics)
        write_df(writer, "candidate_absorption_features", candidate_features)

        add_figure_index(writer)

    print("")
    print("AMSE Excel workbook complete.")
    print(EXCEL_OUT)

if __name__ == "__main__":
    main()
