# AMSE — Atmospheric Multi-Spectral Explorer

Data-reduction repository for AMSE, a compact 600–1000 nm fiber-fed CCD
spectrometer (Celestron NexStar 5SE telescope + Mightex HRS-NIR-050
spectrometer, 50 µm slit) flown through a side window of the Blue Origin
New Shepard capsule on mission **NS-12, 2019-12-11**, reaching ≈346,700 ft
(≈105.6 km). The instrument recorded spectra roughly every 8.6 s through
ascent and descent, with the goal of characterizing atmospheric O₂/H₂O
absorption and searching for mesospheric OH emission.

This project was originally flown and presented as a poster in 2020
(`docs/source_material/`, once populated — see below) and is being
rebuilt in 2026 as a reproducible Python pipeline toward a peer-reviewed
paper.

## Status (as of this archival commit)

**A critical parsing bug was found in the original reduction pipeline
(`scripts/01`–`04`) and has not yet been fixed in the code tracked at this
commit.** The regex used to extract numeric samples from the raw
`test_data_*.txt` files does not handle scientific notation correctly
(`1.379E+0003` is silently mis-split into two separate numbers). This
silently corrupted every value in `outputs/reduced_products/amse_raw_spectral_matrix.npy`
and everything derived from it, including the saturation flagging. A
corrected parser, regression tests, and a from-scratch re-reduction into
a separate versioned output directory are being developed as a follow-up
change — see the repository's issue/PR history after this commit for
progress.

**Every file under `outputs/reduced_products/` and `outputs/figures/` at
this commit — including the large `*.npy` matrices and
`amse_flight_data_workbook.xlsx` — is the legacy, parser-bug-affected
reduction.** These are tracked deliberately, not ignored, so the
project's original (buggy) results remain available for historical
provenance and for a direct before/after comparison once the corrected
re-reduction lands. **They are preserved for reference only and must not
be used as inputs to, or cited as results of, the corrected analysis.**
Numeric values in these files (raw counts, saturation flags, coadds,
wavelength assignments) should be treated as invalid until re-derived
from the raw `data/raw_spectra/` files with the corrected parser.

The wavelength solution currently in use is a **provisional
"poster-constrained in-flight wavelength calibration"**: pixel-to-nm
mapping was derived from atmospheric O₂ A-band, O₂ B-band, Hα, and H₂O
features identified directly in AMSE flight data (persistence/fade
behavior confirmed between a ground and an apogee spectrum), cross-checked
against a digitized figure from the 2020 poster. It is sufficient for
feature placement and relative altitude trends; it is **not** an absolute
radiometric calibration and is treated as provisional pending confirmation
from the corrected whole-flight re-reduction.

## Repository layout

```
data/
  raw_spectra/            138 raw AMSE spectrum files (text), zipped and extracted
  nasa_flight_report/      NASA/Blue Origin P-9 payload trajectory (capsule branch)
  processed/                (currently empty)
docs/                       (currently empty — source material archival pending)
notebooks/                  (currently empty)
src/amse/                   (currently empty — no installable package yet)
scripts/                    01-09: first-look reduction, diagnostics, coadds,
                             poster-recreation, workbook build (see docstrings;
                             known parser bug, see Status above)
outputs/
  figures/                  PNGs from scripts/01-09
  reduced_products/         CSV/NPY/XLSX products from scripts/01-09
  apogee_39_47_bad_channel_masked/   apogee-specific mini-pipeline (script 09)
```

## Environment

Python 3.14, virtual environment in `.venv/` (not tracked; recreate with
`python -m venv .venv` and install numpy, pandas, scipy, matplotlib,
pyarrow, xlsxwriter — a pinned `requirements.txt` is not yet checked in).

## Mission timing reference

Local instrument clock is UTC−6 (CST; Blue Origin's West Texas site, no
DST in December), independently confirmed against NASA payload telemetry
and the Blue Origin NS-12 webcast. Liftoff: 2019-12-11 17:53:03.220 UTC
(NASA mission-elapsed-time zero, ≈8 s before measured liftoff). Apogee:
≈17:57:08 UTC, ≈346,700 ft. See project notes for the full event timeline
and the capsule-vs-booster telemetry-branch distinction in the NASA file.
