# AMSE Program Log

## 2019-12-11 — Flight

The Atmospheric Multi-Spectral Explorer (AMSE) flew as a payload aboard
Blue Origin's New Shepard, mission **NS-12**, on 2019-12-11, reaching
apogee of approximately 346,700 ft (105.6 km). AMSE was a volunteer,
unfunded student/faculty experiment — a compact 600–1000 nm fiber-fed
spectrometer developed and operated by the volunteer AMSE team, with
mission support through NASA's Flight Opportunities program. The
instrument functioned through ascent and descent and returned 138
spectra, and the project's preliminary results (strong pre-launch O2/H2O
absorption, O2 A-band persisting to altitude, sky-brightness decreasing
with altitude) were presented as a poster in February 2020. For a small,
unfunded student payload to fly, operate for the full ascent/descent
profile, and return a usable dataset was an early success for the effort
that would go on to become the San Diego Space Institute (SDSI).

## 2025-2026 — Preserved and rebuilt

The complete raw flight data (138 spectra), the 2020 poster, the NASA P9
capsule/payload trajectory and event record, and the original (2020-era)
analysis materials were preserved and, starting late 2025, an effort
began to rebuild the analysis as a reproducible, documented pipeline
aimed at a publication-quality re-reduction. The raw AMSE spectra and the
NASA P9 trajectory data are archived in their original available forms.

## 2026-09 — Forensic reanalysis

In September 2026, a forensic reanalysis of the existing repository was
carried out with AI assistance (Codex, Claude). This reanalysis:

- Inventoried the repository, confirmed the raw data (138 spectra,
  contiguous, no gaps/corruption) and independently re-derived the
  local-instrument-clock-to-UTC time synchronization — instrument
  timestamps were recorded in CST; UTC is obtained by adding 6 hours —
  cross-validated against both the NASA P9 capsule/payload trajectory and
  event record (used to assign capsule altitude to each spectrum) and the
  Blue Origin NS-12 webcast/transcript.
- **Discovered that the legacy reduction pipeline's regex-based numeric
  parser silently corrupts scientific-notation samples** (e.g.
  `1.379E+0003` is mis-split into two separate numbers), which
  invalidates the raw counts, saturation flags, and every derived
  product built from them in the current `outputs/` tree.
- Digitized the 2020 poster's solar-spectrum figure and combined it with
  ground/apogee persistence-and-fade behavior in the actual (correctly
  re-parsed) flight spectra to derive a **provisional
  "poster-constrained in-flight wavelength calibration"** — sufficient
  for feature placement and relative-altitude trends, not an absolute
  radiometric calibration, and not yet confirmed against a full,
  corrected whole-flight re-reduction.

## 2026-09 — Phase 1 closeout

GitHub archival of the repository is complete:

- `3a9826c` — archival baseline (raw AMSE spectra, NASA P9 trajectory zip,
  poster-era materials, and all legacy scripts/outputs)
- `0ef0ff1` — NASA P9 CSV archival completion (unzipped copy tracked
  directly, alongside its zip)
- `04adc4c` — initial program log

As of this checkpoint: all raw AMSE spectra and NASA P9 mission/capsule
trajectory data, poster-era materials, and legacy analysis outputs are
preserved in the repository. The legacy regex-parser defect and the
provisional poster-constrained wavelength calibration are documented, but
**no corrected re-reduction of the flight data has yet been performed.**

Phase 2 — the parser fix, regression tests, corrected altitude-resolved
ascent/descent coadds, validation against the provisional calibration,
and publication drafting — is **intentionally deferred**, not started.
The repository is being left at a clean, reproducible archival
checkpoint.
