# EEN1095 — Low-Sensitivity Digital Filters

**Design and Evaluation of Low-Sensitivity Digital Filters Implemented in Python Under Varying Numerical Precision Levels**

| | |
|---|---|
| **Student** | Darshan Mahadeva Naika |
| **Student ID** | A00090581 |
| **Supervisor** | Martin Collier |
| **Module** | EEN1095 — Dublin City University |
| **Submission** | August 2026 |
| **GitHub** | https://github.com/darshanmahadevanaika2-sudo/EEN1095-Low-Sensitivity-Digital-Filters |

---

## 1. Project Overview

This project evaluates three IIR digital filter structures — Direct-Form, Parallel Form, and Ladder (Second-Order Sections) — across four Python floating-point precision levels: float16, float32, float64, and mpmath (50 decimal digits).

**Central finding:** Filter structure, not arithmetic precision, is the primary driver of numerical robustness. The Ladder (SOS) structure remains stable at every order and precision level where Direct-Form fails.

**Key result:** At VLF (fs = 1 kHz), Direct-Form float16 SNR = −3.1 dB vs Ladder float16 SNR = 39.5 dB — a **42.6 dB improvement** from structural choice alone.

---

## 2. How to Run the Project

### Requirements

```
Python 3.11+
pip install numpy scipy mpmath matplotlib
```

### Run the Complete Pipeline

```bash
cd EEN1095-Low-Sensitivity-Digital-Filters/src
python filter_analysis_pipeline.py
```

Expected runtime: **~1-2 minutes**. All 9/9 steps should show **PASS**. Results saved to `results/`.

### Run Individual Steps

```bash
cd EEN1095-Low-Sensitivity-Digital-Filters/src
python ../tests/audio_filter_test.py
```

Replace `audio_filter_test.py` with any file from the `tests/` folder.

---

## 3. Repository Structure

```
EEN1095-Low-Sensitivity-Digital-Filters/
│
├── src/                                 ← Run pipeline from here
│   ├── filter_design.py                 ← Filter design + precision conversion
│   ├── metrics.py                       ← Five sensitivity metrics
│   ├── experiment.py                    ← Experiment runner
│   └── filter_analysis_pipeline.py      ← MAIN ENTRY POINT
│
├── tests/                               ← All test files (Steps 1-9)
│   ├── direct_form_evaluation.py        ← Step 1: Direct-Form baseline
│   ├── signal_validation.py             ← Step 2: Signal validation
│   ├── degradation_analysis.py          ← Step 3: Butterworth order sweep
│   ├── degradation_analysis_ellip.py    ← Step 4: Elliptic order sweep
│   ├── plot_responses.py                ← Step 5: Response plots
│   ├── ladder_filter.py                 ← Step 6: Ladder (SOS) evaluation
│   ├── comparison.py                    ← Step 7: Direct-Form vs Ladder
│   ├── audio_filter_test.py             ← Step 8: Audio + VLF signal tests
│   └── competing_approach.py            ← Step 9: Three-way comparison
│
├── results/                             ← All outputs
│   ├── *.png  (9 plots)
│   ├── *.wav  (78 audio files)
│   └── results.json
│
├── docs/
│   └── ELLIPTIC_FILTER_NOTES.md
│
└── README.md                            ← This file
```

---

## 4. Core Items for Detailed Review

| # | File | Description |
|---|---|---|
| 1 | `README.md` | Overview of repository, core items, run instructions |
| 2 | `src/filter_analysis_pipeline.py` | Main entry point — runs all 9 steps in ~1-2 minutes |
| 3 | `src/filter_design.py` | Core foundation — filter design and precision conversion |
| 4 | `src/metrics.py` | All 5 sensitivity metrics implementation |
| 5 | `tests/direct_form_evaluation.py` | Proves the problem — 23 unstable configurations found |
| 6 | `tests/audio_filter_test.py` | Most comprehensive test — audio + VLF, SNR, overflow monitoring |
| 7 | `tests/competing_approach.py` | Three-way comparison — Ladder 24× more accurate than Parallel Form |
| 8 | `results/comparison_butter_order14.png` | Key result plot — Fig. 2 in paper |
| 9 | `results/competing_comparison_butter_order14.png` | Three-way comparison plot — Fig. 4 in paper |
| 10 | `results/vlf_response_comparison.png` | VLF result — Fig. 3 in paper |

---

## 5. Key Results

| Finding | Result |
|---|---|
| Direct-Form first unstable (Butterworth) | float16: order 14 (SM = −0.068) \| float32: order 24 |
| Direct-Form first unstable (Elliptic) | float16: order 8 \| float32: order 10 \| float64/mpmath: order 14 |
| Ladder (SOS) stability | STABLE at ALL orders and ALL precision levels |
| Audio SNR float16 (fs = 48 kHz) | Direct-Form: 11.3 dB \| Ladder: 18.7 dB (+7.4 dB) |
| VLF SNR float16 (fs = 1 kHz) | Direct-Form: −3.1 dB \| Ladder: 39.5 dB (+42.6 dB) |
| Ladder vs Parallel Form | float16: 24× more accurate \| float32: 33,288× more accurate |
| Pipeline runtime | 9/9 steps PASS in ~1-2 minutes |

---

## 6. Supplementary Material

All material beyond the 10 core items is supplementary:

- `results/*.wav` (78 files) — filtered audio outputs for subjective listening comparison
- `results/*.png` (9 files) — complete set of frequency response and comparison plots
- `results/results.json` — machine-readable Direct-Form evaluation dataset
- `docs/ELLIPTIC_FILTER_NOTES.md` — technical notes on elliptic filter library
- `tests/signal_validation.py`, `tests/degradation_analysis.py`, `tests/ladder_filter.py`, `tests/comparison.py`, `tests/plot_responses.py` — supporting test files

---

*EEN1095 Project Portfolio | Darshan Mahadeva Naika | A00090581 | Dublin City University | August 2026*
