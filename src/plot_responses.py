"""
Amplitude and phase response plots.

For each case, plots |H(f)| in dB (and phase in degrees, where useful)
vs normalised frequency, overlaying all four precision levels
(float16, float32, float64, mpmath) on the same axes.

Cases chosen to sit at/near the instability boundaries found in Task B:
  - Butterworth order 14  (float16 unstable here, SM = -0.068)
  - Elliptic    order 10  (float32 unstable here, SM = -0.0067)

Phase is shown for Butterworth (clean). For Elliptic, phase is omitted -
its multiple deep notches cause repeated near-360deg phase rotations
that make np.unwrap produce confusing offset jumps once masked; the
magnitude response alone already makes the sensitivity comparison clear.
"""

import numpy as np
from scipy import signal
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

mpmath.mp.dps = 50

PRECISIONS = ['float16', 'float32', 'float64', 'mpmath']
COLORS = {'float16': 'tab:red', 'float32': 'tab:orange', 'float64': 'tab:blue', 'mpmath': 'tab:green'}
CUTOFF = 0.3
N_FREQ = 8192


def convert(coeffs, precision):
    if precision == 'float16':
        return coeffs.astype(np.float16)
    elif precision == 'float32':
        return coeffs.astype(np.float32)
    elif precision == 'float64':
        return coeffs.astype(np.float64)
    elif precision == 'mpmath':
        return np.array([float(mpmath.mpf(str(c))) for c in coeffs], dtype=np.float64)


def plot_case(ftype, order, b_ref, a_ref, filename,
               mag_ylim=(-100, 20), phase=True, phase_mask_db=-60):
    if phase:
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    else:
        fig, ax_mag = plt.subplots(1, 1, figsize=(8, 4.5))

    for precision in PRECISIONS:
        b_q = convert(b_ref, precision).astype(np.float64)
        a_q = convert(a_ref, precision).astype(np.float64)

        w, H = signal.freqz(b_q, a_q, worN=N_FREQ)
        freq_norm = w / np.pi

        mag_db = 20 * np.log10(np.abs(H) + 1e-300)

        style = '--' if precision == 'float16' else '-'
        lw = 2 if precision in ('float16', 'float32') else 1.2

        ax_mag.plot(freq_norm, mag_db, style, color=COLORS[precision], lw=lw, label=precision)

        if phase:
            phase_deg = np.degrees(np.unwrap(np.angle(H)))
            phase_masked = phase_deg.copy()
            phase_masked[mag_db < phase_mask_db] = np.nan
            ax_phase.plot(freq_norm, phase_masked, style, color=COLORS[precision], lw=lw, label=precision)

    ax_mag.set_ylabel('|H(f)| (dB)')
    ax_mag.set_title(f'{ftype.capitalize()} order {order} - Magnitude Response')
    ax_mag.axvline(CUTOFF, color='gray', ls=':', lw=1)
    ax_mag.set_ylim(*mag_ylim)
    ax_mag.legend(loc='lower left', fontsize=9)
    ax_mag.grid(True, alpha=0.3)
    if not phase:
        ax_mag.set_xlabel('Normalised Frequency (x pi rad/sample)')

    if phase:
        ax_phase.set_ylabel('Phase (degrees)')
        ax_phase.set_xlabel('Normalised Frequency (x pi rad/sample)')
        ax_phase.set_title(f'{ftype.capitalize()} order {order} - Phase Response (|H| > {phase_mask_db} dB only)')
        ax_phase.axvline(CUTOFF, color='gray', ls=':', lw=1)
        ax_phase.legend(loc='lower left', fontsize=9)
        ax_phase.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved {filename}")


import os
RESULTS_DIR = os.path.join('..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Case 1: Butterworth order 14 (float16 unstable, SM = -0.068)
# Shows magnitude AND phase - clean comparison
b_ref, a_ref = signal.butter(14, CUTOFF, btype='low', output='ba')
b_ref, a_ref = b_ref.astype(np.float64), a_ref.astype(np.float64)
plot_case('butterworth', 14, b_ref, a_ref, os.path.join(RESULTS_DIR, 'response_butter_order14.png'),
          mag_ylim=(-100, 20), phase=True, phase_mask_db=-60)

# Case 2: Elliptic order 10 (float32 unstable, SM = -0.0067)
# Magnitude only - phase is omitted (see module docstring)
b_ref, a_ref = signal.ellip(10, 1.0, 40.0, CUTOFF, btype='low', output='ba')
b_ref, a_ref = b_ref.astype(np.float64), a_ref.astype(np.float64)
plot_case('elliptic', 10, b_ref, a_ref, os.path.join(RESULTS_DIR, 'response_ellip_order10.png'),
          mag_ylim=(-100, 10), phase=False)
