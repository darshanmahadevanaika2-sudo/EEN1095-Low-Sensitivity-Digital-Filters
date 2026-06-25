"""
Comparison: Direct-Form (BA) vs Ladder (SOS)

Compares the two filter structures side-by-side across all 4 precision
levels using stability margin, max frequency response error, and
response plots.

Direct-Form (BA): one high-order polynomial — high sensitivity
Ladder (SOS):     cascaded 2nd-order sections — low sensitivity

"""

import numpy as np
from scipy import signal
import mpmath
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mpmath.mp.dps = 50

PRECISIONS = ['float16', 'float32', 'float64', 'mpmath']
RESULTS_DIR = os.path.join('..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def convert_ba(coeffs, precision):
    if precision == 'mpmath':
        return np.array([float(mpmath.mpf(str(c))) for c in coeffs], dtype=np.float64)
    dtype = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}[precision]
    return coeffs.astype(dtype).astype(np.float64)


def convert_sos(sos, precision):
    if precision == 'mpmath':
        return np.array([[float(mpmath.mpf(str(c))) for c in row]
                         for row in sos], dtype=np.float64)
    dtype = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}[precision]
    return sos.astype(dtype).astype(np.float64)


def stability_margin_ba(a):
    poles = np.roots(a.astype(np.float64))
    return float(1.0 - np.max(np.abs(poles)))


def stability_margin_sos(sos):
    min_sm = np.inf
    for section in sos:
        poles = np.roots(section[3:].astype(np.float64))
        min_sm = min(min_sm, float(1.0 - np.max(np.abs(poles))))
    return min_sm


def compare(ftype='butter', order=10, cutoff=0.3):
    """Side-by-side comparison of BA vs SOS at all 4 precisions."""
    if ftype == 'butter':
        b_ref, a_ref = signal.butter(order, cutoff, btype='low', output='ba')
        sos_ref = signal.butter(order, cutoff, btype='low', output='sos')
    elif ftype == 'ellip':
        b_ref, a_ref = signal.ellip(order, 1.0, 40.0, cutoff, btype='low', output='ba')
        sos_ref = signal.ellip(order, 1.0, 40.0, cutoff, btype='low', output='sos')

    b_ref = b_ref.astype(np.float64)
    a_ref = a_ref.astype(np.float64)
    sos_ref = sos_ref.astype(np.float64)

    _, H_ref = signal.freqz(b_ref, a_ref, worN=1024)
    H_ref_mag = np.abs(H_ref)

    print(f"\n{'='*70}")
    print(f"DIRECT-FORM (BA) vs LADDER (SOS) -- {ftype.upper()} order {order}")
    print(f"{'='*70}")
    print(f"{'Structure':<20} {'Precision':<10} {'SM':>10} {'Max|H|err':>12} {'Status':>10}")
    print(f"{'-'*65}")

    for precision in PRECISIONS:
        b_q = convert_ba(b_ref, precision)
        a_q = convert_ba(a_ref, precision)
        sm = stability_margin_ba(a_q)
        _, H_q = signal.freqz(b_q, a_q, worN=1024)
        err = float(np.max(np.abs(np.abs(H_q) - H_ref_mag)))
        print(f"{'Direct-Form (BA)':<20} {precision:<10} {sm:>10.4f} "
              f"{err:>12.2e} {'STABLE' if sm > 0 else 'UNSTABLE*':>10}")

    print()
    for precision in PRECISIONS:
        sos_q = convert_sos(sos_ref, precision)
        sm = stability_margin_sos(sos_q)
        _, H_q = signal.sosfreqz(sos_q, worN=1024)
        err = float(np.max(np.abs(np.abs(H_q) - H_ref_mag)))
        print(f"{'Ladder (SOS)':<20} {precision:<10} {sm:>10.4f} "
              f"{err:>12.2e} {'STABLE' if sm > 0 else 'UNSTABLE*':>10}")

    print("* SM < 0 means unstable")


def plot_comparison(ftype, order, cutoff=0.3):
    """Side-by-side magnitude response plot: BA vs SOS."""
    if ftype == 'butter':
        b_ref, a_ref = signal.butter(order, cutoff, btype='low', output='ba')
        sos_ref = signal.butter(order, cutoff, btype='low', output='sos')
    elif ftype == 'ellip':
        b_ref, a_ref = signal.ellip(order, 1.0, 40.0, cutoff, btype='low', output='ba')
        sos_ref = signal.ellip(order, 1.0, 40.0, cutoff, btype='low', output='sos')

    b_ref = b_ref.astype(np.float64)
    a_ref = a_ref.astype(np.float64)
    sos_ref = sos_ref.astype(np.float64)

    COLORS = {'float16': 'tab:red', 'float32': 'tab:orange',
              'float64': 'tab:blue', 'mpmath': 'tab:green'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for precision in PRECISIONS:
        style = '--' if precision == 'float16' else '-'
        lw = 2 if precision in ('float16', 'float32') else 1.2

        b_q = convert_ba(b_ref, precision)
        a_q = convert_ba(a_ref, precision)
        w, H = signal.freqz(b_q, a_q, worN=1024)
        ax1.plot(w/np.pi, 20*np.log10(np.abs(H)+1e-300),
                 style, color=COLORS[precision], lw=lw, label=precision)

        sos_q = convert_sos(sos_ref, precision)
        w, H = signal.sosfreqz(sos_q, worN=1024)
        ax2.plot(w/np.pi, 20*np.log10(np.abs(H)+1e-300),
                 style, color=COLORS[precision], lw=lw, label=precision)

    for ax, title in [(ax1, 'Direct-Form (BA)'), (ax2, 'Ladder (SOS)')]:
        ax.set_ylim(-100, 20)
        ax.set_xlabel('Normalised Frequency (x pi rad/sample)')
        ax.set_ylabel('|H(f)| (dB)')
        ax.set_title(f'{ftype.capitalize()} order {order} - {title}')
        ax.axvline(cutoff, color='gray', ls=':', lw=1)
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Direct-Form vs Ladder - {ftype.capitalize()} order {order}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f'comparison_{ftype}_order{order}.png')
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved {fname}")


if __name__ == '__main__':
    print("DIRECT-FORM vs LADDER COMPARISON")
    print("Student: Darshan Mahadeva Naika | A00090581 | EEN1095")

    compare('butter', 10)
    compare('butter', 14)
    compare('butter', 24)
    compare('ellip', 10)

    plot_comparison('butter', 14)
    plot_comparison('ellip', 10)
