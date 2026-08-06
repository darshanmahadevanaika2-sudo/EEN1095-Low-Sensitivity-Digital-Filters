"""
Order sweep / degradation analysis.

Sweeps filter order from 2 to 30 (step 2) for a Butterworth low-pass
filter and, at each order and precision level, computes:

  - Stability Margin   SM = 1 - max|pole|   (negative = unstable)
  - Max |H| error      = max |H_precision(w) - H_float64(w)| over the
                         full frequency grid (sensitivity to coefficient
                         quantisation)

This shows the gradual degradation as order increases, leading up to
the point where each precision level becomes unstable.
"""

import numpy as np
from scipy import signal
import mpmath
import warnings
warnings.filterwarnings('ignore')

mpmath.mp.dps = 50

PRECISIONS = ['float16', 'float32', 'float64', 'mpmath']
ORDERS = list(range(2, 31, 2))   # 2, 4, 6, ..., 30
CUTOFF = 0.3
N_FREQ = 1024


def design_filter(order, cutoff=CUTOFF):
    b, a = signal.butter(order, cutoff, btype='low', output='ba')
    return b.astype(np.float64), a.astype(np.float64)


def convert(coeffs, precision):
    if precision == 'float16':
        return coeffs.astype(np.float16)
    elif precision == 'float32':
        return coeffs.astype(np.float32)
    elif precision == 'float64':
        return coeffs.astype(np.float64)
    elif precision == 'mpmath':
        return np.array([float(mpmath.mpf(str(c))) for c in coeffs], dtype=np.float64)


def stability_margin(a):
    a64 = a.astype(np.float64)
    poles = np.roots(a64)
    return 1.0 - np.max(np.abs(poles))


def max_freq_response_error(b_q, a_q, b_ref, a_ref, n_points=N_FREQ):
    _, H_q   = signal.freqz(b_q.astype(np.float64), a_q.astype(np.float64), worN=n_points)
    _, H_ref = signal.freqz(b_ref, a_ref, worN=n_points)
    return float(np.max(np.abs(H_q - H_ref)))


# Run sweep 
results = {p: {'SM': [], 'err': []} for p in PRECISIONS}

for order in ORDERS:
    b_ref, a_ref = design_filter(order)
    for precision in PRECISIONS:
        b_q = convert(b_ref, precision)
        a_q = convert(a_ref, precision)

        if np.any(~np.isfinite(a_q)):
            results[precision]['SM'].append(np.nan)
            results[precision]['err'].append(np.nan)
            continue

        sm = stability_margin(a_q)
        results[precision]['SM'].append(sm)

        if precision == 'float64':
            results[precision]['err'].append(0.0)
        else:
            err = max_freq_response_error(b_q, a_q, b_ref, a_ref)
            results[precision]['err'].append(err)


# Print Table 1: Stability Margin
print("=== BUTTERWORTH LOW-PASS (cutoff=0.3) - STABILITY MARGIN vs ORDER ===")
print(f"{'Order':<7}" + "".join(f"{p:>14}" for p in PRECISIONS))
for i, order in enumerate(ORDERS):
    row = f"{order:<7}"
    for p in PRECISIONS:
        sm = results[p]['SM'][i]
        if np.isnan(sm):
            row += f"{'nan':>14}"
        else:
            flag = "" if sm > 0 else " (UNSTABLE)"
            row += f"{sm:>10.4f}{flag:<4}" if not flag else f"{sm:>10.4f}{'*':<4}"
    print(row)
print("* = unstable (SM < 0)")

# Print Table 2: Max |H| error vs float64 
print("\n=== MAX |H(w)| ERROR vs FLOAT64 REFERENCE - ACROSS FREQUENCY GRID ===")
print(f"{'Order':<7}{'float16':>14}{'float32':>14}{'mpmath':>14}")
for i, order in enumerate(ORDERS):
    row = f"{order:<7}"
    for p in ['float16', 'float32', 'mpmath']:
        err = results[p]['err'][i]
        if np.isnan(err):
            row += f"{'nan':>14}"
        else:
            row += f"{err:>14.2e}"
    print(row)

# First unstable order per precision
print("\n=== FIRST UNSTABLE ORDER (SM crosses below 0) ===")
for p in PRECISIONS:
    sms = results[p]['SM']
    first_unstable = None
    for i, sm in enumerate(sms):
        if np.isnan(sm) or sm < 0:
            first_unstable = ORDERS[i]
            break
    print(f"  {p:<8}: {'order ' + str(first_unstable) if first_unstable else 'stable up to order 30'}")
