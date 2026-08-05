"""
Run Direct-Form IIR filter evaluation across all 4 precision levels.

This script:
1. Designs Butterworth, Chebyshev I/II, and Elliptic IIR filters
   at orders 10, 20, 30
2. Converts coefficients to float16, float32, float64, mpmath
3. Computes all 5 sensitivity metrics for each configuration
4. Prints a clear summary table showing where filters become unstable
"""

import numpy as np
from scipy import signal
import mpmath
import warnings
warnings.filterwarnings('ignore')

# Configuration
mpmath.mp.dps = 50          # 50 decimal digit precision for mpmath

FILTER_TYPES  = ['butter', 'cheby1', 'cheby2', 'ellip']
ORDERS        = [10, 20, 30]
PRECISIONS    = ['float16', 'float32', 'float64', 'mpmath']
CUTOFF        = 0.3         # Normalised cutoff frequency
TRANS_BW      = 0.02        # Transition bandwidth
RP            = 1.0         # Passband ripple (dB) for cheby1/ellip
RS            = 40.0        # Stopband attenuation (dB) for cheby2/ellip
N_FREQ        = 4096        # Frequency grid points
N_TRIALS      = 10          # Perturbation trials for sensitivity norm
PERTURB       = 1e-6        # Perturbation amplitude
N_SAMPLES     = 1000        # Samples for roundoff noise


# Helper: Design filter at float64
def design_filter(ftype, order):
    """Design IIR filter and return b, a coefficients at float64."""
    wp = CUTOFF
    ws = CUTOFF + TRANS_BW
    try:
        if ftype == 'butter':
            b, a = signal.butter(order, wp, btype='low', output='ba')
        elif ftype == 'cheby1':
            b, a = signal.cheby1(order, RP, wp, btype='low', output='ba')
        elif ftype == 'cheby2':
            b, a = signal.cheby2(order, RS, ws, btype='low', output='ba')
        elif ftype == 'ellip':
            b, a = signal.ellip(order, RP, RS, wp, btype='low', output='ba')
        return b.astype(np.float64), a.astype(np.float64)
    except Exception as e:
        return None, None


# Helper: Convert to precision
def convert(coeffs, precision):
    """Convert float64 coefficients to target precision."""
    if precision == 'float16':
        return coeffs.astype(np.float16)
    elif precision == 'float32':
        return coeffs.astype(np.float32)
    elif precision == 'float64':
        return coeffs.astype(np.float64)
    elif precision == 'mpmath':
        return np.array([float(mpmath.mpf(str(c))) for c in coeffs], dtype=np.float64)


# Helper: Get poles
def get_poles(a):
    """Compute poles from denominator polynomial."""
    try:
        return np.roots(np.array(a, dtype=np.float64))
    except:
        return np.array([])


# Metric 1: Pole Displacement
def pole_displacement(poles_ref, poles_q):
    """L2 norm of pole displacement after quantisation."""
    if len(poles_ref) == 0 or len(poles_q) == 0:
        return np.nan
    n = min(len(poles_ref), len(poles_q))
    ref = poles_ref[np.argsort(np.angle(poles_ref))]
    q   = poles_q[np.argsort(np.angle(poles_q))]
    return float(np.linalg.norm(ref[:n] - q[:n]))


# Metric 2: Stability Margin
def stability_margin(poles):
    """
    SM = min{1 - |pk|}
    Positive = stable, Negative = UNSTABLE
    """
    if len(poles) == 0:
        return np.nan, False
    mags = np.abs(poles)
    sm = float(1.0 - np.max(mags))
    return sm, bool(np.all(mags < 1.0))


# Metric 3: Sensitivity Norm
def sensitivity_norm(b_ref, a_ref, b_q, a_q):
    """
    S = average ||H_perturbed - H_reference||_inf over N_TRIALS
    Measures how much the frequency response changes under small perturbations.
    """
    b_r = np.array(b_ref, dtype=np.float64)
    a_r = np.array(a_ref, dtype=np.float64)
    b_q = np.array(b_q,   dtype=np.float64)
    a_q = np.array(a_q,   dtype=np.float64)

    _, H_ref = signal.freqz(b_r, a_r, worN=N_FREQ)
    H_ref_mag = np.abs(H_ref)

    norms = []
    for _ in range(N_TRIALS):
        bp = b_q + np.random.normal(0, PERTURB * max(np.abs(b_q).mean(), 1e-10), b_q.shape)
        ap = a_q + np.random.normal(0, PERTURB * max(np.abs(a_q).mean(), 1e-10), a_q.shape)
        ap[0] = 1.0
        try:
            _, H_p = signal.freqz(bp, ap, worN=N_FREQ)
            norms.append(np.max(np.abs(np.abs(H_p) - H_ref_mag)))
        except:
            pass

    return float(np.mean(norms)) if norms else np.nan


# Metric 4: Frequency Response Deviation
def freq_response_deviation(b_ref, a_ref, b_q, a_q):
    """
    Dev = ||H_quantised - H_reference||_inf
    Direct magnitude error from precision reduction.
    """
    try:
        _, H_ref = signal.freqz(np.array(b_ref, dtype=np.float64),
                                np.array(a_ref, dtype=np.float64), worN=N_FREQ)
        _, H_q   = signal.freqz(np.array(b_q,   dtype=np.float64),
                                np.array(a_q,   dtype=np.float64), worN=N_FREQ)
        dev = float(np.max(np.abs(np.abs(H_q) - np.abs(H_ref))))
        dev_db = 20 * np.log10(dev + 1e-12)
        return dev, dev_db
    except:
        return np.nan, np.nan

# Metric 5: Roundoff Noise
def roundoff_noise(b_ref, a_ref, b_q, a_q):
    """
    Noise = ||y_quantised - y_reference||_2
    Accumulated output error over N_SAMPLES samples.
    """
    np.random.seed(42)
    x = np.random.randn(N_SAMPLES)
    try:
        y_ref = signal.lfilter(np.array(b_ref, dtype=np.float64),
                               np.array(a_ref, dtype=np.float64), x)
        y_q   = signal.lfilter(np.array(b_q,   dtype=np.float64),
                               np.array(a_q,   dtype=np.float64), x)
        return float(np.linalg.norm(y_q - y_ref))
    except:
        return np.nan

# Main Evaluation
def run_direct_form_evaluation():
    """
    Run complete Direct-Form evaluation across all filter types,
    orders, and precision levels.
    """
    print("=" * 80)
    print("STEP 2: DIRECT-FORM IIR FILTER EVALUATION — ALL 4 PRECISION LEVELS")
    print("Student: Darshan Mahadeva Naika | A00090581 | EEN1095")
    print("=" * 80)

    results = {}

    for ftype in FILTER_TYPES:
        results[ftype] = {}
        print(f"\n{'─' * 80}")
        print(f"Filter Type: {ftype.upper()}")
        print(f"{'─' * 80}")
        print(f"{'Order':<8} {'Precision':<10} {'Stable':<10} {'SM':>10} "
              f"{'Pole Disp':>12} {'Sens Norm':>12} {'Dev (dB)':>10} {'RO Noise':>12}")
        print(f"{'─' * 80}")

        for order in ORDERS:
            b_ref, a_ref = design_filter(ftype, order)
            if b_ref is None:
                print(f"  Order {order}: Design failed")
                continue

            poles_ref = get_poles(a_ref)
            results[ftype][order] = {}

            for precision in PRECISIONS:
                b_q = convert(b_ref, precision)
                a_q = convert(a_ref, precision)

                poles_q = get_poles(a_q)
                sm, stable = stability_margin(poles_q)
                pd  = pole_displacement(poles_ref, poles_q)
                sn  = sensitivity_norm(b_ref, a_ref, b_q, a_q)
                dev, dev_db = freq_response_deviation(b_ref, a_ref, b_q, a_q)
                rn  = roundoff_noise(b_ref, a_ref, b_q, a_q)

                stable_str = "STABLE  " if stable else "UNSTABLE"
                sm_str     = f"{sm:+.4f}"
                pd_str     = f"{pd:.2e}" if not np.isnan(pd) else "N/A"
                sn_str     = f"{sn:.2e}" if not np.isnan(sn) else "N/A"
                dev_str    = f"{dev_db:.2f}" if not np.isnan(dev_db) else "N/A"
                rn_str     = f"{rn:.2e}" if not np.isnan(rn) else "N/A"

                print(f"{order:<8} {precision:<10} {stable_str:<10} {sm_str:>10} "
                      f"{pd_str:>12} {sn_str:>12} {dev_str:>10} {rn_str:>12}")

                results[ftype][order][precision] = {
                    'stable':     stable,
                    'sm':         sm,
                    'pole_disp':  pd,
                    'sens_norm':  sn,
                    'freq_dev_db': dev_db,
                    'roundoff':   rn,
                }

    # Summary Table
    print(f"\n{'=' * 80}")
    print("STABILITY SUMMARY — DIRECT-FORM STRUCTURE")
    print(f"{'=' * 80}")
    print(f"{'Filter':<10} {'Order':<8} {'float16':<12} {'float32':<12} "
          f"{'float64':<12} {'mpmath':<12}")
    print(f"{'─' * 80}")

    for ftype in FILTER_TYPES:
        for order in ORDERS:
            if order not in results[ftype]:
                continue
            row = f"{ftype:<10} {order:<8} "
            for prec in PRECISIONS:
                if prec in results[ftype][order]:
                    r = results[ftype][order][prec]
                    status = "STABLE  " if r['stable'] else "UNSTABLE"
                    row += f"{status:<12}"
                else:
                    row += f"{'ERROR':<12}"
            print(row)

    # KEY FINDING — dynamically computed from actual results
    unstable_cases = []
    for ftype in FILTER_TYPES:
        for order in ORDERS:
            if order not in results[ftype]:
                continue
            for prec in PRECISIONS:
                if prec in results[ftype][order]:
                    if not results[ftype][order][prec]['stable']:
                        unstable_cases.append((ftype, order, prec,
                                               results[ftype][order][prec]['sm']))

    print(f"\n{'=' * 80}")
    print("KEY FINDING (dynamically computed):")
    if unstable_cases:
        print(f"Found {len(unstable_cases)} UNSTABLE configurations:")
        for ftype, order, prec, sm in unstable_cases:
            print(f"  {ftype.upper():<10} order {order:<4} {prec:<10} SM={sm:+.4f}")
        min_order_unstable = min(o for _,o,_,_ in unstable_cases)
        precisions_affected = list(dict.fromkeys(pr for _,_,pr,_ in unstable_cases))
        print(f"\nDirect-Form filters become UNSTABLE from order {min_order_unstable} onwards")
        print(f"at precision levels: {', '.join(precisions_affected)}")
        print("This confirms the sensitivity problem that Ladder structures are designed to solve.")
    else:
        print("All configurations STABLE at the tested orders and precisions.")
    print(f"{'=' * 80}")

    return results


if __name__ == '__main__':
    np.random.seed(42)
    results = run_direct_form_evaluation()
    print("\nStep 2 Complete!")
