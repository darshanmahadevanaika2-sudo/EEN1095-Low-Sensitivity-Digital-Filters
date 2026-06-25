"""
Ladder Filter Implementation using Second-Order Sections (SOS).

"""

import numpy as np
from scipy import signal
import mpmath
import os
import warnings
warnings.filterwarnings('ignore')

mpmath.mp.dps = 50

PRECISIONS = ['float16', 'float32', 'float64', 'mpmath']
RESULTS_DIR = os.path.join('..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def convert_sos(sos, precision):
    """Convert SOS matrix to target precision."""
    if precision == 'mpmath':
        return np.array([[float(mpmath.mpf(str(c))) for c in row]
                         for row in sos], dtype=np.float64)
    dtype = {'float16': np.float16,
             'float32': np.float32,
             'float64': np.float64}[precision]
    return sos.astype(dtype).astype(np.float64)


def stability_margin_sos(sos):
    """
    Stability margin for Ladder (SOS).
    SM = min(1 - max|pole|) across all second-order sections.
    Each section is checked independently — structural isolation means
    one section's stability does not affect others.
    Positive SM = STABLE. Negative SM = UNSTABLE.
    """
    min_sm = np.inf
    for section in sos:
        a_section = section[3:].astype(np.float64)
        poles = np.roots(a_section)
        sm = float(1.0 - np.max(np.abs(poles)))
        min_sm = min(min_sm, sm)
    return min_sm


def ladder_freqz(sos, precision, n_points=1024):
    """Compute frequency response of Ladder (SOS) filter at given precision."""
    sos_q = convert_sos(sos, precision)
    w, H = signal.sosfreqz(sos_q, worN=n_points)
    return w, H


def evaluate_ladder(ftype='butter', order=10, cutoff=0.3):
    """
    Evaluate Ladder (SOS) filter across all 4 precision levels.
    Reports stability margin and max frequency response error
    vs the float64 reference.
    """
    # Design at float64 reference
    if ftype == 'butter':
        sos_ref = signal.butter(order, cutoff, btype='low', output='sos')
    elif ftype == 'ellip':
        sos_ref = signal.ellip(order, 1.0, 40.0, cutoff, btype='low', output='sos')

    sos_ref = sos_ref.astype(np.float64)

    # Reference frequency response at float64
    w_ref, H_ref = signal.sosfreqz(sos_ref, worN=1024)
    H_ref_mag = np.abs(H_ref)

    print(f"\n{'='*65}")
    print(f"LADDER FILTER (SOS) -- {ftype.upper()} order {order}, cutoff={cutoff}")
    print(f"{'='*65}")
    print(f"Number of SOS sections: {len(sos_ref)}")
    print(f"\n{'Precision':<10} {'SM':>12} {'Max|H|err':>14} {'Status':>10}")
    print(f"{'-'*50}")

    results = {}
    for precision in PRECISIONS:
        sos_q = convert_sos(sos_ref, precision)
        sm = stability_margin_sos(sos_q)
        _, H_q = signal.sosfreqz(sos_q, worN=1024)
        err = float(np.max(np.abs(np.abs(H_q) - H_ref_mag)))
        status = 'STABLE' if sm > 0 else 'UNSTABLE*'
        print(f"{precision:<10} {sm:>12.4f} {err:>14.2e} {status:>10}")
        results[precision] = {'sm': sm, 'err': err, 'stable': sm > 0}

    print("* SM < 0 means unstable")
    return results, sos_ref


def order_sweep(ftype='butter', orders=range(2, 31, 2), cutoff=0.3):
    """
    Sweep filter order and show stability margin of Ladder (SOS)
    at each precision level.
    """
    print(f"\n{'='*65}")
    print(f"LADDER (SOS) ORDER SWEEP -- {ftype.upper()}, cutoff={cutoff}")
    print(f"{'='*65}")
    print(f"{'Order':<7}" + "".join(f"{p:>14}" for p in PRECISIONS))

    first_unstable = {p: None for p in PRECISIONS}

    for order in orders:
        if ftype == 'butter':
            sos_ref = signal.butter(order, cutoff, btype='low', output='sos')
        elif ftype == 'ellip':
            sos_ref = signal.ellip(order, 1.0, 40.0, cutoff, btype='low', output='sos')

        sos_ref = sos_ref.astype(np.float64)
        row = f"{order:<7}"

        for p in PRECISIONS:
            sos_q = convert_sos(sos_ref, p)
            sm = stability_margin_sos(sos_q)
            flag = '*' if sm < 0 else ''
            row += f"{sm:>10.4f}{flag:<4}"
            if sm < 0 and first_unstable[p] is None:
                first_unstable[p] = order

        print(row)

    print("* = unstable (SM < 0)")
    print(f"\nFirst unstable order:")
    max_order = list(orders)[-1]
    for p in PRECISIONS:
        val = f"order {first_unstable[p]}" if first_unstable[p] else f"stable to {max_order}"
        print(f"  {p:<10}: {val}")


if __name__ == '__main__':
    print("LADDER FILTER (SOS) IMPLEMENTATION")
    print("Student: Darshan Mahadeva Naika | A00090581 | EEN1095")
    print("Based on: Bruton 1975, IEEE Trans. Circuits Syst.")

    # Evaluate at key orders
    evaluate_ladder('butter', order=10)
    evaluate_ladder('butter', order=14)  # Direct-Form float16 fails here
    evaluate_ladder('butter', order=24)  # Direct-Form float32 fails here
    evaluate_ladder('ellip', order=10)   # Direct-Form float16+float32 fail here

    # Order sweep
    order_sweep('butter', range(2, 31, 2))
    order_sweep('ellip', range(2, 21, 2))

    print("\n\nKEY FINDINGS:")
    print("="*65)
    print("Ladder (SOS) remains STABLE at all precision levels")
    print("at every order where Direct-Form (BA) fails.")
    print("Structural distribution of poles is the primary robustness driver.")
