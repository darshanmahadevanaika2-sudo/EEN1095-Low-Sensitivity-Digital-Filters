"""
Main experimental evaluation runner for the multi-precision
digital filter evaluation framework.

Runs all configurations:
  - 3 IIR structures × 4 filter types × 3 orders × 4 precisions = 144 IIR configs
  - 3 FIR tap counts × 4 precisions = 12 FIR configs
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

from filter_design import (
    design_iir_filter, design_fir_filter,
    convert_to_precision, get_poles_from_a, get_poles_from_sos
)
from metrics import compute_iir_metrics, compute_fir_metrics

# ── Experiment Configuration ──────────────────────────────────────────────────
IIR_FILTER_TYPES  = ['butter', 'cheby1', 'cheby2', 'ellip']
IIR_ORDERS        = [10, 20, 30]
FIR_TAPS          = [100, 200, 300]
PRECISION_LEVELS  = ['float16', 'float32', 'float64', 'mpmath']
TRANSITION_BW     = 0.02
CUTOFF            = 0.3

# IIR Structures — currently evaluating Direct-Form
# Ladder and WDF will be added in Stage 3
IIR_STRUCTURES = ['direct_form']  # will add 'ladder', 'wdf'


def run_iir_experiment(filter_type, order, structure='direct_form'):
    """
    Run IIR evaluation for one filter type and order across all precision levels.

    Returns
    -------
    results : dict
        Metrics for each precision level
    """
    print(f"  IIR: {filter_type} order-{order} [{structure}]")

    # Design at float64 baseline
    try:
        sos_ref, b_ref, a_ref = design_iir_filter(
            filter_type, order,
            cutoff=CUTOFF,
            transition_bw=TRANSITION_BW
        )
    except Exception as e:
        print(f"    ERROR designing filter: {e}")
        return None

    # Reference poles
    poles_ref = get_poles_from_a(a_ref, 'float64')

    results = {}

    for precision in PRECISION_LEVELS:
        try:
            # Convert coefficients to target precision
            b_q = convert_to_precision(b_ref, precision)
            a_q = convert_to_precision(a_ref, precision)

            # Handle mpmath pole computation
            if precision == 'mpmath':
                # Convert back to float64 for pole calculation
                b_q_arr = np.array([float(c) for c in b_q], dtype=np.float64)
                a_q_arr = np.array([float(c) for c in a_q], dtype=np.float64)
            else:
                b_q_arr = np.array(b_q, dtype=np.float64)
                a_q_arr = np.array(a_q, dtype=np.float64)

            poles_q = get_poles_from_a(a_q_arr, precision)

            # Compute metrics
            metrics = compute_iir_metrics(
                b_ref, a_ref,
                b_q_arr, a_q_arr,
                poles_ref, poles_q,
                precision
            )
            results[precision] = metrics

            # Print summary line
            stable_str = "STABLE  " if metrics['is_stable'] else "UNSTABLE"
            print(f"    {precision:8s}: pd={metrics['pole_displacement']:.2e} | "
                  f"sm={metrics['stability_margin']:+.4f} | "
                  f"{stable_str} | "
                  f"sn={metrics['sensitivity_norm']:.2e} | "
                  f"rn={metrics['roundoff_noise']:.2e}")

        except Exception as e:
            print(f"    {precision:8s}: ERROR — {e}")
            results[precision] = {'error': str(e)}

    return results


def run_fir_experiment(num_taps):
    """
    Run FIR evaluation for one tap count across all precision levels.

    Returns
    -------
    results : dict
        Metrics for each precision level
    """
    print(f"  FIR: {num_taps} taps")

    # Design at float64 baseline
    h_ref = design_fir_filter(num_taps, cutoff=CUTOFF)
    results = {}

    for precision in PRECISION_LEVELS:
        try:
            h_q = convert_to_precision(h_ref, precision)

            if precision == 'mpmath':
                h_q_arr = np.array([float(c) for c in h_q], dtype=np.float64)
            else:
                h_q_arr = np.array(h_q, dtype=np.float64)

            metrics = compute_fir_metrics(h_ref, h_q_arr, precision)
            results[precision] = metrics

            print(f"    {precision:8s}: dev={metrics['freq_deviation']:.2e} | "
                  f"dev_db={metrics['freq_deviation_db']:.2f} dB | "
                  f"rn={metrics['roundoff_noise']:.2e}")

        except Exception as e:
            print(f"    {precision:8s}: ERROR — {e}")
            results[precision] = {'error': str(e)}

    return results


def run_all_experiments(save_results=True):
    """
    Run the complete experimental evaluation.

    Returns
    -------
    all_results : dict
        Complete results dataset
    """
    all_results = {
        'config': {
            'iir_filter_types': IIR_FILTER_TYPES,
            'iir_orders':       IIR_ORDERS,
            'fir_taps':         FIR_TAPS,
            'precision_levels': PRECISION_LEVELS,
            'transition_bw':    TRANSITION_BW,
            'cutoff':           CUTOFF,
            'n_freq_points':    4096,
            'n_trials':         10,
            'perturb_amp':      1e-6,
            'n_samples':        1000,
        },
        'iir_results': {},
        'fir_results': {},
    }

    print("=" * 70)
    print("MULTI-PRECISION DIGITAL FILTER EVALUATION FRAMEWORK")
    print("Darshan Mahadeva Naika | A00090581")
    print("=" * 70)

    # ── IIR Experiments ───────────────────────────────────────────────────────
    print("\n--- IIR FILTER EXPERIMENTS ---")
    for structure in IIR_STRUCTURES:
        all_results['iir_results'][structure] = {}
        for filter_type in IIR_FILTER_TYPES:
            all_results['iir_results'][structure][filter_type] = {}
            for order in IIR_ORDERS:
                key = f"order_{order}"
                result = run_iir_experiment(filter_type, order, structure)
                if result:
                    all_results['iir_results'][structure][filter_type][key] = result

    # ── FIR Experiments ───────────────────────────────────────────────────────
    print("\n--- FIR FILTER EXPERIMENTS ---")
    for taps in FIR_TAPS:
        key = f"taps_{taps}"
        result = run_fir_experiment(taps)
        all_results['fir_results'][key] = result

    # ── Save Results ──────────────────────────────────────────────────────────
    if save_results:
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} not JSON serializable")

        with open('/home/claude/filter_framework/results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=convert_for_json)
        print("\nResults saved to results.json")

    return all_results


def print_summary(all_results):
    """Print a clean summary table of key findings."""
    print("\n" + "=" * 70)
    print("SUMMARY — IIR STABILITY RESULTS")
    print("=" * 70)
    print(f"{'Structure':<16} {'Filter':<10} {'Order':<8} "
          f"{'float16':<12} {'float32':<12} {'float64':<12} {'mpmath':<12}")
    print("-" * 70)

    for structure, ftype_data in all_results['iir_results'].items():
        for ftype, order_data in ftype_data.items():
            for order_key, prec_data in order_data.items():
                order = order_key.replace('order_', '')
                row = f"{structure:<16} {ftype:<10} {order:<8} "
                for prec in ['float16', 'float32', 'float64', 'mpmath']:
                    if prec in prec_data and 'is_stable' in prec_data[prec]:
                        stable = prec_data[prec]['is_stable']
                        pd = prec_data[prec]['pole_displacement']
                        row += f"{'✓' if stable else '✗'} {pd:.1e}  "
                    else:
                        row += f"{'ERROR':<12}"
                print(row)


if __name__ == '__main__':
    results = run_all_experiments(save_results=True)
    print_summary(results)
    print("\nFramework execution complete!")
