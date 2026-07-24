"""
Steps:
    1. Direct-Form IIR baseline evaluation
    2. Signal validation (impulse vs sine vs freqz)
    3. Degradation analysis — Butterworth order sweep
    4. Degradation analysis — Elliptic order sweep
    5. Amplitude and phase response plots
    6. Ladder filter (SOS) evaluation
    7. Direct-Form vs Ladder comparison
    8. Audio filter test (impulse, sinusoidal, audio, overflow)
    9. Competing approach comparison (Parallel Form vs Ladder)
"""

import sys
import os
import time
import argparse

# Helpers
def run_step(step_num, name, filename):
    """Run a single pipeline step."""
    print(f"\n{'='*65}")
    print(f"STEP {step_num}: {name}")
    print(f"File: src/{filename}")
    print(f"{'='*65}")
    start = time.time()

    try:
        # Execute the script
        with open(filename, 'r') as f:
            code = f.read()
        exec(compile(code, filename, 'exec'), {'__name__': '__main__'})
        elapsed = time.time() - start
        print(f"\n  Step {step_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"\n  Step {step_num} FAILED: {e}")
        return False


def print_header():
    print("="*65)
    print("EEN1095 — LOW SENSITIVITY DIGITAL FILTERS")
    print("Complete Project Pipeline")
    print("Student: Darshan Mahadeva Naika | A00090581")
    print("Supervisor: Martin Collier")
    print("="*65)
    print()
    print("This script runs the complete project in sequence:")
    print()
    print("  Step 1 — Direct-Form IIR baseline evaluation")
    print("  Step 2 — Signal validation")
    print("  Step 3 — Degradation analysis (Butterworth)")
    print("  Step 4 — Degradation analysis (Elliptic)")
    print("  Step 5 — Amplitude and phase response plots")
    print("  Step 6 — Ladder filter (SOS) evaluation")
    print("  Step 7 — Direct-Form vs Ladder comparison")
    print("  Step 8 — Audio filter test")
    print("  Step 9 — Competing approach comparison")
    print()
    print("Results saved to: ../results/")
    print("Expected runtime: 10-15 minutes total")
    print("="*65)


def print_summary(results):
    print(f"\n{'='*65}")
    print("PIPELINE SUMMARY")
    print(f"{'='*65}")

    steps = [
        (1, "Direct-Form IIR baseline"),
        (2, "Signal validation"),
        (3, "Degradation analysis (Butterworth)"),
        (4, "Degradation analysis (Elliptic)"),
        (5, "Response plots"),
        (6, "Ladder filter evaluation"),
        (7, "Direct-Form vs Ladder comparison"),
        (8, "Audio filter test"),
        (9, "Competing approach comparison"),
    ]

    passed = 0
    for num, name in steps:
        status = "PASS" if results.get(num) else "FAIL"
        if results.get(num): passed += 1
        print(f"  Step {num}: {name:<40} {status}")

    print(f"\n  {passed}/9 steps completed successfully")

    if passed == 9:
        print("\n  ALL STEPS COMPLETE")
        print("  Results saved to ../results/")
        print("\n  Key findings:")
        print("  - Direct-Form unstable at float16 order 14, float32 order 24")
        print("  - Ladder (SOS) stable at ALL orders and ALL precisions")
        print("  - Audio SNR: BA float16=11.3dB vs SOS float16=18.7dB")
        print("  - Ladder 24x-33000x more accurate than Parallel Form (Bank 2018)")
        print("  - Structure is the primary robustness driver, not precision")
    else:
        print("\n  Some steps failed. Check output above for details.")


# Main pipeline
def main():
    parser = argparse.ArgumentParser(
        description='Run the complete EEN1095 project pipeline')
    parser.add_argument('--step', type=int,
                        help='Run only this step number (1-9)')
    parser.add_argument('--from', dest='from_step', type=int,
                        help='Run from this step number onwards')
    args = parser.parse_args()

    print_header()

    # Define all steps
    steps = [
        (1, "Direct-Form IIR Baseline Evaluation",    "direct_form_evaluation.py"),
        (2, "Signal Validation",                       "signal_validation.py"),
        (3, "Degradation Analysis — Butterworth",      "degradation_analysis.py"),
        (4, "Degradation Analysis — Elliptic",         "degradation_analysis_ellip.py"),
        (5, "Amplitude and Phase Response Plots",      "plot_responses.py"),
        (6, "Ladder Filter (SOS) Evaluation",          "ladder_filter.py"),
        (7, "Direct-Form vs Ladder Comparison",        "comparison.py"),
        (8, "Audio Filter Test",                       "audio_filter_test.py"),
        (9, "Competing Approach Comparison",           "competing_approach.py"),
    ]

    # Filter steps based on arguments
    if args.step:
        steps = [(n, name, f) for n, name, f in steps if n == args.step]
        if not steps:
            print(f"Error: Step {args.step} not found. Valid steps are 1-9.")
            sys.exit(1)
    elif args.from_step:
        steps = [(n, name, f) for n, name, f in steps if n >= args.from_step]
        if not steps:
            print(f"Error: Step {args.from_step} not found. Valid steps are 1-9.")
            sys.exit(1)

    # Run steps
    results = {}
    total_start = time.time()

    for step_num, name, filename in steps:
        # Check file exists
        if not os.path.exists(filename):
            print(f"\nSkipping Step {step_num} — {filename} not found")
            results[step_num] = False
            continue

        success = run_step(step_num, name, filename)
        results[step_num] = success

        if not success:
            print(f"\nStep {step_num} failed. Continue anyway? (y/n): ", end='')
            try:
                response = input().strip().lower()
                if response != 'y':
                    print("Pipeline stopped.")
                    break
            except:
                print("Pipeline stopped.")
                break

    total_elapsed = time.time() - total_start
    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
    print_summary(results)


if __name__ == '__main__':
    main()
