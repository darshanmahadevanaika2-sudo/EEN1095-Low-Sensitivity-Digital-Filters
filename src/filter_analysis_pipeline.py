"""
filter_analysis_pipeline.py — Complete Project Pipeline Runner

Usage:
    python filter_analysis_pipeline.py              # Run everything
    python filter_analysis_pipeline.py --step 1     # Run only step 1
    python filter_analysis_pipeline.py --from 4     # Run from step 4 onwards

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

# ── Path setup ────────────────────────────────────────────────────────────────
# Pipeline lives in src/ — testing files are in ../testing/
TESTING_DIR = os.path.join('..', 'tests')
RESULTS_DIR = os.path.join('..', 'results')


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_step(step_num, name, filename):
    """Run a single pipeline step from the testing/ folder."""
    print(f"\n{'='*65}")
    print(f"STEP {step_num}: {name}")
    print(f"File: testing/{os.path.basename(filename)}")
    print(f"{'='*65}")
    start = time.time()

    if not os.path.exists(filename):
        print(f"\n  Step {step_num} SKIPPED — file not found: {filename}")
        return False

    try:
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
    print("Repository structure:")
    print("  src/      — core framework (filter_design, metrics, experiment)")
    print("  tests/  — all test and evaluation files")
    print("  results/  — all output plots and audio files")
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


def print_summary(results, steps):
    print(f"\n{'='*65}")
    print("PIPELINE SUMMARY")
    print(f"{'='*65}")

    passed = 0
    for num, name, _ in steps:
        status = "PASS" if results.get(num) else "FAIL"
        if results.get(num): passed += 1
        print(f"  Step {num}: {name:<40} {status}")

    total_steps = len(steps)
    print(f"\n  {passed}/{total_steps} steps completed successfully")

    if passed == total_steps:
        print(f"\n  ALL {total_steps} STEPS COMPLETE")
        print("  Results saved to ../results/")
        print("\n  See individual step outputs above for all key findings.")
        if os.path.exists(RESULTS_DIR):
            png_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.png')]
            wav_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.wav')]
            print(f"    {len(png_files)} plot(s) saved to {RESULTS_DIR}/")
            print(f"    {len(wav_files)} audio file(s) saved to {RESULTS_DIR}/")
    else:
        failed = total_steps - passed
        print(f"\n  {failed} step(s) failed. Check output above for details.")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Run the complete EEN1095 project pipeline')
    parser.add_argument('--step', type=int,
                        help='Run only this step number (1-9)')
    parser.add_argument('--from', dest='from_step', type=int,
                        help='Run from this step number onwards')
    args = parser.parse_args()

    print_header()

    # All test files now live in testing/ folder
    steps = [
        (1, "Direct-Form IIR Baseline Evaluation",
             os.path.join(TESTING_DIR, "direct_form_evaluation.py")),
        (2, "Signal Validation",
             os.path.join(TESTING_DIR, "signal_validation.py")),
        (3, "Degradation Analysis — Butterworth",
             os.path.join(TESTING_DIR, "degradation_analysis.py")),
        (4, "Degradation Analysis — Elliptic",
             os.path.join(TESTING_DIR, "degradation_analysis_ellip.py")),
        (5, "Amplitude and Phase Response Plots",
             os.path.join(TESTING_DIR, "plot_responses.py")),
        (6, "Ladder Filter (SOS) Evaluation",
             os.path.join(TESTING_DIR, "ladder_filter.py")),
        (7, "Direct-Form vs Ladder Comparison",
             os.path.join(TESTING_DIR, "comparison.py")),
        (8, "Audio Filter Test",
             os.path.join(TESTING_DIR, "audio_filter_test.py")),
        (9, "Competing Approach Comparison",
             os.path.join(TESTING_DIR, "competing_approach.py")),
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
    print_summary(results, steps)


if __name__ == '__main__':
    main()
