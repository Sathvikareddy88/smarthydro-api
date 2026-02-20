"""
training/train_all.py
──────────────────────
Master script that trains ALL five models in sequence.
Respects --skip flags so you can retrain individual models.

Run:
  # Train everything
  python training/train_all.py

  # Train only LSTM and Autoencoder
  python training/train_all.py --only lstm autoencoder

  # Quick smoke-test run (small epoch counts)
  python training/train_all.py --quick
"""

import argparse
import time
import sys
import os

MODELS = ["lstm", "ppo", "cnn", "yolo", "autoencoder"]

QUICK_OVERRIDES = {
    "lstm":        {"epochs": 5,  "batch": 64},
    "ppo":         {"timesteps": 10_000, "envs": 2},
    "cnn":         {"epochs": 3,  "batch": 16, "n_synthetic": 40},
    "yolo":        {"epochs": 3,  "batch": 8},
    "autoencoder": {"epochs": 5,  "batch": 256},
}


def _run(name: str, quick: bool):
    start = time.time()
    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {name.upper()}")
    print(f"{'=' * 60}\n")

    try:
        if name == "lstm":
            from training.train_lstm import train
            kwargs = QUICK_OVERRIDES["lstm"] if quick else {}
            train(**kwargs)

        elif name == "ppo":
            from training.train_ppo import train
            kwargs = QUICK_OVERRIDES["ppo"] if quick else {}
            train(**kwargs)

        elif name == "cnn":
            from training.train_cnn import train
            kwargs = QUICK_OVERRIDES["cnn"] if quick else {}
            train(**kwargs)

        elif name == "yolo":
            from training.train_yolo import train
            kwargs = QUICK_OVERRIDES["yolo"] if quick else {}
            train(**kwargs)

        elif name == "autoencoder":
            from training.train_autoencoder import train
            kwargs = QUICK_OVERRIDES["autoencoder"] if quick else {}
            train(**kwargs)

        elapsed = time.time() - start
        print(f"\n  ✅ {name.upper()} completed in {elapsed:.1f}s")
        return True

    except Exception as exc:
        elapsed = time.time() - start
        print(f"\n  ❌ {name.upper()} FAILED after {elapsed:.1f}s: {exc}")
        import traceback; traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Train all SmartHydro ML models")
    parser.add_argument("--only",  nargs="+", choices=MODELS,
                        help="Train only these models (space-separated)")
    parser.add_argument("--skip",  nargs="+", choices=MODELS, default=[],
                        help="Skip these models")
    parser.add_argument("--quick", action="store_true",
                        help="Run with minimal epochs/steps for smoke-testing")
    args = parser.parse_args()

    to_train = args.only if args.only else MODELS
    to_train = [m for m in to_train if m not in args.skip]

    print("\n" + "=" * 60)
    print("  SmartHydro — Full ML Training Pipeline")
    print("=" * 60)
    print(f"  Models to train: {to_train}")
    if args.quick:
        print("  Mode: QUICK (reduced epochs/timesteps)")
    print("=" * 60)

    results = {}
    total_start = time.time()

    for model_name in to_train:
        results[model_name] = _run(model_name, quick=args.quick)

    # ── Summary ──────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print("\n\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {name}")

    saved = [f for f in os.listdir("saved_models") if f.endswith((".h5", ".pt", ".zip"))]
    print(f"\n  Saved model files ({len(saved)}):")
    for f in sorted(saved):
        size_kb = os.path.getsize(os.path.join("saved_models", f)) // 1024
        print(f"    saved_models/{f}  ({size_kb} KB)")

    print(f"\n  Total time: {total_elapsed:.1f}s")
    print("=" * 60)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
