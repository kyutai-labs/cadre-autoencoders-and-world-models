"""Does a converted test split match ours? Compares the test split of a dataset root written by prepare/<env>.py with
our reference fingerprint (fingerprints.json, beside this file): its episode and frame counts, and per action
dimension the mean, std, min and max over all rows, and the std over the even and the odd rows (which tells the
30 -> 20 fps row rules apart). Aggregates only, no data. The tolerance absorbs the known library-version effects
(SciPy's rotation conversions, NumPy's float32 smoothing, the MANO fit on CPU or GPU).

    python prepare/check.py --env robot_manipulation --root <root>
"""

import argparse
import json
from pathlib import Path

import numpy as np

FINGERPRINTS = Path(__file__).with_name("fingerprints.json")
TOLERANCE = 1e-4  # absolute, and relative to the value's magnitude above 1


def fingerprint(root: Path) -> dict:
    """The test split's counts and per-dimension action statistics (float64)."""
    words = (root / "splits" / "test.txt").read_text().split()
    actions = [np.load(root / "actions" / f"{key}.npy").astype(np.float64) for key in words[::2]]
    rows = np.concatenate(actions)
    even, odd = (np.concatenate([a[parity::2] for a in actions]) for parity in (0, 1))
    stats = {"mean": rows.mean(0), "std": rows.std(0), "min": rows.min(0), "max": rows.max(0)}
    stats |= {"std_even": even.std(0), "std_odd": odd.std(0)}
    counts = {"episodes": len(actions), "frames": sum(map(int, words[1::2]))}
    return counts | {name: value.tolist() for name, value in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, choices=["robot_manipulation", "driving", "indoor_navigation",
                                                          "human_manipulation"])  # fmt: skip
    parser.add_argument("--root", type=Path, required=True, help="the dataset root prepare/<env>.py wrote")
    args = parser.parse_args()
    reference, ours = json.loads(FINGERPRINTS.read_text())[args.env], fingerprint(args.root)
    ok = True
    for name in ("episodes", "frames"):
        print(f"{name}: {ours[name]} (ours {reference[name]})")
        ok &= ours[name] == reference[name]
    for name in ("mean", "std", "min", "max", "std_even", "std_odd"):
        want, got = np.array(reference[name]), np.array(ours[name])
        excess = np.abs(got - want) / (TOLERANCE * np.maximum(1, np.abs(want)))
        worst = int(excess.argmax())
        print(f"{name}: largest deviation {abs(got[worst] - want[worst]):.2e} (dimension {worst}), "
              f"{excess[worst]:.2f} x the tolerance")  # fmt: skip
        ok &= bool(excess.max() <= 1)
    print("MATCH: this test split is ours" if ok else "MISMATCH: see above (was the whole test split converted?)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
