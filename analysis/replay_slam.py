"""Rebuild the map of a recorded SLAM run with the current settings.

    python analysis/replay_slam.py data/slam/run_20260923_103629
    python analysis/replay_slam.py data/slam/run_... --set grid_mode=fixed grid_cell_m=0.6

Writes replay_map_raw.png, replay_map_clean.png and replay_map_grid.png into
<run>/replay/ (or --out). No robot needed.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slam import params as params_mod  # noqa: E402
from slam.replay import save_replay  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    params = params_mod.load(dict(item.partition("=")[::2] for item in args.set))
    out = save_replay(args.run_dir, args.out or os.path.join(args.run_dir, "replay"), params)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
