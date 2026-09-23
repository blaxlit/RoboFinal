"""Class Work 8: SLAM - explore an unknown area with the RoboMaster EP.

    python src/slam_explore.py --sim                 # simulator, opens the console
    python src/slam_explore.py                       # real robot (Wi-Fi AP mode)
    python src/slam_explore.py --gt data/slam/ground_truth_example.json
    python src/slam_explore.py --sim --auto --headless   # run a mission and exit

The console (http://localhost:8765) shows the live map, pose, sensor data and
logs, and has every setting and command. Results land in data/slam/run_*/.
"""

import argparse
import os
import sys
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(BASE_DIR, ".venv", "bin", "python")

try:
    import cv2, numpy, yaml  # noqa: E401,F401
except ImportError as exc:
    # Started with a Python that lacks the packages (e.g. the system python3):
    # rerun with the project's virtual environment if there is one.
    if os.path.exists(VENV_PYTHON) and os.path.realpath(sys.prefix) != os.path.realpath(os.path.join(BASE_DIR, ".venv")):
        os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)
    sys.exit(f"Missing package ({exc.name}). Install them with: python3 -m pip install -r requirements.txt")

from slam import params as params_mod  # noqa: E402
from slam.evaluation import load_ground_truth  # noqa: E402
from slam.explorer import Explorer  # noqa: E402

EXAMPLE_GT = os.path.join(BASE_DIR, "data", "slam", "ground_truth_example.json")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="RoboMaster SLAM explorer with a live web console.")
    ap.add_argument("--sim", action="store_true", help="use the simulator instead of the robot")
    ap.add_argument("--connection", default="ap", choices=["ap", "sta", "rndis"], help="robot connection type")
    ap.add_argument("--gt", help="ground-truth map JSON for accuracy (the simulator uses it as its world)")
    ap.add_argument("--port", type=int, default=8765, help="console port")
    ap.add_argument("--host", default="127.0.0.1", help="console address (0.0.0.0 to open it from a phone)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    ap.add_argument("--headless", action="store_true", help="no console; use with --auto")
    ap.add_argument("--auto", action="store_true", help="start the mission immediately")
    ap.add_argument("--time-scale", type=float, default=4.0, help="simulator speed-up")
    ap.add_argument("--seed", type=int, default=1, help="simulator random seed")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "data", "slam"), help="results folder")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override settings, e.g. scan_mode=step")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extra = {}
    for item in args.set:
        key, _, value = item.partition("=")
        extra[key] = value
    try:
        params = params_mod.load(extra)
    except ValueError as exc:
        print(f"Settings error: {exc}", file=sys.stderr)
        return 2

    gt_path = args.gt or (EXAMPLE_GT if args.sim else None)
    gt = load_ground_truth(gt_path) if gt_path else None

    if args.sim:
        from slam.io_sim import SimRobotIO
        io = SimRobotIO(gt, start=(params["gt_start_x"], params["gt_start_y"],
                                   __import__("math").radians(params["gt_start_deg"])),
                        time_scale=args.time_scale, seed=args.seed)
    else:
        from slam.io_real import RealRobotIO
        io = RealRobotIO(args.connection)

    explorer = Explorer(io, params, args.out, gt)
    server = None
    try:
        if not args.headless:
            from slam.console import serve
            server, url = serve(explorer, args.host, args.port)
            print(f"SLAM console: {url}  (Ctrl+C to quit)")
            if not args.no_browser:
                webbrowser.open(url)
        if args.auto:
            explorer.command("start")
        while True:
            time.sleep(0.5)
            if args.headless and args.auto and explorer.state == "DONE" and explorer.cmd_q.empty():
                break
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if explorer.state not in ("IDLE", "DONE"):
            explorer.command("stop")
            time.sleep(0.5)
        if explorer.last_report is None and explorer.stats["scans"]:
            explorer.save_outputs()
        explorer.shutdown()
        if server:
            server.shutdown()
        io.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
