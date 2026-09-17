#!/usr/bin/env bash
# Install the DJI RoboMaster SDK on macOS (Apple Silicon or Intel).
#
# DJI only publishes Linux/Windows wheels, so `pip install robomaster` fails on
# a Mac. The SDK is pure Python except for one native video decoder
# (libmedia_codec). This script:
#   1. creates .venv in the repo with a Python 3.8-3.12 interpreter
#   2. installs the SDK's dependencies (all have native macOS wheels)
#   3. extracts the SDK's Python code from the official wheel
#   4. installs tools/macos/libmedia_codec.py (PyAV based) as the decoder
#   5. runs a self-test
#
# Usage:  bash tools/macos/setup_robomaster_mac.sh [path/to/python3]
set -euo pipefail

SDK_VERSION="0.1.1.68"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"
SHIM="$REPO_DIR/tools/macos/libmedia_codec.py"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "This script is for macOS. On Linux/Windows run: pip install -r requirements.txt"
  exit 1
fi

pick_python() {
  if [[ $# -gt 0 && -n "$1" ]]; then echo "$1"; return; fi
  for candidate in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; sys.exit(0 if (3, 8) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
        command -v "$candidate"; return
      fi
    fi
  done
  return 1
}

PYTHON="$(pick_python "${1:-}")" || {
  echo "Need Python 3.8-3.12 (the SDK uses 'audioop', removed in 3.13)."
  echo "Install one from https://www.python.org/downloads/macos/ and re-run."
  exit 1
}
echo "==> Using $PYTHON ($("$PYTHON" --version 2>&1), $(uname -m))"

if [[ -d "$VENV_DIR" ]] && ! "$VENV_DIR/bin/python" -c 'import sys' >/dev/null 2>&1; then
  echo "==> Existing .venv is broken, recreating"
  rm -rf "$VENV_DIR"
fi
[[ -d "$VENV_DIR" ]] || "$PYTHON" -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"

echo "==> Upgrading pip (old pip does not recognise recent macOS wheels)"
"$PY" -m pip install --quiet --upgrade pip setuptools wheel

echo "==> Installing dependencies"
"$PY" -m pip install --quiet --only-binary=:all: \
  "numpy>=1.18,<2" "opencv-python>=4.5" "PyYAML>=6.0" "netaddr>=0.8" "av>=10" "pynput>=1.7"
# netifaces has no macOS wheels; netifaces-plus is the maintained fork with the same module name.
"$PY" -m pip install --quiet --only-binary=:all: netifaces-plus \
  || "$PY" -m pip install --quiet netifaces

echo "==> Fetching RoboMaster SDK $SDK_VERSION (pure Python part)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
"$PY" -m pip download --quiet --no-deps --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 38 --implementation cp --abi cp38 \
  -d "$TMP_DIR" "robomaster==$SDK_VERSION"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$PY" - "$TMP_DIR" "$SITE" <<'PYEOF'
import glob, os, shutil, sys, zipfile
tmp, site = sys.argv[1], sys.argv[2]
wheel = glob.glob(os.path.join(tmp, "robomaster-*.whl"))[0]
for pkg in ("robomaster", "multi_robomaster"):
    shutil.rmtree(os.path.join(site, pkg), ignore_errors=True)
with zipfile.ZipFile(wheel) as zf:
    for name in zf.namelist():
        if name.split("/")[0] in ("robomaster", "multi_robomaster") and name.endswith(".py"):
            zf.extract(name, site)
print("    extracted SDK into", site)

# numpy.fromstring (binary mode) is deprecated; frombuffer + copy is equivalent and
# keeps frames writable for OpenCV drawing.
media = os.path.join(site, "robomaster", "media.py")
src = open(media, encoding="utf-8").read()
old = "numpy.fromstring(frame, dtype=numpy.ubyte, count=len(frame), sep='')"
if old in src:
    src = src.replace(old, "numpy.frombuffer(frame, dtype=numpy.ubyte, count=len(frame)).copy()")
    open(media, "w", encoding="utf-8").write(src)
    print("    patched robomaster/media.py (numpy.fromstring -> frombuffer)")

# DJI bug: constants are compared with `is` (e.g. `conn_type is CONNECTION_WIFI_STA`).
# That only works when both strings happen to be the same object, so values that
# come from argparse/YAML fail -> "local variable 'proxy_addr' referenced before
# assignment", LED effects / vision names silently ignored. Rewrite `is CONST` to
# `== CONST` (and `is not CONST` to `!= CONST`) using the tokenizer, so strings and
# comments are never touched.
import io, re, tokenize
const = re.compile(r"^[A-Z][A-Z0-9_]+$")
fixed_total = 0
for pkg in ("robomaster", "multi_robomaster"):
    for path in glob.glob(os.path.join(site, pkg, "*.py")):
        text = open(path, encoding="utf-8").read()
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        edits = []  # (row, col_start, col_end, replacement)
        for i, tok in enumerate(toks):
            if tok.type != tokenize.NAME or tok.string != "is":
                continue
            j, negate = i + 1, False
            if toks[j].type == tokenize.NAME and toks[j].string == "not":
                j, negate = j + 1, True
            target = toks[j]
            if target.type == tokenize.NAME and const.match(target.string) and target.start[0] == tok.start[0]:
                edits.append((tok.start[0], tok.start[1], target.start[1], "!= " if negate else "== "))
        if not edits:
            continue
        lines = text.splitlines(True)
        for row, c0, c1, rep in sorted(edits, reverse=True):
            line = lines[row - 1]
            lines[row - 1] = line[:c0] + rep + line[c1:]
        open(path, "w", encoding="utf-8").write("".join(lines))
        fixed_total += len(edits)
print("    fixed %d `is CONSTANT` comparisons (SDK proxy_addr / LED / vision bugs)" % fixed_total)

# DJI bugs hit when a connection fails: `raise print(...)` ("exceptions must derive
# from BaseException"), stop() on a never-started client (AttributeError spam on
# exit) and a config name that does not exist.
client_py = os.path.join(site, "robomaster", "client.py")
src = open(client_py, encoding="utf-8").read()
for old, new in [
    ("raise print('Robot: Can not connect to robot, check connection please.')",
     "raise ConnectionError('Robot: Can not connect to robot, check connection please.')"),
    ("        if self._thread.is_alive():",
     "        if self._thread is not None and self._thread.is_alive():"),
    ("protocol=config.DEFAULT_CONN_PROTO)", "protocol=config.DEFAULT_PROTO_TYPE)"),
]:
    if old in src:
        src = src.replace(old, new)
open(client_py, "w", encoding="utf-8").write(src)
print("    fixed SDK connection-failure crashes in robomaster/client.py")
PYEOF
cp "$SHIM" "$SITE/libmedia_codec.py"
echo "    installed macOS libmedia_codec decoder"

echo "==> Self-test"
"$PY" "$REPO_DIR/tools/macos/check_robomaster.py" --offline

cat <<EOF

Done. Next:
  1. Turn on the robot, set the connection switch to Wi-Fi direct (AP) mode.
  2. On the Mac, join the robot's Wi-Fi (RMEP-xxxxxx; the password is on the sticker).
  3. Check it:   .venv/bin/python tools/macos/check_robomaster.py
  4. Run:        .venv/bin/python src/gimbal_shooter.py
EOF
