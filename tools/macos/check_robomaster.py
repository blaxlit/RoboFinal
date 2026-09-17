"""Check that this Mac can talk to a RoboMaster EP.

    .venv/bin/python tools/macos/check_robomaster.py --offline   # install self-test only
    .venv/bin/python tools/macos/check_robomaster.py             # + connect to the robot (AP mode)
    .venv/bin/python tools/macos/check_robomaster.py --connection sta --snapshot frame.jpg

Each step prints OK/FAIL with a hint, so it is clear *where* a connection breaks.
"""

import argparse
import importlib
import socket
import sys
import threading
import time

ROBOT_AP_IP = "192.168.2.1"
results = []


def report(ok, name, detail="", hint=""):
    results.append(ok)
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok and hint:
        print(f"       hint: {hint}")
    return ok


def check_imports():
    ok = report((3, 8) <= sys.version_info[:2] <= (3, 12), "Python version", sys.version.split()[0],
                "use Python 3.8-3.12 (the SDK imports audioop, removed in 3.13)")
    for module, hint in [
        ("numpy", "run tools/macos/setup_robomaster_mac.sh"),
        ("cv2", "pip install opencv-python"),
        ("yaml", "pip install PyYAML"),
        ("av", "pip install av"),
        ("netaddr", "pip install netaddr"),
        ("netifaces", "pip install netifaces-plus"),
        ("libmedia_codec", "run tools/macos/setup_robomaster_mac.sh (installs the macOS decoder)"),
        ("robomaster", "run tools/macos/setup_robomaster_mac.sh"),
    ]:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "")
            ok &= report(True, f"import {module}", str(version))
        except Exception as error:
            ok &= report(False, f"import {module}", str(error), hint)
    try:
        importlib.import_module("pynput")
        report(True, "import pynput (optional, multi-key WASD)")
    except Exception as error:
        print(f"[warn] pynput not usable ({error}); gimbal_shooter falls back to OpenCV window keys")
    return ok


def check_decoder():
    """Encode a synthetic H.264 clip and decode it in random-sized chunks like the robot stream."""
    import random

    import av
    import numpy as np
    import libmedia_codec

    width, height, n_frames = 320, 180, 20
    buf = bytearray()
    encoder = None
    for name in ("libx264", "h264_videotoolbox", "libopenh264", "h264"):
        try:
            encoder = av.CodecContext.create(name, "w")
            encoder.width, encoder.height, encoder.pix_fmt = width, height, "yuv420p"
            encoder.time_base = av.utils.Fraction(1, 30) if hasattr(av, "utils") else None
            encoder.framerate = 30
            encoder.open()
            break
        except Exception:
            encoder = None
    if encoder is None:
        print("[warn] no H.264 encoder available to self-test the decoder; skipping")
        return True

    for i in range(n_frames):
        image = np.zeros((height, width, 3), np.uint8)
        image[:, :] = (255, 0, 0)          # BGR blue background
        image[40:140, 40 + i * 5:140 + i * 5] = (0, 0, 255)  # moving red square
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = i
        for packet in encoder.encode(frame):
            buf += bytes(packet)
    for packet in encoder.encode(None):
        buf += bytes(packet)

    decoder = libmedia_codec.H264Decoder()
    rng = random.Random(1)
    decoded, pos = [], 0
    while pos < len(buf):
        size = rng.randint(200, 3000)
        decoded += decoder.decode(bytes(buf[pos:pos + size]))
        pos += size
    decoded += decoder.decode(b"\x00\x00\x00\x01\x09\x10")  # access unit delimiter flushes the last frame

    if not decoded:
        return report(False, "H.264 decoder", "no frames decoded")
    frame_bytes, w, h, _ = decoded[0]
    image = np.frombuffer(frame_bytes, np.uint8).reshape((h, w, 3))
    bg = image[10, 10].astype(int)
    square = image[90, 90].astype(int)
    colors_ok = bg[0] > 180 and bg[2] < 80 and square[2] > 180 and square[0] < 80
    return report(len(decoded) >= n_frames - 2 and colors_ok, "H.264 decoder (macOS libmedia_codec)",
                  f"{len(decoded)}/{n_frames} frames, {w}x{h}, BGR order {'correct' if colors_ok else 'WRONG'}")


def local_ip_towards(ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((ip, 9))  # no packet is sent for UDP connect
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def run_with_timeout(fn, timeout):
    box = {}

    def target():
        try:
            box["value"] = fn()
        except Exception as error:  # noqa: BLE001
            box["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"no answer within {timeout:.0f} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def check_robot(connection, snapshot, wiggle, robot_ip=None):
    from robomaster import config, robot

    if robot_ip:
        config.ROBOT_IP_STR = robot_ip   # skip broadcast discovery (often blocked on campus Wi-Fi)
        print(f"[info] using robot IP {robot_ip}")

    if connection == "ap":
        local_ip = local_ip_towards(ROBOT_AP_IP)
        on_robot_wifi = bool(local_ip and local_ip.startswith("192.168.2."))
        if not report(on_robot_wifi, "Mac is on the robot's Wi-Fi", f"local IP {local_ip}",
                      "join the robot hotspot (RMEP-xxxxxx, password on the robot sticker); the robot's "
                      "connection switch must be on the Wi-Fi direct (AP) position"):
            return False

    ep = robot.Robot()
    try:
        kwargs = {"conn_type": connection}
        if connection == "ap":
            kwargs["proto_type"] = "udp"
        try:
            run_with_timeout(lambda: ep.initialize(**kwargs), 20)
            ok = report(True, "SDK connection", f"conn_type={connection}")
        except Exception as error:
            hint = ("power-cycle the robot, close the RoboMaster app on phones/PCs, "
                    "and make sure no other script is connected")
            if connection == "sta" and not robot_ip:
                hint = ("the robot was not found on this network. Campus Wi-Fi usually blocks the robot's "
                        "discovery broadcast: find the robot's IP (RoboMaster app or router list) and re-run "
                        "with --robot-ip <ip>, or use AP mode (join the robot's RMEP-xxxxxx Wi-Fi)")
            return report(False, "SDK connection", str(error), hint)

        def read_battery():
            # The EP has no get_battery(); its level only arrives via subscription.
            box = []
            ep.battery.sub_battery_info(freq=5, callback=lambda percent: box.append(percent))
            try:
                t0 = time.time()
                while not box and time.time() - t0 < 3:
                    time.sleep(0.05)
            finally:
                ep.battery.unsub_battery_info()
            return f"{box[0]}%" if box else None

        for name, fn in [("robot version", ep.get_version), ("serial number", ep.get_sn),
                         ("battery", read_battery)]:
            try:
                value = run_with_timeout(fn, 5)
                ok &= report(value is not None, name, str(value))
            except Exception as error:
                ok &= report(False, name, str(error))

        try:
            ep.camera.start_video_stream(display=False, resolution="720p")
            frames, first, t0 = 0, None, time.time()
            while time.time() - t0 < 4:
                try:
                    img = ep.camera.read_cv2_image(strategy="newest", timeout=1.0)
                except Exception:
                    continue
                frames += 1
                first = img if first is None else first
                if frames >= 60:
                    break
            elapsed = time.time() - t0
            if first is not None:
                detail = f"{frames} frames in {elapsed:.1f} s ({frames / elapsed:.0f} fps), {first.shape[1]}x{first.shape[0]}"
                ok &= report(frames >= 10, "camera video stream", detail)
                if snapshot:
                    import cv2
                    cv2.imwrite(snapshot, first)
                    print(f"       saved {snapshot}")
            else:
                ok &= report(False, "camera video stream", "no frames in 4 s",
                             "check that nothing else is using the stream; try resolution 360p")
        finally:
            try:
                ep.camera.stop_video_stream()
            except Exception:
                pass

        if wiggle:
            ep.gimbal.move(yaw=10, yaw_speed=60).wait_for_completed(timeout=3)
            ep.gimbal.move(yaw=-10, yaw_speed=60).wait_for_completed(timeout=3)
            report(True, "gimbal moved +/-10 deg")
        return ok
    finally:
        try:
            ep.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="only check the installation")
    parser.add_argument("--connection", choices=("ap", "sta", "rndis"), default="ap")
    parser.add_argument("--robot-ip", help="robot IP for sta mode (skips network discovery)")
    parser.add_argument("--snapshot", help="save one camera frame to this path")
    parser.add_argument("--wiggle", action="store_true", help="move the gimbal +/-10 deg as a final test")
    args = parser.parse_args()

    if check_imports():
        check_decoder()
    if not args.offline and all(results):
        check_robot(args.connection, args.snapshot, args.wiggle, args.robot_ip)

    print()
    if all(results):
        print("All checks passed." if not args.offline else "Installation OK (robot not contacted).")
        return 0
    print("Some checks failed - see hints above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
