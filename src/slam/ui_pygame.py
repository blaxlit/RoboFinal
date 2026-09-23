"""pygame console: live map, pose, sensor data, logs, settings and commands.

Runs in the main thread (required by pygame on macOS) and talks to the
Explorer directly. Layout: header with mission buttons, map (left), log
(bottom left), tabbed side panel (right): Status, Control, Settings, Ground
truth, Results.

Keys: Space = STOP, W/A/S/D drive/strafe and Q/E rotate (while no text field is
being edited and keyboard drive is on), F = fit map, +/- = zoom, R = follow robot.
"""

import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pygame

from . import params as params_mod
from .evaluation import load_ground_truth, validate_ground_truth

DEG = math.pi / 180

PALETTES = {
    "dark": dict(
        bg=(14, 17, 22), panel=(22, 26, 33), panel2=(27, 32, 41), line=(42, 48, 59), text=(230, 233, 238),
        muted=(144, 153, 166), accent=(91, 140, 255), accent2=(34, 184, 207), ok=(60, 207, 122),
        warn=(245, 165, 36), bad=(255, 93, 93), violet=(167, 139, 250), traj=(255, 138, 61),
        unknown=(29, 34, 42), free=(49, 58, 71), wall=(238, 241, 245), grid=(40, 46, 56),
        btn=(34, 40, 52), hover=(43, 51, 65), white=(255, 255, 255)),
    "light": dict(
        bg=(243, 244, 246), panel=(255, 255, 255), panel2=(248, 249, 251), line=(223, 226, 231), text=(22, 25, 29),
        muted=(100, 107, 117), accent=(37, 99, 235), accent2=(8, 145, 178), ok=(22, 163, 74),
        warn=(217, 119, 6), bad=(220, 38, 38), violet=(124, 58, 237), traj=(234, 88, 12),
        unknown=(201, 204, 210), free=(255, 255, 255), wall=(28, 32, 38), grid=(214, 217, 222),
        btn=(238, 240, 243), hover=(226, 229, 234), white=(255, 255, 255)),
}

RUNNING_STATES = ("RUNNING", "SCANNING", "MOVING", "PLANNING", "CALIBRATING", "SAVING")
TABS = ["Status", "Control", "Settings", "Ground truth", "Results"]
MODES = [("pan", "Pan"), ("goto", "Go to"), ("pose", "Set pose"), ("border", "Border"),
         ("wall", "GT wall"), ("gtborder", "GT arena")]
MODE_HINTS = {"goto": "Click where the robot should drive",
              "pose": "Drag from the robot's real position toward where it faces",
              "border": "Drag a rectangle: exploration and scoring stay inside it",
              "wall": "Drag along a real wall (arena frame, snapped)",
              "gtborder": "Drag the arena's outer rectangle"}
LAYERS = [("clean", "Clean"), ("maze", "Maze"), ("prob", "Prob"), ("traj", "Path"), ("scan", "Scan"),
          ("plan", "Plan"), ("gt", "GT"), ("grid", "Lines"), ("truth", "Truth")]
CHARTS = [("tof", "ToF"), ("cov", "Coverage"), ("speed", "Speed"), ("match", "Drift fix"), ("loc_error", "Loc err")]
SNAPS = [0.01, 0.05, 0.1, 0.3, 0.6]


def f2(v, d=2):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def fpose(p):
    return f"{f2(p['x'])}, {f2(p['y'])} m  {f2(p['deg'], 1)}°" if p else "–"


# =============================================================================
# immediate-mode widgets
# =============================================================================
class UI:
    def __init__(self):
        pygame.font.init()
        mono = pygame.font.match_font("menlo,consolas,dejavusansmono,couriernew,monospace")
        sans = pygame.font.match_font("helveticaneue,helvetica,segoeui,arial,dejavusans")
        self.fonts = {
            "s": pygame.font.Font(sans, 13), "sb": pygame.font.Font(sans, 13), "xs": pygame.font.Font(sans, 11),
            "h": pygame.font.Font(sans, 15), "big": pygame.font.Font(sans, 26), "m": pygame.font.Font(mono, 12),
        }
        self.fonts["sb"].set_bold(True)
        self.fonts["h"].set_bold(True)
        self.fonts["big"].set_bold(True)
        self.theme = "dark"
        self.pal = PALETTES[self.theme]
        self._cache = {}
        self.focus = None          # key of the text field being edited
        self.buf = ""
        self.drag_key = None       # slider being dragged
        self.clip = None
        self.screen = None

    def set_theme(self, name):
        self.theme = name
        self.pal = PALETTES[name]
        self._cache.clear()

    def begin(self, screen, events):
        self.screen = screen
        self.mouse = pygame.mouse.get_pos()
        self.down = any(e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 for e in events)
        self.clicked = any(e.type == pygame.MOUSEBUTTONUP and e.button == 1 for e in events)
        self.held = pygame.mouse.get_pressed()[0]
        self.text_in = "".join(e.text for e in events if e.type == pygame.TEXTINPUT)
        self.keys = [e.key for e in events if e.type == pygame.KEYDOWN]
        self.clip = None
        self.cursor = pygame.SYSTEM_CURSOR_ARROW
        self.seen = set()          # fields drawn this frame
        if not self.held:
            self.drag_key = None

    def text(self, s, font="s", color="text"):
        key = (s, font, color if isinstance(color, str) else tuple(color), self.theme)
        surf = self._cache.get(key)
        if surf is None:
            col = self.pal[color] if isinstance(color, str) else color
            surf = self.fonts[font].render(str(s), True, col)
            if len(self._cache) > 3000:
                self._cache.clear()
            self._cache[key] = surf
        return surf

    def label(self, s, x, y, font="s", color="text", align="left", maxw=None):
        surf = self.text(s, font, color)
        if maxw and surf.get_width() > maxw:
            while s and self.fonts[font].size(s + "…")[0] > maxw:
                s = s[:-1]
            surf = self.text(s + "…", font, color)
        if align == "right":
            x -= surf.get_width()
        elif align == "center":
            x -= surf.get_width() // 2
        self.screen.blit(surf, (x, y))
        return surf.get_width()

    def wrap(self, s, font, width):
        key = ("wrap", s, font, width)
        if key in self._cache:
            return self._cache[key]
        f = self.fonts[font]
        lines = []
        for para in str(s).split("\n"):
            cur = ""
            for word in para.split(" "):
                nxt = (cur + " " + word) if cur else word
                if f.size(nxt)[0] <= width:
                    cur = nxt
                else:
                    if cur:
                        lines.append(cur)
                    while f.size(word)[0] > width and len(word) > 1:  # very long word (paths)
                        n = len(word)
                        while n > 1 and f.size(word[:n])[0] > width:
                            n -= 1
                        lines.append(word[:n])
                        word = word[n:]
                    cur = word
            lines.append(cur)
        self._cache[key] = lines
        return lines

    def hit(self, rect):
        return rect.collidepoint(self.mouse) and (self.clip is None or self.clip.collidepoint(self.mouse))

    def rrect(self, rect, color, radius=6, width=0):
        pygame.draw.rect(self.screen, self.pal[color] if isinstance(color, str) else color, rect, width,
                         border_radius=radius)

    def button(self, rect, label, kind="normal", on=False, enabled=True, font="s"):
        hover = enabled and self.hit(rect)
        if kind == "primary":
            bg, fg, border = ("accent", "white", "accent")
        elif kind == "danger":
            bg, fg, border = ("bad", "white", "bad")
        elif on:
            bg, fg, border = ("accent", "white", "accent")
        else:
            bg, fg, border = ("hover" if hover else "btn", "text", "line")
        if not enabled:
            fg = "muted"
        self.rrect(rect, bg, 7)
        if kind == "normal" and not on:
            self.rrect(rect, border, 7, 1)
        elif hover:
            self.rrect(rect, "white", 7, 1)
        surf = self.text(label, "sb" if kind != "normal" else font, fg)
        self.screen.blit(surf, surf.get_rect(center=rect.center))
        if hover:
            self.cursor = pygame.SYSTEM_CURSOR_HAND
        return enabled and hover and self.clicked

    def checkbox(self, rect, value, label=None):
        box = pygame.Rect(rect.x, rect.y + (rect.h - 16) // 2, 16, 16)
        self.rrect(box, "accent" if value else "panel2", 4)
        self.rrect(box, "accent" if value else "line", 4, 1)
        if value:
            pygame.draw.lines(self.screen, self.pal["white"], False,
                              [(box.x + 3, box.y + 8), (box.x + 7, box.y + 12), (box.x + 13, box.y + 4)], 2)
        if label:
            self.label(label, box.right + 6, rect.y + (rect.h - 15) // 2, "s", "muted")
        if self.hit(rect) and self.clicked:
            return not value
        return value

    def field(self, key, rect, value, numeric=True, font="m"):
        """Text/number box. Returns the new value when editing ends, else None."""
        self.seen.add(key)
        active = self.focus == key
        hover = self.hit(rect)
        self.rrect(rect, "panel2", 6)
        self.rrect(rect, "accent" if active else ("muted" if hover else "line"), 6, 1)
        if hover:
            self.cursor = pygame.SYSTEM_CURSOR_IBEAM
        shown = self.buf if active else ("" if value is None else (_fmt(value) if numeric else str(value)))
        old_clip = self.screen.get_clip()
        inner = rect.inflate(-10, 0)
        self.screen.set_clip(inner.clip(self.clip) if self.clip else inner)
        surf = self.text(shown, font)
        tx = rect.x + 6 if surf.get_width() < rect.w - 12 else rect.right - 6 - surf.get_width()
        self.screen.blit(surf, (tx, rect.y + (rect.h - surf.get_height()) // 2))
        if active and (time.time() * 2) % 2 < 1.3:
            cx = tx + surf.get_width() + 1
            pygame.draw.line(self.screen, self.pal["text"], (cx, rect.y + 5), (cx, rect.bottom - 5))
        self.screen.set_clip(old_clip)

        if not active:
            if hover and self.clicked:
                self.focus, self.buf = key, shown
            return None
        # editing
        commit = cancel = False
        for k in self.keys:
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
                commit = True
            elif k == pygame.K_ESCAPE:
                cancel = True
            elif k == pygame.K_BACKSPACE:
                self.buf = self.buf[:-1]
        allowed = "0123456789.-+e" if numeric else None
        self.buf += "".join(c for c in self.text_in if allowed is None or c in allowed)
        if self.clicked and not hover:
            commit = True
        if cancel:
            self.focus = None
            return None
        if commit:
            self.focus = None
            if numeric:
                try:
                    return float(self.buf)
                except ValueError:
                    return None
            return self.buf
        return None

    def slider(self, key, rect, value, lo, hi):
        track = pygame.Rect(rect.x, rect.centery - 2, rect.w, 4)
        self.rrect(track, "line", 2)
        t = (value - lo) / (hi - lo) if hi > lo else 0
        fill = pygame.Rect(track.x, track.y, int(track.w * t), 4)
        self.rrect(fill, "accent", 2)
        knob = (int(track.x + track.w * t), rect.centery)
        pygame.draw.circle(self.screen, self.pal["accent"], knob, 7)
        pygame.draw.circle(self.screen, self.pal["white"], knob, 7, 2)
        if self.hit(rect.inflate(0, 8)) and self.down:
            self.drag_key = key
        if self.drag_key == key:
            t = min(1.0, max(0.0, (self.mouse[0] - track.x) / max(track.w, 1)))
            return lo + t * (hi - lo)
        return value


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return s if s not in ("", "-0") else "0"
    return str(v)


# =============================================================================
# the console
# =============================================================================
class Console:
    HEADER = 50
    SIDE = 410
    LOG_H = 190

    def __init__(self, explorer, gt_dir):
        self.ex = explorer
        self.gt_dir = gt_dir
        pygame.init()
        pygame.display.set_caption("RoboMaster SLAM Console")
        info = pygame.display.Info()
        w = min(1480, max(1000, info.current_w - 80))
        h = min(900, max(700, info.current_h - 120))
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.key.start_text_input()
        self.ui = UI()
        self.clock = pygame.time.Clock()
        self.running = True
        # data from the explorer
        self.snap = None
        self.map = None            # dict rows, cols, res, ox, oy, cls, prob
        self.map_version = -1
        self.traj = []
        self.traj_len = 0
        self.events = []
        self.ev_id = 0
        self.gt_version = 0
        self.gt_server = None
        self.param_version = 0
        self.last_poll = 0.0
        # view state
        self.view = {"cx": 0.0, "cy": 0.0, "scale": 90.0, "rot": 0.0}
        self.fitted = False
        self.follow = False
        self.mode = "pan"
        self.layers = {"clean": True, "maze": True, "prob": False, "traj": True, "scan": True, "plan": True,
                       "gt": True, "grid": True, "truth": True}
        self.grid_seen = -1
        self.drag = None
        self._map_cache = (None, None)
        # panel state
        self.tab = "Status"
        self.scroll = {t: 0 for t in TABS}
        self.content_h = {t: 0 for t in TABS}
        self.log_scroll = 0          # lines from the bottom
        self.log_debug = False
        self.chart = "tof"
        self.groups_open = {"Map": True, "Scan": True}
        self.set_filter = ""
        self.inputs = {"turn": 30.0, "move": 0.3, "gx": 0.0, "gy": 0.0, "gim": 90.0, "wall": 0.5,
                       "spx": 0.0, "spy": 0.0, "spt": 0.0, "speed": 0.25, "turn_rate": 45.0}
        self.kbd_drive = True
        self.driving = False
        self.last_drive = 0.0
        self.draft = {"name": "arena", "border": None, "walls": [], "wall_thickness": 0.02}
        self.gt_path = os.path.join(gt_dir, "ground_truth_example.json")
        self.snap_i = 1
        self.files = []
        self.files_t = 0.0
        self.thumb = None
        self.thumb_mtime = None
        self.toast_msg = None
        self.toast_until = 0.0
        self.toast_ok = False
        self.confirm = None       # (message, callback)

    # ------------------------------------------------------------------ helpers
    @property
    def p(self):
        return self.ex.p

    def toast(self, msg, ok=False):
        self.toast_msg, self.toast_until, self.toast_ok = msg, time.time() + 4.0, ok

    def cmd(self, name, **args):
        try:
            self.ex.command(name, args)
        except Exception as exc:
            self.toast(str(exc))

    def set_params(self, **changes):
        try:
            self.ex.set_params(changes)
        except Exception as exc:
            self.toast(str(exc))

    def ask(self, message, callback):
        self.confirm = (message, callback)

    # ------------------------------------------------------------------ data
    def poll(self):
        ex = self.ex
        snap = ex.snapshot(map_version=ex.grid.version, event_id=self.ev_id, gt_version=self.gt_version,
                           param_version=self.param_version, traj_from=self.traj_len)
        self.snap = snap
        g = ex.grid
        if ex.grid_version != self.grid_seen and self.map is not None and self.map.get("grid") is g:
            self.grid_seen = ex.grid_version
            with ex.lock:
                self.map["clean"] = None if ex.snapped is None else ex.snapped.copy()
            self._map_cache = (None, None)
        if g.version != self.map_version or self.map is None or self.map.get("grid") is not g:
            with ex.lock:
                cls = ex.classes()
                prob = g.probability()
                version = g.version
            with ex.lock:
                clean = None if ex.snapped is None else ex.snapped.copy()
            self.map = {"grid": g, "rows": g.rows, "cols": g.cols, "res": g.res, "ox": g.origin_x, "oy": g.origin_y,
                        "cls": cls, "prob": prob, "clean": clean}
            self.grid_seen = ex.grid_version
            self.map_version = version
            self._map_cache = (None, None)
        tr = snap["traj"]
        if tr["from"] == 0:
            self.traj = []
        self.traj.extend(tr["pts"])
        self.traj_len = tr["len"]
        if "params" in snap:
            self.param_version = snap["param_version"]
            self.view["rot"] = float(self.p["map_rotation_deg"])
        if "gt" in snap:
            self.gt_server = snap["gt"]
            self.gt_version = snap["gt_version"]
            if snap["gt"] and not self.draft["walls"] and not self.draft["border"]:
                self.load_draft(snap["gt"]["raw"])
        for e in snap["events"]:
            self.events.append(e)
            self.ev_id = e[0]
        if len(self.events) > 2000:
            self.events = self.events[-1500:]
        if not self.fitted and self.map_rect().w > 50:
            self.fit()
            self.fitted = True
        if self.follow:
            self.view["cx"], self.view["cy"] = snap["pose"]["x"], snap["pose"]["y"]

    def load_draft(self, gt):
        self.draft = {"name": gt.get("name", "arena"), "border": list(gt["border"]) if gt.get("border") else None,
                      "walls": [list(w) for w in gt.get("walls", [])],
                      "wall_thickness": gt.get("wall_thickness", 0.02)}

    # ------------------------------------------------------------------ transforms
    def map_rect(self):
        w, h = self.screen.get_size()
        return pygame.Rect(10, self.HEADER + 10, w - self.SIDE - 30, h - self.HEADER - self.LOG_H - 30)

    def w2s(self, x, y, rect=None):
        rect = rect or self.map_rect()
        v = self.view
        r = v["rot"] * DEG
        dx, dy = x - v["cx"], y - v["cy"]
        rx = math.cos(r) * dx - math.sin(r) * dy
        ry = math.sin(r) * dx + math.cos(r) * dy
        return rect.centerx + rx * v["scale"], rect.centery - ry * v["scale"]

    def s2w(self, px, py, rect=None):
        rect = rect or self.map_rect()
        v = self.view
        r = -v["rot"] * DEG
        rx = (px - rect.centerx) / v["scale"]
        ry = -(py - rect.centery) / v["scale"]
        return v["cx"] + math.cos(r) * rx - math.sin(r) * ry, v["cy"] + math.sin(r) * rx + math.cos(r) * ry

    def to_gt(self, x, y):
        p = self.p
        th = p["gt_start_deg"] * DEG
        return (p["gt_start_x"] + math.cos(th) * x - math.sin(th) * y,
                p["gt_start_y"] + math.sin(th) * x + math.cos(th) * y)

    def from_gt(self, x, y):
        p = self.p
        th = p["gt_start_deg"] * DEG
        dx, dy = x - p["gt_start_x"], y - p["gt_start_y"]
        return math.cos(-th) * dx - math.sin(-th) * dy, math.sin(-th) * dx + math.cos(-th) * dy

    def snapv(self, v):
        s = SNAPS[self.snap_i]
        return round(round(v / s) * s, 4)

    def fit(self):
        m = self.map
        rect = self.map_rect()
        x0, x1, y0, y1 = -1.5, 1.5, -1.5, 1.5
        if m is not None:
            known = np.argwhere(m["cls"] != 0)
            if known.size:
                r0, c0 = known.min(axis=0)
                r1, c1 = known.max(axis=0) + 1
                x0, x1 = m["ox"] + c0 * m["res"], m["ox"] + c1 * m["res"]
                y0, y1 = m["oy"] + r0 * m["res"], m["oy"] + r1 * m["res"]
        for w in self.draft_walls_world():
            x0, x1 = min(x0, w[0], w[2]), max(x1, w[0], w[2])
            y0, y1 = min(y0, w[1], w[3]), max(y1, w[1], w[3])
        self.view["cx"], self.view["cy"] = (x0 + x1) / 2, (y0 + y1) / 2
        span = max(x1 - x0, y1 - y0, 1.0) + 0.6
        self.view["scale"] = max(8.0, min(rect.w, rect.h) / span)

    def draft_walls_world(self):
        d = self.draft
        walls = [list(w) for w in d["walls"]]
        if d["border"]:
            x0, y0, x1, y1 = d["border"]
            walls += [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]
        out = []
        for w in walls:
            a = self.from_gt(w[0], w[1])
            b = self.from_gt(w[2], w[3])
            out.append([a[0], a[1], b[0], b[1]])
        return out

    # ------------------------------------------------------------------ main loop
    def run(self):
        while self.running and self.ex._running:
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    self.running = False
            self.ui.begin(self.screen, events)
            now = time.time()
            if now - self.last_poll > 0.1 or self.snap is None:
                try:
                    self.poll()
                except Exception as exc:
                    self.toast(f"update failed: {exc!r}")
                self.last_poll = now
            self.handle_keys(events)
            self.draw(events)
            if self.ui.focus is not None and self.ui.focus not in self.ui.seen:
                self.ui.focus = None   # the field scrolled away or its tab closed
            pygame.mouse.set_cursor(self.ui.cursor)
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()

    def handle_keys(self, events):
        ui = self.ui
        typing = ui.focus is not None
        for e in events:
            if e.type != pygame.KEYDOWN or typing:
                continue
            if e.key == pygame.K_SPACE:
                self.cmd("stop")
            elif e.key == pygame.K_f:
                self.fit()
            elif e.key == pygame.K_r:
                self.follow = not self.follow
            elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.view["scale"] *= 1.3
            elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.view["scale"] /= 1.3
            elif e.key == pygame.K_ESCAPE and self.confirm:
                self.confirm = None
        # manual drive, 10 Hz while keys are held
        keys = pygame.key.get_pressed()
        held = {k: keys[getattr(pygame, "K_" + k)] for k in "wasdqe"} if not typing and self.kbd_drive else {}
        pad = getattr(self, "_pad_held", set())
        for k in pad:
            held[k] = True
        active = any(held.values())
        if time.time() - self.last_drive > 0.1:
            if active:
                v, w = self.inputs["speed"], self.inputs["turn_rate"]
                self.cmd("manual", vx=(held.get("w", 0) - held.get("s", 0)) * v,
                         vy=(held.get("a", 0) - held.get("d", 0)) * v, wz=(held.get("q", 0) - held.get("e", 0)) * w)
                self.driving = True
                self.last_drive = time.time()
            elif self.driving:
                self.cmd("manual", vx=0, vy=0, wz=0)
                self.driving = False

    # ------------------------------------------------------------------ drawing
    def draw(self, events):
        ui = self.ui
        self.screen.fill(ui.pal["bg"])
        modal = self.confirm is not None
        if modal:  # widgets underneath must not react
            ui.mouse = (-999, -999)
        self.draw_map(events if not modal else [])
        self.draw_log(events if not modal else [])
        self.draw_side(events if not modal else [])
        self.draw_header()
        if self.confirm:
            ui.mouse = pygame.mouse.get_pos()
            self.draw_confirm()
        if self.toast_msg and time.time() < self.toast_until:
            w, h = self.screen.get_size()
            s = ui.text(self.toast_msg, "sb", "white")
            r = pygame.Rect(0, 0, min(w - 40, s.get_width() + 30), 34)
            r.midbottom = (w // 2, h - 16)
            ui.rrect(r, "ok" if self.toast_ok else "bad", 8)
            ui.label(self.toast_msg, r.x + 15, r.y + 9, "sb", "white", maxw=r.w - 30)

    def draw_header(self):
        ui, s = self.ui, self.snap
        w, _ = self.screen.get_size()
        bar = pygame.Rect(0, 0, w, self.HEADER)
        ui.rrect(bar, "panel", 0)
        pygame.draw.line(self.screen, ui.pal["line"], (0, self.HEADER - 1), (w, self.HEADER - 1))
        x = 14
        x += ui.label("RoboMaster SLAM", x, 15, "h") + 12
        if s:
            state = "PAUSED" if s["paused"] else s["state"]
            col = "warn" if s["paused"] else "ok" if state in RUNNING_STATES else "accent" if state == "DONE" else "muted"
            tw = ui.fonts["sb"].size(state)[0] + 18
            pill = pygame.Rect(x, 13, tw, 24)
            ui.rrect(pill, col, 12, 1)
            ui.label(state, pill.centerx, 17, "sb", col, "center")
            x = pill.right + 12
            x += ui.label("SIMULATOR" if s["robot"] == "simulator" else "ROBOT", x, 17, "s", "muted") + 14
            x += ui.label(f"time {s['t']:.1f} s", x, 17, "s", "muted") + 14
            bat = s["battery"]
            x += ui.label(f"battery {bat}%" if bat is not None else "battery –", x, 17, "s", "muted") + 14
        # buttons from the right
        busy = s is not None and s["state"] not in ("IDLE", "DONE")
        specs = [("Theme", "theme", "normal", 64), ("STOP", "stop", "danger", 88), ("Save", "save", "normal", 60),
                 ("Finish & report", "finish", "normal", 130),
                 ("Resume" if s and s["paused"] else "Pause", "pause", "normal", 84),
                 ("Start mission", "start", "primary", 124)]
        bx = w - 12
        for label, key, kind, bw in specs:
            r = pygame.Rect(bx - bw, 9, bw, 32)
            bx -= bw + 8
            enabled = not (key == "start" and busy)
            if ui.button(r, label, kind, enabled=enabled):
                if key == "theme":
                    ui.set_theme("light" if ui.theme == "dark" else "dark")
                    self._map_cache = (None, None)
                elif key == "pause":
                    self.cmd("resume" if s and s["paused"] else "pause")
                else:
                    self.cmd(key)

    # ------------------------------------------------------------------ map
    def draw_map(self, events):
        ui = self.ui
        rect = self.map_rect()
        ui.rrect(rect, "unknown", 10)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(rect)
        ui.clip = rect
        m = self.map
        if m is not None:
            self.blit_map(rect)
        if self.layers["grid"]:
            self.draw_grid(rect)
        p = self.p
        if p["border_enabled"]:
            self.poly([(p["border_min_x"], p["border_min_y"]), (p["border_max_x"], p["border_min_y"]),
                       (p["border_max_x"], p["border_max_y"]), (p["border_min_x"], p["border_max_y"])],
                      "warn", 2, closed=True, dash=True)
        if self.layers["gt"]:
            for wl in self.draft_walls_world():
                self.poly([(wl[0], wl[1]), (wl[2], wl[3])], "violet", 2)
        s = self.snap
        if s and self.layers["maze"] and s.get("grid"):
            for wl in s["grid"].get("unknown", []):
                self.poly([(wl[0], wl[1]), (wl[2], wl[3])], "warn", 2, dash=True)
            for wl in s["grid"].get("walls", []):
                self.poly([(wl[0], wl[1]), (wl[2], wl[3])], "accent2", 3)
        if s and self.layers["plan"] and s.get("plan"):
            plan = s["plan"]
            for f in plan.get("frontiers") or []:
                x, y = self.w2s(f[0], f[1])
                pygame.draw.circle(self.screen, ui.pal["ok"], (int(x), int(y)), int(3 + min(8, math.sqrt(f[2]))), 2)
            if plan.get("path"):
                self.poly(plan["path"], "accent2", 2, dash=True)
            if plan.get("goal"):
                x, y = self.w2s(*plan["goal"])
                pygame.draw.circle(self.screen, ui.pal["accent2"], (int(x), int(y)), 8, 2)
                pygame.draw.circle(self.screen, ui.pal["accent2"], (int(x), int(y)), 2)
        if self.layers["traj"] and len(self.traj) > 1:
            self.poly(self.traj, "traj", 2)
        if s and self.layers["scan"] and s.get("scan"):
            o = self.w2s(s["scan"]["pose"]["x"], s["scan"]["pose"]["y"])
            for q in s["scan"]["pts"]:
                x, y = self.w2s(q[0], q[1])
                if q[2]:
                    pygame.draw.rect(self.screen, ui.pal["bad"], (int(x) - 1, int(y) - 1, 3, 3))
                else:
                    pygame.draw.line(self.screen, _mix(ui.pal["bad"], ui.pal["unknown"], 0.7), o, (x, y))
        if s:
            x, y = self.w2s(s["start"]["x"], s["start"]["y"])
            pygame.draw.circle(self.screen, ui.pal["ok"], (int(x), int(y)), 6)
            ui.label("START", int(x) + 8, int(y) - 18, "xs", "ok")
            if s.get("true_pose") and self.layers["truth"]:
                self.robot(s["true_pose"], "violet", hollow=True)
            self.robot(s["pose"], "accent", gimbal=s["gimbal_deg"], tof=s["tof_m"])
        if self.drag and self.drag.get("preview"):
            self.drag["preview"]()
        self.draw_scale(rect)
        self.map_interaction(rect, events)
        self.map_toolbar(rect)
        # hover readout
        if rect.collidepoint(ui.mouse):
            wx, wy = self.s2w(*ui.mouse)
            txt = f"map x {wx:.2f}  y {wy:.2f}"
            if self.gt_server or self.draft["walls"]:
                gx, gy = self.to_gt(wx, wy)
                txt += f"    arena x {gx:.2f}  y {gy:.2f}"
            r = pygame.Rect(rect.x + 8, rect.bottom - 30, ui.fonts["m"].size(txt)[0] + 16, 22)
            ui.rrect(r, "panel", 6)
            ui.label(txt, r.x + 8, r.y + 4, "m", "muted")
        if self.mode in MODE_HINTS:
            hint = MODE_HINTS[self.mode]
            tw = ui.fonts["s"].size(hint)[0] + 20
            r = pygame.Rect(rect.centerx - tw // 2, rect.y + 84, tw, 24)
            ui.rrect(r, "accent", 6)
            ui.label(hint, r.centerx, r.y + 4, "s", "white", "center")
        self.screen.set_clip(old_clip)
        ui.clip = None
        pygame.draw.rect(self.screen, ui.pal["line"], rect, 1, border_radius=10)

    def map_rgb(self):
        m, pal = self.map, self.ui.pal
        if self.layers["prob"]:
            pr = m["prob"][..., None]
            rgb = (np.array(pal["free"]) * (1 - pr) + np.array(pal["wall"]) * pr).astype(np.uint8)
            unk = (m["cls"] == 0) & (np.abs(m["prob"] - 0.5) < 0.02)
            rgb[unk] = pal["unknown"]
        else:
            lut = np.array([pal["unknown"], pal["free"], pal["wall"]], np.uint8)
            use = m["clean"] if self.layers["clean"] and m.get("clean") is not None else m["cls"]
            rgb = lut[use]
        return rgb

    def blit_map(self, rect):
        m, v = self.map, self.view
        key = (self.map_version, self.grid_seen, self.layers["clean"], id(m["grid"]), self.ui.theme, self.layers["prob"], round(v["cx"], 4),
               round(v["cy"], 4), round(v["scale"], 3), round(v["rot"], 2), rect.size)
        if self._map_cache[0] == key:
            surf, pos = self._map_cache[1]
            self.screen.blit(surf, pos)
            return
        corners = [self.s2w(x, y, rect) for x, y in (rect.topleft, rect.topright, rect.bottomleft, rect.bottomright)]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        c0 = max(0, int(math.floor((min(xs) - m["ox"]) / m["res"])) - 1)
        c1 = min(m["cols"], int(math.ceil((max(xs) - m["ox"]) / m["res"])) + 1)
        r0 = max(0, int(math.floor((min(ys) - m["oy"]) / m["res"])) - 1)
        r1 = min(m["rows"], int(math.ceil((max(ys) - m["oy"]) / m["res"])) + 1)
        if c1 <= c0 or r1 <= r0:
            self._map_cache = (key, (pygame.Surface((1, 1), pygame.SRCALPHA), (0, 0)))
            return
        sub = self.map_rgb()[r0:r1, c0:c1][::-1]            # image rows: largest y first
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(sub.transpose(1, 0, 2))).convert_alpha()
        cell = m["res"] * v["scale"]
        size = (max(1, int(round((c1 - c0) * cell))), max(1, int(round((r1 - r0) * cell))))
        surf = pygame.transform.scale(surf, size)
        if abs(v["rot"]) > 1e-6:
            surf = pygame.transform.rotate(surf, v["rot"])
        cx = m["ox"] + (c0 + c1) / 2 * m["res"]
        cy = m["oy"] + (r0 + r1) / 2 * m["res"]
        sx, sy = self.w2s(cx, cy, rect)
        pos = surf.get_rect(center=(round(sx), round(sy))).topleft
        self._map_cache = (key, (surf, pos))
        self.screen.blit(surf, pos)

    def draw_grid(self, rect):
        sc = self.view["scale"]
        step = 0.1 if sc > 160 else 0.5 if sc > 40 else 1.0
        corners = [self.s2w(x, y, rect) for x, y in (rect.topleft, rect.topright, rect.bottomleft, rect.bottomright)]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        col = self.ui.pal["grid"]
        x = math.floor(min(xs) / step) * step
        while x <= max(xs):
            pygame.draw.line(self.screen, col, self.w2s(x, min(ys)), self.w2s(x, max(ys)))
            x += step
        y = math.floor(min(ys) / step) * step
        while y <= max(ys):
            pygame.draw.line(self.screen, col, self.w2s(min(xs), y), self.w2s(max(xs), y))
            y += step
        o = self.w2s(0, 0)
        pygame.draw.line(self.screen, self.ui.pal["bad"], o, self.w2s(0.3, 0), 2)
        pygame.draw.line(self.screen, self.ui.pal["ok"], o, self.w2s(0, 0.3), 2)

    def poly(self, pts, color, width=2, closed=False, dash=False):
        if len(pts) < 2:
            return
        col = self.ui.pal[color] if isinstance(color, str) else color
        sp = [self.w2s(q[0], q[1]) for q in pts]
        if closed:
            sp.append(sp[0])
        if not dash:
            pygame.draw.lines(self.screen, col, False, sp, width)
            return
        for (x0, y0), (x1, y1) in zip(sp, sp[1:]):
            length = math.hypot(x1 - x0, y1 - y0)
            n = int(length // 9)
            for i in range(0, n + 1, 2):
                a = i / max(n, 1)
                b = min(1.0, (i + 1) / max(n, 1))
                pygame.draw.line(self.screen, col, (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                                 (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), width)

    def robot(self, pose, color, hollow=False, gimbal=None, tof=None):
        pal = self.ui.pal
        x, y = self.w2s(pose["x"], pose["y"])
        sc = self.view["scale"]
        length, wid = max(0.16 * sc, 10), max(0.12 * sc, 7)
        a = pose["deg"] * DEG + self.view["rot"] * DEG

        def pt(fx, fy):  # robot frame (forward, left) in pixels -> screen
            return (x + math.cos(a) * fx - math.sin(a) * fy, y - (math.sin(a) * fx + math.cos(a) * fy))
        tri = [pt(length, 0), pt(-length * 0.7, wid), pt(-length * 0.7, -wid)]
        if gimbal is not None:
            g = a + gimbal * DEG
            r = (min(tof + 0.08, 4.0) * sc) if tof else length * 1.8
            end = (x + math.cos(g) * r, y - math.sin(g) * r)
            n = int(r // 8)
            for i in range(0, n, 2):
                pygame.draw.line(self.screen, pal["bad"], (x + (end[0] - x) * i / n, y + (end[1] - y) * i / n),
                                 (x + (end[0] - x) * (i + 1) / n, y + (end[1] - y) * (i + 1) / n), 2)
        if hollow:
            pygame.draw.polygon(self.screen, pal[color], tri, 2)
        else:
            pygame.draw.polygon(self.screen, pal[color], tri)
            pygame.draw.polygon(self.screen, pal["white"], tri, 1)

    def draw_scale(self, rect):
        sc = self.view["scale"]
        m = 0.1 if sc > 200 else 0.5 if sc > 60 else 1.0
        px = m * sc
        y = rect.bottom - 42
        pygame.draw.line(self.screen, self.ui.pal["text"], (rect.right - 16 - px, y), (rect.right - 16, y), 2)
        self.ui.label(f"{m:g} m", int(rect.right - 16 - px), y - 17, "xs")

    def map_toolbar(self, rect):
        ui = self.ui
        x, y = rect.x + 8, rect.y + 8

        def group(items, x, y):
            widths = [max(40, ui.fonts["s"].size(lbl)[0] + 18) for lbl, *_ in items]
            box = pygame.Rect(x, y, sum(widths) + 4 + 2 * (len(items) - 1), 32)
            ui.rrect(box, "panel", 8)
            ui.rrect(box, "line", 8, 1)
            bx = x + 2
            for (lbl, on, fn), bw in zip(items, widths):
                if ui.button(pygame.Rect(bx, y + 2, bw, 28), lbl, on=on):
                    fn()
                bx += bw + 2
            return box.right + 8

        def zoom(k):
            def fn():
                self.view["scale"] = max(8.0, min(2500.0, self.view["scale"] * k))
            return fn

        def setrot(d):
            def fn():
                deg = ((round(self.view["rot"]) + d + 540) % 360) - 180
                self.set_params(map_rotation_deg=deg)
                self.view["rot"] = deg
            return fn

        x = group([("+", False, zoom(1.3)), ("-", False, zoom(1 / 1.3)), ("Fit", False, self.fit),
                   ("Follow", self.follow, lambda: setattr(self, "follow", not self.follow))], x, y)
        x = group([("Rotate L", False, setrot(90)), (f"{round(self.view['rot'])}°", False, setrot(-round(self.view["rot"]))),
                   ("Rotate R", False, setrot(-90))], x, y)

        def setmode(k):
            return lambda: setattr(self, "mode", k)
        modes = [(lbl, self.mode == k, setmode(k)) for k, lbl in MODES]
        if x + 420 > rect.right:
            x, y = rect.x + 8, y + 38
        x = group(modes, x, y)

        def toggle(k):
            def fn():
                self.layers[k] = not self.layers[k]
                self._map_cache = (None, None)
            return fn
        layers = [(lbl, self.layers[k], toggle(k)) for k, lbl in LAYERS]
        if x + 360 > rect.right:
            x, y = rect.x + 8, y + 38
        group(layers, x, y)
        self._toolbar_bottom = y + 34

    def map_interaction(self, rect, events):
        ui = self.ui
        on_toolbar = ui.mouse[1] < getattr(self, "_toolbar_bottom", rect.y + 40) and ui.mouse[1] > rect.y
        for e in events:
            if e.type == pygame.MOUSEWHEEL and rect.collidepoint(ui.mouse):
                wx, wy = self.s2w(*ui.mouse)
                self.view["scale"] = max(8.0, min(2500.0, self.view["scale"] * math.exp(e.y * 0.12)))
                nx, ny = self.s2w(*ui.mouse)
                self.view["cx"] += wx - nx
                self.view["cy"] += wy - ny
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and rect.collidepoint(e.pos) and not on_toolbar \
                    and ui.focus is None:
                start = self.s2w(*e.pos)
                self.drag = {"px": e.pos, "start": start, "cur": start, "cx": self.view["cx"], "cy": self.view["cy"]}
                if self.mode == "pan":
                    self.follow = False
            elif e.type == pygame.MOUSEMOTION and self.drag:
                d = self.drag
                d["cur"] = self.s2w(*e.pos)
                if self.mode == "pan":
                    r = -self.view["rot"] * DEG
                    dx = (e.pos[0] - d["px"][0]) / self.view["scale"]
                    dy = -(e.pos[1] - d["px"][1]) / self.view["scale"]
                    self.view["cx"] = d["cx"] - (math.cos(r) * dx - math.sin(r) * dy)
                    self.view["cy"] = d["cy"] - (math.sin(r) * dx + math.cos(r) * dy)
                else:
                    d["preview"] = self._preview
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.drag:
                self._finish_drag(e.pos)
                self.drag = None
        if rect.collidepoint(ui.mouse) and not on_toolbar:
            ui.cursor = pygame.SYSTEM_CURSOR_SIZEALL if self.mode == "pan" else pygame.SYSTEM_CURSOR_CROSSHAIR

    def _preview(self):
        d = self.drag
        a, b = d["start"], d["cur"]
        if self.mode == "border":
            self.poly([a, (b[0], a[1]), b, (a[0], b[1])], "warn", 2, closed=True, dash=True)
        elif self.mode == "gtborder":
            g0, g1 = self.to_gt(*a), self.to_gt(*b)
            q = [(self.snapv(g0[0]), self.snapv(g0[1])), (self.snapv(g1[0]), self.snapv(g0[1])),
                 (self.snapv(g1[0]), self.snapv(g1[1])), (self.snapv(g0[0]), self.snapv(g1[1]))]
            self.poly([self.from_gt(*p) for p in q], "violet", 2, closed=True)
        elif self.mode == "wall":
            g0, g1 = self.to_gt(*a), self.to_gt(*b)
            self.poly([self.from_gt(self.snapv(g0[0]), self.snapv(g0[1])),
                       self.from_gt(self.snapv(g1[0]), self.snapv(g1[1]))], "violet", 3)
        elif self.mode == "pose":
            self.poly([a, b], "accent", 3)
            self.robot({"x": a[0], "y": a[1], "deg": math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))},
                       "accent", hollow=True)

    def _finish_drag(self, pos):
        d = self.drag
        a, b = d["start"], self.s2w(*pos)
        moved = math.hypot(pos[0] - d["px"][0], pos[1] - d["px"][1]) > 4
        if self.mode == "goto" and not moved:
            self.cmd("goto", x=a[0], y=a[1])
        elif self.mode == "pose" and moved:
            self.cmd("set_pose", x=a[0], y=a[1], deg=math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
        elif self.mode == "border" and moved:
            self.set_params(border_enabled=True, border_min_x=round(min(a[0], b[0]), 2),
                            border_max_x=round(max(a[0], b[0]), 2), border_min_y=round(min(a[1], b[1]), 2),
                            border_max_y=round(max(a[1], b[1]), 2))
        elif self.mode == "gtborder" and moved:
            g0, g1 = self.to_gt(*a), self.to_gt(*b)
            self.draft["border"] = [self.snapv(min(g0[0], g1[0])), self.snapv(min(g0[1], g1[1])),
                                    self.snapv(max(g0[0], g1[0])), self.snapv(max(g0[1], g1[1]))]
        elif self.mode == "wall" and moved:
            g0, g1 = self.to_gt(*a), self.to_gt(*b)
            wl = [self.snapv(g0[0]), self.snapv(g0[1]), self.snapv(g1[0]), self.snapv(g1[1])]
            if wl[:2] != wl[2:]:
                self.draft["walls"].append(wl)

    # ------------------------------------------------------------------ log
    def draw_log(self, events):
        ui = self.ui
        w, h = self.screen.get_size()
        rect = pygame.Rect(10, h - self.LOG_H - 10, w - self.SIDE - 30, self.LOG_H)
        ui.rrect(rect, "panel", 10)
        ui.rrect(rect, "line", 10, 1)
        ui.label("Log", rect.x + 12, rect.y + 8, "sb")
        self.log_debug = ui.checkbox(pygame.Rect(rect.right - 190, rect.y + 5, 80, 22), self.log_debug, "debug")
        follow = self.log_scroll == 0
        if ui.checkbox(pygame.Rect(rect.right - 100, rect.y + 5, 90, 22), follow, "follow") != follow:
            self.log_scroll = 0 if not follow else 1
        pygame.draw.line(self.screen, ui.pal["line"], (rect.x, rect.y + 32), (rect.right - 1, rect.y + 32))
        body = pygame.Rect(rect.x + 10, rect.y + 38, rect.w - 20, rect.h - 44)
        for e in events:
            if e.type == pygame.MOUSEWHEEL and rect.collidepoint(ui.mouse):
                self.log_scroll = max(0, self.log_scroll + e.y * 3)
        lines = []
        tw = ui.fonts["m"].size("00000.0s  ")[0]
        for _, t, level, msg in self.events[-600:]:
            if level == "debug" and not self.log_debug:
                continue
            col = {"warn": "warn", "error": "bad", "debug": "muted"}.get(level, "text")
            for i, ln in enumerate(ui.wrap(msg, "m", body.w - tw)):
                lines.append((f"{t:7.1f}s" if i == 0 else "", ln, col))
        lh = 16
        n = body.h // lh
        self.log_scroll = min(self.log_scroll, max(0, len(lines) - n))
        view = lines[max(0, len(lines) - n - self.log_scroll):len(lines) - self.log_scroll]
        old = self.screen.get_clip()
        self.screen.set_clip(body)
        for i, (t, ln, col) in enumerate(view):
            y = body.y + i * lh
            ui.label(t, body.x, y, "m", "muted")
            ui.label(ln, body.x + tw, y, "m", col)
        self.screen.set_clip(old)

    # ------------------------------------------------------------------ side panel
    def draw_side(self, events):
        ui = self.ui
        w, h = self.screen.get_size()
        rect = pygame.Rect(w - self.SIDE - 10, self.HEADER + 10, self.SIDE, h - self.HEADER - 20)
        ui.rrect(rect, "panel", 10)
        ui.rrect(rect, "line", 10, 1)
        tw = (rect.w - 12) / len(TABS)
        for i, t in enumerate(TABS):
            r = pygame.Rect(int(rect.x + 6 + i * tw), rect.y + 6, int(tw) - 2, 30)
            on = self.tab == t
            if on:
                ui.rrect(r, "btn", 6)
            if ui.hit(r) and ui.clicked:
                self.tab = t
                if t == "Results":
                    self.refresh_files(force=True)
            ui.label(t, r.centerx, r.y + 7, "sb" if on else "s", "text" if on else "muted", "center")
        pygame.draw.line(self.screen, ui.pal["line"], (rect.x, rect.y + 42), (rect.right - 1, rect.y + 42))
        body = pygame.Rect(rect.x + 12, rect.y + 50, rect.w - 24, rect.h - 58)
        for e in events:
            if e.type == pygame.MOUSEWHEEL and rect.collidepoint(ui.mouse):
                self.scroll[self.tab] -= e.y * 30
        self.scroll[self.tab] = max(0, min(self.scroll[self.tab], max(0, self.content_h[self.tab] - body.h)))
        old = self.screen.get_clip()
        self.screen.set_clip(body)
        ui.clip = body
        y0 = body.y - self.scroll[self.tab]
        draw = {"Status": self.tab_status, "Control": self.tab_control, "Settings": self.tab_settings,
                "Ground truth": self.tab_gt, "Results": self.tab_results}[self.tab]
        end = draw(body.x, y0, body.w)
        self.content_h[self.tab] = end - y0 + 10
        self.screen.set_clip(old)
        ui.clip = None
        if self.content_h[self.tab] > body.h:  # scrollbar
            frac = body.h / self.content_h[self.tab]
            bar_h = max(30, int(body.h * frac))
            top = body.y + int((body.h - bar_h) * self.scroll[self.tab] / max(1, self.content_h[self.tab] - body.h))
            ui.rrect(pygame.Rect(rect.right - 6, top, 3, bar_h), "line", 2)

    def section(self, title, x, y):
        self.ui.label(title.upper(), x, y, "xs", "muted")
        return y + 20

    def kv(self, rows, x, y, w):
        ui = self.ui
        for k, v in rows:
            ui.label(k, x, y, "s", "muted")
            ui.label(v, x + w, y + 1, "m", "text", "right", maxw=w - ui.fonts["s"].size(k)[0] - 12)
            y += 20
        return y

    # --- Status
    def tab_status(self, x, y, w):
        ui, s = self.ui, self.snap
        if not s:
            return y
        m = s.get("metrics") or {}
        half = (w - 8) // 2
        for i, (title, val, sub) in enumerate((
                ("Map Accuracy", f"{m['accuracy_pct']:.1f}%" if "accuracy_pct" in m else "–",
                 f"{m['correct_cells']} / {m['total_cells']} cells · {m['eval_cell_m']} m" if "accuracy_pct" in m
                 else "load a ground truth"),
                ("Coverage", f"{m['coverage_pct']:.1f}%" if "coverage_pct" in m else "–",
                 f"{m['explored_cells']} / {m['total_cells']} cells" if "coverage_pct" in m
                 else f"{m.get('known_area_m2', 0)} m² mapped"))):
            r = pygame.Rect(x + i * (half + 8), y, half, 74)
            ui.rrect(r, "panel2", 8)
            ui.rrect(r, "line", 8, 1)
            ui.label(title, r.x + 10, r.y + 8, "xs", "muted")
            ui.label(val, r.x + 10, r.y + 22, "big")
            ui.label(sub, r.x + 10, r.y + 55, "xs", "muted", maxw=r.w - 16)
        y += 88
        y = self.section("Robot position (x forward at start, y left)", x, y)
        g = self.to_gt(s["pose"]["x"], s["pose"]["y"])
        rows = [("SLAM pose", fpose(s["pose"])), ("Odometry", fpose(s["odom"])), ("Drift correction", fpose(s["corr"])),
                ("Start", fpose(s["start"])),
                ("In arena frame", f"{g[0]:.2f}, {g[1]:.2f} m  {s['pose']['deg'] + self.p['gt_start_deg']:.1f}°"
                 if self.gt_server else "–")]
        if s.get("true_pose"):
            t = s["true_pose"]
            err = math.hypot(t["x"] - s["pose"]["x"], t["y"] - s["pose"]["y"]) * 100
            rows.append(("Simulator truth", f"{fpose(t)} ({err:.1f} cm)"))
        g = s.get("grid")
        if g:
            rows.append(("Maze grid", f"{g['cell_m']:.3f} m cells, {g['theta_deg']:+.1f}°"
                         + ("" if g["cell_trusted"] else " (checking)")))
            if "walls" in m.get("grid", {}):
                mg = m["grid"]
                rows.append(("Edges wall / open, cells", f"{mg['walls']} / {mg['open']}, {mg['cells_seen']} seen"))
        rows += [("ToF ahead", "no reading" if s["tof_m"] is None else f"{s['tof_m']:.3f} m"),
                 ("Gimbal yaw", f"{s['gimbal_deg']:.1f}°")]
        y = self.kv(rows, x, y, w) + 10
        y = self.section("Mission", x, y)
        st = s["stats"]
        rows = [("State", s["state"] + (f" · {s['detail']}" if s["detail"] else "")), ("Scans", str(st["scans"])),
                ("Distance driven", f"{st['distance_m']:.2f} m"), ("Cycles", str(st["iterations"])),
                ("Emergency stops", str(st["blocked_moves"]))]
        if "wall_found" in m:
            rows += [("Walls found / missed / false", f"{m['wall_found']} / {m['wall_missed']} / {m['false_walls']}"),
                     ("Accuracy strict / explored", f"{m['strict_accuracy_pct']:.1f}% / {m['explored_accuracy_pct']:.1f}%")]
        y = self.kv(rows, x, y, w) + 10
        if s.get("report"):
            r = s["report"]
            y = self.section("Mission report", x, y)
            rows = [("Started at", fpose(r["start"])), ("Ended at", fpose(r["end"]))]
            if r["start"].get("gt"):
                rows += [("Start (arena)", fpose(r["start"]["gt"])), ("End (arena)", fpose(r["end"]["gt"]))]
            rows += [("Duration", f"{r['duration_s']:.1f} s")]
            y = self.kv(rows, x, y, w)
            for ln in ui.wrap("Finished: " + r["finish_reason"], "s", w):
                ui.label(ln, x, y, "s", "muted")
                y += 18
            y += 10
        y = self.section("Data over the whole run", x, y)
        bx = x
        for k, lbl in CHARTS:
            bw = ui.fonts["s"].size(lbl)[0] + 16
            if ui.button(pygame.Rect(bx, y, bw, 26), lbl, on=self.chart == k):
                self.chart = k
            bx += bw + 4
        y += 32
        self.draw_chart(pygame.Rect(x, y, w, 130))
        y += 142
        y = self.section("Last scan (robot view)", x, y)
        size = min(w, 240)
        self.draw_polar(pygame.Rect(x + (w - size) // 2, y, size, size))
        return y + size + 6

    def draw_chart(self, rect):
        ui, s = self.ui, self.snap
        ui.rrect(rect, "panel2", 8)
        ui.rrect(rect, "line", 8, 1)
        ser = s.get("series", {})
        sets = {"tof": [("tof", "bad", "m")], "cov": [("coverage", "ok", "%"), ("accuracy", "accent", "%")],
                "speed": [("speed", "traj", "m/s")], "match": [("match", "violet", "cm")],
                "loc_error": [("loc_error", "warn", "cm")]}[self.chart]
        vals = [(t, v) for k, _, _ in sets for t, v in ser.get(k, []) if v is not None]
        if not vals:
            ui.label("no data yet", rect.x + 12, rect.centery - 8, "s", "muted")
            return
        t0, t1 = min(v[0] for v in vals), max(v[0] for v in vals)
        v0, v1 = min(v[1] for v in vals), max(v[1] for v in vals)
        if self.chart == "cov":
            v0, v1 = 0.0, 100.0
        else:
            v0 = min(0.0, v0)
            v1 = v0 + 1 if v1 == v0 else v1 * 1.08
        t1 = t0 + 1 if t1 == t0 else t1
        L, R, T, B = 40, 8, 22, 18

        def X(t):
            return rect.x + L + (t - t0) / (t1 - t0) * (rect.w - L - R)

        def Y(v):
            return rect.bottom - B - (v - v0) / (v1 - v0) * (rect.h - T - B)
        for i in range(5):
            yy = rect.y + T + i * (rect.h - T - B) / 4
            pygame.draw.line(self.screen, ui.pal["line"], (rect.x + L, yy), (rect.right - R, yy))
            val = v1 - i * (v1 - v0) / 4
            ui.label(f"{val:.2f}" if v1 < 5 else f"{val:.0f}", rect.x + 4, int(yy) - 7, "xs", "muted")
        ui.label(f"{t0:.0f} s", rect.x + L, rect.bottom - 16, "xs", "muted")
        ui.label(f"{t1:.0f} s", rect.right - R, rect.bottom - 16, "xs", "muted", "right")
        lx = rect.x + L + 4
        for k, col, unit in sets:
            pts = ser.get(k, [])
            seg = []
            for t, v in pts:
                if v is None:
                    if len(seg) > 1:
                        pygame.draw.lines(self.screen, ui.pal[col], False, seg, 2)
                    seg = []
                else:
                    seg.append((X(t), Y(v)))
            if len(seg) > 1:
                pygame.draw.lines(self.screen, ui.pal[col], False, seg, 2)
            last = pts[-1][1] if pts else None
            lx += ui.label(f"{k} {'–' if last is None else last}{unit}", lx, rect.y + 4, "xs", col) + 14

    def draw_polar(self, rect):
        ui, s = self.ui, self.snap
        cx, cy, R = rect.centerx, rect.centery, rect.w // 2 - 12
        max_r = self.p["tof_max_m"]
        for i in (1, 2, 3):
            pygame.draw.circle(self.screen, ui.pal["line"], (cx, cy), int(R * i / 3), 1)
            ui.label(f"{max_r * i / 3:.1f}m", cx + 3, cy - int(R * i / 3) + 2, "xs", "muted")
        pygame.draw.line(self.screen, ui.pal["line"], (cx, cy - R), (cx, cy + R))
        pygame.draw.line(self.screen, ui.pal["line"], (cx - R, cy), (cx + R, cy))
        ui.label("front", cx, rect.y - 4, "xs", "muted", "center")
        if s.get("scan"):
            p = s["scan"]["pose"]
            th = p["deg"] * DEG
            for q in s["scan"]["pts"]:
                dx, dy = q[0] - p["x"], q[1] - p["y"]
                a = math.atan2(dy, dx) - th
                r = min(math.hypot(dx, dy), max_r) / max_r * R
                pygame.draw.rect(self.screen, ui.pal["bad" if q[2] else "muted"],
                                 (int(cx - math.sin(a) * r) - 1, int(cy - math.cos(a) * r) - 1, 3, 3))
        pygame.draw.polygon(self.screen, ui.pal["accent"], [(cx, cy - 9), (cx - 6, cy + 6), (cx + 6, cy + 6)])

    # --- Control
    def num_row(self, x, y, w, parts):
        """parts: list of str (label) | (key, width) (number field) | (label, callback, width) (button)."""
        ui = self.ui
        cx = x
        for part in parts:
            if isinstance(part, str):
                cx += ui.label(part, cx, y + 5, "s", "muted") + 6
            elif len(part) == 2:
                key, fw = part
                val = ui.field("in-" + key, pygame.Rect(cx, y, fw, 26), self.inputs[key])
                if val is not None:
                    self.inputs[key] = val
                cx += fw + 6
            else:
                lbl, fn, bw = part
                if ui.button(pygame.Rect(cx, y, bw, 26), lbl):
                    fn()
                cx += bw + 6
        return y + 34

    def tab_control(self, x, y, w):
        ui = self.ui
        inp = self.inputs
        y = self.section("Manual drive", x, y)
        for ln in ui.wrap("Hold W/A/S/D to drive and strafe, Q/E to rotate (or hold the pad buttons). Space = STOP.",
                          "s", w):
            ui.label(ln, x, y, "s", "muted")
            y += 18
        y += 4
        pad = [["q", "w", "e"], ["a", "s", "d"]]
        labels = {"q": "Q turn", "w": "W fwd", "e": "E turn", "a": "A left", "s": "S back", "d": "D right"}
        held = set()
        for r, row in enumerate(pad):
            for c, k in enumerate(row):
                rect = pygame.Rect(x + c * 62, y + r * 44, 58, 40)
                ui.button(rect, labels[k])
                if ui.hit(rect) and ui.held:
                    held.add(k)
        self._pad_held = held
        sx = x + 196
        sw = max(60, w - 196 - 48 - 72)
        ui.label("speed", sx, y + 4, "s", "muted")
        inp["speed"] = round(ui.slider("sl-speed", pygame.Rect(sx + 46, y + 2, sw, 20), inp["speed"], 0.05, 0.7), 2)
        ui.label(f"{inp['speed']:.2f} m/s", x + w, y + 4, "m", align="right")
        ui.label("turn", sx, y + 34, "s", "muted")
        inp["turn_rate"] = round(ui.slider("sl-turn", pygame.Rect(sx + 46, y + 32, sw, 20), inp["turn_rate"], 10, 180))
        ui.label(f"{inp['turn_rate']:.0f} °/s", x + w, y + 34, "m", align="right")
        self.kbd_drive = ui.checkbox(pygame.Rect(sx, y + 60, 200, 22), self.kbd_drive, "keyboard drive")
        y += 100
        y = self.section("Rotate robot", x, y)
        bx = x
        for lbl, deg in (("-90°", -90), ("-45°", -45), ("+45°", 45), ("+90°", 90), ("180°", 180)):
            if ui.button(pygame.Rect(bx, y, 64, 28), lbl):
                self.cmd("turn", deg=deg)
            bx += 70
        y += 36
        y = self.num_row(x, y, w, ["by", ("turn", 70), "° (+ = left)",
                                   ("Turn", lambda: self.cmd("turn", deg=inp["turn"]), 60)])
        y = self.section("Move", x, y + 4)
        y = self.num_row(x, y, w, ["forward", ("move", 70), "m", ("Move", lambda: self.cmd("move", m=inp["move"]), 60),
                                   ("Go home", lambda: self.cmd("home"), 80)])
        y = self.num_row(x, y, w, ["go to x", ("gx", 60), "y", ("gy", 60),
                                   ("Go", lambda: self.cmd("goto", x=inp["gx"], y=inp["gy"]), 50)])
        y = self.section("Sensors", x, y + 4)
        bx = x
        for lbl, name, bw in (("Scan now", "scan", 90), ("Auto calibrate", "calibrate", 120),
                              ("Centre gimbal", "gimbal_center", 116)):
            if ui.button(pygame.Rect(bx, y, bw, 28), lbl):
                self.cmd(name)
            bx += bw + 6
        y += 36
        y = self.num_row(x, y, w, ["gimbal to", ("gim", 60), "°",
                                   ("Aim", lambda: self.cmd("gimbal_to", deg=inp["gim"]), 50)])
        y = self.num_row(x, y, w, ["wall at", ("wall", 60), "m from lens",
                                   ("Calibrate ToF", lambda: self.cmd("calibrate_wall", distance=inp["wall"]), 110)])
        y = self.section("Map & pose", x, y + 4)
        y = self.num_row(x, y, w, ["pose x", ("spx", 56), "y", ("spy", 56), "θ", ("spt", 56),
                                   ("Set", lambda: self.cmd("set_pose", x=inp["spx"], y=inp["spy"], deg=inp["spt"]), 44)])
        bx = x + ui.label("rotate estimate", x, y + 5, "s", "muted") + 8
        for d in (-5, -1, 1, 5):
            if ui.button(pygame.Rect(bx, y, 46, 26), f"{d:+d}°"):
                self.cmd("rotate_pose", deg=d)
            bx += 50
        y += 36
        bx = x
        for lbl, fn, bw in (
                ("Clear map", lambda: self.ask("Clear the map? The pose is kept.", lambda: self.cmd("reset_map")), 90),
                ("Clear ignored frontiers", lambda: self.cmd("clear_blacklist"), 170),
                ("New session", lambda: self.ask("Start a new session? Map and trajectory are cleared and a new run "
                                                 "folder is made.", self._new_session), 110)):
            if ui.button(pygame.Rect(bx, y, bw, 28), lbl):
                fn()
            bx += bw + 6
        y += 36
        y = self.section("Maze grid", x, y + 4)
        g = (self.snap or {}).get("grid")
        info = "not found yet" if not g else (f"{g['cell_m']:.3f} m cells, angle {g['theta_deg']:+.1f}°, "
                                              + ("locked" if g["cell_trusted"] else "cell size not confirmed yet"))
        for ln in ui.wrap(info + ". Set Settings > Grid > grid_mode = fixed and grid_cell_m to your tile size "
                          "to lock it at once.", "s", w):
            ui.label(ln, x, y, "s", "muted")
            y += 18
        if ui.button(pygame.Rect(x, y + 4, 130, 28), "Re-detect grid"):
            self.cmd("redetect_grid")
        return y + 40

    def _new_session(self):
        self.cmd("reset_all")
        self.traj, self.traj_len, self.events, self.fitted = [], 0, [], False

    # --- Settings
    def tab_settings(self, x, y, w):
        ui = self.ui
        if ui.button(pygame.Rect(x, y, 116, 28), "Save settings", "primary"):
            try:
                path = params_mod.save_overrides(self.p)
                self.ex.log(f"Settings saved to {path}")
                self.toast("Saved " + os.path.relpath(path), ok=True)
            except Exception as exc:
                self.toast(str(exc))
        if ui.button(pygame.Rect(x + 122, y, 80, 28), "Defaults"):
            self.ask("Reset every setting to its default?", lambda: self.set_params(**params_mod.defaults()))
        val = ui.field("filter", pygame.Rect(x + 208, y, w - 208, 28), self.set_filter or "", numeric=False, font="s")
        if val is not None:
            self.set_filter = val.strip().lower()
        if not self.set_filter and ui.focus != "filter":
            ui.label("filter…", x + 216, y + 6, "s", "muted")
        y += 36
        for ln in ui.wrap("Changes apply immediately (Enter). Blue names differ from the defaults; * = changing it clears "
                          "the map. Rotate robot: + turns left.",
                          "xs", w):
            ui.label(ln, x, y, "xs", "muted")
            y += 15
        y += 6
        groups = {}
        for spec in params_mod.schema():
            if self.set_filter and self.set_filter not in (spec["key"] + " " + spec["help"]).lower():
                continue
            groups.setdefault(spec["group"], []).append(spec)
        for g, items in groups.items():
            is_open = self.groups_open.get(g, False) or bool(self.set_filter)
            head = pygame.Rect(x, y, w, 30)
            ui.rrect(head, "panel2", 6)
            ui.rrect(head, "line", 6, 1)
            ui.label(("- " if is_open else "+ ") + g, x + 10, y + 7, "sb")
            ui.label(str(len(items)), head.right - 10, y + 8, "xs", "muted", "right")
            if ui.hit(head) and ui.clicked:
                self.groups_open[g] = not is_open
            y += 34
            if not is_open:
                continue
            for spec in items:
                key, kind = spec["key"], spec["type"]
                value = self.p[key]
                changed = value != spec["default"]
                ui.label(key + (" *" if spec["reset"] else ""), x + 4, y + 5, "m", "accent" if changed else "text")
                ctl = pygame.Rect(x + w - 110, y, 110, 26)
                if isinstance(kind, list):
                    if ui.button(ctl, f"{value}  (change)"):
                        self.set_params(**{key: kind[(kind.index(value) + 1) % len(kind)]})
                elif kind == "bool":
                    nv = ui.checkbox(pygame.Rect(ctl.right - 20, y, 20, 26), bool(value))
                    if nv != bool(value):
                        self.set_params(**{key: nv})
                else:
                    nv = ui.field("set-" + key, ctl, value)
                    if nv is not None:
                        self.set_params(**{key: nv})
                y += 28
                help_text = spec["help"] + (f"  [{_fmt(spec['min'])} … {_fmt(spec['max'])}]" if spec["min"] is not None else "")
                for ln in ui.wrap(help_text, "xs", w - 8):
                    ui.label(ln, x + 4, y, "xs", "muted")
                    y += 14
                y += 8
                pygame.draw.line(self.screen, ui.pal["line"], (x, y - 4), (x + w, y - 4))
            y += 6
        return y

    # --- Ground truth
    def tab_gt(self, x, y, w):
        ui = self.ui
        y = self.section("Ground-truth map", x, y)
        for ln in ui.wrap("Load the real arena (JSON: border + wall lines in metres, arena frame) or draw it on the map "
                          "with GT arena and GT wall, then press Use.", "s", w):
            ui.label(ln, x, y, "s", "muted")
            y += 18
        y += 4
        ui.label("file", x, y + 5, "s", "muted")
        val = ui.field("gt-path", pygame.Rect(x + 34, y, w - 100, 26), self.gt_path, numeric=False)
        if val is not None:
            self.gt_path = val.strip()
        if ui.button(pygame.Rect(x + w - 60, y, 60, 26), "Load"):
            self.load_gt_file()
        y += 34
        ui.label("name", x, y + 5, "s", "muted")
        val = ui.field("gt-name", pygame.Rect(x + 44, y, w - 44, 26), self.draft["name"], numeric=False, font="s")
        if val is not None:
            self.draft["name"] = val.strip() or "arena"
        y += 34
        ui.label("snap", x, y + 5, "s", "muted")
        if ui.button(pygame.Rect(x + 40, y, 70, 26), f"{SNAPS[self.snap_i] * 100:g} cm"):
            self.snap_i = (self.snap_i + 1) % len(SNAPS)
        ui.label("wall thickness", x + 124, y + 5, "s", "muted")
        val = ui.field("gt-thick", pygame.Rect(x + 222, y, 70, 26), self.draft["wall_thickness"])
        if val is not None:
            self.draft["wall_thickness"] = max(0.001, val)
        y += 36
        bx = x
        for lbl, fn, bw, kind in (("Use this ground truth", self.apply_gt, 160, "primary"),
                                  ("Undo wall", self.undo_wall, 84, "normal"),
                                  ("Clear", lambda: self.ask("Remove the ground truth?", self.clear_gt), 60, "normal")):
            if ui.button(pygame.Rect(bx, y, bw, 28), lbl, kind):
                fn()
            bx += bw + 6
        y += 34
        if ui.button(pygame.Rect(x, y, 150, 28), "Save JSON file"):
            self.save_gt_file()
        y += 38
        d = self.draft
        y = self.kv([("walls", str(len(d["walls"]))),
                     ("arena", ", ".join(f"{v:.2f}" for v in d["border"]) if d["border"] else "–"),
                     ("in use", self.gt_server["name"] if self.gt_server else "none")], x, y, w) + 10
        y = self.section("Robot start in the arena", x, y)
        bx = x
        for lbl, key in (("x", "gt_start_x"), ("y", "gt_start_y"), ("θ", "gt_start_deg")):
            bx += ui.label(lbl, bx, y + 5, "s", "muted") + 6
            val = ui.field("gs-" + key, pygame.Rect(bx, y, 70, 26), self.p[key])
            if val is not None:
                self.set_params(**{key: val})
            bx += 80
        y += 34
        for ln in ui.wrap("The robot's start pose in the arena frame lines the two maps up for scoring.", "s", w):
            ui.label(ln, x, y, "s", "muted")
            y += 18
        return y

    def draft_json(self):
        d = self.draft
        return {"name": d["name"], "border": d["border"], "wall_thickness": d["wall_thickness"], "walls": d["walls"]}

    def load_gt_file(self):
        try:
            gt = load_ground_truth(os.path.expanduser(self.gt_path))
            self.load_draft(gt)
            self.ex.set_ground_truth(gt)
            self.toast(f"Loaded {gt['name']}", ok=True)
        except Exception as exc:
            self.toast(f"Could not load: {exc}")

    def apply_gt(self):
        try:
            self.ex.set_ground_truth(validate_ground_truth(self.draft_json()))
            self.toast("Ground truth in use", ok=True)
        except Exception as exc:
            self.toast(str(exc))

    def undo_wall(self):
        if self.draft["walls"]:
            self.draft["walls"].pop()

    def clear_gt(self):
        self.draft = {"name": "arena", "border": None, "walls": [], "wall_thickness": 0.02}
        self.ex.set_ground_truth(None)

    def save_gt_file(self):
        name = "".join(c if c.isalnum() else "_" for c in self.draft["name"]).strip("_") or "ground_truth"
        path = os.path.join(self.gt_dir, f"{name}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.draft_json(), f, indent=2)
            self.gt_path = path
            self.toast("Saved " + os.path.relpath(path), ok=True)
        except Exception as exc:
            self.toast(str(exc))

    # --- Results
    def refresh_files(self, force=False):
        if not force and time.time() - self.files_t < 2.0:
            return
        self.files_t = time.time()
        d = self.ex.run_dir
        try:
            self.files = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
        except OSError:
            self.files = []
        img = os.path.join(d, "map.png")
        if os.path.exists(img):
            mt = os.path.getmtime(img)
            if mt != self.thumb_mtime:
                try:
                    self.thumb = pygame.image.load(img).convert()
                    self.thumb_mtime = mt
                except pygame.error:
                    self.thumb = None
        else:
            self.thumb = None

    def tab_results(self, x, y, w):
        ui = self.ui
        self.refresh_files()
        y = self.section("Run folder", x, y)
        for ln in ui.wrap(self.ex.run_dir, "m", w):
            ui.label(ln, x, y, "m", "muted")
            y += 16
        y += 6
        bx = x
        for lbl, fn, bw in (("Save now", lambda: self.cmd("save"), 90), ("Open folder", lambda: open_path(self.ex.run_dir), 104),
                            ("Refresh", lambda: self.refresh_files(True), 80)):
            if ui.button(pygame.Rect(bx, y, bw, 28), lbl):
                fn()
            bx += bw + 6
        y += 38
        for f in self.files:
            r = pygame.Rect(x, y, w, 20)
            hover = ui.hit(r)
            ui.label(f, x, y + 2, "m", "accent" if hover else "text")
            if hover:
                ui.cursor = pygame.SYSTEM_CURSOR_HAND
                if ui.clicked:
                    open_path(os.path.join(self.ex.run_dir, f))
            y += 20
        y += 10
        if self.thumb is not None:
            y = self.section("Latest saved map", x, y)
            tw, th = self.thumb.get_size()
            s = w / tw
            img = pygame.transform.smoothscale(self.thumb, (w, int(th * s)))
            self.screen.blit(img, (x, y))
            y += img.get_height() + 6
        return y

    # ------------------------------------------------------------------ modal
    def draw_confirm(self):
        ui = self.ui
        w, h = self.screen.get_size()
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 140))
        self.screen.blit(shade, (0, 0))
        msg, cb = self.confirm
        box = pygame.Rect(0, 0, 420, 150)
        box.center = (w // 2, h // 2)
        ui.rrect(box, "panel", 12)
        ui.rrect(box, "line", 12, 1)
        yy = box.y + 20
        for ln in ui.wrap(msg, "s", box.w - 40):
            ui.label(ln, box.x + 20, yy)
            yy += 20
        if ui.button(pygame.Rect(box.right - 200, box.bottom - 48, 84, 30), "Cancel"):
            self.confirm = None
        if ui.button(pygame.Rect(box.right - 108, box.bottom - 48, 88, 30), "Yes", "primary"):
            self.confirm = None
            cb()


def _mix(a, b, t):
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


def open_path(path):
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # noqa: S606 - opening the user's own results
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


def run_console(explorer, gt_dir):
    Console(explorer, gt_dir).run()
