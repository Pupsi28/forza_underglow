# /// script
# requires-python = ">=3.10"
# dependencies = ["PySide6>=6.5", "numpy"]
# ///
"""Forza Underglow Overlay (Telemetrie-Version).

Liest Forzas "Data Out"-Telemetrie (UDP) und zeichnet eine Unterbodenbeleuchtung
als transparentes Overlay. Keine Bilderkennung: In der Verfolgerkamera steht das
Spieler-Auto immer an ungefähr derselben Bildschirmstelle. Der Lichtschein wird als
Textur auf einer virtuellen Bodenebene berechnet und perspektivisch auf den
Bildschirm projiziert. Die Telemetrie bewegt ihn mit:
  - Driftwinkel / Gierrate  -> Auto dreht sich auf dem Bildschirm, Licht dreht mit
  - Federweg pro Rad        -> eingefederte Ecke näher am Boden = heller
  - Längsbeschleunigung     -> Kamera fällt zurück / holt auf (Zoom)
Seiten des Autos, die von der Kamera wegzeigen, werden von der Karosserie verdeckt.
"""
import argparse
import colorsys
import ctypes
import json
import math
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULTS = {
    "port": 1306,
    "color": "#00e5ff",       # Hex-Farbe oder "rainbow"
    "intensity": 0.9,
    "anchor_x": 0.5,          # Bildschirmposition der Fahrzeugmitte am Boden (0..1)
    "anchor_y": 0.73,
    "scale": 0.85,            # Größe auf dem Bildschirm
    "cam_height": 1.8,        # virtuelle Verfolgerkamera in Metern -> Perspektive
    "cam_dist": 6.0,
    "car_length": 4.4,
    "car_width": 1.9,
    "glow_reach": 0.9,        # wie weit das Licht über den Wagenrand reicht (Meter)
    "slip_follow": 0.6,       # wie stark sich das Auto beim Driften auf dem Bildschirm dreht
    "yaw_lag": 0.0,           # zusätzliche Drehung pro rad/s Gierrate (Vorzeichen ggf. umdrehen)
    "lateral_shift": 0.0,     # seitlicher Versatz pro m/s² Querbeschleunigung (Bildschirmanteil)
    "accel_zoom": 0.004,      # Größenänderung pro m/s² Längsbeschleunigung
    "smoothing": 0.2,         # 0..1, kleiner = ruhiger
}
COLORS = ["#00e5ff", "#ff00d4", "#ff1a1a", "#00ff66", "#2050ff", "#9a30ff", "#ffffff", "#ff8800", "rainbow"]


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            print(f"config.json ignoriert ({e})")
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def color_rgb(c):
    if c == "rainbow":
        return np.array(colorsys.hsv_to_rgb((time.time() * 0.1) % 1, 1, 1), np.float32)
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


class Telemetry(threading.Thread):
    """Empfängt Forza-Data-Out-Pakete. Felder im Fahrzeug-Koordinatensystem: X rechts, Y oben, Z vorne."""

    def __init__(self, port):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.state = None
        self.count = 0

    def run(self):
        while True:
            data = self.sock.recv(2048)
            if len(data) < 84:
                continue
            self.count += 1
            self.state = {
                "t": time.time(),
                "race_on": struct.unpack_from("<i", data, 0)[0] != 0,
                "accel": struct.unpack_from("<3f", data, 20),
                "vel": struct.unpack_from("<3f", data, 32),
                "ang_vel": struct.unpack_from("<3f", data, 44),
                "susp": struct.unpack_from("<4f", data, 68),  # FL, FR, RL, RR; 0 = ausgefedert, 1 = eingefedert
            }


class GlowModel:
    """Lichtschein als Textur auf der Bodenebene (Draufsicht, Zeile 0 = vorne)."""

    PPM = 28  # Texel pro Meter

    def __init__(self, cfg):
        hw, hl, reach = cfg["car_width"] / 2, cfg["car_length"] / 2, cfg["glow_reach"]
        self.hx, self.hz = hw + reach * 2.5, hl + reach * 2.5
        xs = -self.hx + (np.arange(int(2 * self.hx * self.PPM)) + 0.5) / self.PPM
        zs = self.hz - (np.arange(int(2 * self.hz * self.PPM)) + 0.5) / self.PPM
        X, Z = np.meshgrid(xs.astype(np.float32), zs.astype(np.float32))
        self.tw, self.th = X.shape[1], X.shape[0]

        # Abgerundetes Rechteck als Fahrzeugumriss (vorzeichenbehaftete Distanz, außen > 0)
        r = 0.35
        qx, qz = np.abs(X) - (hw - r), np.abs(Z) - (hl - r)
        d = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - r
        g_row, g_col = np.gradient(d)
        nx, nz = g_col, -g_row  # Zeilen laufen nach hinten
        n = np.hypot(nx, nz) + 1e-6
        self.nx, self.nz = nx / n, nz / n

        outside = np.exp(-np.maximum(d, 0) / (0.35 * reach)) + 0.12 * np.exp(-np.maximum(d, 0) / (0.8 * reach))
        inside = 1.3 * np.exp(np.minimum(d, 0) / 0.15)
        # zum Texturrand hin weich ausblenden, sonst sieht man harte Kanten
        edge = np.minimum(self.hx - np.abs(X), self.hz - np.abs(Z))
        fade = np.clip(edge / (0.8 * reach), 0, 1) ** 2
        self.base = (np.where(d > 0, outside, inside) * fade).astype(np.float32)
        self.u = np.clip((X / hw + 1) / 2, 0, 1)  # 0 = links, 1 = rechts
        self.v = np.clip((Z / hl + 1) / 2, 0, 1)  # 0 = hinten, 1 = vorne
        self.outline = self._outline(hw, hl, r)

    @staticmethod
    def _outline(hw, hl, r):
        pts = []
        for cx, cz, a0 in ((hw - r, hl - r, 0), (-hw + r, hl - r, 90), (-hw + r, -hl + r, 180), (hw - r, -hl + r, 270)):
            for a in range(a0, a0 + 91, 15):
                pts.append((cx + r * math.cos(math.radians(a)), cz + r * math.sin(math.radians(a))))
        return pts

    def texture(self, theta, corners, rgb, alpha):
        # Richtung zur Kamera im Fahrzeugsystem: weggedrehte Seiten verdeckt die Karosserie
        tx, tz = math.sin(theta), -math.cos(theta)
        vis = np.clip(0.75 + 0.9 * (self.nx * tx + self.nz * tz), 0, 1)
        fl, fr, rl, rr = corners
        u, v = self.u, self.v
        c = (rl * (1 - u) + rr * u) * (1 - v) + (fl * (1 - u) + fr * u) * v
        a = np.clip(self.base * vis * c, 0, 1) * alpha
        out = np.empty((self.th, self.tw, 4), np.uint8)
        out[..., :3] = (a[..., None] * rgb * 255).astype(np.uint8)  # premultiplied
        out[..., 3] = (a * 255).astype(np.uint8)
        return QtGui.QImage(out.data, self.tw, self.th, self.tw * 4,
                            QtGui.QImage.Format_RGBA8888_Premultiplied).copy()

    def homography(self, cfg, W, H, theta, zoom, shift_x):
        """3x3-Matrix: Texel -> Bildschirm (Texel -> Fahrzeug -> Welt -> Verfolgerkamera)."""
        T = np.array([[1 / self.PPM, 0, -self.hx], [0, -1 / self.PPM, self.hz], [0, 0, 1]])
        return self.ground_to_screen(cfg, W, H, theta, zoom, shift_x) @ T

    @staticmethod
    def ground_to_screen(cfg, W, H, theta, zoom, shift_x):
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        ch, cd = cfg["cam_height"], cfg["cam_dist"]
        phi = math.atan2(ch, cd)  # Kamera blickt auf die Fahrzeugmitte am Boden
        sp, cp = math.sin(phi), math.cos(phi)
        M = np.array([[1, 0, 0], [0, sp, cd * sp - ch * cp], [0, cp, ch * sp + cd * cp]])
        F = cfg["scale"] * zoom * W * 0.75 * math.hypot(ch, cd) / 6.3
        K = np.array([[F, 0, (cfg["anchor_x"] + shift_x) * W], [0, -F, cfg["anchor_y"] * H], [0, 0, 1]])
        return K @ M @ R


def apply_h(Hm, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:3]


class Hotkeys:
    """Globale Tasten per GetAsyncKeyState (funktioniert auch, wenn Forza den Fokus hat). Nur Windows."""

    def __init__(self):
        self.user32 = ctypes.windll.user32 if sys.platform == "win32" else None
        self.prev = {}

    def down(self, vk):
        return bool(self.user32 and self.user32.GetAsyncKeyState(vk) & 0x8000)

    def pressed(self, vk):
        d = self.down(vk)
        was, self.prev[vk] = self.prev.get(vk, False), d
        return d and not was


VK = dict(F8=0x77, F9=0x78, F10=0x79, LEFT=0x25, UP=0x26, RIGHT=0x27, DOWN=0x28,
          PGUP=0x21, PGDN=0x22, END=0x23, HOME=0x24, INS=0x2D, DEL=0x2E, SHIFT=0x10)


class Overlay(QtWidgets.QWidget):
    def __init__(self, geom, cfg, tel):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
                         | QtCore.Qt.Tool | QtCore.Qt.WindowTransparentForInput)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setGeometry(geom)
        self.cfg, self.tel = cfg, tel
        self.model = GlowModel(cfg)
        self.keys = Hotkeys()
        self.enabled, self.calib = True, False
        self.started = time.time()
        # geglätteter Zustand
        self.theta, self.zoom, self.shift, self.opacity = 0.0, 1.0, 0.0, 0.0
        self.corners = np.ones(4)
        self.tex, self.xform, self.outline = None, None, None
        self.dirty = QtCore.QRect()
        self.hud_text, self.hud_rect = "", QtCore.QRect(10, 10, 900, 130)
        self.last_count, self.rate, self.rate_t = 0, 0, time.time()

        self.timer = QtCore.QTimer(self, interval=16, timeout=self.tick)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.start()
        # Spiele, die nach dem Overlay starten, schieben sich gern davor -> regelmäßig nach vorne holen
        self.top_timer = QtCore.QTimer(self, interval=2000, timeout=self.keep_on_top)
        self.top_timer.start()

    def keep_on_top(self):
        if sys.platform == "win32":
            # HWND_TOPMOST, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
            ctypes.windll.user32.SetWindowPos(int(self.winId()), -1, 0, 0, 0, 0, 0x13)
        else:
            self.raise_()

    def handle_keys(self):
        k, cfg = self.keys, self.cfg
        if k.pressed(VK["F10"]):
            self.enabled = not self.enabled
        if k.pressed(VK["F9"]):
            i = COLORS.index(cfg["color"]) if cfg["color"] in COLORS else -1
            cfg["color"] = COLORS[(i + 1) % len(COLORS)]
            save_config(cfg)
        if k.pressed(VK["F8"]):
            self.calib = not self.calib
            if not self.calib:
                save_config(cfg)
        if not self.calib:
            return
        f = 0.2 if k.down(VK["SHIFT"]) else 1.0
        step = {"LEFT": ("anchor_x", -0.001), "RIGHT": ("anchor_x", 0.001),
                "UP": ("anchor_y", -0.001), "DOWN": ("anchor_y", 0.001),
                "PGUP": ("scale", 0.005), "PGDN": ("scale", -0.005),
                "HOME": ("cam_height", 0.02), "END": ("cam_height", -0.02),
                "INS": ("car_length", 0.01), "DEL": ("car_length", -0.01)}
        rebuild = False
        for key, (name, delta) in step.items():
            if k.down(VK[key]):
                cfg[name] = round(max(0.05, cfg[name] + delta * f), 4)
                rebuild |= name == "car_length"
        if rebuild:
            self.model = GlowModel(cfg)

    def tick(self):
        self.handle_keys()
        cfg, st, now = self.cfg, self.tel.state, time.time()
        live = st is not None and now - st["t"] < 0.5 and st["race_on"]

        theta, zoom, shift, corners = 0.0, 1.0, 0.0, np.ones(4)
        if live:
            vx, _, vz = st["vel"]
            speed = math.hypot(vx, vz)
            # Driftwinkel: Kamera folgt eher der Fahrtrichtung -> Auto steht schräg im Bild
            beta = math.atan2(vx, abs(vz)) * min(1.0, speed / 5)
            theta = -cfg["slip_follow"] * beta + cfg["yaw_lag"] * st["ang_vel"][1]
            ax, _, az = st["accel"]
            zoom = 1 - cfg["accel_zoom"] * max(-15, min(15, az))
            shift = cfg["lateral_shift"] * max(-15, min(15, ax))
            corners = np.clip(0.7 + 0.8 * np.array(st["susp"]), 0.3, 1.5)

        a = cfg["smoothing"]
        self.theta += (theta - self.theta) * a
        self.zoom += (zoom - self.zoom) * a
        self.shift += (shift - self.shift) * a
        self.corners += (corners - self.corners) * a
        target = 1.0 if self.enabled and (live or self.calib) else 0.0
        self.opacity += (target - self.opacity) * 0.15

        W, H = self.width(), self.height()
        rect = QtCore.QRect()
        if self.opacity > 0.01:
            Hm = self.model.homography(cfg, W, H, self.theta, self.zoom, self.shift)
            self.tex = self.model.texture(self.theta, self.corners, color_rgb(cfg["color"]),
                                          cfg["intensity"] * self.opacity)
            self.xform = QtGui.QTransform(*Hm.T.flatten())
            m = self.model
            corners_px = apply_h(Hm, [(0, 0), (m.tw, 0), (m.tw, m.th), (0, m.th)])
            lo, hi = corners_px.min(0), corners_px.max(0)
            rect = QtCore.QRect(int(lo[0]) - 4, int(lo[1]) - 4, int(hi[0] - lo[0]) + 8, int(hi[1] - lo[1]) + 8)
            self.outline = None
            if self.calib:
                Hg = GlowModel.ground_to_screen(cfg, W, H, self.theta, self.zoom, self.shift)
                self.outline = QtGui.QPolygonF([QtCore.QPointF(*p) for p in apply_h(Hg, m.outline)])
        else:
            self.tex = None
        # nur geänderte Bereiche neu zeichnen (Vollbild-Neuzeichnen wäre teuer)
        dirty = rect.united(self.dirty)
        if not dirty.isEmpty():
            self.update(dirty)
        self.dirty = rect
        self.update_hud(live, now)

    def update_hud(self, live, now):
        if now - self.rate_t >= 1:
            self.rate, self.last_count, self.rate_t = self.tel.count - self.last_count, self.tel.count, now
        cfg = self.cfg
        if self.calib:
            text = ("KALIBRIERUNG  (F8 = fertig & speichern)\n"
                    "Pfeiltasten: verschieben   Bild auf/ab: Größe   Pos1/Ende: Perspektive   Einfg/Entf: Länge\n"
                    "Shift = fein   |   Umriss = Fahrzeug-Fußabdruck, an die Reifen anlegen\n"
                    f"x={cfg['anchor_x']:.3f} y={cfg['anchor_y']:.3f} Größe={cfg['scale']:.3f} "
                    f"Kamerahöhe={cfg['cam_height']:.2f} Länge={cfg['car_length']:.2f}   Telemetrie: {self.rate}/s")
        elif self.rate == 0:
            text = (f"Warte auf Forza-Telemetrie an 127.0.0.1:{cfg['port']} ...\n"
                    "Forza: Einstellungen > HUD und Gameplay > Data Out = An, IP 127.0.0.1, Port wie oben")
        elif now - self.started < 8:
            text = f"Underglow aktiv ({self.rate} Pakete/s)   F8 kalibrieren   F9 Farbe   F10 an/aus"
        else:
            text = ""
        if text != self.hud_text:
            self.hud_text = text
            self.update(self.hud_rect)

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        if self.tex is not None:
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            p.setTransform(self.xform)
            p.drawImage(0, 0, self.tex)
            p.resetTransform()
        if self.calib and self.outline is not None:
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 2, QtCore.Qt.DashLine))
            p.drawPolygon(self.outline)
        if self.hud_text:
            p.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
            r = p.fontMetrics().boundingRect(self.hud_rect, 0, self.hud_text).adjusted(-8, -6, 8, 6)
            r.moveTopLeft(QtCore.QPoint(20, 20))
            p.fillRect(r, QtGui.QColor(0, 0, 0, 170))
            p.setPen(QtGui.QColor(0, 229, 255))
            p.drawText(r.adjusted(8, 6, -8, -6), 0, self.hud_text)


def main():
    ap = argparse.ArgumentParser(description="Underglow-Overlay für Forza Horizon (Telemetrie)")
    ap.add_argument("--screen", type=int, default=0, help="Bildschirm-Index (0 = primär)")
    args = ap.parse_args()
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Strg+C im Terminal beendet
    app = QtWidgets.QApplication(sys.argv)

    cfg = load_config()
    if not CONFIG_PATH.exists():
        save_config(cfg)
    try:
        tel = Telemetry(cfg["port"])
    except OSError as e:
        sys.exit(f"Port {cfg['port']} ist belegt ({e}). Läuft das Overlay schon?")
    tel.start()

    screens = app.screens()
    screen = screens[args.screen] if args.screen < len(screens) else app.primaryScreen()
    overlay = Overlay(screen.geometry(), cfg, tel)
    overlay.show()
    print(f"Underglow läuft (Telemetrie auf Port {cfg['port']}). Strg+C oder Fenster schließen zum Beenden.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
