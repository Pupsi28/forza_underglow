# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "ultralytics>=8.3.0", "mss>=9.0", "PySide6>=6.5", "opencv-python>=4.8", "numpy",
#   "torch", "torchvision",
# ]
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]
# torchvision = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///
"""Forza Underglow Overlay.

Erfasst den Bildschirm, segmentiert Autos mit YOLO und zeichnet eine
Unterbodenbeleuchtung als transparentes, klick-durchlässiges Overlay.

Perspektiven-Trick: Statt einer festen Ellipse wird aus der Segmentierungsmaske
die Unterkante des Autos (Reifen/Schweller) bestimmt. Deren konvexe Hülle ist
eine gute Näherung des Fußabdrucks auf dem Boden – egal ob Seiten-, Heck- oder
Schrägansicht. Die Karosserie verdeckt den Lichtschein von oben.
"""
import argparse
import colorsys
import ctypes
import signal
import sys
import threading
import time

import cv2
import mss
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO

VEHICLES = [2, 3, 7]  # COCO: car, motorcycle, truck


def parse_args():
    p = argparse.ArgumentParser(description="Underglow-Overlay für Forza Horizon")
    p.add_argument("--color", default="#00e5ff", help="Hex-Farbe oder 'rainbow'")
    p.add_argument("--intensity", type=float, default=0.85, help="0..1")
    p.add_argument("--spread", type=float, default=1.0, help="Größe des Lichtscheins")
    p.add_argument("--model", default="yolo11n-seg.pt", help="YOLO-Segmentierungsmodell")
    p.add_argument("--conf", type=float, default=0.25, help="Erkennungsschwelle")
    p.add_argument("--monitor", type=int, default=1, help="Monitor-Index (mss, 1 = primär)")
    p.add_argument("--work-width", type=int, default=640, help="Interne Auflösung")
    p.add_argument("--smooth", type=float, default=0.45, help="Zeitliche Glättung 0..1 (kleiner = ruhiger)")
    p.add_argument("--max-fps", type=float, default=60)
    p.add_argument("--no-hud", action="store_true", help="Statusanzeige oben links ausblenden")
    return p.parse_args()


def hex_rgb(s):
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


class GlowRenderer:
    def __init__(self, args, size):
        self.args = args
        self.w, self.h = size
        self.acc = np.zeros((self.h, self.w), np.float32)
        self.rainbow = args.color.lower() == "rainbow"
        self.rgb = None if self.rainbow else hex_rgb(args.color)

    def car_glow(self, poly):
        """Lichtschein + Verdeckung für ein einzelnes Auto (Polygon in Arbeitskoordinaten)."""
        mask = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
        if w < 12 or h < 8:
            return None, None

        # Unterkante: unterster Maskenpixel pro Spalte
        sub = mask[y:y + h, x:x + w]
        cols = np.where(sub.any(0))[0]
        bottom = h - 1 - np.argmax(sub[::-1, cols], axis=0)
        pts = np.stack([cols + x, bottom + y], 1)
        pts = pts[pts[:, 1] > y + 0.5 * h]  # nur echte Unterkante, keine Dachlinie
        if len(pts) < 3:
            return None, None

        size = max(w, h) * self.args.spread
        foot = np.zeros((self.h, self.w), np.float32)
        cv2.fillConvexPoly(foot, cv2.convexHull(pts.astype(np.int32)), 1.0)
        # Lichtschein: horizontal breiter als vertikal (Bodenebene ist verkürzt)
        # Lichtpool leicht nach unten versetzt, damit er auf dem Boden statt am Blech liegt
        pool = np.roll(foot, int(size * 0.025), axis=0)
        glow = cv2.GaussianBlur(pool, (0, 0), sigmaX=size * 0.09, sigmaY=size * 0.05) * 2.4
        glow += cv2.GaussianBlur(foot, (0, 0), sigmaX=size * 0.02) * 0.6
        # Heller LED-Saum direkt an der Unterkante
        strip = np.zeros_like(foot)
        cv2.polylines(strip, [pts.reshape(-1, 1, 2).astype(np.int32)], False, 1.0,
                      max(1, int(size * 0.012)))
        glow += cv2.GaussianBlur(strip, (0, 0), size * 0.01) * 0.7

        # Karosserie verdeckt das Licht – nach unten hin (Schweller) lässt sie es durch
        ramp = np.clip((np.arange(self.h, dtype=np.float32) - (y + 0.65 * h)) / (0.35 * h), 0, 1)
        occl = mask.astype(np.float32) * (1 - 0.45 * ramp)[:, None]
        return glow, occl

    def render(self, polys):
        glow = np.zeros((self.h, self.w), np.float32)
        occl = np.zeros_like(glow)
        for poly in polys:
            g, o = self.car_glow(poly)
            if g is not None:
                np.maximum(glow, g, out=glow)
                np.maximum(occl, o, out=occl)
        glow = np.clip(glow, 0, 1) * (1 - 0.95 * occl)

        a = self.args.smooth
        self.acc = self.acc * (1 - a) + glow * a

        rgb = self.rgb
        if self.rainbow:
            rgb = np.array(colorsys.hsv_to_rgb((time.time() * 0.1) % 1, 1, 1), np.float32)
        alpha = self.acc * self.args.intensity
        out = np.empty((self.h, self.w, 4), np.uint8)
        out[..., :3] = (alpha[..., None] * rgb * 255).astype(np.uint8)  # premultiplied
        out[..., 3] = (alpha * 255).astype(np.uint8)
        return out


class Bridge(QtCore.QObject):
    frame = QtCore.Signal(QtGui.QImage)
    status = QtCore.Signal(str)


class Overlay(QtWidgets.QWidget):
    def __init__(self, geom, hud=True):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
                         | QtCore.Qt.Tool | QtCore.Qt.WindowTransparentForInput)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setGeometry(geom)
        self.img = None
        self.hud = hud
        self.status = "Lade Modell ..."
        # Spiele, die nach dem Overlay starten, schieben sich gern davor -> regelmäßig nach vorne holen
        self.topmost_timer = QtCore.QTimer(self, interval=2000, timeout=self.keep_on_top)
        self.topmost_timer.start()

    def keep_on_top(self):
        if sys.platform == "win32":
            # HWND_TOPMOST, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
            ctypes.windll.user32.SetWindowPos(int(self.winId()), -1, 0, 0, 0, 0, 0x13)
        else:
            self.raise_()

    def exclude_from_capture(self):
        # Windows 10 2004+: Overlay unsichtbar für die eigene Bildschirmaufnahme
        # (verhindert, dass YOLO den eigenen Lichtschein sieht).
        if sys.platform == "win32":
            ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x11)

    def set_image(self, img):
        self.img = img
        self.update()

    def set_status(self, text):
        self.status = text
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        if self.img is not None:
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            p.drawImage(self.rect(), self.img)
        if self.hud:
            p.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
            r = p.fontMetrics().boundingRect(self.status).adjusted(-8, -4, 8, 4).translated(20, 30)
            p.fillRect(r, QtGui.QColor(0, 0, 0, 160))
            p.setPen(QtGui.QColor(0, 229, 255))
            p.drawText(r, QtCore.Qt.AlignCenter, self.status)


def worker(args, bridge, stop):
    model = YOLO(args.model)
    device = "GPU" if _cuda() else "CPU (langsam)"
    with mss.mss() as sct:
        mon = sct.monitors[args.monitor]
        ww = args.work_width
        wh = int(mon["height"] * ww / mon["width"])
        renderer = GlowRenderer(args, (ww, wh))
        frame_dt = 1 / args.max_fps
        t_fps, n = time.time(), 0
        while not stop.is_set():
            t0 = time.time()
            shot = np.asarray(sct.grab(mon))[..., :3]
            black = shot[::16, ::16].mean() < 3
            frame = cv2.resize(shot, (ww, wh), interpolation=cv2.INTER_AREA)
            res = model.predict(frame, imgsz=ww, conf=args.conf, classes=VEHICLES, verbose=False)[0]
            polys = [p for p in res.masks.xy if len(p) >= 3] if res.masks is not None else []
            out = renderer.render(polys)
            img = QtGui.QImage(out.data, ww, wh, ww * 4, QtGui.QImage.Format_RGBA8888_Premultiplied).copy()
            bridge.frame.emit(img)

            n += 1
            if time.time() - t_fps > 1:
                if black:
                    msg = "Bildschirmaufnahme ist schwarz - Forza auf 'Randloses Fenster' stellen"
                else:
                    msg = f"Underglow aktiv | {device} | {n / (time.time() - t_fps):.0f} FPS | {len(polys)} Auto(s)"
                bridge.status.emit(msg)
                print("\r" + msg + "     ", end="", flush=True)
                t_fps, n = time.time(), 0
            time.sleep(max(0.0, frame_dt - (time.time() - t0)))


def _cuda():
    import torch
    return torch.cuda.is_available()


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Strg+C im Terminal beendet
    app = QtWidgets.QApplication(sys.argv)

    with mss.mss() as sct:
        mon = sct.monitors[args.monitor]
    dpr = app.primaryScreen().devicePixelRatio()  # mss liefert physische Pixel
    geom = QtCore.QRect(*(int(round(mon[k] / dpr)) for k in ("left", "top", "width", "height")))

    overlay = Overlay(geom, hud=not args.no_hud)
    overlay.show()
    overlay.exclude_from_capture()

    bridge = Bridge()
    bridge.frame.connect(overlay.set_image)
    bridge.status.connect(overlay.set_status)
    stop = threading.Event()
    threading.Thread(target=worker, args=(args, bridge, stop), daemon=True).start()
    print("Underglow läuft – Strg+C im Terminal zum Beenden.")
    code = app.exec()
    stop.set()
    sys.exit(code)


if __name__ == "__main__":
    main()
