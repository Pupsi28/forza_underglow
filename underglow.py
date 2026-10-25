# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "ultralytics>=8.3.0", "mss>=9.0", "PySide6>=6.5", "opencv-python>=4.8", "numpy",
#   "torch", "torchvision", "dxcam; sys_platform == 'win32'",
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
    p.add_argument("--all-cars", action="store_true", help="Alle Autos beleuchten, nicht nur das eigene")
    p.add_argument("--model", default="yolo11n-seg.pt", help="YOLO-Segmentierungsmodell")
    p.add_argument("--conf", type=float, default=0.25, help="Erkennungsschwelle")
    p.add_argument("--imgsz", type=int, default=416, help="Eingangsgröße der Erkennung (kleiner = schneller)")
    p.add_argument("--monitor", type=int, default=1, help="Monitor-Index (1 = primär)")
    p.add_argument("--work-width", type=int, default=640, help="Interne Auflösung")
    p.add_argument("--smooth", type=float, default=0.5, help="Zeitliche Glättung 0..1 (kleiner = ruhiger)")
    p.add_argument("--max-fps", type=float, default=60)
    p.add_argument("--no-hud", action="store_true", help="Statusanzeige oben links ausblenden")
    return p.parse_args()


def hex_rgb(s):
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


class GlowRenderer:
    """Rechnet mit halber Arbeitsauflösung und nur in einem Ausschnitt um jedes Auto (schnell)."""

    def __init__(self, args, work_size):
        self.args = args
        self.scale = 0.5
        self.w, self.h = int(work_size[0] * self.scale), int(work_size[1] * self.scale)
        self.acc = np.zeros((self.h, self.w), np.float32)
        self.rainbow = args.color.lower() == "rainbow"
        self.rgb = None if self.rainbow else hex_rgb(args.color)

    def car_glow(self, poly, glow, occl):
        """Lichtschein + Verdeckung für ein Auto in glow/occl eintragen."""
        poly = (poly * self.scale).astype(np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        if w < 6 or h < 4:
            return
        size = max(w, h) * self.args.spread
        m = int(size * 0.4) + 2
        x0, y0, x1, y1 = max(x - m, 0), max(y - m, 0), min(x + w + m, self.w), min(y + h + m, self.h)
        rw, rh, lx, ly = x1 - x0, y1 - y0, x - x0, y - y0

        mask = np.zeros((rh, rw), np.uint8)
        cv2.fillPoly(mask, [poly - (x0, y0)], 1)

        # Unterkante: unterster Maskenpixel pro Spalte
        sub = mask[ly:ly + h, lx:lx + w]
        cols = np.where(sub.any(0))[0]
        if len(cols) < 3:
            return
        bottom = h - 1 - np.argmax(sub[::-1, cols], axis=0)
        pts = np.stack([cols + lx, bottom + ly], 1)
        pts = pts[pts[:, 1] > ly + 0.5 * h]  # nur echte Unterkante, keine Dachlinie
        if len(pts) < 3:
            return

        foot = np.zeros((rh, rw), np.float32)
        cv2.fillConvexPoly(foot, cv2.convexHull(pts.astype(np.int32)), 1.0)
        # Lichtpool leicht nach unten versetzt (liegt auf dem Boden statt am Blech),
        # horizontal breiter als vertikal (Bodenebene ist perspektivisch verkürzt)
        pool = np.roll(foot, int(size * 0.025), axis=0)
        g = cv2.GaussianBlur(pool, (0, 0), sigmaX=size * 0.09, sigmaY=size * 0.05) * 2.4
        g += cv2.GaussianBlur(foot, (0, 0), sigmaX=size * 0.02) * 0.6
        # Heller LED-Saum direkt an der Unterkante
        strip = np.zeros_like(foot)
        cv2.polylines(strip, [pts.reshape(-1, 1, 2).astype(np.int32)], False, 1.0,
                      max(1, int(size * 0.012)))
        g += cv2.GaussianBlur(strip, (0, 0), max(size * 0.01, 0.5)) * 0.7

        # Karosserie verdeckt das Licht – nach unten hin (Schweller) lässt sie es durch
        ramp = np.clip((np.arange(rh, dtype=np.float32) - (ly + 0.65 * h)) / (0.35 * h), 0, 1)
        o = mask.astype(np.float32) * (1 - 0.45 * ramp)[:, None]
        np.maximum(glow[y0:y1, x0:x1], g, out=glow[y0:y1, x0:x1])
        np.maximum(occl[y0:y1, x0:x1], o, out=occl[y0:y1, x0:x1])

    def render(self, polys):
        """Liefert (RGBA-Bild, geänderter Bereich als (x0, y0, x1, y1) relativ 0..1 oder None)."""
        glow = np.zeros((self.h, self.w), np.float32)
        occl = np.zeros_like(glow)
        for poly in polys:
            self.car_glow(poly, glow, occl)
        glow = np.clip(glow, 0, 1) * (1 - 0.95 * occl)

        a = self.args.smooth
        self.acc = self.acc * (1 - a) + glow * a
        self.acc[self.acc < 0.004] = 0

        rgb = self.rgb
        if self.rainbow:
            rgb = np.array(colorsys.hsv_to_rgb((time.time() * 0.1) % 1, 1, 1), np.float32)
        alpha = self.acc * self.args.intensity
        out = np.empty((self.h, self.w, 4), np.uint8)
        out[..., :3] = (alpha[..., None] * rgb * 255).astype(np.uint8)  # premultiplied
        out[..., 3] = (alpha * 255).astype(np.uint8)

        rows, cols = np.where(self.acc.any(1))[0], np.where(self.acc.any(0))[0]
        rect = None
        if len(rows):
            rect = (cols[0] / self.w, rows[0] / self.h, (cols[-1] + 1) / self.w, (rows[-1] + 1) / self.h)
        return out, rect


class PlayerPicker:
    """Wählt das Spieler-Auto: in der Verfolgerkamera groß, mittig, unten im Bild.
    Einmal gefunden, folgt die Auswahl dem Auto (Überlappung mit dem letzten Frame)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.last = None  # letzte Box (x0, y0, x1, y1)
        self.lost = 0

    @staticmethod
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0

    def pick(self, boxes):
        best, best_score = None, 0.0
        for i, b in enumerate(boxes):
            bw, bh = b[2] - b[0], b[3] - b[1]
            cx, bottom = (b[0] + b[2]) / 2 / self.w, b[3] / self.h
            area = bw * bh / (self.w * self.h)
            # Verfolgerkamera-Zone: horizontal mittig, untere Bildhälfte, nicht winzig
            score = 0.0
            if abs(cx - 0.5) < 0.3 and bottom > 0.55 and bw / self.w > 0.1:
                score = area * (1.5 - abs(cx - 0.5)) * bottom
            if self.last is not None:
                score += self.iou(b, self.last) * 2  # Kontinuität schlägt Position
            if score > best_score:
                best, best_score = i, score
        if best is None:
            self.lost += 1
            if self.lost > 15:  # ~0.25 s weg -> neu suchen
                self.last = None
            return None
        self.last, self.lost = boxes[best], 0
        return best


class Capture:
    """Windows: DXGI Desktop Duplication (dxcam, läuft auf der GPU, sehr schnell).
    Sonst / Fallback: mss."""

    def __init__(self, monitor, fps):
        self.cam = None
        if sys.platform == "win32":
            try:
                import dxcam
                self.cam = dxcam.create(output_idx=monitor - 1, output_color="BGR")
                self.cam.start(target_fps=int(fps), video_mode=True)
                if self.cam.get_latest_frame() is None:
                    raise RuntimeError("kein Bild")
                self.name = "DXGI"
            except Exception as e:  # noqa: BLE001
                print(f"dxcam nicht verfügbar ({e}), nutze mss")
                self.cam = None
        if self.cam is None:
            self.sct = mss.mss()
            self.mon = self.sct.monitors[monitor]
            self.name = "GDI"

    def grab(self):
        if self.cam is not None:
            return self.cam.get_latest_frame()  # wartet auf das nächste Bild
        return np.asarray(self.sct.grab(self.mon))[..., :3]


class Bridge(QtCore.QObject):
    frame = QtCore.Signal(QtGui.QImage, object)
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
        self.dirty = QtCore.QRect()
        self.hud = hud
        self.hud_rect = QtCore.QRect(0, 0, 1600, 70)
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

    def set_image(self, img, rect):
        # Nur den Bereich neu zeichnen, in dem Licht war oder ist – ein Vollbild-Neuzeichnen
        # des transparenten Fensters wäre der teuerste Schritt.
        self.img = img
        r = QtCore.QRect()
        if rect is not None:
            W, H = self.width(), self.height()
            r = QtCore.QRect(QtCore.QPoint(int(rect[0] * W), int(rect[1] * H)),
                             QtCore.QPoint(int(rect[2] * W), int(rect[3] * H))).adjusted(-4, -4, 4, 4)
        dirty = r.united(self.dirty)
        if not dirty.isEmpty():
            self.update(dirty)
        self.dirty = r

    def set_status(self, text):
        self.status = text
        if self.hud:
            self.update(self.hud_rect)

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
    cuda = _cuda()
    device = "GPU" if cuda else "CPU (langsam)"
    cap = Capture(args.monitor, args.max_fps)
    shot = cap.grab()
    ww = args.work_width
    wh = int(shot.shape[0] * ww / shot.shape[1])
    renderer = GlowRenderer(args, (ww, wh))
    picker = PlayerPicker(ww, wh)
    frame_dt = 1 / args.max_fps
    t_fps, n = time.time(), 0
    t_cap = t_det = t_glow = 0.0
    while not stop.is_set():
        t0 = time.perf_counter()
        shot = cap.grab()
        if shot is None:
            continue
        black = shot[::32, ::32].mean() < 3
        frame = cv2.resize(shot, (ww, wh), interpolation=cv2.INTER_LINEAR)
        t1 = time.perf_counter()

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, classes=VEHICLES,
                            half=cuda, verbose=False)[0]
        polys = []
        if res.masks is not None:
            xy = res.masks.xy
            if args.all_cars:
                polys = [p for p in xy if len(p) >= 3]
            else:
                i = picker.pick(res.boxes.xyxy.cpu().numpy())
                if i is not None and len(xy[i]) >= 3:
                    polys = [xy[i]]
        elif not args.all_cars:
            picker.pick([])
        t2 = time.perf_counter()

        out, rect = renderer.render(polys)
        img = QtGui.QImage(out.data, renderer.w, renderer.h, renderer.w * 4,
                           QtGui.QImage.Format_RGBA8888_Premultiplied).copy()
        bridge.frame.emit(img, rect)
        t3 = time.perf_counter()
        t_cap, t_det, t_glow = t_cap + t1 - t0, t_det + t2 - t1, t_glow + t3 - t2

        n += 1
        if time.time() - t_fps > 1:
            if black:
                msg = "Bildschirmaufnahme ist schwarz - Forza auf 'Randloses Fenster' stellen"
            else:
                ms = lambda t: f"{t / n * 1000:.0f}"  # noqa: E731
                who = "alle Autos" if args.all_cars else ("Spieler-Auto" if polys else "kein Spieler-Auto")
                msg = (f"Underglow | {device} | {n / (time.time() - t_fps):.0f} FPS | {who} | "
                       f"Aufnahme ({cap.name}) {ms(t_cap)} ms, Erkennung {ms(t_det)} ms, Glow {ms(t_glow)} ms")
            bridge.status.emit(msg)
            print("\r" + msg + "     ", end="", flush=True)
            t_fps, n = time.time(), 0
            t_cap = t_det = t_glow = 0.0
        if cap.cam is None:  # dxcam taktet selbst
            time.sleep(max(0.0, frame_dt - (time.perf_counter() - t0)))


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
