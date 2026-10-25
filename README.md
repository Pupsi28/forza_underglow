# Forza Underglow Overlay

Erkennt Autos auf dem Bildschirm (YOLO-Segmentierung) und legt in Echtzeit eine
Unterbodenbeleuchtung als transparentes, klick-durchlässiges Overlay darüber.

## Start
- **Windows:** `run.bat` doppelklicken
- **Linux:** `./run.sh`

Beim ersten Start werden die Abhängigkeiten installiert und das Modell (~6 MB) geladen.
Forza im **randlosen Fenstermodus** laufen lassen (Vollbild-exklusiv verdeckt Overlays).
Beenden: Strg+C im Terminal.

## Optionen
```
run.bat --color "#ff00aa"      # Farbe
run.bat --color rainbow        # Farbwechsel
run.bat --intensity 0.6 --spread 1.3
run.bat --model yolo11s-seg.pt # genauer, langsamer
```

## Wie es funktioniert
Aus der Segmentierungsmaske wird die Unterkante des Autos (Reifen/Schweller) ermittelt.
Deren konvexe Hülle ≈ Fußabdruck auf dem Boden → funktioniert in Seiten-, Heck- und
Schrägansicht. Der Schein wird weichgezeichnet, von der Karosserie verdeckt und zeitlich
geglättet, damit er bei Kamerabewegungen nicht flackert.

## Hinweise
- **GPU:** Für flüssige FPS eine CUDA-Version von PyTorch installieren
  (`.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124`).
  Auf CPU läuft es, aber eher mit 5–15 FPS.
- Unter Windows wird das Overlay aus der Bildschirmaufnahme ausgeblendet. Unter Linux geht das
  nicht – dort sieht die Erkennung ihren eigenen Schein (meist unkritisch). Linux braucht X11
  mit Compositor.
- Reines Screen-Capture-Overlay, kein Eingriff in den Spielprozess.
# forza_underglow
