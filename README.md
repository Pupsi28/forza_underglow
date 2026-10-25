# Forza Underglow Overlay

Erkennt Autos auf dem Bildschirm (YOLO-Segmentierung) und legt in Echtzeit eine
Unterbodenbeleuchtung als transparentes, klick-durchlässiges Overlay darüber.

## Start
- **Windows:** `run.bat` doppelklicken
- **Linux:** `./run.sh`

Es muss **nichts vorher installiert werden**. `run.bat` holt sich beim ersten Start
[uv](https://docs.astral.sh/uv/), das wiederum Python, PyTorch (mit GPU-Support) und das
Erkennungsmodell herunterlädt (einmalig ca. 3 GB). Danach startet es in Sekunden.

1. Forza starten, in den Grafikeinstellungen **Randloses Fenster** statt Vollbild wählen.
2. `run.bat` starten.
3. Oben links erscheint eine Statuszeile (FPS, Anzahl erkannter Autos). Ausblenden mit `--no-hud`.

Beenden: Fenster schließen oder Strg+C im Terminal.

### Fehlersuche
| Statuszeile | Bedeutung |
|---|---|
| nicht sichtbar | Forza läuft im exklusiven Vollbild → auf randloses Fenster umstellen |
| „Bildschirmaufnahme ist schwarz" | dito |
| „0 Auto(s)" | Erkennung greift nicht → `run.bat --conf 0.15` oder `--model yolo11s-seg.pt` |
| „CPU (langsam)" | keine NVIDIA-GPU gefunden, läuft trotzdem, nur mit weniger FPS |

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
- **GPU:** Unter Windows wird automatisch PyTorch mit CUDA installiert (NVIDIA). Ohne NVIDIA-GPU
  läuft es auf der CPU mit ca. 5–15 FPS.
- Unter Windows wird das Overlay aus der Bildschirmaufnahme ausgeblendet. Unter Linux geht das
  nicht – dort sieht die Erkennung ihren eigenen Schein (meist unkritisch). Linux braucht X11
  mit Compositor.
- Reines Screen-Capture-Overlay, kein Eingriff in den Spielprozess.
# forza_underglow
