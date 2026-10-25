# Forza Underglow Overlay

Unterbodenbeleuchtung für Forza Horizon 6 als transparentes Overlay. Statt Bilderkennung
nutzt es Forzas **Data-Out-Telemetrie**: läuft ohne Verzögerung mit 60 FPS, flackert nicht
und braucht praktisch keine GPU-Leistung.

## Einrichtung (einmalig)
1. In Forza: **Einstellungen → HUD und Gameplay → Data Out = An**,
   IP `127.0.0.1`, Port `1306`.
2. Forza im **randlosen Fenstermodus** spielen (exklusives Vollbild verdeckt Overlays).
3. `run.bat` starten. Beim ersten Mal lädt es automatisch Python + Pakete (~100 MB),
   vorher muss nichts installiert werden.
4. Im Spiel in der Verfolgerkamera **F8** drücken und den gestrichelten Umriss an die Reifen
   anlegen (siehe unten). **F8** speichert.

## Tasten (funktionieren, während Forza im Vordergrund ist)
| Taste | Funktion |
|---|---|
| F8 | Kalibrierung an/aus (speichert beim Verlassen) |
| F9 | Farbe wechseln (inkl. Regenbogen) |
| F10 | Underglow an/aus (z. B. für Cockpit-Ansicht) |

Im Kalibriermodus: **Pfeiltasten** verschieben, **Bild auf/ab** Größe, **Pos1/Ende**
Perspektive, **Einfg/Entf** Fahrzeuglänge, **Shift** gedrückt halten für feine Schritte.
Am besten stehend kalibrieren, damit die Pfeiltasten nicht das Auto lenken.

## Wie es funktioniert
In der Verfolgerkamera steht das eigene Auto immer an ungefähr derselben Bildschirmstelle.
Der Lichtschein wird auf einer virtuellen Bodenebene berechnet und perspektivisch projiziert.
Die Telemetrie bewegt ihn mit:
- **Driftwinkel** → Auto und Licht drehen sich im Bild
- **Federweg pro Rad** → eingefederte Ecken leuchten heller
- **Beschleunigen/Bremsen** → Kamera fällt zurück/holt auf
- Seiten, die von der Kamera wegzeigen, verdeckt die Karosserie

## Feintuning (`config.json`, wird beim ersten Start angelegt)
- `color`, `intensity`, `glow_reach`: Farbe, Helligkeit, Reichweite des Lichts
- `slip_follow`: wie stark sich der Schein beim Driften dreht
- `yaw_lag`, `lateral_shift`: Kamera-Nachziehen in Kurven (Standard aus; falls genutzt
  und es in die falsche Richtung geht, Vorzeichen umdrehen)
- `smoothing`: kleiner = ruhiger, größer = direkter
- `port`: falls in Forza ein anderer Port eingestellt ist

## Grenzen
- Gedacht für die **Verfolgerkamera**. In Cockpit-/Motorhaubenansicht mit F10 ausschalten.
  Fotomodus/freie Kamera funktioniert nicht (die Telemetrie kennt die Kameraposition nicht).
- Fahrzeuge unterschiedlicher Größe sitzen evtl. leicht anders → kurz mit F8 nachjustieren.
- Linux: läuft, aber ohne globale Tasten (Einstellungen dann direkt in `config.json`).
