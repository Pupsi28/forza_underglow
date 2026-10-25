@echo off
cd /d "%~dp0"
title Forza Underglow

rem uv installiert Python + alle Pakete automatisch (nur beim ersten Start, danach sofort)
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul || (
  echo Erster Start: installiere uv ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)

echo Starte Underglow (erster Start laedt ca. 3 GB herunter, das dauert etwas) ...
uv run underglow.py %*
pause
