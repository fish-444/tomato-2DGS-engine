#!/usr/bin/env bash
# Waits until OneDrive has the capture locally, then Steps 0-2 (photos, board poses, contract, masks, fixed-pose triangulation).
# 2026-10-21 23:10: the user moved the capture folder into the project root ("토마토 촬영/"); the local-count probe finds it by
# its file names (ASCII only) so the Korean folder name never passes through a PowerShell command line.
set -e
ROOT="<PROJECT_ROOT>"; cd "$ROOT"
SRC="$ROOT/토마토 촬영"
CH=build/p1_charuco_a3_20261021_01; RC=build/p1_recon_charuco_20261021_01; LOGF=$RC/run_to_sparse.log
PROBE='$d = Get-ChildItem -Directory | Where-Object { (Get-ChildItem $_.FullName -Filter "20261021_06*_iOS*.heic" -ErrorAction SilentlyContinue).Count -gt 50 } | Select-Object -First 1; if ($d) { (Get-ChildItem $d.FullName -Filter *.heic | Where-Object { ([int]$_.Attributes -band 0x400000) -eq 0 }).Count } else { -1 }'
for i in $(seq 1 480); do
  n=$(powershell -NoProfile -Command "$PROBE" | tr -d '\r')
  echo "$(date +%H:%M) local=$n" >> $LOGF
  [ "$n" -ge 85 ] && break
  sleep 60
done
echo "$(date +%H:%M) photos" >> $LOGF; python -X utf8 -m tomato3d.recon.charuco_photos --src "$SRC" --out $CH >> $LOGF 2>&1
echo "$(date +%H:%M) poses" >> $LOGF; python -X utf8 -m tomato3d.recon.charuco_a3 --jpg $CH/jpg --out $CH >> $LOGF 2>&1
echo "$(date +%H:%M) contract" >> $LOGF; python -X utf8 -m tomato3d.recon.charuco_contract --poses $CH --recon $RC >> $LOGF 2>&1
echo "$(date +%H:%M) masks" >> $LOGF; python -X utf8 -m tomato3d.recon.charuco_masks --poses $CH --out $RC >> $LOGF 2>&1
echo "$(date +%H:%M) sparse" >> $LOGF; python -X utf8 -m tomato3d.recon.charuco_sparse --poses $CH --recon $RC >> $LOGF 2>&1
echo "$(date +%H:%M) DONE" >> $LOGF
