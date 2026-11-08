# Detached local jobs. Background shells of the chat session stop when a turn ends; Start-Process children survive turn ends
# but not an app restart (seen 00:41), and WMI-created processes were stopped within seconds on this machine (tested 00:43),
# so jobs resume: finished views/stages are skipped on a relaunch.
# usage: powershell -ExecutionPolicy Bypass -File run_detached.ps1 -Job sparse|leaflets|sheets
# The project root is derived from this script's folder: PS 5.1 reads BOM-less scripts as ANSI, so no non-ASCII literal here.
param([string]$Job)
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $root
$ch = "build/p1_charuco_a3_20261021_01"; $rc = "build/p1_recon_charuco_20261021_01"
$work = "<SCRATCH_DIR>/scratchpad/colmap_work3"
switch ($Job) {
  "sparse"   { $args_ = @("-X", "utf8", "-m", "tomato3d.recon.charuco_sparse", "--poses", $ch, "--recon", $rc, "--work", $work) }
  "leaflets" { $args_ = @("-X", "utf8", "-m", "tomato3d.recon.charuco_leaflets", "--poses", $ch, "--recon", $rc) }
  "sheets"   { $args_ = @("-X", "utf8", "-m", "tomato3d.recon.charuco_sheets", "--poses", $ch, "--recon", $rc) }
}
$p = Start-Process -FilePath "python" -ArgumentList $args_ -WorkingDirectory $root -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput "$rc/$Job.out.log" -RedirectStandardError "$rc/$Job.err.log"
"$($p.Id)" | Out-File -Encoding ascii "$rc/$Job.pid"
"started $Job pid $($p.Id)"
