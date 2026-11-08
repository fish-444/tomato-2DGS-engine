# Detached local jobs for R3-1 (chat background shells stop at the end of a turn). Jobs resume: finished views are skipped.
# usage: powershell -ExecutionPolicy Bypass -File run_detached.ps1 -Job masks_a|masks_b|depth|sparse
param([string]$Job)
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $root
$ch = "build/p1_charuco_a3_20261021_01"; $v2 = "build/p1_recon_wholeplant_20261022_01"; $rw = "build/p1_recon_r3-1_20261022_01"
$work = "<SCRATCH_DIR>/scratchpad/colmap_r31"
switch ($Job) {
  "masks_a" { $args_ = @("-X", "utf8", "-m", "tomato3d.recon.r31_masks", "--poses", $ch, "--v2", $v2, "--out", $rw, "--stage", "a") }
  "masks_b" { $args_ = @("-X", "utf8", "-m", "tomato3d.recon.r31_masks", "--poses", $ch, "--v2", $v2, "--out", $rw, "--stage", "b", "--renders", "$rw/renders_base") }
  "depth"   { $args_ = @("-X", "utf8", "-m", "tomato3d.recon.r31_depth", "--poses", $ch, "--out", $rw) }
  "sparse"  { $args_ = @("-X", "utf8", "-m", "tomato3d.recon.r31_sparse", "--poses", $ch, "--out", $rw, "--work", $work, "--masks", "masks_a", "--tag", "a") }
}
$p = Start-Process -FilePath "python" -ArgumentList $args_ -WorkingDirectory $root -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput "$rw/$Job.out.log" -RedirectStandardError "$rw/$Job.err.log"
"$($p.Id)" | Out-File -Encoding ascii "$rw/$Job.pid"
"started $Job pid $($p.Id)"
