#!/usr/bin/env python3
"""Step 2 - whole-plant masks. --method colour (charuco_masks: colour in the visual hull), wholeplant (wholeplant_masks: SAM stem step), r31 (r31_masks: the R3-1b steps M1-M5, stage a / b).
The remaining options are the chosen module's own: python scripts/02_masks.py --method r31 --help"""
import runpy, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


if __name__ == "__main__":
    m = {"colour": "charuco_masks", "wholeplant": "wholeplant_masks", "r31": "r31_masks"}
    k = sys.argv[sys.argv.index("--method") + 1] if "--method" in sys.argv else "colour"
    i = sys.argv.index("--method") if "--method" in sys.argv else None
    if i is not None:
        del sys.argv[i:i + 2]
    try:
        runpy.run_module("tomato3d.recon." + m[k], run_name="__main__")
    except ModuleNotFoundError as ex:
        if any(s in str(ex) for s in ("leafseg", "topview")):
            sys.exit("이 단계(SAM 로더 leafseg.run / topview.measure2d)의 모듈은 저장소에 없습니다 (docs/ONEZIP_DIAGNOSIS_KO.md). "
                     "같은 마스크 단계는 onezip/run_recon.py (--mask auto) 와 onezip/stem_masks.py (--mask auto+stem) 로 돌립니다.\n" + str(ex))
        raise
