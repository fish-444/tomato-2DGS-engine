# Third-party components

This repository's own code is MIT (see `LICENSE`). It uses, and in some cases was written against, the components below. They are **referenced, not copied**: no third-party source is in this repository.
Licence names are the ones the projects state; check the upstream licence before redistributing anything built with them.

| Component | Used for | Where | Licence (as stated upstream) |
|---|---|---|---|
| gsplat 1.5.3 | 2DGS rasterisation and training | `src/tomato3d/recon/kernel_2dgs_r31.py` (installed at run time) | Apache-2.0 |
| SAM 2 / SAM 2.1 (`sam2==1.1.0`, `sam2.1_hiera_tiny.pt`) | mask steps, candidate masks | `onezip/run_recon.py`, `onezip/stem_masks.py`, `organs/all3d_cands_job.py` | Apache-2.0 |
| pycolmap 4.2.1 | initial points: SIFT matching and triangulation with the board poses held fixed (no pose estimation) | `recon/wholeplant_sparse.py`, `recon/charuco_sparse.py` | BSD-3-Clause |
| OpenCV (`opencv-contrib-python`) | ChArUco detection, undistortion, image operations | `recon/charuco_*.py` and others | Apache-2.0 |
| **Ultralytics** | imported by `organs/all3d_cands_job.py` (candidate masks) only; not needed for stage 7 or the scoring | optional, install only for that step | **AGPL-3.0** (a separate commercial licence exists). Running that step with it installed makes your use subject to the AGPL terms. |
| NumPy, SciPy, scikit-learn, scikit-image, Pillow, pillow-heif, matplotlib, psutil, skan | numerical work, image IO | everywhere | BSD / HPND / MIT-style (pillow-heif links libheif, LGPL) |
| PyTorch | trainer and SAM | `recon/kernel_2dgs_r31.py`, `onezip/` | BSD-3-Clause |
| hydra-core 1.3.8, iopath 0.1.10 | SAM 2 dependencies | `onezip/run_recon.py` | MIT / Apache-2.0 |
| Depth Anything V2 Small | **not used** in this repository (the R3-1b reconstruction had depth-based initial points; they are not part of the shipped path, see `docs/ONEZIP_DIAGNOSIS_KO.md`) | - | Apache-2.0 |

Model weights (SAM 2.1 tiny fine-tuned by the authors; the M1 instance model) are **not** in the repository: names, sizes and sha256 are in `DATA.md`.
Not included on purpose: files derived from the VTC fork (`vtc_growth_fork.py`, `vtc_development.py`), whose permission is still open; any code that needs an API key.

