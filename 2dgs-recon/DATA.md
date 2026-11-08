# DATA.md — 저장소에 넣지 않은 자료 (microtom-2dgs-recon)

원자료·대용량 산출물은 올리지 않고 여기 목록만 둡니다. 공유: **요청 시 (on request)** — 저자에게 연락.
경로는 원 프로젝트 기준 (`<PROJECT_ROOT>/...`). 폴더 행은 파일별 sha256 을 계산하지 않았음 (OneDrive 클라우드 파일을 내려받지 않기 위해); 필요하면 요청 시 매니페스트 제공.

## 외부 의존 (저장소 밖)

| 항목 | 위치 | 비고 |
|---|---|---|
| 원본 사진 (HEIC/JPG), 마스크, 사람 라벨, 골격 클릭 로그, held-out pick | `data/records/`, `<PHOTO_ROOT>`, `build/p1_all_3dfirst_20261027_01/user_*_click.jsonl` | 공유하지 않음 (프롬프트 §2) |
| SAM 2.1 미세조정 가중치 `sam2.1_hiera_tiny_leaflet_v*` | Kaggle 데이터셋 | 요청 시 |

## 목록 (크기순)

| 경로 | 크기 | sha256 / 파일 수 | 만든 폴더 (결과) | 이유 |
|---|---|---|---|---|
| `build/p1_recon_charuco_20261021_01/sparse/` | 305.8 MB | 폴더, 14 파일 | ChArUco pose (board 10x7, 38 mm) | excluded dir (outputs/uploads/raw/cache), 14 files [.txt:8,.bin:5,.db:1] |
| `build/mtg_build_20261102_01/kaggle/results_print/out/` | 236.7 MB | 폴더, 365 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 365 files [.obj:179,.csv:179,.log:5,.txt:1,.json |
| `build/mtg_build_20261102_01/kaggle/results_import_dtheta/out/` | 236.6 MB | 폴더, 365 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 365 files [.obj:179,.csv:179,.log:5,.txt:1,.json |
| `build/mtg_build_20261102_01/kaggle/results_import/out/` | 216.6 MB | 폴더, 231 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 231 files [.obj:113,.csv:113,.log:3,.txt:1,.json |
| `build/p1_recon_r3-1_20261022_01/depth/` | 187.6 MB | 폴더, 61 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 61 files [.npz:61] |
| `build/p1_recon_all_20261025_01/outputs/` | 164.2 MB | 폴더, 53 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 53 files [.ply:34,.json:10,.html:9] |
| `build/traits_recon_20261101_01/kaggle_out/` | 116.0 MB | 폴더, 41 파일 | traits + MTG templates (v1..v2.4, curvature/span/arrangement v1) | excluded dir (outputs/uploads/raw/cache), 41 files [.py:13,.csv:12,.npz:10,.json:3,.txt:2] |
| `build/traits_recon_20261101_01/kaggle_data/` | 115.5 MB | 폴더, 36 파일 | traits + MTG templates (v1..v2.4, curvature/span/arrangement v1) | excluded dir (outputs/uploads/raw/cache), 36 files [.py:13,.npz:10,.csv:10,.json:2,.txt:1] |
| `build/p1_recon_r3-1_20261022_01/_sparse_a/` | 110.1 MB | 폴더, 78 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 78 files [.png:61,.txt:8,.bin:5,.json:3,.ply:1] |
| `build/p1_tprox_others_20261101_01/runs/` | 98.5 MB | 폴더, 234 파일 | T-Prox 11 plants | excluded dir (outputs/uploads/raw/cache), 234 files [.json:120,.npz:87,.py:15,.log:12] |
| `build/mtg_build_20261102_01/kaggle/tri_ds/` | 75.4 MB | 폴더, 46 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 46 files [.csv:44,.json:2] |
| `build/p1_tprox_sam2_20261107_01/kaggle_others/` | 72.6 MB | 폴더, 58 파일 | T-Prox + SAM 2 arm A (0.844) | excluded dir (outputs/uploads/raw/cache), 58 files [.npz:44,.json:12,.py:2] |
| `build/p1_all_3dfirst_20261027_01/e_03/` | 62.4 MB | 폴더, 191 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 191 files [.json:149,.jpg:36,.log:3,.npz:2,.done |
| `build/p1_recon_r3-1_20261022_01/kaggle_stage/data/` | 59.2 MB | 폴더, 793 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 793 files [.png:715,.jpg:65,.ply:11,.json:2] |
| `build/mtg_build_v2_clicks_20261103_02/kaggle/results_02/out/` | 59.0 MB | 폴더, 25 파일 | MTG 11-03 package _02 (imported chain) | excluded dir (outputs/uploads/raw/cache), 25 files [.obj:11,.csv:11,.txt:1,.json:1,.log:1] |
| `build/p1_recon_charuco_20261021_01/kernel_out/` | 52.2 MB | 폴더, 8 파일 | ChArUco pose (board 10x7, 38 mm) | excluded dir (outputs/uploads/raw/cache), 8 files [.jpg:4,.txt:1,.json:1,.ply:1,.npz:1] |
| `build/p1_recon_r3-1_20261022_01/kaggle_stage/upload/` | 49.7 MB | 폴더, 2 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 2 files [.json:1,.zip:1] |
| `build/p1_recon_charuco_20261021_01/leaflets/` | 48.5 MB | 폴더, 261 파일 | ChArUco pose (board 10x7, 38 mm) | excluded dir (outputs/uploads/raw/cache), 261 files [.json:131,.png:65,.jpg:65] |
| `build/p1_all_3dfirst_20261027_01/e_05/` | 44.4 MB | 폴더, 239 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 239 files [.json:174,.jpg:48,.log:6,.npz:5,.done |
| `build/p1_all_3dfirst_20261027_01/t2_02/` | 42.7 MB | 폴더, 287 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 287 files [.json:211,.jpg:62,.npz:8,.done:4,.csv |
| `build/p1_all_3dfirst_20261027_01/n_04/` | 42.3 MB | 폴더, 275 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 275 files [.json:199,.jpg:56,.npz:12,.done:4,.tx |
| `build/p1_all_3dfirst_20261027_01/n_01/` | 41.0 MB | 폴더, 251 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 251 files [.json:185,.jpg:54,.npz:6,.done:4,.csv |
| `build/p1_all_3dfirst_20261027_01/n_05/` | 38.6 MB | 폴더, 203 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 203 files [.json:153,.jpg:38,.npz:6,.done:4,.csv |
| `build/p1_recon_all_20261025_01/k3a/` | 38.1 MB | 폴더, 1596 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 1596 files [.png:1547,.json:20,.jpg:18,.ply:11] |
| `build/p1_recon_charuco_20261021_01/kaggle_stage/upload/` | 37.7 MB | 폴더, 2 파일 | ChArUco pose (board 10x7, 38 mm) | excluded dir (outputs/uploads/raw/cache), 2 files [.json:1,.zip:1] |
| `build/p1_recon_charuco_20261021_01/kaggle_stage/data/` | 37.6 MB | 폴더, 134 파일 | ChArUco pose (board 10x7, 38 mm) | excluded dir (outputs/uploads/raw/cache), 134 files [.jpg:65,.png:65,.json:2,.ply:2] |
| `build/mtg_fit_surfels_20261103_01/kaggle/results_user/out/` | 37.5 MB | 폴더, 17 파일 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | excluded dir (outputs/uploads/raw/cache), 17 files [.obj:7,.csv:7,.txt:1,.json:1,.log:1] |
| `build/mtg_fit_surfels_20261103_01/kaggle/results_clicks/out/` | 37.5 MB | 폴더, 17 파일 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | excluded dir (outputs/uploads/raw/cache), 17 files [.obj:7,.csv:7,.txt:1,.json:1,.log:1] |
| `build/p1_all_3dfirst_20261027_01/t2_05/` | 35.9 MB | 폴더, 243 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 243 files [.json:179,.jpg:52,.npz:6,.done:4,.csv |
| `build/p1_charuco_a3_20261021_01/_TRASH/` | 34.5 MB | 폴더, 34 파일 | ChArUco A3 pot board | excluded dir (outputs/uploads/raw/cache), 34 files [.jpg:18,.txt:7,.json:6,.csv:3] |
| `build/p1_recon_r3-1_20261022_01/kernel_out_1c/` | 30.9 MB | 폴더, 5 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 5 files [.ply:2,.npz:2,.json:1] |
| `build/p1_2dgs_full_20261018_01/dwt_03_2dgs_discs.html` | 30.7 MB | 018b92fe22a45b7601751be696d9fbc22cc3d80269de28ae200ae6e5465c65c8 | 2DGS full kernel | viewer html with embedded data (>1 MB) |
| `build/p1_skeleton_layer_20261101_01/r8pc/` | 29.7 MB | 폴더, 90 파일 | skeleton_v1 / v1.1 format + axis tool | excluded dir (outputs/uploads/raw/cache), 90 files [.json:43,.npz:39,.py:4,.log:4] |
| `build/mtg_build_v2_clicks_20261105_03/static/` | 29.1 MB | 폴더, 181 파일 | MTG _03 (leaflet split real run) | excluded dir (outputs/uploads/raw/cache), 181 files [.csv:85,.json:84,.obj:12] |
| `build/p1_all_3dfirst_20261027_01/t2_04/` | 28.9 MB | 폴더, 195 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 195 files [.json:143,.jpg:40,.npz:6,.done:4,.csv |
| `build/p1_all_3dfirst_20261027_01/n_03/` | 28.9 MB | 폴더, 217 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 217 files [.json:160,.jpg:46,.npz:5,.done:4,.csv |
| `build/p1_all_3dfirst_20261027_01/e_01/` | 26.1 MB | 폴더, 183 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 183 files [.json:130,.jpg:36,.log:6,.npz:5,.done |
| `build/p1_all_3dfirst_20261027_01/e_07/` | 24.4 MB | 폴더, 177 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 177 files [.json:125,.jpg:34,.log:6,.npz:6,.done |
| `build/p1_all_3dfirst_20261027_01/n_02/` | 23.3 MB | 폴더, 169 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 169 files [.json:124,.jpg:34,.npz:5,.done:4,.csv |
| `build/mtg_build_20261102_01/dynamic/` | 23.2 MB | 폴더, 503 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 503 files [.json:240,.csv:231,.obj:32] |
| `build/p1_all_3dfirst_20261027_01/e_06/` | 21.3 MB | 폴더, 67 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 67 files [.json:35,.jpg:26,.log:3,.npz:2,.done:1 |
| `build/mtg_build_20261102_01/kaggle/results_ctrl0/out/` | 20.9 MB | 폴더, 146 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 146 files [.obj:71,.csv:71,.log:2,.txt:1,.json:1 |
| `build/p1_all_3dfirst_20261027_01/e_04/` | 20.1 MB | 폴더, 159 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 159 files [.json:112,.jpg:30,.log:6,.npz:5,.done |
| `build/p1_recon_all_20261025_01/e_03/` | 19.8 MB | 폴더, 7 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 7 files [.jpg:2,.txt:1,.log:1,.ply:1,.npz:1] |
| `build/p1_recon_all_20261025_01/stage2_new/` | 19.7 MB | 폴더, 63 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 63 files [.json:27,.csv:24,.jpg:12] |
| `build/p1_2dgs_full_20261018_01/kernel/out/` | 19.0 MB | 폴더, 8 파일 | 2DGS full kernel | excluded dir (outputs/uploads/raw/cache), 8 files [.json:5,.pt:2,.txt:1] |
| `build/p1_recon_r3-1_20261022_01/renders_base/` | 18.8 MB | 폴더, 3 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 3 files [.npz:2,.txt:1] |
| `build/p1_recon_all_20261025_01/e_05/` | 18.5 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_recon_all_20261025_01/n_04/` | 18.4 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_recon_all_20261025_01/e_06/` | 17.9 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_recon_all_20261025_01/n_05/` | 16.0 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_all_3dfirst_20261027_01/n_06/` | 15.5 MB | 폴더, 107 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 107 files [.json:77,.jpg:18,.npz:6,.done:4,.csv: |
| `build/p1_recon_all_20261025_01/n_01/` | 15.5 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_recon_all_20261025_01/e_02/` | 14.9 MB | 폴더, 7 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 7 files [.jpg:2,.txt:1,.log:1,.ply:1,.npz:1] |
| `build/p1_recon_all_20261025_01/n_06/` | 14.2 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_recon_all_20261025_01/t2_02/` | 14.1 MB | 폴더, 7 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 7 files [.json:2,.jpg:2,.txt:1,.ply:1,.npz:1] |
| `build/p1_recon_r3-1_20261022_01/kernel_out_1b/` | 13.9 MB | 폴더, 3 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 3 files [.ply:1,.npz:1,.json:1] |
| `build/p1_recon_r3-1_20261022_01/kernel_out_1a/` | 13.9 MB | 폴더, 3 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 3 files [.ply:1,.npz:1,.json:1] |
| `build/p1_recon_all_20261025_01/e_01/` | 13.5 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_all_3dfirst_20261027_01/e_02/` | 12.9 MB | 폴더, 43 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 43 files [.json:23,.jpg:14,.log:3,.npz:2,.done:1 |
| `build/p1_recon_all_20261025_01/e_07/` | 12.5 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/mtg_build_v2_clicks_20261103_02/static/` | 12.3 MB | 폴더, 181 파일 | MTG 11-03 package _02 (imported chain) | excluded dir (outputs/uploads/raw/cache), 181 files [.csv:85,.json:84,.obj:12] |
| `build/mtg_build_v2_clicks_20261103_01/static/` | 12.2 MB | 폴더, 173 파일 | MTG 11-03 package (imported chain) | excluded dir (outputs/uploads/raw/cache), 173 files [.csv:81,.json:80,.obj:12] |
| `build/mtg_build_20261102_01/static/` | 11.6 MB | 폴더, 192 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 192 files [.json:93,.csv:87,.obj:12] |
| `build/p1_recon_all_20261025_01/t2_05/` | 11.6 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_recon_d1028_20261028_01/stage2/` | 11.0 MB | 폴더, 38 파일 | e_* recon batch | excluded dir (outputs/uploads/raw/cache), 38 files [.json:17,.csv:14,.jpg:7] |
| `build/p1_recon_r3-1_20261022_01/outputs/` | 10.9 MB | 폴더, 4 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 4 files [.ply:3,.json:1] |
| `build/p1_recon_all_20261025_01/e_04/` | 10.6 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_recon_all_20261025_01/n_02/` | 10.4 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_tprox_sam2_20261107_01/kout1/` | 10.0 MB | 폴더, 42 파일 | T-Prox + SAM 2 arm A (0.844) | excluded dir (outputs/uploads/raw/cache), 42 files [.npz:40,.log:1,.json:1] |
| `build/p1_recon_all_20261025_01/t2_04/` | 10.0 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_recon_charuco_20261021_01/sheets/` | 9.0 MB | 폴더, 68 파일 | ChArUco pose (board 10x7, 38 mm) | excluded dir (outputs/uploads/raw/cache), 68 files [.jpg:68] |
| `build/p1_recon_r3-1_20261022_01/_TRASH/` | 8.9 MB | 폴더, 4 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 4 files [.ply:3,.html:1] |
| `build/p1_recon_all_20261025_01/n_03/` | 8.2 MB | 폴더, 6 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 6 files [.jpg:2,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_recon_r3-1_20261022_01/viewer_input/surfels_full.ply` | 7.7 MB | 17dbb9b96d3fd9b90e16c25fe682cad5c1384fbf148e24661cb3e3e02d713350 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | data file (not uploaded) |
| `build/p1_recon_r3-1_20261022_01/kernel_out_base/` | 5.6 MB | 폴더, 9 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 9 files [.jpg:5,.txt:1,.ply:1,.npz:1,.json:1] |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060602722_iOS.jpg` | 5.4 MB | fe25be41dbc0c1e9980ee473aed055eff24ab918207632377585380e46cf6920 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060559521_iOS.jpg` | 5.4 MB | 1897525a38d87f7a0926229b737d95c8450279b276c4f3bf3c0282e7a9c8a0f0 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060600613_iOS.jpg` | 5.4 MB | 4f48c311613ab54c1c93096c8ff554926d4db8153533353b717aff143cc2546e | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060605333_iOS.jpg` | 5.4 MB | d8f0dd6728c74edb1535147d1c0f26febb73a1548b41563b753d5a7b737c3c88 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060603671_iOS.jpg` | 5.3 MB | 1f409e939e6be2540b409985e5528436f0d814477761e4f3dafe2c3e08a5c37d | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060542310_iOS.jpg` | 5.3 MB | a32400df4887d0e6b09cde8a8acd6c9a8889fa613d80fedea445047c2b515c87 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060313903_iOS.jpg` | 5.3 MB | c8cfd1322a34809fc5c4b87f2ae68b4104efd0e5e5e9c7f18dae46f2ae039281 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060544649_iOS.jpg` | 5.3 MB | dad5b2efa122b4bd6ef730e670693ce5fa7047a67be0f9f51b256bfecefd282f | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060540841_iOS.jpg` | 5.2 MB | dca9eb3435cd021156461c33e3ba2fb6aed052d9ebf5131c156cb3a7c15709d4 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060557359_iOS.jpg` | 5.2 MB | ff55b15b93f75611447e905ead7ecaf9e86db0bf5156bbec2bb7972f65b4141e | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060546510_iOS.jpg` | 5.2 MB | 377205e29648cf8b76d3669a9c02926bde87f7148130dbd42fcfd27cd74f04db | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060545604_iOS.jpg` | 5.2 MB | b2ccf6e1a335315252329ba4314a5a56ebec03133f34a3c74818e62223b21ba7 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060537292_iOS.jpg` | 5.2 MB | 1df4379170dca72f1425ec95d1cf8f63f06f40ac0d5e6ea6b2283aac26206915 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060529992_iOS.jpg` | 5.2 MB | 6d61279586bffebef6dfda40061dbbe7e57deed8f1a5c0f1b1f70d0f8c08852d | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060538353_iOS.jpg` | 5.2 MB | d35796e940657759f188ba37a32c3ecf2105237d70e7a56a8b02b5cfea9fa389 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060554103_iOS.jpg` | 5.2 MB | 38d03b62ebfc660c1ea8b0dd746e2bc4cb68001bc26d79d03319131d7cb43453 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060539367_iOS.jpg` | 5.2 MB | 8f01bcef87864a78bf4df62808e96235d7deca36faa1ec83436164783a4b3405 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060528683_iOS.jpg` | 5.2 MB | c6d29353b186d3483de55408fdf011462fe69ca372e35628e5b092f36e6be5e7 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060547546_iOS.jpg` | 5.1 MB | 54a764968cf231986ede20f2ae253e7f8a435d694138b42ce0ec335ef5290a41 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060527538_iOS.jpg` | 5.1 MB | 027776947dd4d657f7571af8154a29fa698c552dc8e1114b66b163d2ca94ffec | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060549196_iOS.jpg` | 5.1 MB | bc12eb5be3dd7b8e33aab73d19f8c3cbe939398c6ad5ddd9067361994bfa895b | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060524278_iOS.jpg` | 5.1 MB | abb241f2eea76471a0bb766a5185742dbc637b05ef5289782efd2700d0612c1c | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060414567_iOS.jpg` | 5.1 MB | 5ee510a193542af0660942f4cc4d70dd774c89eaeaaed9b9e29a6a4890ac46f0 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060531086_iOS.jpg` | 5.1 MB | c61e1f3bcf952b47f519022bc5bd610f738086741f4d7b86b2322f0f8fa89865 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060531950_iOS.jpg` | 5.1 MB | 2784dc4e1fbcda02c0b3a0e3a61ee5669f2af18838e1e6149b4c95966fd6259b | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060551789_iOS.jpg` | 5.1 MB | b6e76b6b82b23fbb26710e5d92b1e4ca4b699a44a4d8797b1630b8f881081fe9 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060532792_iOS.jpg` | 5.1 MB | 2732917b94c57f0e46d5cb508233b86373bbf88449080aca2d548477d2070b88 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060536184_iOS.jpg` | 5.1 MB | f14ebcba0c483e0352193e377f83310f5c9ad8da49125220b218807a1b7582f5 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060534548_iOS.jpg` | 5.1 MB | 5202017591e5422bb3e5edc7713b05f3dd88457eb2a5470e8d4abe3fe5ecb20f | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060555768_iOS.jpg` | 5.0 MB | c418671850f4583243bb9347c5930f8533d3d733cad7b6beec06a7336e4b26eb | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060533609_iOS.jpg` | 5.0 MB | c22c93d0e0cedff42a7b9a3821cab2f5518b1e4697fe6c145fdf7f63febe20da | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060553242_iOS.jpg` | 5.0 MB | b2f739159bc43c881830d1784290fd27acf425899082e6c2213289137123a43e | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060416893_iOS.jpg` | 4.9 MB | 473766d282bed402e48f7f14d0e118e93146be49f42e6849de5eee99809791af | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060355657_iOS.jpg` | 4.9 MB | 9209d05919c43bc1d089eed89b6416158b4a91a5757f99502bad21659a98baf0 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060315956_iOS.jpg` | 4.9 MB | 06d3e0a3530fc04489df6dd3b71ebf8c6253246e3772af67650db4df3a28da39 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060317552_iOS.jpg` | 4.9 MB | 8b0696fc95cfa3a303c19a03f1736112b5524ec7054010ef1d50ee267d4c8fa2 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060418991_iOS.jpg` | 4.9 MB | d9051576da7bfd2a3e57e54ad4412b9829deb27f432111a9f56d266c6fc166ac | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060348489_iOS.jpg` | 4.9 MB | 184ecc8dd833720889bac340016d5fcab39b4c794de32ec74b271ba930a5a215 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060354234_iOS.jpg` | 4.8 MB | 5c1f474ec9a270b0ccc44ab652eb64dd70da7c336ca551855cbd67253980b42c | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060420263_iOS.jpg` | 4.8 MB | 86129ca3de1733e74be8b7e6a97bf1230e251c8c5fae1088bae69b4ad7c1465b | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060352245_iOS.jpg` | 4.8 MB | ed76050793e70627dcd346db0349fa14e55f9c5c38875d0e94ab71b022f1a9e9 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060442059_iOS.jpg` | 4.8 MB | 5806d89449a3fabdd442860d9fa21c75e7877274f619387964fba8a3a4e50d43 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060319179_iOS.jpg` | 4.8 MB | f438aa39c847a4ac06d1b380914809ddcb05b09b09dddd71745e9255aadc12d0 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060437164_iOS.jpg` | 4.8 MB | aa91e892c3bf09d46f1ce488adcaef318b35781958f3e5f8ab2b9fb3d97ca9c4 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060422747_iOS.jpg` | 4.7 MB | 0f6c278fc67d202063f66bbe250d469492eca463df082f8b8658713da7b3729d | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060440865_iOS.jpg` | 4.7 MB | 061f60091b2564521925fa3a4d4c397fb5b71a1c3d5604dabf0930981a103f0b | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060328024_iOS.jpg` | 4.7 MB | 8160fb676bd0ec4048b4b8b398e63bbc1e14411a7391ef298d0b23df6929d579 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060350518_iOS.jpg` | 4.7 MB | a4b8f68345747aeab0c4463ea3cf8d467295811c8814d3679ef584d8c23c0dcd | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060439622_iOS.jpg` | 4.7 MB | 24ed8ff22f4054f1d07a8798aa8c1533015bf99ee10f464feaea11781a0869cf | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060248144_iOS.jpg` | 4.7 MB | 21ab66c680690efa94e660a15df808890f5c6adf84b90ff1e48d14abc8a50ad3 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060329819_iOS.jpg` | 4.7 MB | d4ff57af1fcbeb2f5fe9948b5d58ab8764a1a768ce89b1be17e9a3528ace2dc1 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060421660_iOS.jpg` | 4.7 MB | b7a194c53c22bbb5cff70d81e2ad5897b0055188aeae3f08516992bed1b6baa2 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060430586_iOS.jpg` | 4.6 MB | 90434b83880d077dbc9dad19d9a3fe782c106248ee0bcf890ba7ff0f59101557 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060346661_iOS.jpg` | 4.6 MB | 60ef42a32b1b20835c5232b87fdbe89d5784902bbe634a13ab557c58979f32a1 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060321744_iOS.jpg` | 4.6 MB | 438349b17846c67725734822afaf5d9c9c65bb38a3bd23609f702d0fb3209fd9 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060428635_iOS.jpg` | 4.6 MB | 8f316b557b29c2516c28a0a0afdf2fb46e7324b636fe3f17cec2ed41ec5ee814 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060423912_iOS.jpg` | 4.6 MB | d20679319bfd732dda5d45f377d4f3f1d866ec4401171f18821e50ef09e1959e | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060332829_iOS.jpg` | 4.6 MB | fb6500a7028ffec94c0abcb30b8905628d32b9fe1b1760c887baf1c42049816c | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060326166_iOS.jpg` | 4.6 MB | e5e908ef2c68f12909fc67c0b8bf54217846b699493f498930aaa670c5ea05e1 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060337367_iOS.jpg` | 4.6 MB | 0fdc39101ef1267a30a03d1818ca9303d5973112088b94d15a3957dec92bd0e3 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060235403_iOS.jpg` | 4.5 MB | f4286ab369d760a91b60df22b88b3a7dc42ba74a461003121721d84586051bcc | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060429762_iOS.jpg` | 4.5 MB | c8449d1a9f0b9ed763369aba9f43b5d3774960337e1238279b3897a7ead3ef76 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060433835_iOS.jpg` | 4.5 MB | a34830b1418ac42d5dec42cbe7ddd1ac84e9816194249ed25b87f454bc09e790 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_recon_r3-1_20261022_01/review/` | 4.5 MB | 폴더, 3 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 3 files [.html:2,.png:1] |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060340484_iOS.jpg` | 4.5 MB | ef7238b934c05b344dc581a83e8a2e565a723309d8ba7df30ef5beb158fd107f | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060323966_iOS.jpg` | 4.5 MB | db809aa41872c46d7548ef0acc5c05e68da2fb7cda310ea9ef012b5e91fc3045 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060331680_iOS.jpg` | 4.5 MB | 7294f78ae3c23893691d148fb5b0bfe6bbb8352745a3816b3579b19c965ea9aa | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060335139_iOS.jpg` | 4.5 MB | 46ce853ef149412be95ac9eebdbec225a69bde77d1c2ca0a324b4cfda235f841 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060344652_iOS.jpg` | 4.5 MB | 11dc4d23f21c58c088da4b7b70dec3058679d049c8ce23e3b251f0f63b2d81f7 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060338861_iOS.jpg` | 4.4 MB | 758e18501f4e8838a44af727396132569cfbd73108b45a48113dfb3170f12cff | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060341729_iOS.jpg` | 4.4 MB | adc38f0c490b1320c30e762d607bcb49585a89fe0b38c90381e5e280941c954f | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060343438_iOS.jpg` | 4.4 MB | 893e498333f4cbca3dfefa06a61d7a1f1c92fd2e092d5f0d7f68fc30622f8d0c | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060427460_iOS.jpg` | 4.3 MB | 089c41051018bb44be4bb17f67dcef4ebade6cd1690267cc9eb842c482dbe1d9 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060443572_iOS.jpg` | 4.3 MB | 0f4b190932b5428b30c68036f539d8dbd16430de6e2b52429d5573812b8ec9b8 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060446991_iOS.jpg` | 4.2 MB | f2c23eb2916197e2b0fea80cbe5660740d6415f7cfaa13239e480d335a731331 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060445656_iOS.jpg` | 4.2 MB | 06b5b0c027178357bdd349e5fe3910e20db94d9ad2ab155907d2a245042f13cd | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060456328_iOS.jpg` | 4.2 MB | 4fe5d87e18485b055dc88fd529efc97dea4aeaa1b3b32a9796356f66c9004185 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060455325_iOS.jpg` | 4.1 MB | 7bbd6733ba485e418779d16c026fc1659348ce86a0d12e82daea9a5402b933b2 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060457442_iOS.jpg` | 4.1 MB | 62bbc804cd3feac6178641f5c0659da9bffc8a93bf107f713a58c6ec2d865afd | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060458583_iOS.jpg` | 4.0 MB | fb55e047d3221558b2287bb24cdc800ea9067a3a32faf55ff769ff5d9297240b | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060448335_iOS.jpg` | 4.0 MB | 9b9f04a3c8a8d7367a51202d69c5e758a6d52106c4cd2515fc6224cc52819e80 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060454093_iOS.jpg` | 4.0 MB | 536870282c31857c813f18675b313cd99a46f02d3e539f3d49aa14368112113d | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060452133_iOS.jpg` | 4.0 MB | 17d7434f7d20265d7622d8db831298b37dc662c9bd1f5e8bc3788497bf54b213 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_tprox_sam2_20261107_01/kout_others/` | 3.9 MB | 폴더, 24 파일 | T-Prox + SAM 2 arm A (0.844) | excluded dir (outputs/uploads/raw/cache), 24 files [.npz:22,.log:1,.json:1] |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060450923_iOS.jpg` | 3.9 MB | 6f33dc4dc73845beeacea7778169f3c9cdba77e9fdecbfbe73141f7ca522edc0 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/jpg/20261021_060449755_iOS.jpg` | 3.9 MB | fb7f18c42dc36d4695aa203df037f2dd35ed17fcf8576ab868dd0b4f58266cc6 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_tprox_sam2_20261107_01/kaggle_data/` | 3.8 MB | 폴더, 9 파일 | T-Prox + SAM 2 arm A (0.844) | excluded dir (outputs/uploads/raw/cache), 9 files [.npz:6,.py:2,.json:1] |
| `build/p1_recon_r3-1_20261022_01/soft_a/` | 3.2 MB | 폴더, 65 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 65 files [.png:65] |
| `build/p1_recon_r3-1_20261022_01/viewer/` | 3.1 MB | 폴더, 1 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 1 files [.html:1] |
| `build/p1_pc_absorb_20261027_01/kaggle_data/` | 3.1 MB | 폴더, 2 파일 | R8_pc random walker (LATEST_PIPELINE) | excluded dir (outputs/uploads/raw/cache), 2 files [.json:1,.npz:1] |
| `build/mtg_build_20261102_01/kaggle/import_ds/` | 2.7 MB | 폴더, 118 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 118 files [.csv:113,.json:3,.bin:2] |
| `build/p1_skeleton_assign_20261101_01/clicks_export_20261101_1849/` | 2.7 MB | 폴더, 13 파일 | skeleton route accuracy (A4) | excluded dir (outputs/uploads/raw/cache), 13 files [.png:11,.csv:1,.json:1] |
| `build/p1_skeleton_assign_20261101_01/lines_check_20261101_2300/` | 2.6 MB | 폴더, 10 파일 | skeleton route accuracy (A4) | excluded dir (outputs/uploads/raw/cache), 10 files [.png:9,.csv:1] |
| `build/p1_recon_r3-1_20261022_01/sheets/` | 2.5 MB | 폴더, 3 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 3 files [.jpg:3] |
| `build/mtg_build_20261102_01/kaggle/print_ds/` | 2.5 MB | 폴더, 4 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 4 files [.json:2,.bin:2] |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_e_05_engine.plant_tree_v1.json` | 2.4 MB | 435870efbc8a16fc56692ae507a3eade0f1cce27e21f6e5e1bd2334f204e99d4 | MTG _04 = current static MTG set (8/11 pass) | large derived table/geometry (>2 MB); listed with sha256 |
| `build/p1_recon_all_20261025_01/stage2_new_n06/` | 2.4 MB | 폴더, 7 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 7 files [.json:3,.jpg:2,.csv:2] |
| `build/p1_recon_d1028_20261028_01/k3a/` | 2.4 MB | 폴더, 15 파일 | e_* recon batch | excluded dir (outputs/uploads/raw/cache), 15 files [.json:8,.jpg:7] |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_n_04_engine.plant_tree_v1.json` | 2.2 MB | 022bad4678761ac7c2b3204cf41d8be0c7746b58ddd79799d62e3ec3ba17083a | MTG _04 = current static MTG set (8/11 pass) | large derived table/geometry (>2 MB); listed with sha256 |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_n_01_engine.plant_tree_v1.json` | 2.1 MB | c0758eb036cdf60993ff3bd248eb060ff972701f861d306b40d7a6e9345ae4b0 | MTG _04 = current static MTG set (8/11 pass) | large derived table/geometry (>2 MB); listed with sha256 |
| `build/p1_recon_r3-1_20261022_01/soft_b/` | 2.1 MB | 폴더, 65 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 65 files [.png:65] |
| `build/mtg_build_20261102_01/kaggle/ctrl0_ds/` | 2.0 MB | 폴더, 75 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 75 files [.csv:71,.json:2,.bin:2] |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_n_05_engine.plant_tree_v1.json` | 1.9 MB | 7ed6336f0de1c8ae2045e6103a1f7e273409fdc8a37f8a43772ad78d52a9252c | MTG _04 = current static MTG set (8/11 pass) | large derived table/geometry (>2 MB); listed with sha256 |
| `build/mtg_build_20261102_01/kaggle/import_ds_dtheta/scenarios.tar.bin` | 1.9 MB | d19ca018b8df3badcf0ccc534ad859818735b32ad5f4dcc9965d8734a12121ef | MTG build | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060600613_iOS.jpg` | 1.9 MB | 930882bd9f564223fce3e66cbd644016362bcbbc5c83ec673d96e790d489b025 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060559521_iOS.jpg` | 1.9 MB | 37726215f6f753402388721b59af41db600244144661f0773cd6c0c68274c65e | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060317552_iOS.jpg` | 1.9 MB | 8fc887eafdbc7ccf88016d775bb5c7be7dbf628c6d19b32ea88945db60d1a782 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060414567_iOS.jpg` | 1.9 MB | 61194eb6300590e492fec0e33bcf62c8715f24d2db5e8a02e303a9380d2a0a28 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060603671_iOS.jpg` | 1.9 MB | e9f08e0d415d98715bfaba8d1876ac99f35d79035ba98aa419fc46359a87bad5 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060348489_iOS.jpg` | 1.9 MB | 6b9796163c405424b893875afd6d24fef7374cc63eec76cf10d1a383ee3fbd7c | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060605333_iOS.jpg` | 1.9 MB | 4839a3241056fa487e96b6432d2dbb3ff488e1316478a8241f9a59132d2a6446 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060319179_iOS.jpg` | 1.8 MB | 9f08552dddc7881acaf92f69921164b0b4547fb584a67788d86f67af9d56c8f0 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060313903_iOS.jpg` | 1.8 MB | 2bdbf775bd16b936d243bbc155d1114b483c4914d93c37e322fdccf01de91bdd | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060315956_iOS.jpg` | 1.8 MB | 40a739b243dd9536394e11956ec596e13e885dea16e3d4fcc38635b84289d181 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060542310_iOS.jpg` | 1.8 MB | 0ff91d7b6520d7497357d49808e9985c6a2724dbe55392d5d7298691b57ac344 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060416893_iOS.jpg` | 1.8 MB | 1c2a3a3f6a41b84b8ab0d870a6b03fc57f8808c32b86c1c7b115eb0ec05e25d4 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060350518_iOS.jpg` | 1.8 MB | 9983b8d2a70841083f5fa7fadad8d6eb5219fad8794eaebee1b37cd2d1aab20b | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060544649_iOS.jpg` | 1.8 MB | 7d3a1bf7666ec064e9859461999f9caa81822f3ddf2af2099c4c566444dd10c7 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060418991_iOS.jpg` | 1.8 MB | 79563a2094427a5dba39034df717eb446e394f6109f07bd14433dc46b94360f1 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060352245_iOS.jpg` | 1.8 MB | 002673927c30837c28331cddef9c7aae819b5f06c4b59ec882b72097a647e00b | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060557359_iOS.jpg` | 1.8 MB | 0d8044a754f0d128f9382beeb1403d9c33bd5a578d0407d92e89292fcd7bb1ad | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060420263_iOS.jpg` | 1.8 MB | e6df104125a1debdc280dadb478a59b1786ae20dfde2cba51ca625dd2099a0d2 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060540841_iOS.jpg` | 1.8 MB | 3b44e6350f6766aa7029a536092e51813f20235e754ecbc2f4da672d56b8672c | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060547546_iOS.jpg` | 1.8 MB | eb13b11c44527af3587da90a2ca60643cd3e508c0514f52581ac0e6f1e2b54d6 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060554103_iOS.jpg` | 1.8 MB | 472fd2e7d9db8a19366583174323e665d1307b0bcc9b7fe8f094066de29050c1 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060551789_iOS.jpg` | 1.8 MB | 8fa9492bbc92d96ecaf07c9f73487981c277c4bbd3de237323b349acdd0fab29 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060549196_iOS.jpg` | 1.8 MB | 886aee16c670ffb74ead4273c17f73aaa1bf2c933c98168dd52c717c68d90b81 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060354234_iOS.jpg` | 1.8 MB | 03f961b9ab877fac0f23909d21150db4bcea2b283fffb3fb03f76cd84f5732bc | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060545604_iOS.jpg` | 1.8 MB | fe3cacdc1eb30c6873e68f320bfa18044689d0dcba85eab9463950a2f2e2d7a0 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060346661_iOS.jpg` | 1.8 MB | 0591b8954a1e4c4d03b271a9b9fc0e3527da52ab97f85d6cfdcb8b553fa42d65 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060355657_iOS.jpg` | 1.8 MB | 9cf96490a475f37edc1450c06ec24f78fa36e94c397b49f5b99ad0b6ca06b231 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060328024_iOS.jpg` | 1.8 MB | 8f4fe709976d6782dc16240b3a662a915b0ffa1e4a0d5f190f3b6e691cf29c25 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060538353_iOS.jpg` | 1.8 MB | bbbdbc5306b880dcf85617b8f8a41d7a402c1cc2a53d7bcfdfc37930a7ad1140 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060329819_iOS.jpg` | 1.8 MB | 5cbca26f32117e4183b3dd5352d6fbade45279d5ba56b05cda43f7e1b139114c | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060528683_iOS.jpg` | 1.8 MB | 37c237968fd9885b9ed2fbddbfd1e8f57a91610da0456f1a124cc2e19f72637d | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060321744_iOS.jpg` | 1.8 MB | 2a28148c162d77d37b6b9be9c0833e810caa02b1e06af48697f418d175b43a46 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060337367_iOS.jpg` | 1.8 MB | 3276294ea102306622130dcf6afbc566a6f37e3fa66e7eb89545a6f48c7f6b1c | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060340484_iOS.jpg` | 1.8 MB | 03bafc40379addfc10a52eef24d0ec892d2478701f1467af7209f5cfda685036 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060527538_iOS.jpg` | 1.8 MB | 1dce765c1e4a1ee55633366070655f9de25722946a576fbf94c56b3d1bc638c7 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060531950_iOS.jpg` | 1.7 MB | 1878f4555980c30984aa49942e5da0bd89808a0fe8a59a046c31e4f8ba0b7337 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060555768_iOS.jpg` | 1.7 MB | 971332e6c440fbbef55499e49da245fbd1da4ebfa3c8124be8e4eb6662a8893c | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060326166_iOS.jpg` | 1.7 MB | d5de53681aadffeb296446fd5a39a08780fd81da4f342f1c6e85523ddc00b3cd | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060532792_iOS.jpg` | 1.7 MB | 467b9cff9d78373e086ac87b19754aaf6fa1ddbe8ac4355af32343be2c7c9b54 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060421660_iOS.jpg` | 1.7 MB | 0358d462b7d824e3d419dd1d7e53666580b87ee686d8f2c1a90abe43b933a22f | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060338861_iOS.jpg` | 1.7 MB | 384707024f28c33837e4d4f2e28c2bc1d3024b5358ae11681225c9e280e50f1a | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060524278_iOS.jpg` | 1.7 MB | ab6417ee45c61a49a99aa22d2c400c6100550b266d3c470fa6c5df0aa753262a | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060422747_iOS.jpg` | 1.7 MB | 6de6bde706552703ab62dc56bc8c9467e2e3cb8e7598aa9a5f1a48e6ab491e54 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060341729_iOS.jpg` | 1.7 MB | a8a8ccb98d96847c144784a4d2ab16d200018594a73d3560cbca9d188071c1d4 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060331680_iOS.jpg` | 1.7 MB | 0006c61deb7c99e7c93f84faea87f9040cff2eaeded502f0503a192fbfb7db08 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060332829_iOS.jpg` | 1.7 MB | 6f96ed66fd06b8678c910dac6875785201a114829d67ffc853da700111ef6509 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060430586_iOS.jpg` | 1.7 MB | 126be02fafac4b1a9c803e5c6d6150e927eeaf8d384c2180af9c2bf9c11d699b | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060335139_iOS.jpg` | 1.7 MB | 83a8c163f543583627970ef35b24ffed64825eb7c9974c4abbd8a47569366034 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060323966_iOS.jpg` | 1.7 MB | 0d2eca28f54ee3469db63a337e466a8938857f2a6ed1ea1f22b14b8370ff6819 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060344652_iOS.jpg` | 1.7 MB | c5de6692dd96d0e7a716f29f5bab395ae6a1aadeb71956aa24d72799de0b550a | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060429762_iOS.jpg` | 1.7 MB | 545373cde41df9f1630f637d5c1108de1d2955e876e3e599fa593e1534bb37e2 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060428635_iOS.jpg` | 1.7 MB | 5c7c06b61a3afa7c3d91bf644909977abd76ab58355489181abade8c0dfb0ec8 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060343438_iOS.jpg` | 1.7 MB | 8cd6737ef73cff13f6cede290c5bab739dfc999d8f8f79cfec7d06fca4273b6e | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060423912_iOS.jpg` | 1.7 MB | 16ae81a013a578c4d7e739905e3d619729d24199a24a48cdb620d3e464fed486 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/qc_sheet.jpg` | 1.6 MB | 1be5fb987135303db4da7367afafee12d0d92053b35edaaafdecfb2642acd12c | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060442059_iOS.jpg` | 1.6 MB | e0be5538289c891eac2c4ca7eb64dc4e48a80e377c05928a389c1de08ea480b4 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060427460_iOS.jpg` | 1.6 MB | 45ba8318171703c2834ab866f261be410fc0654b43728a89747ce868606007e2 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060440865_iOS.jpg` | 1.6 MB | 577f00e823d1dfc7e8767b8caaf288c5fa081efb3e2b625c6767b210455516f7 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060437164_iOS.jpg` | 1.6 MB | e734fdd24688d7af1eec5267fa2b75ab0abe1c6af337b6c7c92dc7b65c8e73fd | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_recon_charuco_20261021_01/viewer/` | 1.6 MB | 폴더, 4 파일 | ChArUco pose (board 10x7, 38 mm) | excluded dir (outputs/uploads/raw/cache), 4 files [.png:3,.html:1] |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060439622_iOS.jpg` | 1.5 MB | 5d43ac23bf59a6ac0246ae1a69feb668b92e5fd4e5f6c1ca29c023f978d565f7 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060248144_iOS.jpg` | 1.5 MB | 33541ef623764d383c431d99ec50717f0d718bb9f4bc55ed2c7938e257c110cc | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060235403_iOS.jpg` | 1.5 MB | bb323d5bc818d654bf44ae7f5d2057b03e08c366587327927ab689300d1cd1e8 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060454093_iOS.jpg` | 1.4 MB | 80d1b0343aa4b25bf8009314a19b515f9e18f19191e420336d393ef5540ca5a1 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060448335_iOS.jpg` | 1.4 MB | 8619834938dc5fb04f3eba97ec408724103a3c76e8bcc7b83b3e85614c40fa1f | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060450923_iOS.jpg` | 1.4 MB | 9626deb9375e177d7b2e3896029660cfb3e668e01dba7a678d3f7326856cc471 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_charuco_a3_20261021_01/undist_3024/20261021_060449755_iOS.jpg` | 1.4 MB | af4e5f52e59e60f7c069c3a48be3ac2ea7d7b78b4bcced5121a85d9289847da3 | ChArUco A3 pot board | data file (not uploaded) |
| `build/p1_recon_all_20261025_01/check_sheet.jpg` | 1.3 MB | 96e6d1eb6785257a2d674047c8c55279ef950c8982333b939fe479c66174aeb8 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | data file (not uploaded) |
| `tomato3d/organs/__pycache__/` | 1.3 MB | 폴더, 28 파일 | labelling: 3D-first groups, skeleton tool 8835, T-Prox, R8_pc, V-P | excluded dir (outputs/uploads/raw/cache), 28 files [.pyc:28] |
| `build/p1_recon_r3-1_20261022_01/masks_a/` | 1.2 MB | 폴더, 65 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 65 files [.png:65] |
| `build/p1_recon_all_20261025_01/pts_all/` | 1.1 MB | 폴더, 19 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 19 files [.json:10,.ply:9] |
| `build/p1_recon_r3-1_20261022_01/masks_b/` | 1.1 MB | 폴더, 65 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 65 files [.png:65] |
| `build/p1_recon_r3-1_20261022_01/bands_a/` | 948.0 KB | 폴더, 65 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 65 files [.png:65] |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_e_05_engine.obj` | 933.6 KB | d75bdcea3b262bc086feb464bf1500923b0af4b2961ba822241f41a32998c813 | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/p1_recon_charuco_20261021_01/masks/` | 893.5 KB | 폴더, 65 파일 | ChArUco pose (board 10x7, 38 mm) | excluded dir (outputs/uploads/raw/cache), 65 files [.png:65] |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_e_05.npz` | 882.2 KB | f9124038100fa210d4a881800629bb0d81023bb1d49aaa9f34458185e313a92f | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_e_05.npz` | 881.2 KB | 5751b5371017f2cc3a8281d6c48c041f8edbb6b0825b59bb78bed098538f18da | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_n_01_engine.obj` | 833.6 KB | a5aee81de8e3ade3d95e6330c72f672a0a60f973e73f92d7068acae2ad29f2d6 | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_n_04_engine.obj` | 833.3 KB | a3739f63ddb619874179f60e5c7f90256a88ffd8b8956673e42ff2d7327a69c1 | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_n_01_fitted_engine.obj` | 749.9 KB | a82a9b86b314f46876399329bac23126906ac3e0a6c69b2ead789b2a0481d716 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_n_01_clicks_engine.obj` | 749.9 KB | 8327d506e7f59219b5b919dc694a60db72b0d17f0330b5d14b54360a97520e23 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/p1_pc_absorb_20261027_01/kaggle_out/` | 743.5 KB | 폴더, 4 파일 | R8_pc random walker (LATEST_PIPELINE) | excluded dir (outputs/uploads/raw/cache), 4 files [.npz:3,.json:1] |
| `build/p1_tprox_others_20261101_01/labels_tprox_n_05.npz` | 742.7 KB | e70d916ff956de18f3945a8a4389b9d7efec7c6a1cc149951b456e1b9524d23d | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_tprox_e_05.npz` | 742.2 KB | 50ba7ce0d96b3513f337ad2a5e118c8fedeb3cffa283ee1db6a76050bcf41507 | T-Prox 11 plants | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_n_05_engine.obj` | 728.9 KB | 8f7bba232241af88b91571389dbe98221df3c45ff48e272072484f60288439af | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_tprox_n_04.npz` | 718.8 KB | e386de0ec4189323080734a16152eab0e7ad904dc85e3e8f112f2aca6397024e | T-Prox 11 plants | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_n_02_fitted_engine.obj` | 704.1 KB | 863e698f46e5c469e6ef644e76ceafbf00828fd70684553c300fc7439c5a694d | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_n_02_clicks_engine.obj` | 703.9 KB | 86cefa2257232f1e9a7e472830f418db276dae8ae1b0bab2ced2a99eb9ba99ef | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_n_04_clicks_engine.obj` | 692.3 KB | d99a5837ef4bd58284dab92a9603fc048b545bca4a4b7734ee62a54cbc06dd6c | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_n_04_fitted_engine.obj` | 692.1 KB | 711ba3d6018f93dbd699cc9bbec9d282a43cb9e5c078815ed8c962b39422c767 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_R3-1b_engine.obj` | 686.8 KB | 2fdd5e00285bd1e1560a22a0a9148754fd89ae6449bdd9ce8af9bca9d7c4e377 | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_e_05_fitted_engine.obj` | 686.5 KB | c7a8efc15c33a052870255d416ac6cb2830c41474354c0b546211ccf64caff3a | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_e_05_clicks_engine.obj` | 686.3 KB | c256aed2dcba3da2454e64934fad08b7b783aff3c8fd5f63033a8bcfb8a187ed | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/p1_recon_r3-1_20261022_01/bands_b/` | 663.9 KB | 폴더, 65 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 65 files [.png:65] |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_e_01_engine.obj` | 661.0 KB | 6f80721241e3fbdc6c397ac5dd11e915645bd97a0eb0791909d76add1b9b5f09 | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_e_07_engine.obj` | 653.0 KB | 075b8a0863a53c4af9c910d720171144ec1778a44bbb0ace1f4f25ff505c0078 | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_e_01.npz` | 651.2 KB | ca89eda7539748668fdc4f38b11424d8f082f7260779078ba8c37faa300be512 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_e_01.npz` | 651.0 KB | f5ebe160d7db27e919fb34fb1a84eaed18ec120b3bb87ef112394c71bc688419 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_tprox_t2_02.npz` | 622.7 KB | 5d992a4ad0845d67b31b8109a78e4aeb403b7f6f6ddee18b01234492abcb7c24 | T-Prox 11 plants | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_e_07.npz` | 616.6 KB | 9803973862a75e2be94430f41200040be6bc16a5929ba2b1a88554efff44a032 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_e_07.npz` | 615.5 KB | bd997306fd03180d5242fe1bc8cacb131410ee29e322eb963a1f31990a0d1946 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_n_05.npz` | 590.9 KB | 0d12274cc04ba0246dc1ed327e93ec2cd2193e6fcb160fc21f005407f59e0616 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_n_05.npz` | 590.9 KB | 72e2aec8224249b2f8c5970c80c5de46e86c6b3d8eca1d85fd4cc09cf3766600 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_e_07_fitted_engine.obj` | 587.7 KB | 4c9131e0b6274bd28394aed910c235bd0fdc610096bad12d1e32856cbf8ac063 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_e_07_clicks_engine.obj` | 587.4 KB | 8066f3c3e93a45a9892ea26c069e93da83c889726e39b8045446d8022bb7bcda | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_n_04.npz` | 576.3 KB | 9ec074711cb3eff2539ea4308e8839a9a6eaa77acc581be368da1f8fefefbee3 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_n_04.npz` | 574.8 KB | 6f9cc02daeaaa70f1b8f59437c5770af809e7788a3cdccbd34a730a41485dc6a | leaflet split (amendment 5) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_tprox_n_01.npz` | 572.2 KB | 9e6f24332f5813ba2fc6d873ac1c3e71884ac5af260ba85b08ff0c6226201545 | T-Prox 11 plants | data file (not uploaded) |
| `build/mtg_build_20261102_01/kaggle/import_ds_dtheta/work_model.tar.bin` | 550.0 KB | 1350a971ab8234b99abdfd9680ba18cf3a40ce299a611e93c0fc56f842a67bb4 | MTG build | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_n_05_fitted_engine.obj` | 541.8 KB | eb0ac487c73b70f17af52b8335fb4cc1b2a8e9ff59e42dd938ef4264f01f3225 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_n_05_clicks_engine.obj` | 541.8 KB | 682eac4878d83ec28d5b4a7122eaf3feb18d4bbb6d698742a6a13827381c4875 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_e_04.npz` | 531.8 KB | 01f9d44116ac7129db75f98e1388618c2648435adb1ce648bb1eaa3028fa9269 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_e_04.npz` | 531.5 KB | 050930427837820fea378bb399eebfc0940f57d306e2e2f906ad39681234a105 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_20261102_01/kaggle/import_ds_dtheta/work_model_geo7_unpatched.tar.bin` | 520.0 KB | bafdd87f728ae79b29979246e1317864833b672c9e8e6edaf7959958978c072e | MTG build | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261103_02/kaggle/import_02/work_model.tar.bin` | 520.0 KB | bafdd87f728ae79b29979246e1317864833b672c9e8e6edaf7959958978c072e | MTG 11-03 package _02 (imported chain) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/kaggle/import_clicks/work_model.tar.bin` | 520.0 KB | bafdd87f728ae79b29979246e1317864833b672c9e8e6edaf7959958978c072e | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/kaggle/import_user/work_model.tar.bin` | 520.0 KB | bafdd87f728ae79b29979246e1317864833b672c9e8e6edaf7959958978c072e | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_tprox_n_06.npz` | 505.0 KB | ce78f0659217afe99959651916aebdbf13317c4a10c0558e42ae024298dd6b08 | T-Prox 11 plants | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_e_04_clicks_engine.obj` | 499.7 KB | f79a1457947b18ffe37efb0e452345969bc189fbcec76725ab968477b8d460e2 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_e_04_fitted_engine.obj` | 499.7 KB | 67084fbefb3390b0b68787bdae62c0296aec05a5a188f9dd24b51433fa1417c1 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_n_02_engine.obj` | 499.7 KB | 9e26067d55e831a74c541ae29ddf522300cfc2d76c9f3baf5530b322d904fb0a | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_e_01_fitted_engine.obj` | 499.2 KB | 391491a5731eee75e07f7a5e372531634ebc37fd56af31b1a0c84c475b446216 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_e_01_clicks_engine.obj` | 498.6 KB | 59e56bb184a4431d180503e4ea7bee13bd5c6c04a3884636109c10b45e2cdf44 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_n_06.npz` | 493.6 KB | 6f7cf386825f27dfad4d97c83544e7ef8f92b21ed63454f82e5bdac20278f122 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_n_06.npz` | 493.0 KB | 673479a805c39ef6a6a696d6efa1a889059e6183e673a621134ac9e0934af350 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_20261102_01/work_dtheta_print/` | 491.3 KB | 폴더, 27 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 27 files [.rgg:14,.csv:8,.diff:1,.xml:1,.gs:1] |
| `build/mtg_build_20261102_01/work_dtheta/` | 489.8 KB | 폴더, 27 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 27 files [.rgg:14,.csv:8,.diff:1,.xml:1,.gs:1] |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_n_01.npz` | 487.2 KB | a1c33ab4049727944d286f4d70d04bef77086948dc69080d3d90b6f654d8a4ff | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_n_01.npz` | 486.0 KB | f26b4cb651665eaf73a4133ebc1748aff724032cb421cdddf2bd300848630a26 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/p1_recon_r3-1_20261022_01/plant_points_mono.ply` | 480.1 KB | 7aeba9f40609d7559600882b59823f0eed67e6c887c7bcb32e615b67a0971b57 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_n_06_engine.obj` | 474.1 KB | 8365a676c854cf50ad51ee1303fe3bcd662204a86fcbc3c54725765eaa28a6b7 | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_e_04_engine.obj` | 468.2 KB | 74f458ef0b154f5ce7fedf4f2071cee648cf53cb86c29a7d31378fc05f4b4aa2 | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_tprox_e_01.npz` | 465.4 KB | 8d74f279ff2a91c5a5ee4d931c8c60ac854001e63e9c04ec69af23d48ed05c8f | T-Prox 11 plants | data file (not uploaded) |
| `build/mtg_build_20261102_01/kaggle/results_light/numba_cache/` | 463.9 KB | 폴더, 10 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 10 files [.nbc:5,.nbi:5] |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_n_03_fitted_engine.obj` | 453.9 KB | 36b8fba67720be6a8c60daf30790dc923b454bdbaa91143baeb982e8aafcd16b | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_n_03_clicks_engine.obj` | 453.9 KB | 24a21f76ba9af447c1eea8d14cc0032ac1cd9042149f372e158954f2b5ce1e38 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/p1_skeleton_layer_20261101_01/axis/` | 450.3 KB | 폴더, 20 파일 | skeleton_v1 / v1.1 format + axis tool | excluded dir (outputs/uploads/raw/cache), 20 files [.json:8,.csv:6,.log:3,.png:1,.npz:1] |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_R3-1b_fitted_engine.obj` | 434.8 KB | 9b7f19984291395e15a479c115196a90fe1e634398cc506cce487561b71f207b | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_R3-1b_clicks_engine.obj` | 434.6 KB | 6f26b6552a4d290145f56b30c56040b39d5a8c256a223b34e1b8173af3f483b6 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_tprox_t2_05.npz` | 420.5 KB | 1b2dff7bb5be9cebb0343fbecba9e3fbc3c53e037abfbc7c1eae8b6ec308b4f2 | T-Prox 11 plants | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_user/mtg_n_06_fitted_engine.obj` | 410.9 KB | 5c57a4845ae059d36225febfa26c6fa329f4ef7e5f392d981cbb2566ea853bcb | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_n_06_clicks_engine.obj` | 410.9 KB | 8ddc2865230f2b692773f15a9591c1bfceb7cdac206afe0690c828c760e55b2f | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_tprox_n_02.npz` | 381.8 KB | 4dc1e6c08a8d452372d57e8e3f1efe9a187b990337845b45f75e0675f148e63a | T-Prox 11 plants | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_template_mean_v2draft_engine.obj` | 379.9 KB | 679e6b15b2d5adb7db8430efc3fe16a57951f0b08d65ac914c3ceab2e1722454 | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_n_02.npz` | 377.4 KB | 799244f4fa1181c761eb3639393bdfc6141d38d5d432136417deb78da4100857 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_n_02.npz` | 376.8 KB | 48fdc58c1eef50512c8a58f20e50ca33432c988c3f256e89ae1cdad9c90b4250 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/p1_vp_recluster_20261026_01/run1_with_vp_share/surfel_clusters.npz` | 360.8 KB | c6d1cd6f5e466e425bc5b34b51ad99428386b6bdab7cf76310ec36f8d033a43f | V-P pilot | data file (not uploaded) |
| `build/p1_vp_recluster_20261026_01/surfel_clusters.npz` | 350.4 KB | 64f736a79b9fe0f9da0913d17fb208c5d6dc7741e3f9476c5f8d134431bc8a90 | V-P pilot | data file (not uploaded) |
| `build/mtg_build_v2_clicks_20261105_04/static/mtg_n_03_engine.obj` | 337.4 KB | 5b3a3396adc01cb47a39e067f22aefc59870d0ad213ec698acd4c55bb8a751a8 | MTG _04 = current static MTG set (8/11 pass) | data file (not uploaded) |
| `build/p1_recon_all_20261025_01/pts_n04/` | 329.5 KB | 폴더, 3 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 3 files [.json:2,.ply:1] |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_R3-1b.npz` | 328.4 KB | 8a5e63ae16f9575b7ad3fc378804027626ad87fb1132a7ef0cca7750295af4a3 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_R3-1b.npz` | 328.3 KB | 006ffc29fcc17537b265bb0d26584f90a4a7f2da63c429c2b26aa8b798a3fa9e | leaflet split (amendment 5) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_tprox_e_04.npz` | 327.4 KB | a9482ddf3136839d6f692c284e2eea122745780f00d864937067dad4d4001eea | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_recon_charuco_20261021_01/visual_hull.ply` | 294.5 KB | 1fad62627939d86e166ea0634a2f74f0abda78bba002e3091b0ffdc37b70cf5a | ChArUco pose (board 10x7, 38 mm) | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/withheld_amendment3/split_n_03.npz` | 275.4 KB | 2371ffc2c3d94cb2ed91a4cd3213dc7e6a7b0550424051d1c072afa13743e3da | leaflet split (amendment 5) | data file (not uploaded) |
| `build/p1_recon_r3-1_20261022_01/plant_points_a.ply` | 275.1 KB | 5d13a42f049a93af267d17eed31875cc617d2bc6ca8a77fd0914bd04c0a30256 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | data file (not uploaded) |
| `build/mtg_build_v2_clicks_leaflets_20261105_01/split_n_03.npz` | 274.9 KB | ee512d8aa5b520ee168f99681fe4b55b15839888a439008eb8e372ba69025239 | leaflet split (amendment 5) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_tprox_n_03.npz` | 268.6 KB | fea769da1c2c592fedda8a8f4a912a971f39930bab5bcb14172917ba35126317 | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_vp_recluster_20261026_01/photo_features_base.npz` | 266.0 KB | 978c43a5234b091f8ef35395f221f87a0807c80d9c615ba397c854976e82ae72 | V-P pilot | data file (not uploaded) |
| `build/p1_recon_all_20261025_01/census_new/` | 263.7 KB | 폴더, 5 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 5 files [.csv:3,.json:1,.md:1] |
| `build/p1_recon_charuco_20261021_01/_TRASH/` | 250.9 KB | 폴더, 25 파일 | ChArUco pose (board 10x7, 38 mm) | excluded dir (outputs/uploads/raw/cache), 25 files [.log:17,.pid:4,.txt:1,.db:1,.db-shm:1] |
| `build/p1_pc_absorb_20261027_01/R8_pc_rachis/labels_rachis.npz` | 238.5 KB | ddcbac13dce1d34c6d25102e1b417a24592cc592ce5177f3fa05beb4ca8cfe5d | R8_pc random walker (LATEST_PIPELINE) | data file (not uploaded) |
| `build/p1_pc_absorb_20261027_01/R8_pc/labels.npz` | 238.1 KB | e1ff792be1c4efdff7e0a6238b95a2d532ca4eb0d92a5f66676dc9bdc554267b | R8_pc random walker (LATEST_PIPELINE) | data file (not uploaded) |
| `build/p1_pc_absorb_20261027_01/smoke_test/set_A/labels.npz` | 211.2 KB | 5a7fa51e144f0d4529352488306ba0ded94ea26e0f717395dc9054510eb689be | R8_pc random walker (LATEST_PIPELINE) | data file (not uploaded) |
| `build/p1_pc_absorb_20261027_01/smoke_test/R8_pc/labels.npz` | 211.2 KB | c17361574008df2115f1aabcacbfa7fc3bccbf6df41bcbb0d4ecd3fd0e7bba7b | R8_pc random walker (LATEST_PIPELINE) | data file (not uploaded) |
| `build/p1_pc_absorb_20261027_01/smoke_test/set_C/labels.npz` | 211.2 KB | 0da4965b3b3655233d212a654c25db7aec10751e27ec90d2ea1dee89b8e4cd0c | R8_pc random walker (LATEST_PIPELINE) | data file (not uploaded) |
| `build/p1_recon_r3-1_20261022_01/plant_points_mono_only.ply` | 205.2 KB | 57135341ecaabc0d842f8dd3e2e0c7214e49a38c4a801d4d307eaa8090c06eff | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_distance_n_05.npz` | 187.4 KB | 3f9d478c91774f4b55cc373270fbaa21faf6beb455db68f02b9e97d33fbadfdb | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_distance_n_04.npz` | 183.2 KB | 00ec88e4a7abeafaf6717030ee58689c9a2e020d26a8e16605067ddfc34c2013 | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_distance_e_05.npz` | 176.0 KB | e2ea96486179c6e5e0594ffca0a5aeee5b116fa5ff332a2c129c78f3f7ffe753 | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_distance_t2_02.npz` | 168.9 KB | e9c840b7ae2fcc59d1306a31d0c440e61e8db499bee503c2eb08654dac9d31e6 | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_distance_n_06.npz` | 153.7 KB | 257702eed474e329de41d17008ed02a167110031a9198e379cc06b0179ffcf56 | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_recon_all_20261025_01/pts_n01/` | 153.6 KB | 폴더, 3 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 3 files [.json:2,.ply:1] |
| `build/p1_tprox_others_20261101_01/labels_distance_n_01.npz` | 153.3 KB | 2b86346051faf503dc17216388cb86770d34b93cb9abfd288b1f18e360819335 | T-Prox 11 plants | data file (not uploaded) |
| `tomato3d/leafseg/__pycache__/` | 147.0 KB | 폴더, 7 파일 | H2 leaflet masks (SAM) | excluded dir (outputs/uploads/raw/cache), 7 files [.pyc:7] |
| `build/p1_vp_cleanup_20261026_01/surfel_clusters.npz` | 143.7 KB | b75236d584ce744af15db7cfaa3061a6de62342a3a69fec0b42eeb0e4c82128d | V-P pilot | data file (not uploaded) |
| `build/p1_recon_r3-1_20261022_01/human/` | 136.8 KB | 폴더, 4 파일 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | excluded dir (outputs/uploads/raw/cache), 4 files [.json:4] |
| `build/p1_pc_absorb_20261027_01/R8_pc_unmix/labels_unmix.npz` | 134.5 KB | a5658a45e8d8f4c6db0d03841f3f29fa856cbb3319fcee41d41cabc8e154464a | R8_pc random walker (LATEST_PIPELINE) | data file (not uploaded) |
| `build/p1_pc_absorb_20261027_01/R8_pc_unmix_tuned/labels_unmix.npz` | 133.7 KB | e9a9a639dcbce5c523e750afaba03702386a8f9e5f6092e7de056178497eabf1 | R8_pc random walker (LATEST_PIPELINE) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_distance_e_01.npz` | 132.6 KB | 7a4fffc37ea65237262d1a5a06ea8c3d726ed8fc3f7658aa57b416dfb41cf537 | T-Prox 11 plants | data file (not uploaded) |
| `tomato3d/recon/__pycache__/` | 132.5 KB | 폴더, 10 파일 | 2DGS recon (R3-1b method) + silhouette checks | excluded dir (outputs/uploads/raw/cache), 10 files [.pyc:10] |
| `build/mtg_build_v2_clicks_20261103_02/kaggle/import_02/scenarios.tar.bin` | 130.0 KB | 20bd4ebae97078012188096bb8394bbb02d7cb04958e8f5e0c56cf9811b85901 | MTG 11-03 package _02 (imported chain) | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_distance_t2_05.npz` | 129.9 KB | 66179ab5da3d8adade1b63d9a5ca42d97c94e359f0cd5cd3f338496ad9ca3a0a | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_distance_n_02.npz` | 113.7 KB | 57c006c83a40aeb0f090b7de68a69ea56694f4190c1e82ff8d69bb94aa1b8092 | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_recon_all_20261025_01/pts_t202/` | 110.5 KB | 폴더, 3 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 3 files [.json:2,.ply:1] |
| `build/mtg_build_v2_clicks_20261103_01/dynamic/` | 110.2 KB | 폴더, 7 파일 | MTG 11-03 package (imported chain) | excluded dir (outputs/uploads/raw/cache), 7 files [.csv:7] |
| `build/mtg_build_v2_clicks_20261103_02/dynamic/` | 110.2 KB | 폴더, 7 파일 | MTG 11-03 package _02 (imported chain) | excluded dir (outputs/uploads/raw/cache), 7 files [.csv:7] |
| `build/mtg_build_v2_clicks_20261105_03/dynamic/` | 110.2 KB | 폴더, 7 파일 | MTG _03 (leaflet split real run) | excluded dir (outputs/uploads/raw/cache), 7 files [.csv:7] |
| `build/p1_tprox_others_20261101_01/labels_distance_e_04.npz` | 105.8 KB | 46bd9f2401dae7da97f78e6e538653176fbab7d6d9aabdf804ec51c6da52461d | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_tprox_others_20261101_01/labels_distance_n_03.npz` | 82.8 KB | beb2231884d78ad7ab9d3b3cd25f8873d8f5810f3ce273f772d3d7f72edbf804 | T-Prox 11 plants | data file (not uploaded) |
| `build/p1_skeleton_layer_20261101_01/engine_n_04/` | 81.5 KB | 폴더, 10 파일 | skeleton_v1 / v1.1 format + axis tool | excluded dir (outputs/uploads/raw/cache), 10 files [.csv:6,.json:4] |
| `build/mtg_fit_surfels_20261103_01/kaggle/import_clicks/scenarios.tar.bin` | 80.0 KB | 1953bcdcd16f28e8408f4a12950c2715ee694e39ff1ea242c3d6c3dd008112f7 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/mtg_fit_surfels_20261103_01/kaggle/import_user/scenarios.tar.bin` | 80.0 KB | 8480cf801f32422ccaf26d575e30cc1db1e1a845ab8a90f1fd05f3c63b74a0ae | MTG fit (fit_plant / run_real, imported by _03/_04 split) | data file (not uploaded) |
| `build/p1_recon_charuco_20261021_01/plant_points.ply` | 35.3 KB | 1c25d93965f7e2903d010769368ffed56526673f5e9af92e69f601b6fa6b6615 | ChArUco pose (board 10x7, 38 mm) | data file (not uploaded) |
| `build/p1_recon_r3-1_20261022_01/plant_points_mono_tag.npy` | 32.1 KB | 8e191ca20bb62ca5d5460f3c0a054e737f58c23cc5a8d45c5f997996fce60dc5 | R3-1b 2DGS + SAM 2.1 masks + held-out IoU | data file (not uploaded) |
| `build/mtg_build_20261102_01/kaggle/kernels/` | 30.2 KB | 폴더, 10 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 10 files [.json:5,.py:5] |
| `build/mtg_build_20261102_01/tests/one_row_engine.obj` | 25.6 KB | 4f940882f7fd3720e48a1e9dec83f41f6b8afeb8483b57cd2def7fbba3167c54 | MTG build | data file (not uploaded) |
| `tomato3d/plant_tree/__pycache__/` | 22.4 KB | 폴더, 2 파일 | plant_tree_v1 (MTG structure) | excluded dir (outputs/uploads/raw/cache), 2 files [.pyc:2] |
| `build/mtg_build_20261102_01/inputs/` | 15.2 KB | 폴더, 1 파일 | MTG build | excluded dir (outputs/uploads/raw/cache), 1 files [.json:1] |
| `build/mtg_fit_surfels_20261103_01/inputs/` | 12.5 KB | 폴더, 1 파일 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | excluded dir (outputs/uploads/raw/cache), 1 files [.json:1] |
| `build/mtg_fit_surfels_20261103_01/kaggle/kernels/` | 12.4 KB | 폴더, 4 파일 | MTG fit (fit_plant / run_real, imported by _03/_04 split) | excluded dir (outputs/uploads/raw/cache), 4 files [.json:2,.py:2] |
| `build/p1_recon_all_20261025_01/pts_d1028/` | 10.5 KB | 폴더, 8 파일 | 9-plant R3-1b-method recon (held-out IoU 0.747-0.896) | excluded dir (outputs/uploads/raw/cache), 8 files [.json:8] |
| `build/traits_recon_20261101_01/inputs/` | 10.3 KB | 폴더, 1 파일 | traits + MTG templates (v1..v2.4, curvature/span/arrangement v1) | excluded dir (outputs/uploads/raw/cache), 1 files [.xlsx:1] |
| `build/mtg_build_v2_clicks_20261103_02/kaggle/kernels/` | 6.2 KB | 폴더, 2 파일 | MTG 11-03 package _02 (imported chain) | excluded dir (outputs/uploads/raw/cache), 2 files [.json:1,.py:1] |
| `build/p1_all_3dfirst_20261027_01/ext/` | 658 B | 폴더, 3 파일 | 3D-first groups + 8835 click tool | excluded dir (outputs/uploads/raw/cache), 3 files [.json:2,.done:1] |

## 사진 파생 의심 그림 (사용자 결정으로 제외)

| 경로 | 크기 | sha256 |
|---|---|---|
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_R3-1b.png` | 597.9 KB | 7b12adcefbd0dd45c82d58465e23de943ba518e704b64b559293bed74f39883c |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_R3-1b_run1.png` | 597.4 KB | 94576bef041774ccadff05beb0f7a7a49cac0adb57b3fbafd9ac82e58d9de00f |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_R3-1b_run2.png` | 597.4 KB | 94576bef041774ccadff05beb0f7a7a49cac0adb57b3fbafd9ac82e58d9de00f |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_01.png` | 651.9 KB | 76c3c07fb1517196aabf40db396707c6a73e37bd7a0cacfc4882bd2b77ebf8e9 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_01_run1.png` | 645.4 KB | c734d2ee1d13f29a6762e5d37db27bc60324a3c3644179d367f000460ea2c6d9 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_01_run2.png` | 651.9 KB | 76c3c07fb1517196aabf40db396707c6a73e37bd7a0cacfc4882bd2b77ebf8e9 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_04.png` | 652.2 KB | b924ce2b1ba7363cf00e19bb2db36583b2d691a7aa41c5a1a13c490316d2c2b1 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_04_run1.png` | 651.8 KB | e43a6f54bdd143bf9a8f8be7519dff4da0ab16fb0463b4b019d63fbc7f418943 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_04_run2.png` | 652.2 KB | b924ce2b1ba7363cf00e19bb2db36583b2d691a7aa41c5a1a13c490316d2c2b1 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_05.png` | 786.7 KB | 24d75749b00a5b2590769f05feadc243f5df8133fb5c282026e3777c0f2d916c |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_05_run1.png` | 785.3 KB | 4393ea9cf3bb6700cb34efd14e259dd21c25d15c7c0f2522a6fde33890f19b36 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_05_run2.png` | 787.7 KB | 2c77b08bbeff9737f3eb0a60255a3814d92062eb5799334b15ef31596cafdb58 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_07.png` | 797.7 KB | cfa07acda1b9c594071428194f5e48a1a9eecbbfe7e5f4940d6ff2d709b9d840 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_07_run1.png` | 789.4 KB | 434d64ff0bc45039fb54f743e074f9873e4ca0a3569d74bf39e0e2466fc6261d |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_e_07_run2.png` | 792.3 KB | 0ee314921184d5e80f66e9e4217b7d803cc3bce4634eab46e29bba1730fe9b2e |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_01.png` | 627.4 KB | 65d25862acf62e234a43fc5ccadb402bdcbf5aa563f6c7531f023d820ec267d7 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_01_run1.png` | 627.4 KB | 65d25862acf62e234a43fc5ccadb402bdcbf5aa563f6c7531f023d820ec267d7 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_01_run2.png` | 627.4 KB | 65d25862acf62e234a43fc5ccadb402bdcbf5aa563f6c7531f023d820ec267d7 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_02.png` | 662.4 KB | a32e8a79e2b074a5af885e2c796309c74cf6cad694d01746fd6d889a1eb697f3 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_02_run1.png` | 662.4 KB | a32e8a79e2b074a5af885e2c796309c74cf6cad694d01746fd6d889a1eb697f3 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_02_run2.png` | 662.4 KB | a32e8a79e2b074a5af885e2c796309c74cf6cad694d01746fd6d889a1eb697f3 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_03.png` | 633.4 KB | 7d6f859872e3138d2a693a67a9ef0b8ed555906b0086f94afa592c88de01ebbc |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_03_run1.png` | 633.4 KB | 7d6f859872e3138d2a693a67a9ef0b8ed555906b0086f94afa592c88de01ebbc |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_03_run2.png` | 633.4 KB | 7d6f859872e3138d2a693a67a9ef0b8ed555906b0086f94afa592c88de01ebbc |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_04.png` | 762.0 KB | a235437a48f3e0d52c4c701adf99760610479d9acb38cbd5aec925e3f9455db3 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_04_run1.png` | 762.0 KB | a235437a48f3e0d52c4c701adf99760610479d9acb38cbd5aec925e3f9455db3 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_04_run2.png` | 762.0 KB | a235437a48f3e0d52c4c701adf99760610479d9acb38cbd5aec925e3f9455db3 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_05.png` | 912.5 KB | 0f9396a454f5478d3538739d4db7e1a8b54844caeac374607e13f056e92e6bbd |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_05_run1.png` | 913.8 KB | 662f6402590799427e86243892878157333a41455ee8f0f7d5dc578194c8c2cd |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_05_run2.png` | 913.8 KB | 662f6402590799427e86243892878157333a41455ee8f0f7d5dc578194c8c2cd |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_06.png` | 732.6 KB | 4faae65c09c4810ebd681bcdc10c2f8c25aff71dd8085236ca960fa68e16ab51 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_06_run1.png` | 731.7 KB | fd0d31d482e340b512211cb20b2e01dd839f2430089062b629e9cdfe275b1851 |
| `build/mtg_build_v2_clicks_curv_20261105_01/figures/overlay_n_06_run2.png` | 732.6 KB | 4faae65c09c4810ebd681bcdc10c2f8c25aff71dd8085236ca960fa68e16ab51 |

## 공개판에서 뺀 VTC 파생 파일

상위 [`VTC_WITHHELD.md`](../VTC_WITHHELD.md) 참조 (파일 목록·크기·sha256·영향받는 스크립트).
