# DATA.md — 저장소에 넣지 않은 자료 (microtom-dynamic-fspm)

원자료·대용량 산출물은 올리지 않고 여기 목록만 둡니다. 공유: **요청 시 (on request)** — 저자에게 연락.
경로는 원 프로젝트 기준 (`<PROJECT_ROOT>/...`). 폴더 행은 파일별 sha256 을 계산하지 않았음 (OneDrive 클라우드 파일을 내려받지 않기 위해); 필요하면 요청 시 매니페스트 제공.

## 외부 의존 (저장소 밖)

| 항목 | 위치 | 비고 |
|---|---|---|
| DEV 엔진 사본 | `C:/fsp_build/engine_dev_20261020` (`<FSP_BUILD_ROOT>`) | `prepare_v2.py`·이데오타입·GUI 재빌드에 필요. MR6/MR7 미병합 수정 포함. VTC 파생 → 원저자 허락 전 공유 불가 |
| GroIMP 2.2.1 | `C:/Program Files/GroIMP 2.2.1` (`<GROIMP_HOME>`) | https://download.grogra.de/ |
| Kaggle 데이터셋 `engv2-*`, `tomato-h-input`, `tomato-dreftray-input-*` | kaggle.com/dolgfish | 커널 입력 (코드 묶음·GroIMP·triangle 휠) |
| 복원 surfel·MTG 입력 | 저장소 `microtom-2dgs-recon` 의 DATA.md (`p1_recon_all_20261025_01/outputs/`, `mtg_build_v2_clicks_20261105_04/static/`) | 2 MB 이하 MTG 파일은 recon 저장소에 들어 있음 |

## 목록 (크기순)

| 경로 | 크기 | sha256 / 파일 수 | 만든 폴더 (결과) | 이유 |
|---|---|---|---|---|
| `build/engine_gui_canopy_20261106_01/gui/gui_runs/` | 5.8 GB | 폴더, 26 파일 | GUI canopy tool | excluded dir (outputs/uploads/raw/cache), 26 files [.csv:24,.json:2] |
| `build/engine_v2_calib_20261104_01/dataset/` | 1.1 GB | 폴더, 176 파일 | Engine v2 / v2.1 / v2.2 slaA+slaR frozen + calibration C4A + validator V12 | excluded dir (outputs/uploads/raw/cache), 176 files [.gz:88,.json:88] |
| `build/engine_v2_calib_20261104_01/kout/` | 266.8 MB | 폴더, 3060 파일 | Engine v2 / v2.1 / v2.2 slaA+slaR frozen + calibration C4A + validator V12 | excluded dir (outputs/uploads/raw/cache), 3060 files [.json:1861,.txt:985,.log:148,.npz:63 |
| `build/engine_ideotype_20261105_01/dataset/` | 76.4 MB | 폴더, 32 파일 | ideotype s10h40 (stopped; stale RUNNING.lock) | excluded dir (outputs/uploads/raw/cache), 32 files [.gz:16,.json:16] |
| `build/engine_gui_canopy_20261106_01/dataset/` | 63.9 MB | 폴더, 52 파일 | GUI canopy tool | excluded dir (outputs/uploads/raw/cache), 52 files [.gz:26,.json:26] |
| `build/engine_ideotype_20261105_01/kout/` | 23.3 MB | 폴더, 147 파일 | ideotype s10h40 (stopped; stale RUNNING.lock) | excluded dir (outputs/uploads/raw/cache), 147 files [.json:83,.txt:44,.log:13,.npz:7] |
| `build/engine_v2_calib_20261104_01/kout_c6_on_v2/` | 23.0 MB | 폴더, 515 파일 | Engine v2 / v2.1 / v2.2 slaA+slaR frozen + calibration C4A + validator V12 | excluded dir (outputs/uploads/raw/cache), 515 files [.json:329,.txt:165,.npz:12,.log:9] |
| `build/engine_gui_canopy_20261106_01/kout/` | 8.4 MB | 폴더, 475 파일 | GUI canopy tool | excluded dir (outputs/uploads/raw/cache), 475 files [.json:160,.csv:121,.log:97,.txt:97] |
| `build/engine_gui_canopy_20261106_01/gui/_build_slaR/` | 3.6 MB | 폴더, 4 파일 | GUI canopy tool | excluded dir (outputs/uploads/raw/cache), 4 files [.rgg:2,.json:1,.csv:1] |
| `build/engine_gui_canopy_20261106_01/gui/_build_slaA/` | 3.6 MB | 폴더, 4 파일 | GUI canopy tool | excluded dir (outputs/uploads/raw/cache), 4 files [.rgg:2,.json:1,.csv:1] |
| `build/engine_gui_canopy_20261106_01/gui/leaflet_outline_uv.csv` | 3.5 MB | be98c746835738c7e9211fffe9a0f43f9861183d8d7437fd800ab908153da986 | GUI canopy tool | large derived table/geometry (>2 MB); listed with sha256 |
| `build/engine_v2_calib_20261104_01/work/leaflet_outline_uv.csv` | 3.5 MB | be98c746835738c7e9211fffe9a0f43f9861183d8d7437fd800ab908153da986 | Engine v2 / v2.1 / v2.2 slaA+slaR frozen + calibration C4A + validator V12 | large derived table/geometry (>2 MB); listed with sha256 |
| `build/engine_v2_calib_20261104_01/kout_failed/` | 1.4 MB | 폴더, 169 파일 | Engine v2 / v2.1 / v2.2 slaA+slaR frozen + calibration C4A + validator V12 | excluded dir (outputs/uploads/raw/cache), 169 files [.json:67,.log:35,.txt:35,.csv:32] |
| `build/engine_v2_calib_20261104_01/kernels/` | 653.7 KB | 폴더, 274 파일 | Engine v2 / v2.1 / v2.2 slaA+slaR frozen + calibration C4A + validator V12 | excluded dir (outputs/uploads/raw/cache), 274 files [.json:137,.py:137] |
| `build/engine_gui_canopy_20261106_01/kernels/` | 158.3 KB | 폴더, 68 파일 | GUI canopy tool | excluded dir (outputs/uploads/raw/cache), 68 files [.json:34,.py:34] |
| `build/engine_ideotype_20261105_01/kernels/` | 89.6 KB | 폴더, 38 파일 | ideotype s10h40 (stopped; stale RUNNING.lock) | excluded dir (outputs/uploads/raw/cache), 38 files [.json:19,.py:19] |

## 공개판에서 뺀 VTC 파생 파일

상위 [`VTC_WITHHELD.md`](../VTC_WITHHELD.md) 참조 (파일 목록·크기·sha256·영향받는 스크립트).
