# microtom-dynamic-fspm — Micro-Tom 동적 FSPM (GroIMP, VTC 파생)



> **공개판 안내 (tomato-2DGS-engine):** VTC 파생 파일(모델 `.rgg`·`.gsz`, `src/tools/vtc_growth_fork.py`, `src/tools/vtc_development.py`, 모델 비교 `.diff`)은 원저자 허락 전이라 이 저장소에 넣지 않았습니다. 목록·sha256·그 때문에 돌지 않는 스크립트는 상위 [`VTC_WITHHELD.md`](../VTC_WITHHELD.md).
> *Public edition: VTC-derived files are withheld; see [`../VTC_WITHHELD.md`](../VTC_WITHHELD.md) for the list, hashes and the scripts that need them.*

## 무엇인가
GroIMP 2.2.1 에서 도는 Micro-Tom 성장·광 모델입니다. VTC (Virtual Tomato Crop, K. Smolenova / WUR) 모델을 고쳐 만들었습니다.
- 구조 입력은 `microtom-2dgs-recon` 의 MTG 템플릿 v2.4 입니다.
- 동결 판은 v2, v2.1, v2.2 두 갈래(`slaA`, `slaR`) 입니다.
- 함께 들어 있는 것: 보정·검증기 스크립트, 이데오타입·tray32 동적 장면 분석, GUI 캐노피 도구.

## 폴더
- `src/` = 원 프로젝트와 **같은 상대 경로** (`src/build/<작업 폴더>/`, 필요한 `src/tools/`, `src/tomato3d/`).
- 동결 모델은 `src/build/engine_v2_calib_20261104_01/work/` 에 있습니다. 경로를 바꾸면 해시 기록과 스크립트가 어긋나므로 옮기지 않았습니다.
- 입력 템플릿: `src/build/traits_recon_20261101_01/MTG_TEMPLATE_v2_4.csv` (recon 저장소 산출물, 21 KB 라 복사).
- 계수 정리표: `src/handoff/상수_계수_식_정리_정적_동적_2026-11-05.md`.
- `configs/paths.yaml`, `docs/SUPERSEDED.md` (철회·후보 판), `DATA.md` (DEV 엔진 사본·GroIMP·Kaggle 데이터셋 등 외부), `THIRD_PARTY.md`.

## 동결 모델 sha256 (줄끝 CRLF 바이트 기준; `.gitattributes` 가 바이트를 그대로 보존)
| 판 | 파일 | sha256 |
|---|---|---|
| v2 | `work/Model_v2.rgg` | `fa53fe89a660e58383a9c52905b4e9c6b09ffd8338d7df0334f55d2898faf090` |
| v2.1 | `work/Model_v2_1.rgg` | `0a408633872ec31f5d8bfff6e3ecfee671b12140c5a47ff56bae78fc459b41ac` |
| v2.2 slaA | `work/Model_v2_2_slaA.rgg` | `8b04647ecc2271225b04f6ae2e8666fb0b9d8e621a692562f32f8cc82c9e09a6` |
| v2.2 slaR | `work/Model_v2_2_slaR.rgg` | `30e0df8c3544b08b599c4e5fa4252287b9eef4be9edc180344658640e0600be6` |
| slaA 매개변수 | `work/parametersMicroTom_v2_2_slaA.rgg` | `004927904f093f5374cf4f629047d1cb068b2fe1094a7666051ad27c8fa15d43` |
| slaR 매개변수 | `work/parametersMicroTom_v2_2_slaR.rgg` | `b4e5b879aec0f1570bddd80951b1180aa187e99be322ed1381cc1abd6b94c9a5` |

`slaA` = SLA ×1.3271 (잎 면적 목표 수용), `slaR` = SLA ×1.15 (잎 면적이 목표보다 작음). 두 갈래 모두 키 중앙값을 약 130 mm 에 맞췄습니다 (slaA 129.7 mm·잎 36.3 cm², slaR 130.5 mm·잎 24.3 cm²; 출처 `src/build/engine_v2_calib_20261104_01/acceptance_v2_1_v2_2.csv`, 16 seed). 철회된 단일 갈래 v2.2 (`*_s4.10_withdrawn`)·후보 판은 기록용입니다.

## 실행
| 내용 | 진입점 |
|---|---|
| GUI 캐노피 도구 (간격·램프·차트) | GroIMP 2.2.1 에서 `src/build/engine_gui_canopy_20261106_01/gui/TomatoCanopyGUI_slaA.gsz` (또는 `_slaR`) 열기. 또는 `START_GUI_CANOPY.bat` / `LIVE_CHARTS.bat` (설치 경로 `C:/fsp_build`, `C:/Program Files/GroIMP 2.2.1` 가정 — `configs/paths.yaml`). GUI 모델 맨 위 `GC_HOME` 은 `<PROJECT_ROOT>` 자리표시자이므로 먼저 자기 경로로 바꿀 것. |
| 동결 판 빌드·보정 (C4A 등) | `src/build/engine_v2_calib_20261104_01/prepare_v2.py`, `make_params_v2.py` — **DEV 엔진 사본 (`C:/fsp_build/engine_dev_20261020`) 필요, 저장소에 없음** (`DATA.md`) |
| 검증기 (V12 포함) | `src/build/engine_v2_calib_20261104_01/kaggle_kernel/v2_validate.py` |
| 이데오타입 (s10h40) | `src/build/engine_ideotype_20261105_01/ideo_patch.py`, `prepare_ideo.py`, `analyze_ideo.py` (중단된 실행; 표 `paired_effects_s10h40.csv`, `plants_summary_s10h40.csv`) |
| tray32 동적 장면 | `src/build/engine_tray32_dyn_20261107_01/analyze_*.py`, 그림 `make_*_fig*.py` |

실행 묶음은 Kaggle CPU 에서 돌렸습니다 (커널 이름은 `KERNELS_*.json`; 계정 이름 `dolgfish` 를 그대로 둠). `pip install -r requirements.txt` 는 분석·그림 스크립트용입니다.

## 상태 라벨
| 라벨 | 뜻 |
|---|---|
| `not_validation` | 시나리오·모델 결과이며 측정으로 검증되지 않음 (이데오타입·tray32 는 시나리오) |
| `relative_light` | 빛은 상대값 |
| `conditional_on_unvalidated_scale` | MTG 구축 자료(복원 스케일)에 맞춘 조건부 보정 |
| `skeleton_labels_ungraded` | MTG 입력의 복엽 라벨은 파일럿 외 채점 없음 |
| `assumed_lamp_model` | 램프는 GroIMP AreaLight 가정 모델; 법선에서 약 18–20° 넘으면 cos 법칙보다 많이 냄 |

v2.x 는 MTG 구축 자료에 맞춘 보정(`calibrated_on_mtg_building_data`)이며 독립 측정으로 검증한 것이 아닙니다. SLA 는 문헌 범위 밖입니다.

## 내부 이름
| 이름 | 뜻 |
|---|---|
| C4A | 보정 블록: 복엽 축 길이를 순위별로 맞춤 (`C4A_rachis_mult*.json`) |
| C1–C6, C4C/C4L/… | 보정 블록 (`BLOCK_*.json`, `CONTRACT.md` Amendments) |
| V12 | 검증기 항목: 동시 생장 순서에서 자체 탄소로 크는 사슬 길이 |
| `slaA` / `slaR` | v2.2 두 갈래 (위) |
| m00 | 기준 개체 번호 |
| VTC / NRH | 원 모델 / 원 모델의 영양·자원 처리 함수 |

## 라이선스 — 주의
- 코드 MIT (`LICENSE`), 그림·표 CC BY 4.0 (`LICENSE-CC-BY-4.0.md`).
- **VTC 파생 파일은 이 공개 저장소에 없습니다** (`../VTC_WITHHELD.md`). 대상: `Model_*.rgg`, `Model_gui_canopy_*.rgg`, `parametersMicroTom_*.rgg`, `TomatoCanopyGUI_*.gsz`, `src/tools/vtc_growth_fork.py`, `vtc_development.py`.
- VTC 원본(`vtc-main`)에는 LICENSE 파일이 없어 재배포 조건을 알 수 없습니다. 그래서 공개판에서는 뺐고, 넣으려면 원저자(K. Smolenova / WUR) 허락이 필요합니다. 위 sha256 표는 파일을 따로 받은 사람이 확인하는 용도입니다. 원 논문: https://doi.org/10.1093/insilicoplants/diaf022

---
**English summary.**
- Micro-Tom growth and light FSPM for GroIMP 2.2.1, derived from VTC (K. Smolenova / WUR).
- Frozen models v2, v2.1 and v2.2 in two arms (`slaA` SLA ×1.327, `slaR` SLA ×1.15; both height median ≈ 130 mm), with sha256 above (CRLF bytes, preserved by `.gitattributes`).
- Calibration (C4A) and validator (V12) scripts, ideotype and dynamic tray32 analyses, and a GroIMP GUI canopy tool (`.gsz`, `.bat`).
- Rebuilding needs the external DEV engine copy (`C:/fsp_build/engine_dev_20261020`) and GroIMP; both are listed in `DATA.md`.
- Results are calibrated on MTG-building data and are not validated; the lamp is an assumed model.
- **The VTC-derived files are withheld from this public repository** (VTC has no licence file); the hashes above let holders of those files verify them.
