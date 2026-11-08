# microtom-static-light — 복원 개체 위의 정적 광 모델

**저자:** 서현우, 김민웅, 이주원, 최재임 (경희대학교 스마트팜과학과) · 버전 0.1.0 (2026-11-08) · 공개 저장소 `tomato-2DGS-engine` 의 하위 폴더

> **공개판 안내 (tomato-2DGS-engine):** VTC 파생 파일(모델 `.rgg`·`.gsz`, `src/tools/vtc_growth_fork.py`, `src/tools/vtc_development.py`, 모델 비교 `.diff`)은 원저자 허락 전이라 이 저장소에 넣지 않았습니다. 목록·sha256·그 때문에 돌지 않는 스크립트는 상위 [`VTC_WITHHELD.md`](../VTC_WITHHELD.md).
> *Public edition: VTC-derived files are withheld; see [`../VTC_WITHHELD.md`](../VTC_WITHHELD.md) for the list, hashes and the scripts that need them.*

## 무엇인가
2DGS 로 복원한 Micro-Tom 개체(surfel)에 그림자 광선 추적기로 램프 빛을 쏘아 잎·개체별 상대 흡수광을 계산합니다. 여기에는 다음이 포함됩니다.
- 잎 광학 (τ rule 2, 같은 잎 법선 평활)
- 캐노피 양 (LAI 범위, 5 mm 층 투과)
- 32개체 트레이 장면 (tray32 v2/v3)
- 소엽 배치 비교 (원판 D vs 삼각형 T)
- GroIMP 와의 closure 점검

입력(surfel·MTG)은 `microtom-2dgs-recon` 이 만듭니다. 이 저장소에는 넣지 않았고 이름·크기·sha256 은 `DATA.md` 에 있습니다.

## 폴더
- `src/` = 원 프로젝트와 **같은 상대 경로** (`src/build/<작업 폴더>/`, 필요한 `src/tools/`, `src/tomato3d/`). 결과 CSV·그림은 각 작업 폴더 안에 있습니다.
- `configs/paths.yaml` (자리표시자 뜻), `docs/SUPERSEDED.md`, `DATA.md`, `THIRD_PARTY.md`.
- 46점 실측 광장 U (빈 판, PAR meter, 원측정값): `src/data/records/env_20261013/ppfd_grid_20261013.csv`.

## 추적기 두 판
| 판 | 위치 | `tracer.py` sha256 | 쓰인 곳 |
|---|---|---|---|
| v1 (검은 잎) | `src/build/engine_static_20261026_01/static/` | `84e435abaac3ec0ca949464009df9530bb5b7a307afffe145d43c6cf0dcb8dd8` | 첫 채택 실행, 9개체 Tier 0/1 |
| tau 묶음 (현행, v1_tau) | `src/build/engine_static_tau_20261031_01/kaggle_data/static/` + `tau_config.json` | `5844e991c51c638c5902c2632a134a5fb1780e99a0a2d2add2c8436713c6e4f3` | v1_tau 기준, rule 2, tray32, LAI, D vs T |

같은 잎 법선 평활 모듈: `src/build/engine_static_normals_v2_20261104_01/kaggle_flat/normals.py` (r 2 mm, `normals_v2_config.json`). v1 판은 `src/build/engine_static_normals_20261104_01/kaggle_data/normals.py`.
이 모듈들은 원래 Kaggle 업로드 폴더 안에만 있었습니다. 결과를 만든 코드라서 코드 파일만 넣었습니다.

## 주요 스크립트 (정확한 명령·입력 해시는 각 폴더 `CONTRACT.md`)
| 내용 | 진입점 | 결과 표 |
|---|---|---|
| τ rule 2 (줄기 = 주축 선 d_stem 1.5 mm, 나머지 τ 0.07 / a 0.85) | `src/build/engine_static_tau_rule2_20261105_01/kaggle_k1*/`, `make_table_unified.py` | `RESULT_TABLE_rule2_unified.csv` |
| v1 vs v1_tau 기준 + GroIMP τ closure | `src/build/engine_static_v1tau_reference_20261104_01/` | `RESULT_TABLE_v1_vs_v1tau.csv` |
| LAI 범위 (min / max_fill / up_area), 5 mm 층 투과 | `src/build/engine_static_lai_range_20261105_01/lai_range.py`, `kaggle_k15*/` | `lai_transmittance_min_max_cell.csv`, `layer_transmittance_min_max_cell.csv` |
| tray32 v2 / v3 | `src/build/static_tray32_v3_20261105_01/tray32_main.py`, `tray32_lib.py` | `efficiency_by_scene.csv`, `scenario_effects.csv` |
| 소엽 배치 Part C (D vs T, 공면 비율) | `src/build/engine_static_leaflet_arrangement_20261105_01/`, `src/build/engine_coplanar_sep_20261101_01/`, `src/build/engine_disc_vs_tri_20261101_01/` | `c2_summary.csv` |
| static ↔ GroIMP closure | `src/build/engine_static_20261026_01/compare_shading.py`, `groimp_mesh_probe.py` | 각 폴더 JSON/CSV |

설치: `pip install -r requirements.txt`. 광 계산 커널은 Kaggle CPU 에서 돌렸습니다 (입력 데이터셋은 `DATA.md`).

## 꼭 알아둘 사실
- **closure** 는 같은 삼각형 메시를 두 추적기(우리 정적 추적기와 GroIMP)에 넣어 비교합니다.
- **D vs T** 는 surfel 원판(D)을 다음과 비교합니다: 복원된 골격과 잎 면적 위에 엔진 법칙으로 만든 소엽 메시(T). 원판을 바꾼 것이 아닙니다. 표현 비교이며 검증이 아닙니다.
- 표본: 결과 표의 개체 14개 (`RESULT_TABLE_v1_vs_v1tau.csv`). **사람 라벨이 있는 개체는 파일럿 개체 1개** — 다른 개체의 잎(복엽) 단위 값은 채점 없는 라벨 위의 값입니다 (`skeleton_labels_ungraded`).
- 빛은 상대값이고 램프 배광은 가정 모델입니다. 시나리오 비교이며 보정·검증된 결과가 아닙니다.

## 상태 라벨
| 라벨 | 뜻 |
|---|---|
| `not_validation` | 시나리오·모델 결과이며 측정으로 검증되지 않음 |
| `relative_light` | 빛은 상대값 (램프 절대 출력 미보정) |
| `conditional_on_unvalidated_scale` | 복원 스케일(mm)이 독립 측정으로 검증되지 않은 조건부 값 |
| `skeleton_labels_ungraded` | 파일럿 외 개체의 복엽 라벨은 채점 없음 |
| `assumed_lamp_model` | 램프 배광은 가정 모델 |

## 내부 이름
| 이름 | 뜻 |
|---|---|
| `R3-1b` | 파일럿 개체의 파일 이름 |
| τ rule 2 | 줄기 제외 모든 surfel 투과율 0.07·흡수율 0.85, 줄기 τ 0 |
| d_ex | 자기 가림 제외 거리 (2.4 mm) |
| D / T | surfel 원판 / 엔진 법칙 소엽 삼각형 |
| K12–K18, K15 | Kaggle 커널 묶음 이름 (K15 = LAI 범위; `_cell` = 셀 크기 채움 원판) |
| tray32 | 32개체를 4 트레이에 둔 장면 |
| `dolgfish` | Kaggle 계정 이름 (커널·데이터셋 id 에 남겨 둠) |

## 라이선스
코드 MIT (`LICENSE`), 그림·표 CC BY 4.0 (`LICENSE-CC-BY-4.0.md`). 공용 모듈 사슬로 들어온 `src/tools/vtc_growth_fork.py`, `vtc_development.py` 는 VTC 조각을 담고 있어 이 공개 저장소에서 뺐습니다 (`../VTC_WITHHELD.md`).
한글 파일 이름은 `git config core.quotepath false` 로 보기 편합니다. 줄끝은 원본 그대로 저장합니다.

---
**English summary.** Shadow-ray light model on 2DGS-reconstructed Micro-Tom plants:
- Leaf optics: τ rule 2 (τ 0.07 / a 0.85 except a 1.5 mm stem band) and same-leaf normal smoothing (r 2 mm).
- Canopy quantities: LAI range and 5 mm layer transmittance.
- Scenes: tray32 v2/v3, and disc (D) vs engine-rule leaflet triangles (T).
- Closure: the same triangle mesh in our tracer and in GroIMP.

Two tracer versions are included: v1, and the current tau bundle (sha 5844e991…). Results are relative light under an assumed lamp model: scenario comparisons, not validation. The tables cover 14 plants, and only the pilot plant has human labels. Inputs come from `microtom-2dgs-recon` and are listed in `DATA.md`. Licensing: code MIT, figures/tables CC BY 4.0; the VTC fragments are excluded.
