# microtom-2dgs-recon — Micro-Tom 2DGS 복원과 기관 라벨링


> **공개판 안내 (tomato-2DGS-engine):** VTC 파생 파일(모델 `.rgg`·`.gsz`, `src/tools/vtc_growth_fork.py`, `src/tools/vtc_development.py`, 모델 비교 `.diff`)은 원저자 허락 전이라 이 저장소에 넣지 않았습니다. 목록·sha256·그 때문에 돌지 않는 스크립트는 상위 [`VTC_WITHHELD.md`](../VTC_WITHHELD.md).
> *Public edition: VTC-derived files are withheld; see [`../VTC_WITHHELD.md`](../VTC_WITHHELD.md) for the list, hashes and the scripts that need them.*

## 무엇인가
턴테이블 사진으로 Micro-Tom 개체를 2D Gaussian splatting(2DGS) surfel 로 복원합니다. 이어서 복엽(잎)을 나누고 골격·형질·MTG(식물 구조 표)를 만듭니다.
이 저장소가 만든 surfel·MTG 는 `microtom-static-light` (정적 광)과 `microtom-dynamic-fspm` (동적 FSPM)의 입력입니다.

```
사진 → ChArUco 자세 → SAM 2.1 마스크 → 2DGS 학습(Kaggle T4) → 실루엣 점검(held-out IoU)
     → 복엽 라벨(3D-first 묶음, 골격 클릭 도구, T-Prox, R8_pc, V-P) → 형질 → MTG (_04) → 정적 / 동적 저장소
```

## 폴더
- `src/` = 원 프로젝트와 **같은 상대 경로**로 복사 (`src/tomato3d/`, `src/tools/`, `src/build/<작업 폴더>/`). 스크립트 안의 상대 경로가 그대로 맞도록 옮기지 않았습니다. 결과 표(CSV)와 그림(PNG)은 각 작업 폴더 안에 있습니다.
- `configs/paths.yaml`: 개인 절대 경로 대신 넣은 자리표시자(`<PROJECT_ROOT>` 등)의 뜻.
- `docs/SUPERSEDED.md`: 대체·철회·초안 파일 목록 (현행 아님).
- `DATA.md`: 올리지 않은 원자료·대용량 산출물. `THIRD_PARTY.md`: 외부 패키지·라이선스.

## 주요 스크립트 (정확한 실행 명령과 입력 해시는 같은 폴더의 `CONTRACT*.md` 에 있음)
| 단계 | 진입점 |
|---|---|
| ChArUco 자세 | `src/tomato3d/calib/`, `src/tomato3d/recon/charuco_*.py`, `src/build/p1_recon_charuco_20261021_01/` |
| 마스크 (SAM 2.1, 설정만) | `src/tomato3d/recon/kernel_all_masks.py`, `src/tomato3d/recon/r31_*.py`, `src/tomato3d/leafseg/` |
| 2DGS 학습 (Kaggle T4, gsplat 1.5.3) | 파일럿: `src/build/p1_recon_r3-1_20261022_01/kaggle_stage/kernel/run.py` · 여러 개체: `src/tomato3d/recon/kernel_all_train.py`, `src/build/p1_recon_all_20261025_01/train_queue.py` |
| 실루엣 면적 보정·held-out IoU | `src/build/p1_recon_r3-1_20261022_01/scores_*.json`, `src/build/p1_recon_all_20261025_01/triage_all.json` |
| 복엽 라벨 도구 (포트 8835) | `src/tomato3d/organs/all3d_page.py`, `skeleton_store.py`, `skeleton_io.py` (형식 정의) |
| T-Prox / R8_pc / V-P | `src/tomato3d/organs/skeleton_tprox.py`, `pc_rachis.py`, `skeleton_r8pc.py`, `src/build/p1_vp_*` |
| 형질 | `src/tools/plant_traits.py`, `src/build/traits_recon_20261101_01/` |
| MTG (_04) | `src/build/mtg_build_v2_clicks_curv_20261105_01/build_04.py` → `src/build/mtg_build_v2_clicks_20261105_04/` |

설치: `pip install -r requirements.txt` (이 PC 기준 버전). 2DGS·SAM 커널은 Kaggle 에서 자체 설치합니다. 대부분의 스크립트는 `DATA.md` 의 입력(사진·surfel)이 있어야 돌아갑니다.

## 동결(frozen) 파일과 sha256
| 파일 | sha256 |
|---|---|
| `src/build/traits_recon_20261101_01/MTG_TEMPLATE_v1.csv` | `ee4b43da797159ef2f5d1a6bf7fa727d95cdf6f94a67ca28dadd5e21d61a228e` |
| `src/build/traits_recon_20261101_01/MTG_TEMPLATE_v2_4.csv` | `6901657ea95d7e32acce44cbf2eddaf7149d099f816b7624f27ac5f458ca7f72` |
| `src/build/traits_recon_20261101_01/MTG_LEAF_CURVATURE_v1.csv` | `8cdf7ff764e65440a1854aa8500ee9b1caa69b96313376cb9b74c413f7370119` |
| `src/build/traits_recon_20261101_01/MTG_LEAFLET_SPAN_v1.csv` | `826b0844f0c5067f0398ef52680aba287eeb178adbbdc5213ef506abfefcbf9a` |
| `src/build/traits_recon_20261101_01/MTG_LEAFLET_ARRANGEMENT_v1.csv` | `faa67d5e926003814d7a6105b31b002aebf6b8e3d93e6245c8d11b2a8cd1daba` |

현재 정적 MTG 세트 = `src/build/mtg_build_v2_clicks_20261105_04/` (11개체 중 8개체 점검 통과; 파일럿 개체·`n_03`·`e_01` 은 `PLANT_FLAGS.json` 에 표시하고 사용자 판단으로 채택).

## 상태 라벨 (결과를 읽을 때)
| 라벨 | 뜻 |
|---|---|
| `not_validation` | 시나리오·모델 결과이며 측정으로 검증되지 않음 |
| `relative_light` | 빛은 상대값 |
| `conditional_on_unvalidated_scale` | 복원 스케일(mm)이 독립 측정으로 검증되지 않은 조건부 값 |
| `skeleton_labels_ungraded` | 골격·복엽 라벨에 held-out 채점이 없음 (사람 라벨이 있는 개체는 파일럿 개체 하나뿐) |
| `assumed_lamp_model` | 램프 배광은 가정 모델 |

표본 수: 복원 개체 다수(파일 이름 `n_0x`, `t2_0x`, `e_0x`), **사람 라벨이 있는 개체는 파일럿 개체 1개**. 다른 개체의 배정 비율은 정확도가 아니라 배정 몫입니다.

## 내부 이름
| 이름 | 뜻 |
|---|---|
| `R3-1b` | 파일럿 개체의 파일 이름 |
| 2DGS | 2D Gaussian splatting (면 원판 surfel 로 복원) |
| T-Prox | 골격 선에 가까운 surfel 을 씨앗으로 하는 복엽 배정 |
| R8_pc | 점군 random walker 흡수로 복엽 라벨을 넓히는 방법 |
| V-P | 사진 우선 복엽 군집 정리 (pilot) |
| `_04` | MTG 구축 4번째 실행 = 현행 정적 MTG 세트 |
| `dolgfish` | Kaggle 계정 이름 (커널·데이터셋 id 에 남겨 둠) |

## 알아둘 것
- 개인 절대 경로는 사본에서만 자리표시자로 바꿨습니다 (`configs/paths.yaml`). 다른 코드는 바꾸지 않았습니다.
- 한글 파일 이름이 있으면 `git config core.quotepath false` 를 권장합니다.
- 줄끝은 원본 그대로 저장합니다 (`.gitattributes` 의 `* -text`).
- 라이선스: 코드 MIT (`LICENSE`), 그림·표 CC BY 4.0 (`LICENSE-CC-BY-4.0.md`). `src/tools/vtc_growth_fork.py`, `vtc_development.py` 는 VTC 조각을 담고 있어 이 공개 저장소에서 뺐습니다 (`../VTC_WITHHELD.md`).

---
**English summary.** Turntable photos of Micro-Tom plants → ChArUco poses → SAM 2.1 masks → 2DGS surfel reconstruction (Kaggle, gsplat 1.5.3) → held-out silhouette checks → compound-leaf labelling (3D-first groups, skeleton click tool, T-Prox, R8_pc random walker, V-P) → traits → MTG (`_04`). Outputs feed `microtom-static-light` and `microtom-dynamic-fspm`. `src/` mirrors the original project layout so relative paths in scripts still work; each work folder's `CONTRACT*.md` holds the exact commands and input hashes. Only one plant (the pilot, file name `R3-1b`) has human labels; other plants' label shares are not accuracies. Raw photos, masks, human labels, click logs and surfel files are not included (see `DATA.md`). Code MIT, figures/tables CC BY 4.0, VTC fragments excluded.
