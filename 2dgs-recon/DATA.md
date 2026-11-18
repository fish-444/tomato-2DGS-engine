# DATA — 무엇이 저장소에 있고, 무엇이 밖에 있는가

예시 개체는 **R3-1b (파일럿 개체)** 하나뿐입니다. R3-1b 는 2026-10-21 에 A3 ChArUco 보드 위에서 찍은 화분 토마토 한 개체의 복원 이름입니다. 다른 개체의 자료는 이 저장소에 없습니다.
라이선스: 예시 자료는 CC BY 4.0 (`LICENSE-CC-BY-4.0.md`).

## 1. 저장소 안: `example/r3-1b/` (약 9.4 MB)
전체 목록과 sha256 은 `example/r3-1b/MANIFEST.csv`, 기대값은 `example/r3-1b/EXPECTED.json`.

| 폴더 | 내용 |
|---|---|
| `surfels/` | R3-1b 서펠 `R3-1b_surfels_full.ply` (전체 53,645개, 불투명도 ≥ 0.5 인 것 46,719개) |
| `frozen_run/` | 2026-11-01 T-Prox 실행의 고정 묶음 `r8pc_bundle.npz`, 그 실행의 라벨 `R8_pc/labels.npz`, `R8_pc_rachis/labels_rachis.npz`, `bundle_meta.json` |
| `groups/` | 그룹 `patches_filled.npz` (stacked-sheets 경로: 4a 패치 + 채우기) |
| `skeleton/` | `skeleton_v1.json` (복엽 선 11개, 좌표계 `board_m_zup`, 사람 트레이스에서 만든 것) |
| `poses/` | 카메라 자세 `poses.json`, 내부 행렬 `intrinsics.json`, 점검 `qc.csv`, 사진 목록과 sha256 `photos.csv` |
| `scores/` | R3-1b 실루엣 점수 `scores_R3-1b.json`, R3-1 계약 `contract_R3-1.json`, 경로별 정확도 표 `r31b_route_accuracy.json` |
| `human_labels/` | 사람 라벨: 클릭 352개 `leaf_labels.json`(4+1장의 사진에서 사용자가 잎 번호를 클릭), 서펠에 옮긴 라벨 `local_surfel_labels.npz`, 채점용 15,650건 `human_instances.npz`, 사용한 사진 `views_chosen.json` |
| `human_masks/` | 보류 사진 4장의 전체 식물 사람 마스크 `human_masks.json` (RLE) |

저장소에 **넣지 못한** 파일: 11-01 실행의 자체 검사 기록 `R8_pc/results.json`, `R8_pc_rachis/results.json` — 만든 날(2026-11-17) 로컬 사본이 읽히지 않았습니다(클라우드 동기화 오류). 재현에는 필요 없고, `scripts/07_tprox.py` 가 새로 씁니다.

## 2. 저장소 밖: 사진
- 파일럿 사진 **85장** (HEIC, 6048 × 6048 px, 정확히 같은 파일 1장 포함 → 서로 다른 것 84장). R3-1b 는 이 중 65장을 썼습니다(학습 61 + 보류 4).
- 목록과 sha256: `example/r3-1b/poses/photos.csv` (2026-10-21 실행이 기록한 값. 이번에 다시 계산하지는 않았습니다).
- 내려받기: Kaggle 데이터셋 `dolgfish/onezip-pilot-photos-20261021` (85장, 262,526,011 bytes). **현재 비공개입니다.** 공개로 바꾸는 것은 소유자가 Kaggle 데이터셋 설정에서 하며, 바꾸면 이 줄을 공개 주소로 고칩니다.
  사진에는 작업 공간의 배경과 손이 일부 찍혀 있습니다.

## 3. 저장소 밖: 모델 가중치 (저장소에 넣지 않음, 요청 시 제공)
| 이름 | 용도 | 크기 | sha256 |
|---|---|---|---|
| `sam2.1_hiera_tiny_leaflet_v5_4plants.pt` | 잎 마스크 단계(SAM 2.1 tiny 를 저자들의 라벨 1,163개로 미세조정) | 156,003,537 bytes | `161b9c6caaf3a8a31df43e5f1aecc672f02409d2ff4b161515d7cd25bd02e73d` |
| `M1_s0_best.pt` | 소엽 후보 마스크 단계(`all3d_cands_job.py`) | 20,591,844 bytes | `43e78232378092b03bd9118c7c331af845329fcec09b259f275c41f16e84bd3f` |
| `sam2.1_hiera_tiny.pt` (SAM 2.1 기본 가중치, `onezip/run_recon.py` 의 `--mask auto` 가 내려받음) | 자동 마스크 | - | `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` |

해시는 Kaggle 의 비공개 데이터셋에 있는 같은 파일에서 계산했습니다(로컬 사본이 읽히지 않아서).
`--mask auto+stem` 은 첫 번째 가중치를 `--weights <폴더>` 로 받습니다. 줄기·엽축·잎자루를 찾는 **학습된 검출기는 없습니다**(이 프로젝트에서 만든 적이 없음).

## 4. 없는 것
- 다른 개체의 사진·복원·라벨·표. 사람 라벨이 있는 개체는 R3-1b 하나뿐이라, 이 저장소의 일치 수치는 모두 그 개체의 값입니다.
- 사람이 8835 포트 페이지에서 찍은 클릭 기록: 이 저장소의 R3-1b 스켈레톤은 페이지 클릭이 아니라 사람이 그린 잎 선(트레이스)에서 만든 것이라 해당 기록이 없습니다(`skeleton_v1.json` 의 `simulated_from_human_traces`).
