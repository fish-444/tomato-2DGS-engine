# microtom-2dgs-recon

화분에 심은 토마토 한 개체를 A3 ChArUco 보드 위에서 돌려 찍은 사진으로 **2DGS 서펠 복원**을 만들고, 그 서펠에 **T-Prox 복엽 라벨**(어느 서펠이 어느 복엽인가)을 붙이는 코드와, 그 방법이 확인된 유일한 예시(R3-1b)입니다.
(English summary at the end.)

## 먼저 읽으세요: 상태

| 라벨 | 뜻 |
|---|---|
| `R3-1b_unaccepted` | 예시 복원은 합격 기준을 통과하지 못한 채 쓰는 복원입니다. |
| `conditional_on_unvalidated_scale` | 길이(mm)는 인쇄한 보드에서 온 값이고 독립 실측으로 검증하지 않았습니다. |
| `tprox_unverified_until_heldout` | T-Prox 라벨의 정확도는 사람 라벨이 있는 **한 개체**에서만 쟀습니다. 다른 개체의 출력은 "배정 비율"일 뿐 정확도가 아닙니다. |

- **R3-1b (파일럿 개체)**: 2026-10-21 에 찍은 화분 토마토 한 개체의 복원 이름입니다. 이 저장소의 예시는 이 개체 하나뿐이고, 사람 라벨(잎 번호 클릭)이 있는 개체도 이것 하나뿐입니다.
- 위 수치(0.750 등)는 모두 이 한 개체의 값입니다. 다른 개체에 대한 정확도를 주장하지 않습니다.
- 사람이 8835 포트 페이지에서 찍은 클릭으로 만든 스켈레톤과 그룹이 필요합니다. 줄기·엽축·잎자루를 찾는 **학습된 검출기는 없습니다**.
- 0.750 실행은 가중치 묶음 **A (기하만)** 입니다. 묶음 B (0.7 / 0.3 색 포함)는 발표 그림에만 나오며 이 저장소의 실행이 아닙니다.

## 저장소만으로 되는 것 (CPU, 데이터 내려받기 없음)

```
pip install -r requirements.txt
python scripts/08_score_pilot.py                    # 11-01 실행의 T-Prox 라벨을 사람 라벨로 채점 -> 0.7503 [0.739, 0.761]  (수 초)
python scripts/08_score_pilot.py --distance         # 거리 기준선 -> 0.6065
python scripts/07_tprox.py --rebuild-seeds          # 스켈레톤+그룹+축 분류에서 씨앗을 다시 만들어 묶음과 같은지 확인한 뒤 고정 계산을 돌림
python scripts/08_score_pilot.py --labels results/r3-1b_tprox/R8_pc_rachis/labels_rachis.npz
```

- `07_tprox.py` 는 고정된 `pc_absorb.py`(묶음 A)와 `pc_rachis.py` 를 돌립니다. Kaggle CPU 커널에서 **212초, 가장 큰 하위 프로세스(`pc_absorb.py`)의 최대 메모리 약 0.7 GB** 였습니다(전체 메모리는 재지 않았습니다).
- **검사한 것**: 정리한 코드로 `07_tprox.py` 를 Kaggle CPU 에서 돌려, 11-01 실행의 모든 배열(라벨, 출처, p1, p2, top1, top2 등 23개)이 **완전히 같음**을 확인했습니다(2026-11-18). 씨앗 재생성(`--rebuild-seeds`)과 두 채점 값은 이 PC 에서도 확인했습니다. 기대값은 `example/r3-1b/EXPECTED.json`.
- `scripts/08` 의 사람 라벨은 15,650건이며, R3-1b 서펠 위의 값입니다. 다른 복원의 서펠에는 `--radius-mm`(기본 2.0) 이내 가장 가까운 서펠로 옮겨 채점합니다(이 옮기는 단계는 검증되지 않았습니다).

## 데이터나 GPU 가 필요한 것 (1–6단계)

| 단계 | 스크립트 | 필요한 것 |
|---|---|---|
| 1 자세 | `scripts/01_poses.py` | 사진 (`DATA.md`) |
| 2 마스크 | `scripts/02_masks.py` | 사진 + SAM 가중치. R3-1b 의 마스크 단계(M1–M5)는 `onezip/stem_masks.py` 로 옮겨 두었습니다 |
| 3 초기 점 | `scripts/03_sparse.py` | pycolmap (보드 자세를 고정한 삼각측량, SfM 자세 추정 없음) |
| 4 학습 | `scripts/04_train.py` | CUDA GPU (Kaggle T4 에서 약 33분/회) |
| 5 실루엣 점수 | `scripts/05_silhouette_score.py` | 보류 사진 |
| 6 그룹 | `scripts/06_groups.py` | Kaggle 에서 실행하는 것이 보통입니다(`docs/GROUPS_AND_TPROX_KO.md`) |

전체를 한 번에 돌리는 길은 **`onezip/`** 입니다: `python onezip/build_zip.py` 로 Kaggle / Colab 용 zip 을 만들고, 사진 폴더와 함께 올려 `RUN_KAGGLE.ipynb` 를 실행합니다(`onezip/README_KO.md`).
저장소의 `src/` 만으로는 안 돌아가는 곳(`docs` 에도 적음): SAM 로더(`leafseg.run`, `topview.measure2d`), 사진 가시성 계산(`seed2d`), 사람 라벨 채점에 쓰던 `userlabel`·`vote2d` 는 이 저장소에 넣지 않았습니다. one-zip 은 자기 SAM 로더를 갖고 있어 한 번에 돌아갑니다.

## 알려진 실패와 현재 상태 (2026-11-18 기준 보고서 그대로)

같은 개체, 같은 스켈레톤, 같은 고정 T-Prox 인데 사람 라벨 일치가 PC 실행에서는 **0.750**, one-zip 자동 마스크 복원에서는 **0.525** 였습니다. 원인 분해(`docs/ONEZIP_DIAGNOSIS_KO.md`):
- 차이의 대부분(−0.19)은 **복원**에서 옵니다. one-zip 복원에서는 위아래로 겹친 잎 4·5 의 구분이 약해져 잎 5 의 잎몸 씨앗이 0개가 되었고, 그 자리를 잎 4 가 차지합니다.
- 채점 이전, 사진 해상도 어댑터, 축 분류 출처는 원인이 아닙니다. 그룹 병합이 −0.036, 선을 잎으로 읽는 방식이 −0.006 입니다(후자는 고침).
- 줄기: R3-1b 의 마스크 단계를 옮긴 `--mask auto+stem` 은 마스크와 서펠 수를 R3-1b 쪽으로 끌어올렸지만(서펠 39,045 → 45,444) **줄기 밑 서펠은 R3-1b 의 45개 중 6개**에 그쳤습니다. R3-1b 에 있던 단안 깊이 초기 점이 원인인지는 가르지 않았습니다.
- `auto+stem` 복원에 7단계를 돌린 값(2026-11-18, 같은 스켈레톤·같은 고정 계산): 배정 비율 0.947, 사람 라벨 일치 보고 절반 **0.567**(`auto` 복원은 0.525–0.531, PC 실행은 0.750). 잎 5 는 0.330(0.25 → 0.33), 잎 6 은 0.011. 줄기가 조금 복원돼도 겹친 잎 4·5 의 문제는 거의 그대로입니다. 그림 `figures/fig_stem_region.png`. 이 한 개체에서 1회 돌린 값입니다.

## 구조
```
src/tomato3d/{recon,organs,...}/   라이브러리 (고정 코드 pc_absorb.py 는 원본과 바이트 동일)
scripts/01_poses … 08_score_pilot  단계별 진입점 (--help)
onezip/                            Kaggle/Colab 용 한 번에 돌리는 패키지와 빌더, pc_side/ = PC 쪽 그룹 만들기
example/r3-1b/                     예시 자료 + EXPECTED.json + MANIFEST.csv
configs/                           paths.yaml, r3-1b.yaml (파일럿 실행이 쓴 값의 기록; 코드가 읽지는 않음)
results/ figures/ docs/            표, 그림, 설명 (CC BY 4.0)
DATA.md  THIRD_PARTY.md  CITATION.cff  LICENSE  LICENSE-CC-BY-4.0.md
```
코드를 정리하면서 **계산은 바꾸지 않았습니다**(식, 문턱값, 가중치, 씨앗, 기본값, 출력 형식). 다른 개체용 코드, 형질·MTG·광 모델, 대체된 변형, Roboflow 층은 뺐고, 그 목록은 정리한 쪽의 기록에만 있고 저장소에는 넣지 않았습니다(파일 이름에 다른 개체 이름이 들어 있어서).
저장소에는 Kaggle 사용자 이름(`dolgfish`)이 소유자의 결정으로 남아 있습니다(`configs/paths.yaml`, 커널 메타데이터를 만드는 스크립트).

## 라이선스, 제3자
- 코드 MIT (`LICENSE`), 그림·표·문서·예시 자료 CC BY 4.0 (`LICENSE-CC-BY-4.0.md`).
- gsplat 1.5.3, SAM 2 / `sam2==1.1.0`, pycolmap, OpenCV, **Ultralytics (AGPL-3.0, 후보 마스크 단계에서만 import)** 등은 가져오지 않고 참조합니다: `THIRD_PARTY.md`. VTC 파생 파일은 넣지 않았습니다.
- 인용: 저장소 최상위의 `CITATION.cff`.

## English summary
Code and one worked example (the pilot plant "R3-1b") for reconstructing a potted tomato plant from photos taken on an A3 ChArUco board (2DGS surfels, camera poses from the board only, no SfM pose estimation)
and for assigning each surfel to a compound leaf (T-Prox: groups from sheets/patches, human skeleton lines as seeds, frozen random-walker absorption with weight set A = geometry only).
Everything is checked on **one plant only**; labels `R3-1b_unaccepted`, `conditional_on_unvalidated_scale`, `tprox_unverified_until_heldout` apply and no accuracy is claimed for other plants.
Run `python scripts/08_score_pilot.py` (agreement with the pilot's human leaf labels, report half 0.7503, 95 % CI 0.739-0.761; distance baseline 0.6065) and
`python scripts/07_tprox.py --rebuild-seeds` (frozen stage 7; reproduces all 23 arrays of the 2026-11-01 run, verified on a Kaggle CPU kernel in 212 s).
Photos and model weights are not in the repository (`DATA.md`). A person must draw the skeleton lines and make the groups; there is no trained stem, rachis or petiole detector.
Known failure: with the automatic masks of `onezip/` the same plant scores 0.525 instead of 0.750, mostly because the reconstruction separates two vertically stacked leaves less well (`docs/ONEZIP_DIAGNOSIS_KO.md`).
Code is MIT, figures/tables/docs/example data CC BY 4.0; third-party components are referenced, not copied (`THIRD_PARTY.md`, note Ultralytics AGPL-3.0).
