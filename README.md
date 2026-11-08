# tomato-2DGS-engine



Micro-Tom 토마토를 회전대 사진으로 3D 복원(2DGS)하고, 그 위에서 빛을 계산하고(정적 광 모델), GroIMP 성장 모델(동적 FSPM)로 키우는 작업을 묶은 저장소입니다.

| 폴더 | 내용 | 설명 |
|---|---|---|
| [`2dgs-recon/`](2dgs-recon/README.md) | ChArUco 자세 → SAM 2.1 마스크 → 2DGS surfel 복원 (Kaggle, gsplat 1.5.3) → 복엽 라벨링 (T-Prox 등) → 형질 → MTG 템플릿 | 다른 두 폴더의 입력 |
| [`static-light/`](static-light/README.md) | 복원 개체 위 그림자 광선 추적: 잎 투과 τ, LAI·층별 투과, tray32 장면, 원판 대 삼각형, GroIMP 대조 | 상대 광, 시나리오 비교 |
| [`dynamic-fspm/`](dynamic-fspm/README.md) | GroIMP 2.2.1 성장·광 모델 (VTC 를 고침) 의 보정·검증기·이데오타입·tray32 동적 장면·GUI 캐노피 도구 | **VTC 파생 모델 파일은 없음** (아래) |

## 쓰는 법
- 각 폴더의 `src/` 가 원 프로젝트의 최상위 폴더와 **같은 상대 경로**입니다 (`src/build/<작업 폴더>/`, `src/tools/`, `src/tomato3d/`). 스크립트는 그 `src/` 에서 실행합니다. 예: `cd static-light/src && python -X utf8 build/static_tray32_v3_20261105_01/make_figures_v3.py`.
- 폴더마다 `requirements.txt`, `DATA.md` (저장소에 넣지 않은 원자료·대용량 산출물 목록), `configs/paths.yaml` (경로 자리표시자 `<PROJECT_ROOT>` 등의 뜻), `docs/SUPERSEDED.md` 가 있습니다.
- 작업 폴더마다 `CONTRACT*.md` 에 정확한 명령·입력 해시·판정 기준이 있습니다.
- 공용 모듈 (`tools/`, `tomato3d/`) 은 폴더마다 필요한 만큼 따로 들어 있어 폴더끼리 겹치는 파일이 있습니다.

## 상태
- 결과는 대부분 **검증이 아닌 시나리오·보정 결과**입니다. 빛은 가정한 램프 모델에서의 상대값입니다. 폴더 README 의 상태 라벨을 보세요.
- 원 사진과 사진에서 만든 그림은 넣지 않았습니다.

## VTC 파생 파일 — 공개판에서 뺌
동적 모델은 VTC (Virtual Tomato Crop; K. Smolenova / WUR, https://doi.org/10.1093/insilicoplants/diaf022) 를 고쳐 만들었습니다. VTC 원본에는 라이선스 파일이 없어, VTC 파생 파일 33개 (모델 `.rgg`·`.gsz`, `tools/vtc_growth_fork.py`, `tools/vtc_development.py`, 모델 `.diff`) 는 넣지 않았습니다. 목록·sha256 과 그 때문에 돌지 않는 스크립트 111개는 [`VTC_WITHHELD.md`](VTC_WITHHELD.md) 에 있습니다. 표·그림으로 남긴 결과는 그대로 들어 있습니다.

## 라이선스
- 코드: MIT ([`LICENSE`](LICENSE)). 그림·표: CC BY 4.0 ([`LICENSE-CC-BY-4.0.md`](LICENSE-CC-BY-4.0.md)).
- 제3자 패키지 (gsplat, SAM 2, GroIMP, Ultralytics (AGPL-3.0), triangle, openalea 등) 는 복사하지 않고 설치해서 씁니다. 폴더별 `THIRD_PARTY.md` 참조.
- 인용: [`CITATION.cff`](CITATION.cff).

---
**English.** Micro-Tom tomato phenotyping in three folders:
- `2dgs-recon/`: turntable photos → ChArUco poses → SAM 2.1 masks → 2DGS surfel reconstruction → compound-leaf labelling → traits → MTG template.
- `static-light/`: a shadow-ray light model on the reconstructed plants (relative light, scenario comparisons).
- `dynamic-fspm/`: calibration, validator, ideotype and tray32 analyses for a GroIMP 2.2.1 growth model derived from VTC.

Each folder's `src/` mirrors the original project layout, so run scripts from that `src/`. Results are mostly scenario or calibration results, not validation; only one pilot plant has human labels. VTC has no licence file, so the 33 VTC-derived files are withheld; the list, hashes and the 111 scripts that need them are in `VTC_WITHHELD.md`. Code is MIT; figures and tables are CC BY 4.0. Raw photos and photo-derived figures are not included.
