# THIRD_PARTY.md (microtom-2dgs-recon)

이 저장소에 제3자 코드는 복사하지 않았습니다. 아래는 import 하는 패키지와 외부 프로그램입니다. 라이선스는 빌드 세션이 아는 범위이며 공개 전에 다시 확인해야 합니다.

| 이름 | 버전 (출처) | URL | 라이선스 | import 횟수 |
|---|---|---|---|---|
| NumPy | 2.2.4 | https://numpy.org | BSD-3-Clause | 307 |
| SciPy | 1.18.0 | https://scipy.org | BSD-3-Clause | 195 |
| opencv-python | 4.13.0.92 | https://opencv.org | Apache-2.0 | 108 |
| Pillow | 12.3.0 | https://python-pillow.org | MIT-CMU (HPND) | 102 |
| Matplotlib | 3.11.0 | https://matplotlib.org | Matplotlib licence (PSF-based) | 38 |
| PyTorch | 2.13.0 (this PC; Kaggle image version on the kernel log) | https://pytorch.org | BSD-3-Clause | 25 |
| pillow-heif | 1.8.1 | https://github.com/bigcat88/pillow_heif | BSD-3-Clause (bundles libheif, LGPL) | 19 |
| scikit-learn | 1.9.0 | https://scikit-learn.org | BSD-3-Clause | 18 |
| gsplat (2D Gaussian splatting rasteriser) | 1.5.3 (pip line in the Kaggle kernels) | https://github.com/nerfstudio-project/gsplat | Apache-2.0 | 18 |
| Shapely | 2.1.2 | https://github.com/shapely/shapely | BSD-3-Clause | 14 |
| SAM 2.1 (sam2 package, checkpoint sam2.1_hiera_tiny) | package version not recorded on this PC - read it from the Kaggle kernel log (open item) | https://github.com/facebookresearch/sam2 | Apache-2.0 (code and checkpoints) | 11 |
| pandas | 3.0.3 | https://pandas.pydata.org | BSD-3-Clause | 10 |
| Ultralytics | 8.4.131 | https://github.com/ultralytics/ultralytics | AGPL-3.0 (copyleft; check before any public release) | 8 |
| scikit-image | 0.26.0 | https://scikit-image.org | BSD-3-Clause | 8 |
| triangle (Python wrapper of J. R. Shewchuk's Triangle) | not installed on this PC (Kaggle wheel) | https://github.com/drufat/triangle | wrapper LGPL-3.0; Triangle C code has its own non-commercial terms - check | 5 |
| openpyxl | 3.1.5 | https://openpyxl.readthedocs.io | MIT | 4 |
| pydantic | 2.13.4 | https://github.com/pydantic/pydantic | MIT | 3 |
| segmentation-models-pytorch | 0.5.0 | https://github.com/qubvel-org/segmentation_models.pytorch | MIT | 3 |
| psutil | 7.0.0 | https://github.com/giampaolo/psutil | BSD-3-Clause | 3 |
| pycolmap (COLMAP) | 4.2.0 | https://github.com/colmap/colmap | BSD-3-Clause | 2 |
| OpenAlea (lpy / plantgl, optional import) | not installed on this PC | https://github.com/openalea | CeCILL-C | 2 |
| python-dotenv | 1.2.2 | https://github.com/theskumar/python-dotenv | BSD-3-Clause | 1 |
| skan | 0.13.1 | https://github.com/jni/skan | BSD-3-Clause | 1 |
| Transformers | 5.14.1 | https://github.com/huggingface/transformers | Apache-2.0 | 1 |

## 외부 프로그램

| 이름 | 버전 | URL | 라이선스 |
|---|---|---|---|
| GroIMP | 2.2.1 | https://grogra.de / https://download.grogra.de/ | GPL-3.0 (설치본 사용, 코드 미포함) |
| COLMAP (pycolmap) | 4.2.0 | https://colmap.github.io | BSD-3-Clause |

## VTC 조각을 담은 도구 (공개판에는 없음 — `../VTC_WITHHELD.md`)

- `src/tools/vtc_growth_fork.py`: VTC 원본 NRH 함수를 바이트 그대로 조각으로 복사해 넣음 (K. Smolenova / WUR 2022-2025, J. Evers FSPM_BASIC). 프로젝트 기록상 license_status = unresolved.
- `src/tools/vtc_development.py`: 위 fork 를 실행·수집 (VTC 출처 경로는 `<DOWNLOADS>/vtc-main` 로 가림).
- 공용 모듈 import 사슬로 들어왔음. 공개판에서는 뺐음; 넣으려면 VTC 원저자 허락 필요.

## 참고

- 2DGS 학습은 `pip install gsplat==1.5.3` 을 Kaggle 커널 안에서 실행 (코드 미포함).
- SAM 2.1 은 `sam2` 패키지 import 만 함; 미세조정 가중치는 DATA.md.
- Ultralytics (AGPL-3.0) 를 import 하는 파일이 있음 (`src/tomato3d/leafseg/`). Ultralytics 코드는 복사하지 않고 import 만 함; 이 파일을 Ultralytics 와 함께 배포·서비스하면 AGPL-3.0 조건이 적용됨.
