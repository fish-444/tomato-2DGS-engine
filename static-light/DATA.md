# DATA.md — 저장소에 넣지 않은 자료 (microtom-static-light)

원자료·대용량 산출물은 올리지 않고 여기 목록만 둡니다. 공유: **요청 시 (on request)** — 저자에게 연락.
경로는 원 프로젝트 기준 (`<PROJECT_ROOT>/...`). 폴더 행은 파일별 sha256 을 계산하지 않았음 (OneDrive 클라우드 파일을 내려받지 않기 위해); 필요하면 요청 시 매니페스트 제공.

## 외부 의존 (저장소 밖)

| 항목 | 위치 | 비고 |
|---|---|---|
| 복원 surfel·MTG 입력 | 저장소 `microtom-2dgs-recon` 의 DATA.md (`p1_recon_all_20261025_01/outputs/`, `mtg_build_v2_clicks_20261105_04/static/`) | 2 MB 이하 MTG 파일은 recon 저장소에 들어 있음 |

## 목록 (크기순)

| 경로 | 크기 | sha256 / 파일 수 | 만든 폴더 (결과) | 이유 |
|---|---|---|---|---|
| `build/engine_static_nine_20261028_01/kaggle_data/` | 131.2 MB | 폴더, 61 파일 | static nine Tier 0/1 (d_ex 2.4) | excluded dir (outputs/uploads/raw/cache), 61 files [.csv:20,.json:14,.npz:11,.ply:10,.py:5 |
| `build/engine_static_nine_20261028_01/kaggle_out/` | 58.9 MB | 폴더, 32 파일 | static nine Tier 0/1 (d_ex 2.4) | excluded dir (outputs/uploads/raw/cache), 32 files [.json:12,(none):10,.npz:10] |
| `build/engine_static_normals_20261104_01/out_k1/` | 44.6 MB | 폴더, 47 파일 | same-leaf normal smoothing v1 | excluded dir (outputs/uploads/raw/cache), 47 files [.json:29,.npz:15,.log:2,(none):1] |
| `build/engine_static_normals_20261104_01/kaggle_data/` | 44.4 MB | 폴더, 50 파일 | same-leaf normal smoothing v1 | excluded dir (outputs/uploads/raw/cache), 50 files [.npz:28,.obj:11,.json:6,.ply:4,.py:1] |
| `build/engine_static_normals_20261104_01/kaggle_flat/` | 44.4 MB | 폴더, 50 파일 | same-leaf normal smoothing v1 | excluded dir (outputs/uploads/raw/cache), 50 files [.npz:28,.obj:11,.json:6,.ply:4,.py:1] |
| `build/engine_static_tau_20261031_01/out_k2/` | 32.2 MB | 폴더, 30 파일 | v1_tau T2b | excluded dir (outputs/uploads/raw/cache), 30 files [.json:10,(none):9,.npz:9,.log:2] |
| `build/engine_static_normals_v2_20261104_01/kaggle_flat/` | 19.0 MB | 폴더, 11 파일 | same-leaf normal smoothing r 2 mm | excluded dir (outputs/uploads/raw/cache), 11 files [.obj:7,.json:2,.py:1,.npz:1] |
| `build/engine_static_r8pc_20261028_01/kaggle_out/` | 15.1 MB | 폴더, 2 파일 | R3-1b R7 vs R8_pc static | excluded dir (outputs/uploads/raw/cache), 2 files [.json:1,.npz:1] |
| `build/engine_static_v1tau_reference_20261104_01/out_k8b/` | 13.7 MB | 폴더, 18 파일 | v1_tau adopted reference + GroIMP tau closure | excluded dir (outputs/uploads/raw/cache), 18 files [.json:6,(none):5,.npz:5,.log:2] |
| `build/engine_static_v1tau_reference_20261104_01/out_k8a/` | 11.3 MB | 폴더, 18 파일 | v1_tau adopted reference + GroIMP tau closure | excluded dir (outputs/uploads/raw/cache), 18 files [.json:6,(none):5,.npz:5,.log:2] |
| `build/engine_static_v1tau_reference_20261104_01/out_k8c/` | 9.8 MB | 폴더, 15 파일 | v1_tau adopted reference + GroIMP tau closure | excluded dir (outputs/uploads/raw/cache), 15 files [.json:5,(none):4,.npz:4,.log:2] |
| `build/engine_static_tau_rule2_20261105_01/kaggle_flat/` | 8.7 MB | 폴더, 29 파일 | optics rule 2 (d_stem 1.5 mm) RESULT_TABLE_rule2_unified | excluded dir (outputs/uploads/raw/cache), 29 files [.npz:28,.json:1] |
| `build/engine_coplanar_sep_20261101_01/kaggle_data/` | 6.9 MB | 폴더, 8 파일 | coplanar vs bending separation | excluded dir (outputs/uploads/raw/cache), 8 files [.csv:6,.json:2] |
| `build/engine_static_v1tau_reference_20261104_01/out_k5b/` | 6.5 MB | 폴더, 18 파일 | v1_tau adopted reference + GroIMP tau closure | excluded dir (outputs/uploads/raw/cache), 18 files [.json:6,(none):5,.npz:5,.log:2] |
| `build/engine_coplanar_sep_20261101_01/out_k1/` | 6.3 MB | 폴더, 22 파일 | coplanar vs bending separation | excluded dir (outputs/uploads/raw/cache), 22 files [.log:8,.csv:6,.rgg:6,.npz:1,.json:1] |
| `build/engine_static_v1tau_reference_20261104_01/out_k5a/` | 5.5 MB | 폴더, 18 파일 | v1_tau adopted reference + GroIMP tau closure | excluded dir (outputs/uploads/raw/cache), 18 files [.json:6,(none):5,.npz:5,.log:2] |
| `build/engine_static_r8pc_20261028_01/kaggle_data/` | 5.3 MB | 폴더, 10 파일 | R3-1b R7 vs R8_pc static | excluded dir (outputs/uploads/raw/cache), 10 files [.py:5,.json:2,.npz:1,.ply:1,.csv:1] |
| `build/engine_static_tau_rule2_20261105_01/out_k12b/` | 5.2 MB | 폴더, 24 파일 | optics rule 2 (d_stem 1.5 mm) RESULT_TABLE_rule2_unified | excluded dir (outputs/uploads/raw/cache), 24 files [.json:8,(none):7,.npz:7,.log:2] |
| `build/engine_static_tau_rule2_20261105_01/out_k12a/` | 5.1 MB | 폴더, 24 파일 | optics rule 2 (d_stem 1.5 mm) RESULT_TABLE_rule2_unified | excluded dir (outputs/uploads/raw/cache), 24 files [.json:8,(none):7,.npz:7,.log:2] |
| `build/engine_static_r8pc_20261028_01/per_surfel_light.npz` | 5.1 MB | 93c75200ef7d2255b57220eed45c957f6882b579e9c032923c9aa92310363022 | R3-1b R7 vs R8_pc static | data file (not uploaded) |
| `build/static_tray32_v3_20261105_01/out_t1/` | 4.8 MB | 폴더, 87 파일 | tray32 v3 (efficiency_by_scene) | excluded dir (outputs/uploads/raw/cache), 87 files [.json:82,.log:4,.py:1] |
| `build/engine_static_v1tau_reference_20261104_01/out_k5c/` | 4.8 MB | 폴더, 15 파일 | v1_tau adopted reference + GroIMP tau closure | excluded dir (outputs/uploads/raw/cache), 15 files [.json:5,(none):4,.npz:4,.log:2] |
| `build/static_tray32_v3_20261105_01/out_t3/` | 4.4 MB | 폴더, 76 파일 | tray32 v3 (efficiency_by_scene) | excluded dir (outputs/uploads/raw/cache), 76 files [.json:73,.log:2,.py:1] |
| `build/engine_static_leaflet_arrangement_20261105_01/out_k19/` | 4.4 MB | 폴더, 13 파일 | leaflet arrangement Part C | excluded dir (outputs/uploads/raw/cache), 13 files [.log:5,.csv:3,.rgg:3,.npz:1,.json:1] |
| `build/static_tray32_v3_20261105_01/leaves_by_scene.csv` | 3.4 MB | e59b529dd7795e47dc08e581aacf357f8610f00363869191cee499007caf3618 | tray32 v3 (efficiency_by_scene) | large derived table/geometry (>2 MB); listed with sha256 |
| `build/engine_static_lai_range_20261105_01/kaggle_flat_b/` | 3.2 MB | 폴더, 99 파일 | LAI range + 5 mm layer transmittance (_cell) | excluded dir (outputs/uploads/raw/cache), 99 files [.npz:84,.json:15] |
| `build/engine_static_lai_range_20261105_01/kaggle_flat/` | 3.2 MB | 폴더, 99 파일 | LAI range + 5 mm layer transmittance (_cell) | excluded dir (outputs/uploads/raw/cache), 99 files [.npz:84,.json:15] |
| `build/engine_static_lai_range_20261105_01/kaggle_flat_c/` | 3.2 MB | 폴더, 99 파일 | LAI range + 5 mm layer transmittance (_cell) | excluded dir (outputs/uploads/raw/cache), 99 files [.npz:84,.json:15] |
| `build/static_tray32_v3_20261105_01/out_t2/` | 3.2 MB | 폴더, 76 파일 | tray32 v3 (efficiency_by_scene) | excluded dir (outputs/uploads/raw/cache), 76 files [.json:73,.log:2,.py:1] |
| `build/static_tray32_v3_20261105_01/out_t4/` | 3.2 MB | 폴더, 76 파일 | tray32 v3 (efficiency_by_scene) | excluded dir (outputs/uploads/raw/cache), 76 files [.json:73,.log:2,.py:1] |
| `build/engine_disc_vs_tri_20261101_01/out_k1/` | 2.7 MB | 폴더, 10 파일 | D vs T (representation comparison) | excluded dir (outputs/uploads/raw/cache), 10 files [.log:4,.csv:2,.rgg:2,.npz:1,.json:1] |
| `build/engine_static_tau_20261031_01/out_k1b/` | 2.6 MB | 폴더, 5 파일 | v1_tau T2b | excluded dir (outputs/uploads/raw/cache), 5 files [.log:2,.json:2,.npz:1] |
| `build/engine_static_v1tau_reference_20261104_01/out_k67/` | 2.4 MB | 폴더, 15 파일 | v1_tau adopted reference + GroIMP tau closure | excluded dir (outputs/uploads/raw/cache), 15 files [.log:6,.csv:4,.rgg:4,.json:1] |
| `build/static_tray32_v2_20261104_01/out_t1/` | 2.2 MB | 폴더, 46 파일 | tray32 v2 | excluded dir (outputs/uploads/raw/cache), 46 files [.json:41,.log:4,.py:1] |
| `build/static_tray32_v2_20261104_01/out_t3/` | 2.2 MB | 폴더, 40 파일 | tray32 v2 | excluded dir (outputs/uploads/raw/cache), 40 files [.json:37,.log:2,.py:1] |
| `build/static_tray32_v2_20261104_01/out_t2/` | 1.6 MB | 폴더, 40 파일 | tray32 v2 | excluded dir (outputs/uploads/raw/cache), 40 files [.json:37,.log:2,.py:1] |
| `build/static_tray32_v2_20261104_01/out_t4/` | 1.6 MB | 폴더, 40 파일 | tray32 v2 | excluded dir (outputs/uploads/raw/cache), 40 files [.json:37,.log:2,.py:1] |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/refk__n_05.npz` | 1.3 MB | fc67b0711d273400dd53cb92c8f25c6138fda14b7c9f87f83e51e1b38a9889ce | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/refk__n_04.npz` | 1.3 MB | 2054eb7dfe8e90c19f9221b02bd6e40228c12fe7510ae83b50d237abff54647c | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/refk__t2_02.npz` | 1.2 MB | a80209e4937c27b8a74865c31cb5af3784bf7cc9da000b2747942836c44b18d6 | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/refk__n_06.npz` | 1.1 MB | 45cee6cc6287b2bcd326d6cdf146fcf6b5e274017f2fb9d8b5d892147fc732b4 | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/refk__n_01.npz` | 1.1 MB | 19c00a107fa3a7f86485d4cf86257da528366ee3ab88c9d2ec675e3635ec38bf | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_leaflet_arrangement_20261105_01/T21.obj` | 1.0 MB | 26330956c151f70b3f8d954ac7f08d866d68ae562a9a901e9cb988fc644b7f3f | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/refk__t2_05.npz` | 993.3 KB | 5b8a95009bfeab3e8b78d4c67f4faf6961aefc84df12fdefacd57cef6ec0cc5e | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__e_05.obj` | 933.6 KB | d75bdcea3b262bc086feb464bf1500923b0af4b2961ba822241f41a32998c813 | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/refk__t2_04.npz` | 922.9 KB | d369d57f21e7797f7fc1e73c496913700822d06beab58344cc7c0765c46d64e6 | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_disc_vs_tri_20261101_01/kaggle_data/` | 882.4 KB | 폴더, 4 파일 | D vs T (representation comparison) | excluded dir (outputs/uploads/raw/cache), 4 files [.json:2,.csv:2] |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/refk__n_02.npz` | 853.9 KB | 4ab1fc499c326bdb7b70758df71716b26471c92ce36b70df731659bc77757d3a | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__n_01.obj` | 833.6 KB | a5aee81de8e3ade3d95e6330c72f672a0a60f973e73f92d7068acae2ad29f2d6 | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__n_04.obj` | 833.3 KB | a3739f63ddb619874179f60e5c7f90256a88ffd8b8956673e42ff2d7327a69c1 | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_disc_vs_tri_20261101_01/T.obj` | 758.1 KB | 02a6234ccce8399a5b68ffbd9d13d8444b970173709d165c3733abf72d95450f | D vs T (representation comparison) | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/refk__R3-1b.npz` | 750.6 KB | f699fb4d822eb3395076b5cb093a895542fbe50b781ee75e5cfcd1b41e84fec2 | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__n_05.obj` | 728.9 KB | 8f7bba232241af88b91571389dbe98221df3c45ff48e272072484f60288439af | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__R3-1b.obj` | 686.8 KB | 2fdd5e00285bd1e1560a22a0a9148754fd89ae6449bdd9ce8af9bca9d7c4e377 | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/mtg__e_05.obj` | 686.0 KB | 84f0c486a4fcc1fe73755dcd22b9b8cc784a192ee0fa2d68b6e307046b2f0f66 | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__e_01.obj` | 661.0 KB | 6f80721241e3fbdc6c397ac5dd11e915645bd97a0eb0791909d76add1b9b5f09 | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__e_07.obj` | 653.0 KB | 075b8a0863a53c4af9c910d720171144ec1778a44bbb0ace1f4f25ff505c0078 | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/refk__n_03.npz` | 634.2 KB | 6931da93f89108e1b63f40cbbb73a031ee2ca5732b323ac72f639b81179d2976 | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/mtg__e_07.obj` | 587.0 KB | 49a0f255008db534be4f5ef3b6e271f3dbd0e91cd6f2af290555124f91e0ecfe | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_tau_rule2_20261105_01/out_k16/` | 560.9 KB | 폴더, 6 파일 | optics rule 2 (d_stem 1.5 mm) RESULT_TABLE_rule2_unified | excluded dir (outputs/uploads/raw/cache), 6 files [.log:2,.json:2,(none):1,.npz:1] |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__n_02.obj` | 499.7 KB | 9e26067d55e831a74c541ae29ddf522300cfc2d76c9f3baf5530b322d904fb0a | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/mtg__e_04.obj` | 499.5 KB | 4a6d93d39fa1e6c73e5471409a7d0c59baeaff538f5a48b4bd118c7749a1720c | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/kaggle_ref/mtg__e_01.obj` | 497.8 KB | 9dfd9c6c2012063ca3bdab3b62d0cca3a9c12de43f843c75c8bc3264e598262a | v1_tau adopted reference + GroIMP tau closure | data file (not uploaded) |
| `build/engine_static_normals_v2_20261104_01/out_k3a/` | 483.9 KB | 폴더, 8 파일 | same-leaf normal smoothing r 2 mm | excluded dir (outputs/uploads/raw/cache), 8 files [.json:6,.log:2] |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__n_06.obj` | 474.1 KB | 8365a676c854cf50ad51ee1303fe3bcd662204a86fcbc3c54725765eaa28a6b7 | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__e_04.obj` | 468.2 KB | 74f458ef0b154f5ce7fedf4f2071cee648cf53cb86c29a7d31378fc05f4b4aa2 | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_20261026_01/results/` | 389.7 KB | 폴더, 8 파일 | static v1 tracer (S0-S3 PASS) + GroIMP closure probe | excluded dir (outputs/uploads/raw/cache), 8 files [.json:4,.md:4] |
| `build/engine_static_lai_range_20261105_01/out_k15c/` | 356.0 KB | 폴더, 17 파일 | LAI range + 5 mm layer transmittance (_cell) | excluded dir (outputs/uploads/raw/cache), 17 files [.json:8,(none):7,.log:2] |
| `build/engine_static_lai_range_20261105_01/out_k15a/` | 356.0 KB | 폴더, 17 파일 | LAI range + 5 mm layer transmittance (_cell) | excluded dir (outputs/uploads/raw/cache), 17 files [.json:8,(none):7,.log:2] |
| `build/engine_static_normals_v2_20261104_01/out_k3b/` | 348.3 KB | 폴더, 7 파일 | same-leaf normal smoothing r 2 mm | excluded dir (outputs/uploads/raw/cache), 7 files [.json:5,.log:2] |
| `build/engine_static_lai_range_20261105_01/out_k15b/` | 346.8 KB | 폴더, 17 파일 | LAI range + 5 mm layer transmittance (_cell) | excluded dir (outputs/uploads/raw/cache), 17 files [.json:8,(none):7,.log:2] |
| `build/engine_static_lai_range_20261105_01/out_k15d/` | 346.8 KB | 폴더, 17 파일 | LAI range + 5 mm layer transmittance (_cell) | excluded dir (outputs/uploads/raw/cache), 17 files [.json:8,(none):7,.log:2] |
| `build/engine_static_leaflet_arrangement_20261105_01/kaggle_c2/mtg04__n_03.obj` | 337.4 KB | 5b3a3396adc01cb47a39e067f22aefc59870d0ad213ec698acd4c55bb8a751a8 | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_static_tau_20261031_01/kaggle_data/` | 303.6 KB | 폴더, 10 파일 | v1_tau T2b | excluded dir (outputs/uploads/raw/cache), 10 files [.py:7,.json:2,.npz:1] |
| `build/engine_static_tau_rule2_20261105_01/out_k14/` | 259.9 KB | 폴더, 8 파일 | optics rule 2 (d_stem 1.5 mm) RESULT_TABLE_rule2_unified | excluded dir (outputs/uploads/raw/cache), 8 files [.json:5,.log:2,.py:1] |
| `build/engine_static_tau_rule2_20261105_01/out_k18/` | 259.9 KB | 폴더, 8 파일 | optics rule 2 (d_stem 1.5 mm) RESULT_TABLE_rule2_unified | excluded dir (outputs/uploads/raw/cache), 8 files [.json:5,.log:2,.py:1] |
| `build/engine_static_leaflet_arrangement_20261105_01/T21.ply` | 221.8 KB | 0f1fbbb0b8a467ce28e718d88edb69baaaa9c860c7f7e9f12a65796d6139283e | leaflet arrangement Part C | data file (not uploaded) |
| `build/engine_disc_vs_tri_20261101_01/T.ply` | 161.5 KB | 3e4960f000b33b06bc4fb13c14e3cd17afbc767fc098df8acafa1ab8adaa1d66 | D vs T (representation comparison) | data file (not uploaded) |
| `build/engine_static_v1tau_reference_20261104_01/out_k11/` | 82.8 KB | 폴더, 3 파일 | v1_tau adopted reference + GroIMP tau closure | excluded dir (outputs/uploads/raw/cache), 3 files [.log:2,.json:1] |
| `build/engine_static_lai_range_20261105_01/out_k15a_gatefail_sumorder/` | 51.8 KB | 폴더, 7 파일 | LAI range + 5 mm layer transmittance (_cell) | excluded dir (outputs/uploads/raw/cache), 7 files [.json:4,.log:2,(none):1] |
| `build/engine_static_normals_20261104_01/out_k2/` | 36.6 KB | 폴더, 3 파일 | same-leaf normal smoothing v1 | excluded dir (outputs/uploads/raw/cache), 3 files [.log:2,.json:1] |
| `build/engine_static_tau_rule2_20261105_01/out_k13/` | 31.5 KB | 폴더, 3 파일 | optics rule 2 (d_stem 1.5 mm) RESULT_TABLE_rule2_unified | excluded dir (outputs/uploads/raw/cache), 3 files [.log:2,.json:1] |
| `build/engine_static_tau_rule2_20261105_01/out_k17/` | 31.5 KB | 폴더, 3 파일 | optics rule 2 (d_stem 1.5 mm) RESULT_TABLE_rule2_unified | excluded dir (outputs/uploads/raw/cache), 3 files [.log:2,.json:1] |
| `build/engine_static_v1tau_reference_20261104_01/out_k9/` | 31.4 KB | 폴더, 3 파일 | v1_tau adopted reference + GroIMP tau closure | excluded dir (outputs/uploads/raw/cache), 3 files [.log:2,.json:1] |
| `build/engine_disc_vs_tri_20261101_01/T_diamond.obj` | 23.2 KB | a81e7e25a39ae81213ca30748d588d036983602610c4ae50b4412b2b34417773 | D vs T (representation comparison) | data file (not uploaded) |
| `build/engine_static_leaflet_arrangement_20261105_01/out_k20/` | 21.1 KB | 폴더, 14 파일 | leaflet arrangement Part C | excluded dir (outputs/uploads/raw/cache), 14 files [.json:12,.log:2] |
| `build/engine_static_v1tau_reference_20261104_01/out_k10/` | 21.1 KB | 폴더, 14 파일 | v1_tau adopted reference + GroIMP tau closure | excluded dir (outputs/uploads/raw/cache), 14 files [.json:12,.log:2] |
| `build/engine_static_tau_20261031_01/out_k3/` | 12.0 KB | 폴더, 3 파일 | v1_tau T2b | excluded dir (outputs/uploads/raw/cache), 3 files [.log:2,.json:1] |
| `build/engine_static_lai_range_20261105_01/out_k15b_gatefail_sumorder/` | 10.6 KB | 폴더, 5 파일 | LAI range + 5 mm layer transmittance (_cell) | excluded dir (outputs/uploads/raw/cache), 5 files [.json:3,.log:2] |
| `build/engine_static_tau_20261031_01/out_k1/` | 7.2 KB | 폴더, 3 파일 | v1_tau T2b | excluded dir (outputs/uploads/raw/cache), 3 files [.log:2,.json:1] |
| `build/engine_disc_vs_tri_20261101_01/T_diamond.ply` | 5.0 KB | baa0350f2427f4f945fcc934d735add57fc71c07fbb2b45497c7b907971fcad4 | D vs T (representation comparison) | data file (not uploaded) |
| `build/engine_static_normals_20261104_01/out_k1_v1_error/` | 2.3 KB | 폴더, 1 파일 | same-leaf normal smoothing v1 | excluded dir (outputs/uploads/raw/cache), 1 files [.log:1] |

## 공개판에서 뺀 VTC 파생 파일

상위 [`VTC_WITHHELD.md`](../VTC_WITHHELD.md) 참조 (파일 목록·크기·sha256·영향받는 스크립트).
