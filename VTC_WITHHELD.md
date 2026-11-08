# VTC_WITHHELD — 공개판에서 뺀 VTC 파생 파일

VTC (Virtual Tomato Crop; K. Smolenova / WUR, J. Evers FSPM_BASIC 에서 고침; 논문 https://doi.org/10.1093/insilicoplants/diaf022) 원본에는 LICENSE 파일과 저작권 문구가 없습니다. 재배포 조건을 알 수 없으므로 아래 파일은 이 공개 저장소에 넣지 않았습니다 (2026-11-08 결정). 원저자 허락을 받으면 다시 넣을 수 있습니다.

- 모델 `.rgg`·`.gsz`: VTC 모델을 고친 사본 (동결 v2·v2.1·v2.2 slaA/slaR, GUI 캐노피 모델).
- `src/tools/vtc_growth_fork.py`: VTC 원본 NRH 함수를 바이트 그대로 담음. `src/tools/vtc_development.py`: 그 fork 를 실행 (VTC 코드 자체는 없으나 fork 를 import 하므로 함께 뺌).
- `Model_*_vs_*.diff`, `parametersMicroTom_*_vs_*.diff`: 위 모델 사이의 차이 (모델 줄을 그대로 담음).

sha256 은 원 파일 바이트 (줄끝 CRLF 그대로) 기준입니다. 파일을 따로 받은 사람은 이 값으로 같은 판인지 확인할 수 있습니다.

## 뺀 파일 (33개)

| 폴더 | 경로 | 바이트 | sha256 |
|---|---|---|---|
| `2dgs-recon` | `src/tools/vtc_development.py` | 26,376 | `f288bdc29d244dc3f2bda31db0008113fdaac0fa4e7fc47e5fe13475308ac1df` |
| `2dgs-recon` | `src/tools/vtc_growth_fork.py` | 13,646 | `454f12a50bf237dab9322f3ae6ba56421852a21787165a3aaed2246fd989dffb` |
| `static-light` | `src/tools/vtc_development.py` | 26,376 | `f288bdc29d244dc3f2bda31db0008113fdaac0fa4e7fc47e5fe13475308ac1df` |
| `static-light` | `src/tools/vtc_growth_fork.py` | 13,646 | `454f12a50bf237dab9322f3ae6ba56421852a21787165a3aaed2246fd989dffb` |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/gui/Model_gui_canopy_slaA.rgg` | 118,479 | `d36ae6514ecc6d691021a8a7944d98a6ede87015e1118170decd108c0acd74f7` |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/gui/Model_gui_canopy_slaR.rgg` | 118,487 | `d5d42a7a75cc91a8dbee644a0779f40f5a761b23bf2cd789063a02746197ca04` |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/gui/TomatoCanopyGUI_slaA.gsz` | 36,467 | `bd936358418f2f489e0f06bc6b129f4458eae1f1984df4654d8fa4f1dd768662` |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/gui/TomatoCanopyGUI_slaA_original.gsz` | 36,467 | `bd936358418f2f489e0f06bc6b129f4458eae1f1984df4654d8fa4f1dd768662` |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/gui/TomatoCanopyGUI_slaR.gsz` | 36,474 | `68218a3b658461067a56808cfd20b59935192f4550d518047c2274cedd9c004c` |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/gui/TomatoCanopyGUI_slaR_original.gsz` | 36,474 | `68218a3b658461067a56808cfd20b59935192f4550d518047c2274cedd9c004c` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v1_m00_reference.rgg` | 83,263 | `c448d602af11be1325cf83169ef90b832516a4c39f455250c06a5f4e0b5ba33c` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2.rgg` | 90,255 | `fa53fe89a660e58383a9c52905b4e9c6b09ffd8338d7df0334f55d2898faf090` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_1.rgg` | 101,126 | `0a408633872ec31f5d8bfff6e3ecfee671b12140c5a47ff56bae78fc459b41ac` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_1_vs_v1.diff` | 25,541 | `ea69cc2fd3558f10939ff7df489c7f3a4a49adcdc7441a86d038d4ced8945ef0` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_1_vs_v2.diff` | 17,855 | `e63c8f39f92ee80ecaf08848ca567b61551e7ed807da652e9ad245d67e43beee` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_2_candidate_s5.85.rgg` | 100,994 | `8f33ac2b062a7c4e0e6be11901921fa5ea3a17f8c3958d175c99832a126df203` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_2_s4.10_withdrawn.rgg` | 100,993 | `168226c92e9f1ca90d98aff1010c0cd3606b2b95049b46d6efd049b2f81fabe6` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_2_slaA.rgg` | 100,996 | `8b04647ecc2271225b04f6ae2e8666fb0b9d8e621a692562f32f8cc82c9e09a6` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_2_slaA_vs_v2_1.diff` | 3,378 | `2eae1d1911af98c0d0b207a1c8b39ee4a77edcee9c82a3b9b71f834c9c2db872` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_2_slaR.rgg` | 101,005 | `30e0df8c3544b08b599c4e5fa4252287b9eef4be9edc180344658640e0600be6` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_2_slaR_vs_v2_1.diff` | 3,251 | `08c32c21a549a5c26b11889d1ebb321177bac2bc070a2a6b20505089c7a5ef79` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_2_vs_v2_1_candidate_s5.85.diff` | 3,122 | `827e2dc8256b5b6ff36153f7d5581771edeff94211abd4646ca4ccb0085fa2c3` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_2_vs_v2_1_s4.10_withdrawn.diff` | 3,375 | `7f63dc70ca9a4a40ea7cf2fbefd7b1b75dc16c601ea20310ef7a70ac79b5ec18` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/Model_v2_vs_v1.diff` | 12,782 | `e5f7704f5d66c0068ebac6a94c64fc956e719b73e057b3b8eef7085e7bdbe481` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/parametersMicroTom_v2.rgg` | 11,635 | `9e925fffaaef5f1277f8cff2406547d1b0859271fc9d260aabd0b7a2fa647cc5` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/parametersMicroTom_v2_1.rgg` | 11,635 | `9e925fffaaef5f1277f8cff2406547d1b0859271fc9d260aabd0b7a2fa647cc5` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/parametersMicroTom_v2_2_candidate_s5.85.rgg` | 11,655 | `004927904f093f5374cf4f629047d1cb068b2fe1094a7666051ad27c8fa15d43` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/parametersMicroTom_v2_2_s4.10_withdrawn.rgg` | 11,655 | `004927904f093f5374cf4f629047d1cb068b2fe1094a7666051ad27c8fa15d43` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/parametersMicroTom_v2_2_slaA.rgg` | 11,655 | `004927904f093f5374cf4f629047d1cb068b2fe1094a7666051ad27c8fa15d43` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/parametersMicroTom_v2_2_slaR.rgg` | 11,666 | `b4e5b879aec0f1570bddd80951b1180aa187e99be322ed1381cc1abd6b94c9a5` |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/work/parametersMicroTom_v2_vs_v1.diff` | 121 | `83bcf83abf993754fdcefd9d80886858c89222704614d51a2dbf42bb39b81764` |
| `dynamic-fspm` | `src/tools/vtc_development.py` | 26,376 | `f288bdc29d244dc3f2bda31db0008113fdaac0fa4e7fc47e5fe13475308ac1df` |
| `dynamic-fspm` | `src/tools/vtc_growth_fork.py` | 13,646 | `454f12a50bf237dab9322f3ae6ba56421852a21787165a3aaed2246fd989dffb` |

## 그래서 돌지 않는 파일 (111개)

자동 점검 결과입니다: 위 모듈을 (다른 모듈을 거쳐서라도) import 하는 `.py`, 또는 뺀 모델 파일 이름을 읽는 스크립트. 코드는 고치지 않았습니다. 파일 수: `2dgs-recon` 32, `static-light` 28, `dynamic-fspm` 51.

주요 영향: `2dgs-recon` 의 복엽 골격 사슬 (`tomato3d/organs/skeleton_*`, T-Prox 채점 `p1_tprox_sam2_*`) 은 `skeleton_tree` → `tools.vtc_development_observation` → … → `tools.vtc_development` 경로로 import 가 끊깁니다. 표로 남긴 결과 (CSV·JSON·그림) 는 그대로 들어 있습니다.

| 폴더 | 파일 | 이유 |
|---|---|---|
| `2dgs-recon` | `src/build/p1_tprox_sam2_20261107_01/grade_picks.py` | imports tomato3d.organs.skeleton_dz10_labels -> ... |
| `2dgs-recon` | `src/build/p1_tprox_sam2_20261107_01/perleaf_fig.py` | imports tomato3d.organs.skeleton_r31b_eval -> ... |
| `2dgs-recon` | `src/build/p1_tprox_sam2_20261107_01/score.py` | imports tomato3d.organs.skeleton_r31b_eval -> ... |
| `2dgs-recon` | `src/tomato3d/organs/all3d_page.py` | imports tomato3d.organs.skeleton_propagate -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_axis.py` | imports tomato3d.organs.skeleton_tree -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_carry.py` | imports tomato3d.organs.skeleton_tree -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_dz10_labels.py` | imports tomato3d.organs.skeleton_propagate -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_multicue.py` | imports tomato3d.organs.skeleton_r31b_eval -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_propagate.py` | imports tomato3d.organs.skeleton_tree -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_r31b_complete.py` | imports tomato3d.organs.skeleton_r31b_eval -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_r31b_eval.py` | imports tomato3d.organs.skeleton_r8pc -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_r8pc.py` | imports tomato3d.organs.skeleton_propagate -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_store.py` | imports tomato3d.organs.skeleton_axis -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_tprox.py` | imports tomato3d.organs.skeleton_propagate -> ... |
| `2dgs-recon` | `src/tomato3d/organs/skeleton_tree.py` | imports tools.vtc_development_observation -> ... |
| `2dgs-recon` | `src/tools/fsp_axes.py` | imports tools.fsp_axes_segments -> ... |
| `2dgs-recon` | `src/tools/fsp_axes_segments.py` | imports tools.fsp_engine_hooks -> ... |
| `2dgs-recon` | `src/tools/fsp_engine_hooks.py` | imports tools.fsp_incident_diagnostics -> ... |
| `2dgs-recon` | `src/tools/fsp_incident_diagnostics.py` | imports tools.fsp_stage_parameters -> ... |
| `2dgs-recon` | `src/tools/fsp_nitrogen_uptake.py` | imports tools.fsp_substrate_water -> tools.vtc_development |
| `2dgs-recon` | `src/tools/fsp_outline_policy.py` | imports tools.vtc_development_io -> ... |
| `2dgs-recon` | `src/tools/fsp_projected_shading.py` | imports tools.vtc_development |
| `2dgs-recon` | `src/tools/fsp_stage_parameters.py` | imports tools.microtom_budbreak -> ... |
| `2dgs-recon` | `src/tools/fsp_substrate_water.py` | imports tools.vtc_development |
| `2dgs-recon` | `src/tools/microtom_budbreak.py` | imports tools.microtom_environment -> ... |
| `2dgs-recon` | `src/tools/microtom_environment.py` | imports tools.vtc_development_io -> ... |
| `2dgs-recon` | `src/tools/microtom_parameters.py` | imports tools.vtc_development_io -> ... |
| `2dgs-recon` | `src/tools/tomato_fsp_intake.py` | imports tools.tomato_fsp_io -> ... |
| `2dgs-recon` | `src/tools/tomato_fsp_io.py` | imports tools.fsp_axes -> ... |
| `2dgs-recon` | `src/tools/vtc_development_io.py` | imports tools.fsp_nitrogen_uptake -> ... |
| `2dgs-recon` | `src/tools/vtc_development_observation.py` | imports tools.fsp_axes -> ... |
| `2dgs-recon` | `src/tools/vtc_growth_bridge.py` | imports tools.fsp_incident_diagnostics -> ... |
| `static-light` | `src/build/engine_disc_vs_tri_20261101_01/make_T.py` | imports tomato3d.organs.skeleton_tree -> ... |
| `static-light` | `src/build/engine_static_20261026_01/groimp_noise_pair.py` | imports tools.microtom_hook_runs -> ... |
| `static-light` | `src/tomato3d/organs/skeleton_tree.py` | imports tools.vtc_development_observation -> ... |
| `static-light` | `src/tools/fsp_axes.py` | imports tools.fsp_axes_segments -> ... |
| `static-light` | `src/tools/fsp_axes_segments.py` | imports tools.fsp_engine_hooks -> ... |
| `static-light` | `src/tools/fsp_engine_hooks.py` | imports tools.fsp_incident_diagnostics -> ... |
| `static-light` | `src/tools/fsp_engine_hooks_acceptance.py` | imports tools.fsp_axes_segments -> ... |
| `static-light` | `src/tools/fsp_incident_diagnostics.py` | imports tools.fsp_stage_parameters -> ... |
| `static-light` | `src/tools/fsp_nitrogen_uptake.py` | imports tools.fsp_substrate_water -> tools.vtc_development |
| `static-light` | `src/tools/fsp_observation_intake.py` | imports tools.tomato_fsp_io -> ... |
| `static-light` | `src/tools/fsp_outline_policy.py` | imports tools.vtc_development_io -> ... |
| `static-light` | `src/tools/fsp_projected_shading.py` | imports tools.vtc_development |
| `static-light` | `src/tools/fsp_reserve_expansion_acceptance.py` | imports tools.fsp_engine_hooks -> ... |
| `static-light` | `src/tools/fsp_stage_parameters.py` | imports tools.microtom_budbreak -> ... |
| `static-light` | `src/tools/fsp_substrate_water.py` | imports tools.vtc_development |
| `static-light` | `src/tools/microtom_budbreak.py` | imports tools.microtom_environment -> ... |
| `static-light` | `src/tools/microtom_environment.py` | imports tools.vtc_development_io -> ... |
| `static-light` | `src/tools/microtom_forecast.py` | imports tools.fsp_axes -> ... |
| `static-light` | `src/tools/microtom_hook_runs.py` | imports tools.fsp_engine_hooks -> ... |
| `static-light` | `src/tools/microtom_parameters.py` | imports tools.vtc_development_io -> ... |
| `static-light` | `src/tools/tomato_fsp_acceptance.py` | imports tools.tomato_fsp_io -> ... |
| `static-light` | `src/tools/tomato_fsp_intake.py` | imports tools.tomato_fsp_io -> ... |
| `static-light` | `src/tools/tomato_fsp_io.py` | imports tools.fsp_axes -> ... |
| `static-light` | `src/tools/tomato_fsp_pipeline.py` | imports tools.fsp_axes -> ... |
| `static-light` | `src/tools/vtc_development_io.py` | imports tools.fsp_nitrogen_uptake -> ... |
| `static-light` | `src/tools/vtc_development_observation.py` | imports tools.fsp_axes -> ... |
| `static-light` | `src/tools/vtc_growth_bridge.py` | imports tools.fsp_incident_diagnostics -> ... |
| `static-light` | `src/build/engine_static_leaflet_arrangement_20261105_01/make_T21.py` | reads withheld model file(s): Model_v2_1.rgg |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/kaggle_kernel/gui_case.py` | imports tools.vtc_growth_bridge -> ... |
| `dynamic-fspm` | `src/build/engine_ideotype_20261105_01/kaggle_kernel/ideo_case.py` | imports tools.vtc_growth_bridge -> ... |
| `dynamic-fspm` | `src/build/engine_ideotype_20261105_01/prepare_ideo.py` | imports tools.fsp_engine_hooks -> ... |
| `dynamic-fspm` | `src/build/engine_ideotype_20261105_01/prepare_ideo_v21only.py` | imports tools.fsp_engine_hooks -> ... |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/kaggle_kernel/ev_case.py` | imports tools.vtc_growth_bridge -> ... |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/kaggle_kernel/ev_case_pre_c4a.py` | imports tools.vtc_growth_bridge -> ... |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/kaggle_kernel/ev_case_pre_c4c.py` | imports tools.vtc_growth_bridge -> ... |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/kaggle_kernel/v2_validate.py` | imports tools.vtc_development_io -> ... |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/kaggle_kernel/v2_validate_pre_v11.py` | imports tools.vtc_development_io -> ... |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/kaggle_kernel/v2_validate_pre_v12.py` | imports tools.vtc_development_io -> ... |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/prepare_v2.py` | imports tools.fsp_engine_hooks -> ... |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/prepare_v2_pre_c4c.py` | imports tools.fsp_engine_hooks -> ... |
| `dynamic-fspm` | `src/tools/fsp_axes.py` | imports tools.fsp_engine_hooks -> ... |
| `dynamic-fspm` | `src/tools/fsp_engine_hooks.py` | imports tools.fsp_nitrogen_uptake -> ... |
| `dynamic-fspm` | `src/tools/fsp_engine_hooks_acceptance.py` | imports tools.fsp_engine_hooks -> ... |
| `dynamic-fspm` | `src/tools/fsp_extensions.py` | imports tools.fsp_engine_hooks -> ... |
| `dynamic-fspm` | `src/tools/fsp_incident_diagnostics.py` | imports tools.fsp_stage_parameters -> ... |
| `dynamic-fspm` | `src/tools/fsp_nitrogen_uptake.py` | imports tools.fsp_substrate_water -> tools.vtc_development |
| `dynamic-fspm` | `src/tools/fsp_observation_intake.py` | imports tools.tomato_fsp_io -> ... |
| `dynamic-fspm` | `src/tools/fsp_outline_policy.py` | imports tools.vtc_development_io -> ... |
| `dynamic-fspm` | `src/tools/fsp_projected_shading.py` | imports tools.vtc_development |
| `dynamic-fspm` | `src/tools/fsp_reserve_expansion_acceptance.py` | imports tools.fsp_engine_hooks -> ... |
| `dynamic-fspm` | `src/tools/fsp_stage_parameters.py` | imports tools.fsp_engine_hooks -> ... |
| `dynamic-fspm` | `src/tools/fsp_substrate_water.py` | imports tools.vtc_development |
| `dynamic-fspm` | `src/tools/microtom_budbreak.py` | imports tools.microtom_environment -> ... |
| `dynamic-fspm` | `src/tools/microtom_environment.py` | imports tools.vtc_development_io -> ... |
| `dynamic-fspm` | `src/tools/microtom_forecast.py` | imports tools.fsp_axes -> ... |
| `dynamic-fspm` | `src/tools/microtom_hook_runs.py` | imports tools.fsp_engine_hooks -> ... |
| `dynamic-fspm` | `src/tools/microtom_parameters.py` | imports tools.fsp_extensions -> ... |
| `dynamic-fspm` | `src/tools/tomato_fsp_acceptance.py` | imports tools.tomato_fsp_io -> ... |
| `dynamic-fspm` | `src/tools/tomato_fsp_intake.py` | imports tools.tomato_fsp_io -> ... |
| `dynamic-fspm` | `src/tools/tomato_fsp_io.py` | imports tools.fsp_axes -> ... |
| `dynamic-fspm` | `src/tools/tomato_fsp_native_boundary.py` | imports tools.fsp_extensions -> ... |
| `dynamic-fspm` | `src/tools/tomato_fsp_pipeline.py` | imports tools.fsp_axes -> ... |
| `dynamic-fspm` | `src/tools/vtc_development_io.py` | imports tools.fsp_axes -> ... |
| `dynamic-fspm` | `src/tools/vtc_development_observation.py` | imports tools.fsp_axes -> ... |
| `dynamic-fspm` | `src/tools/vtc_growth_bridge.py` | imports tools.fsp_incident_diagnostics -> ... |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/make_gsz.py` | reads withheld model file(s): Model_*.rgg / parametersMicroTom_*.rgg (pattern) |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/make_launchers.py` | reads withheld model file(s): Model_*.rgg / parametersMicroTom_*.rgg (pattern) |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/START_GUI_CANOPY.bat` | reads withheld model file(s): Model_*.rgg / parametersMicroTom_*.rgg (pattern) |
| `dynamic-fspm` | `src/build/engine_gui_canopy_20261106_01/START_GUI_CANOPY_lf_broken.bat` | reads withheld model file(s): Model_*.rgg / parametersMicroTom_*.rgg (pattern) |
| `dynamic-fspm` | `src/build/engine_ideotype_20261105_01/kaggle_kernel/ideo_validate.py` | reads withheld model file(s): Model_*.rgg / parametersMicroTom_*.rgg (pattern) |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/make_params_v2.py` | reads withheld model file(s): Model_v2.rgg |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/make_tables_figs.py` | reads withheld model file(s): Model_*.rgg / parametersMicroTom_*.rgg (pattern) |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/make_tables_figs_v2_1.py` | reads withheld model file(s): Model_*.rgg / parametersMicroTom_*.rgg (pattern) |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/START_GUI_V2.bat` | reads withheld model file(s): Model_v2_2_slaA.rgg |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/START_GUI_V2_0.bat` | reads withheld model file(s): Model_v2.rgg |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/START_GUI_V2_1.bat` | reads withheld model file(s): Model_v2_1.rgg |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/START_GUI_V2_2R.bat` | reads withheld model file(s): Model_v2_2_slaR.rgg |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/START_GUI_V2_s4.10_withdrawn.bat` | reads withheld model file(s): Model_*.rgg / parametersMicroTom_*.rgg (pattern) |
| `dynamic-fspm` | `src/build/engine_v2_calib_20261104_01/v2_patch.py` | reads withheld model file(s): Model_v2.rgg, parametersMicroTom_v2.rgg |

---
**English.** VTC has no licence file, so the 33 VTC-derived files above (model `.rgg`/`.gsz`, `tools/vtc_growth_fork.py` which embeds the original NRH function, `tools/vtc_development.py` which drives it, and model `.diff` files) are withheld from this public repository; sha256 values are listed for verification. The 111 files listed second import those modules (directly or transitively) or read the withheld models, so they will not run here. Shipped tables and figures are unaffected.
