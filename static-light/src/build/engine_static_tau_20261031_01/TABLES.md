# v1_tau tables

light_model=v1_tau · assumed_leaf_optics (tau=0.07, a=0.85, reflection dropped) · relative_light · conditional_on_unvalidated_scale · after_checkpoint_v1

## Plant level (d_ex 2.4 mm; a = 1 unless noted; change vs v1)
| plant | v1 mean light | τ0.07 mean light | Δ % | Δ % with a 0.85 | self-shading v1 → τ0.07 | lower third Δ % | middle Δ % | upper Δ % | τ0.05 / τ0.10 Δ % | overflow rays |
|---|---|---|---|---|---|---|---|---|---|---|
| R3-1b | 0.247 | 0.256 | +3.7 | -11.9 | 0.619 → 0.605 | +2.2 | +4.0 | +3.9 | +2.6 / +5.3 | 0 |
| t2_02 | 0.273 | 0.286 | +4.8 | -10.9 | 0.655 → 0.639 | +5.5 | +5.0 | +4.2 | +3.4 / +7.0 | 0 |
| n_04 | 0.308 | 0.322 | +4.3 | -11.3 | 0.593 → 0.575 | +3.7 | +5.1 | +4.0 | +3.1 / +6.3 | 0 |
| n_01 | 0.263 | 0.274 | +4.0 | -11.6 | 0.607 → 0.592 | +3.2 | +5.7 | +3.1 | +2.8 / +5.8 | 0 |
| t2_05 | 0.359 | 0.370 | +2.9 | -12.6 | 0.432 → 0.416 | +2.3 | +3.2 | +2.7 | +2.0 / +4.1 | 0 |
| n_06 | 0.372 | 0.385 | +3.5 | -12.1 | 0.504 → 0.487 | +2.3 | +4.6 | +3.0 | +2.4 / +5.0 | 0 |
| n_05 | 0.335 | 0.347 | +3.8 | -11.8 | 0.541 → 0.524 | +2.9 | +5.0 | +2.8 | +2.7 / +5.5 | 0 |
| t2_04 | 0.332 | 0.342 | +3.2 | -12.3 | 0.474 → 0.457 | +2.2 | +3.1 | +3.5 | +2.3 / +4.6 | 0 |
| n_02 | 0.311 | 0.322 | +3.6 | -12.0 | 0.541 → 0.525 | +4.0 | +3.7 | +3.4 | +2.5 / +5.2 | 0 |
| n_03 | 0.400 | 0.412 | +3.0 | -12.4 | 0.457 → 0.441 | +2.7 | +3.7 | +2.6 | +2.2 / +4.4 | 0 |

## Per leaf (R3-1b R8_pc blades, V-P leaves 1 and 7; n_04 clicked leaf 1 lower bound)
| plant | leaf | mean light v1 | mean light τ0.07 | Δ % (a = 1) |
|---|---|---|---|---|
| R3-1b | vp_leaf_1 | 0.305 | 0.313 | +2.5 |
| R3-1b | vp_leaf_7 | 0.315 | 0.326 | +3.5 |
| R3-1b | r8_blade_1 | 0.324 | 0.331 | +2.2 |
| R3-1b | r8_blade_2 | 0.391 | 0.399 | +2.0 |
| R3-1b | r8_blade_3 | 0.402 | 0.408 | +1.3 |
| R3-1b | r8_blade_4 | 0.386 | 0.396 | +2.6 |
| R3-1b | r8_blade_5 | 0.407 | 0.419 | +2.8 |
| R3-1b | r8_blade_6 | 0.213 | 0.221 | +3.7 |
| R3-1b | r8_blade_7 | 0.404 | 0.415 | +2.5 |
| R3-1b | r8_blade_8 | 0.404 | 0.412 | +2.0 |
| n_04 | click_leaf_1_lower_bound | 0.284 | 0.292 | +2.9 |

## Scenarios R3-1b alone, lamp: stable 44 / 44
| scenario | unit | rel Δ v1 | rel Δ τ0.07 | class |
|---|---|---|---|---|
| lamp_250mm | R3-1b | +2.8 % | +2.7 % | stable |
| lamp_250mm | R3-1b:vp_leaf_1 | +6.9 % | +6.8 % | stable |
| lamp_250mm | R3-1b:vp_leaf_7 | -3.3 % | -3.2 % | stable |
| lamp_250mm | R3-1b:r8_blade_1 | +6.7 % | +6.6 % | stable |
| lamp_250mm | R3-1b:r8_blade_2 | +8.7 % | +8.6 % | stable |
| lamp_250mm | R3-1b:r8_blade_3 | +8.7 % | +8.7 % | stable |
| lamp_250mm | R3-1b:r8_blade_4 | +3.4 % | +3.4 % | stable |
| lamp_250mm | R3-1b:r8_blade_5 | -2.3 % | -2.2 % | stable |
| lamp_250mm | R3-1b:r8_blade_6 | -3.7 % | -3.8 % | stable |
| lamp_250mm | R3-1b:r8_blade_7 | -3.3 % | -3.2 % | stable |
| lamp_250mm | R3-1b:r8_blade_8 | +5.3 % | +5.2 % | stable |
| lamp_350mm | R3-1b | -5.1 % | -5.0 % | stable |
| lamp_350mm | R3-1b:vp_leaf_1 | -6.9 % | -6.9 % | stable |
| lamp_350mm | R3-1b:vp_leaf_7 | -2.0 % | -2.1 % | stable |
| lamp_350mm | R3-1b:r8_blade_1 | -6.8 % | -6.8 % | stable |
| lamp_350mm | R3-1b:r8_blade_2 | -8.4 % | -8.3 % | stable |
| lamp_350mm | R3-1b:r8_blade_3 | -7.9 % | -7.9 % | stable |
| lamp_350mm | R3-1b:r8_blade_4 | -5.9 % | -5.9 % | stable |
| lamp_350mm | R3-1b:r8_blade_5 | -2.8 % | -2.9 % | stable |
| lamp_350mm | R3-1b:r8_blade_6 | -1.5 % | -1.4 % | stable |
| lamp_350mm | R3-1b:r8_blade_7 | -2.0 % | -2.1 % | stable |
| lamp_350mm | R3-1b:r8_blade_8 | -6.4 % | -6.4 % | stable |
| lateral_y_-50mm | R3-1b | -12.7 % | -12.8 % | stable |
| lateral_y_-50mm | R3-1b:vp_leaf_1 | -9.6 % | -9.7 % | stable |
| lateral_y_-50mm | R3-1b:vp_leaf_7 | -16.5 % | -16.4 % | stable |
| lateral_y_-50mm | R3-1b:r8_blade_1 | -9.9 % | -10.0 % | stable |
| lateral_y_-50mm | R3-1b:r8_blade_2 | -9.2 % | -9.4 % | stable |
| lateral_y_-50mm | R3-1b:r8_blade_3 | -9.9 % | -9.9 % | stable |
| lateral_y_-50mm | R3-1b:r8_blade_4 | -12.0 % | -12.0 % | stable |
| lateral_y_-50mm | R3-1b:r8_blade_5 | -16.5 % | -16.4 % | stable |
| lateral_y_-50mm | R3-1b:r8_blade_6 | -15.8 % | -16.0 % | stable |
| lateral_y_-50mm | R3-1b:r8_blade_7 | -16.5 % | -16.4 % | stable |
| lateral_y_-50mm | R3-1b:r8_blade_8 | -10.5 % | -10.6 % | stable |
| lateral_y_+50mm | R3-1b | +15.6 % | +15.6 % | stable |
| lateral_y_+50mm | R3-1b:vp_leaf_1 | +9.6 % | +9.8 % | stable |
| lateral_y_+50mm | R3-1b:vp_leaf_7 | +22.7 % | +22.6 % | stable |
| lateral_y_+50mm | R3-1b:r8_blade_1 | +10.0 % | +10.1 % | stable |
| lateral_y_+50mm | R3-1b:r8_blade_2 | +7.8 % | +8.0 % | stable |
| lateral_y_+50mm | R3-1b:r8_blade_3 | +7.8 % | +7.9 % | stable |
| lateral_y_+50mm | R3-1b:r8_blade_4 | +14.3 % | +14.4 % | stable |
| lateral_y_+50mm | R3-1b:r8_blade_5 | +21.9 % | +21.7 % | stable |
| lateral_y_+50mm | R3-1b:r8_blade_6 | +22.9 % | +23.1 % | stable |
| lateral_y_+50mm | R3-1b:r8_blade_7 | +22.7 % | +22.6 % | stable |
| lateral_y_+50mm | R3-1b:r8_blade_8 | +11.7 % | +11.8 % | stable |

## Scenarios tray 60 mm: stable 36 / 36
| scenario | unit | rel Δ v1 | rel Δ τ0.07 | class |
|---|---|---|---|---|
| lamp_250mm | n_04 | +7.7 % | +7.7 % | stable |
| lamp_250mm | n_04:click_leaf_1_lower_bound | +18.4 % | +18.2 % | stable |
| lamp_250mm | t2_02 | +9.0 % | +9.0 % | stable |
| lamp_250mm | n_01 | -3.7 % | -3.6 % | stable |
| lamp_250mm | t2_05 | -1.6 % | -1.6 % | stable |
| lamp_250mm | tray_total | +4.4 % | +4.4 % | stable |
| lamp_350mm | n_04 | -7.7 % | -7.7 % | stable |
| lamp_350mm | n_04:click_leaf_1_lower_bound | -12.6 % | -12.5 % | stable |
| lamp_350mm | t2_02 | -8.3 % | -8.3 % | stable |
| lamp_350mm | n_01 | -1.9 % | -1.9 % | stable |
| lamp_350mm | t2_05 | -1.9 % | -1.9 % | stable |
| lamp_350mm | tray_total | -5.9 % | -5.9 % | stable |
| lateral_y_-50mm | n_04 | -11.3 % | -11.3 % | stable |
| lateral_y_-50mm | n_04:click_leaf_1_lower_bound | +9.0 % | +8.7 % | stable |
| lateral_y_-50mm | t2_02 | -11.8 % | -11.8 % | stable |
| lateral_y_-50mm | n_01 | -18.9 % | -18.8 % | stable |
| lateral_y_-50mm | t2_05 | -16.8 % | -16.9 % | stable |
| lateral_y_-50mm | tray_total | -13.8 % | -13.8 % | stable |
| lateral_y_+50mm | n_04 | +10.9 % | +10.9 % | stable |
| lateral_y_+50mm | n_04:click_leaf_1_lower_bound | -11.2 % | -10.9 % | stable |
| lateral_y_+50mm | t2_02 | +10.5 % | +10.6 % | stable |
| lateral_y_+50mm | n_01 | +25.2 % | +25.1 % | stable |
| lateral_y_+50mm | t2_05 | +23.1 % | +23.1 % | stable |
| lateral_y_+50mm | tray_total | +15.4 % | +15.4 % | stable |
| spacing_90mm | n_04 | +6.9 % | +6.8 % | stable |
| spacing_90mm | n_04:click_leaf_1_lower_bound | +73.8 % | +72.6 % | stable |
| spacing_90mm | t2_02 | +9.4 % | +9.2 % | stable |
| spacing_90mm | n_01 | +22.4 % | +22.2 % | stable |
| spacing_90mm | t2_05 | +80.4 % | +77.6 % | stable |
| spacing_90mm | tray_total | +17.6 % | +17.2 % | stable |
| spacing_120mm | n_04 | +8.5 % | +8.2 % | stable |
| spacing_120mm | n_04:click_leaf_1_lower_bound | +115.6 % | +113.1 % | stable |
| spacing_120mm | t2_02 | +12.2 % | +11.9 % | stable |
| spacing_120mm | n_01 | +31.3 % | +30.9 % | stable |
| spacing_120mm | t2_05 | +164.6 % | +157.8 % | stable |
| spacing_120mm | tray_total | +28.2 % | +27.5 % | stable |
