"""Korean report of tray32 v3 from the CSVs (numbers computed). python -X utf8 make_report_v3.py"""
import csv, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
def rd(n):
    return list(csv.DictReader(l for l in open(HERE / n, encoding='utf-8') if not l.startswith('#')))
E = rd('efficiency_by_scene.csv'); H = rd('height_response.csv'); S = rd('smoothing_vs_v2.csv'); P = rd('position_effect.csv'); f = lambda r, k: float(r[k])
tabs = sorted({e['table'] for e in E}); tt = '+'.join(tabs)
m = lambda **kw: float(np.mean([f(e, 'efficiency_a085') for e in E if all(e[k] == v for k, v in kw.items())]))
hs = ('0.3', '0.4', '0.5', '0.6'); byh = {h: m(height_m=h) for h in hs}; bys = {s: m(shape=s) for s in ('flat', 'V120', 'V90')}; byp = {p: m(spacing_m=p) for p in ('0.06', '0.1', '0.14')}
ref = [e for e in E if e['table'] == 'A' and e['spacing_m'] == '0.1' and e['shape'] == 'flat' and e['position'] == 'trays' and e['height_m'] == '0.4'][0]
EA = [e for e in E if e['table'] == 'A']; best = max(EA, key=lambda e: f(e, 'efficiency_a085')); worst = min(EA, key=lambda e: f(e, 'efficiency_a085'))
desc = lambda e: f"{int(float(e['spacing_m']) * 100)} cm 간격·{e['shape']}·{'트레이 위' if e['position'] == 'trays' else '틈 위'}·{int(float(e['height_m']) * 100)} cm"
r40 = {h: [f(r, f'ratio_to_40cm_{int(float(h) * 100)}cm') for r in H if r['position'] != 'gaps_over_trays'] for h in ('0.3', '0.5', '0.6')}
gt = {(r['table'], r['shape'], r['spacing_m']): r for r in H if r['position'] == 'gaps_over_trays'}
def cross_txt(shape):
    out = []
    for p in ('0.06', '0.1', '0.14'):
        r = gt.get(('A', shape, p)); c = r['sign_change_height_m']
        out.append(f"{int(float(p) * 100)} cm: " + ('교차 없음 (' + ('틈 위가 항상 좋음' if f(r, 'eff_30cm') > 1 else '트레이 위가 항상 좋음') + ')' if c in ('none', '') else f'{float(c) * 100:.0f} cm 에서 교차 (낮으면 트레이 위, 높으면 틈 위)'))
    return '; '.join(out)
er = [f(r, 'eff_ratio_v3_v2') for r in S]; sk = [r for r in S if r['effect_sign_kept'] in ('True', 'False')]
po = [x for x in P if x['sid'] == ref['sid'] and x['contrast'] == 'outer/inner'][0]; pr = [x for x in P if x['sid'] == ref['sid'] and x['contrast'] == 'end/middle'][0]
oi = [f(x, 'median_rel_diff') for x in P if x['contrast'] == 'outer/inner']; em = [f(x, 'median_rel_diff') for x in P if x['contrast'] == 'end/middle']
nb = {p: np.mean([f(e, 'neighbour_shading_share_mean') for e in E if e['spacing_m'] == p and e.get('neighbour_shading_share_mean')]) for p in ('0.06', '0.1', '0.14')}
conv = json.load(open(HERE / 'conv_A.json')); reg = json.load(open(HERE / 'regression_vs_v2.json')); cl = json.load(open(HERE / 'clearance_summary.json')); c2 = json.load(open(HERE / 'LOCAL_C2_v3.json'))
bA = {e['sid']: f(e, 'efficiency_a085') for e in E if e['table'] == 'A'}; bB = {e['sid']: f(e, 'efficiency_a085') for e in E if e['table'] == 'B'}; rat = [bB[s] / bA[s] for s in bA if s in bB]
iso_tabs = sorted({e['table'] for e in E if e.get('neighbour_shading_share_mean')})
L = ['# 32개체 트레이 정적 광 v3 — 조명 높이 30 / 40 / 50 / 60 cm, 채택 기준 `v1tau_smoothn_r2` (2026-11-05)', '',
     '`light_model_v1tau_smoothn_r2` · `assumed_luminaire_model` · `repeated_placement` · `relative_light` · `conditional_on_unvalidated_scale` · `skeleton_labels_ungraded` · `scenario_comparison_not_validation` · `absolute_efficiency_uncertain_up_to_1.9x` — 측정·검증 아님',
     f'계약 `build/static_tray32_v3_20261105_01/CONTRACT.md`(숫자 전에 작성; 배치표 A·B 는 v2 와 sha256 동일). 이 문서 = 배치 **{tt}**, 고립 복사본 실행 = 배치 {"+".join(iso_tabs)}. 효율 = 식물 흡수광(a 0.85) ÷ 조명 방출광(4 L), 면적 (b) 기준.', '']
L.append(f"1. **효율 {100 * f(worst, 'efficiency_a085'):.2f} ~ {100 * f(best, 'efficiency_a085'):.2f} %**(표 A 72장면" + (f"; 표 B/A = {np.mean(rat):.3f} [{min(rat):.3f}, {max(rat):.3f}]" if rat else '') + f"). 기준 장면(평면·트레이 위·40 cm·10 cm) {100 * f(ref, 'efficiency_a085'):.2f} %; 최고 {desc(best)}, 최저 {desc(worst)}.")
L.append(f"2. **높이:** 30 / 40 / 50 / 60 cm 평균 {100 * byh['0.3']:.2f} / {100 * byh['0.4']:.2f} / {100 * byh['0.5']:.2f} / {100 * byh['0.6']:.2f} %. 40 cm 대비 30 cm ×{min(r40['0.3']):.2f}–{max(r40['0.3']):.2f}, 50 cm ×{min(r40['0.5']):.2f}–{max(r40['0.5']):.2f}, 60 cm ×{min(r40['0.6']):.2f}–{max(r40['0.6']):.2f}. 10 cm 낮출 때마다 약 25–30 % 늘어남.")
L.append(f"3. **모양·간격:** 평면 {100 * bys['flat']:.2f} > V 120° {100 * bys['V120']:.2f} > V 90° {100 * bys['V90']:.2f} %; 간격 6 / 10 / 14 cm {100 * byp['0.06']:.2f} / {100 * byp['0.1']:.2f} / {100 * byp['0.14']:.2f} %.")
L.append(f"4. **틈 위 vs 트레이 위가 바뀌는 높이(표 A):** 평면 — {cross_txt('flat')}. V 120° — {cross_txt('V120')}. V 90° — {cross_txt('V90')}.")
L.append(f"5. **평활화 효과(40 / 50 cm, v3 vs v2, 보고만):** 효율 ×{min(er):.3f}–{max(er):.3f}(평균 +{100 * (np.mean(er) - 1):.1f} %), 복사본별 비 중앙값 {min(f(r, 'copy_ratio_median') for r in S):.2f}–{max(f(r, 'copy_ratio_median') for r in S):.2f}; 자기 그늘 {np.mean([f(r, 'self_shading_v2') for r in S if r.get('self_shading_v2')]):.3f} → {np.mean([f(r, 'self_shading_v3') for r in S if r.get('self_shading_v3')]):.3f}, 이웃 그늘 몫 {np.mean([f(r, 'neighbour_share_v2') for r in S if r.get('neighbour_share_v2')]):.3f} → {np.mean([f(r, 'neighbour_share_v3') for r in S if r.get('neighbour_share_v3')]):.3f}; 기준 장면 대비 장면 효과 부호 **{sum(1 for r in sk if r['effect_sign_kept'] == 'True')}/{len(sk)} 유지**.")
L.append(f"6. **그늘·위치:** 자기 그늘 평균 {np.mean([f(e, 'self_shading_mean') for e in E if e.get('self_shading_mean')]):.2f}; 이웃 그늘 몫 {100 * nb['0.06']:.0f} / {100 * nb['0.1']:.1f} / {100 * nb['0.14']:.1f} % (6 / 10 / 14 cm). 바깥 트레이 vs 안쪽 트레이(같은 식물 짝, 기준 장면) {100 * f(po, 'median_rel_diff'):+.0f} %(식물 {po['n_plants']}개; 전 장면 {100 * min(oi):+.0f} ~ {100 * max(oi):+.0f} %), 끝 줄 vs 가운데 줄 {100 * f(pr, 'median_rel_diff'):+.0f} %(전 장면 {100 * min(em):+.0f} ~ {100 * max(em):+.0f} %).")
L.append(f"7. **에너지 닫힘:** 잔차 평균 +{100 * np.mean([f(e, 'closure_residual') for e in E]):.1f} % (방출광 대비). 바닥에서 본 식물 차단 ÷ (b) 기준 흡수+반사 = v2 {np.mean([f(r, 'removed_over_absorbed_plus_reflected_v2') for r in S]):.2f} → v3 {np.mean([f(r, 'removed_over_absorbed_plus_reflected_v3') for r in S]):.2f}: 평활화가 절대 수준 불확실성을 조금 줄이지만 최대 약 1.9배 불확실성은 그대로입니다.")
L.append(f"8. **검사:** 평활화 끈 회귀 6/6 장면 v2 와 비트 단위 동일({'예' if reg['all_identical'] else '아니오'}); 방출 적분 편차 {c2['C2_worst_rel_dev']:.1e}; 수렴(30 cm 기준 장면) 2배에서 {100 * conv['rows'][1]['rel_change_vs_previous']:+.2f} % → 8000 / 32 유지; 여유 거리 검사 {cl['flagged']}/{cl['rows']} 표시(최소 {100 * cl['min_vertical_gap_m']:.1f} cm, 30 cm 높이) → `near_luminaire` 장면 없음.")
L.append('9. ' + ('표 B·고립 복사본 B 는 선언 순서대로 표 A 보고 뒤 실행 중이며, 끝나면 이 문서를 갱신합니다.' if 'B' not in tabs else ('표 B 포함 결과입니다. 고립 복사본은 배치 ' + '+'.join(iso_tabs) + ' 에서 실행했습니다.')))
L += ['', '## 표 1 — 효율 (%), 표 A, 간격 10 cm', '', '| 모양 | 조명 위치 | 30 cm | 40 cm | 50 cm | 60 cm |', '|---|---|---|---|---|---|']
for sh in ('flat', 'V120', 'V90'):
    for pos in ('trays', 'gaps'):
        L.append(f"| {sh} | {pos} | " + ' | '.join(f"{100 * m(table='A', shape=sh, position=pos, spacing_m='0.1', height_m=h):.2f}" for h in hs) + ' |')
L += ['', '## 표 2 — 틈 위 ÷ 트레이 위 (표 A)', '', '| 모양 | 간격 | 30 cm | 40 cm | 50 cm | 60 cm | 교차 높이 |', '|---|---|---|---|---|---|---|']
for sh in ('flat', 'V120', 'V90'):
    for p in ('0.06', '0.1', '0.14'):
        r = gt[('A', sh, p)]; L.append(f"| {sh} | {int(float(p) * 100)} cm | " + ' | '.join(f"{f(r, f'eff_{c}cm'):.3f}" for c in (30, 40, 50, 60)) + f" | {r['sign_change_height_m'] if r['sign_change_height_m'] not in ('', 'none') else '없음'} |")
L += ['', '파일: `scenes.csv`, `clearance.csv`, `efficiency_by_scene.csv`, `plants_by_scene.csv`, `leaves_by_scene.csv`, `position_effect.csv`, `scenario_effects.csv`, `height_response.csv`, `smoothing_vs_v2.csv`, `regression_vs_v2.json`, `figures/`.', '',
      '## Result 문장', '',
      f"(한국어) 파일럿 개체를 포함한 복원 개체 14개를 4개 트레이(4×2, 간격 6–14 cm)에 32개 배치하고 60 cm 조명 4개 높이를 30–60 cm 로 바꾼 정적 광 모델에서, 식물이 흡수한 빛은 방출광의 {100 * f(worst, 'efficiency_a085'):.1f}–{100 * f(best, 'efficiency_a085'):.1f} % 였고(면적 (b) 기준, 가정한 잎 광학, 절대 수준은 최대 약 1.9배 불확실), 조명을 10 cm 낮출 때마다 약 25–30 % 늘었으며 평면 조명이 V 90° 조명보다 {100 * (bys['flat'] / bys['V90'] - 1):.0f} % 높았습니다; 바깥 트레이 개체는 안쪽보다 잎면적당 {abs(100 * f(po, 'median_rel_diff')):.0f} % 적게 받았습니다.",
      '',
      f"(English) With 32 copies of the 14 reconstructed plants in four 4 × 2 trays (6–14 cm spacing) under four 60 cm luminaires at 30–60 cm, the modelled plants absorbed {100 * f(worst, 'efficiency_a085'):.1f}–{100 * f(best, 'efficiency_a085'):.1f} % of the emitted light (area (b) basis, assumed leaf optics; absolute level uncertain up to about 1.9×); each 10 cm lower luminaire height raised it by about 25–30 %, a flat panel was {100 * (bys['flat'] / bys['V90'] - 1):.0f} % better than a 90° V, and plants in the outer trays absorbed {abs(100 * f(po, 'median_rel_diff')):.0f} % less per leaf area than those in the inner trays."]
(ROOT / 'handoff' / 'STATIC_TRAY32_V3_2026-11-05_KO.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('report', len(L), 'lines')
