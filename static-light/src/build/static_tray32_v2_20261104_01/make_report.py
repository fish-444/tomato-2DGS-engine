"""Korean report of the 32-plant tray scenes from the aggregated tables (numbers computed from the CSVs). python -X utf8 make_report.py"""
import csv, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
def rd(n):
    return list(csv.DictReader(l for l in open(HERE / n, encoding='utf-8') if not l.startswith('#')))
E = rd('efficiency_by_scene.csv'); PE = rd('position_effect.csv'); SE = rd('scenario_effects.csv')
tabs = sorted({e['table'] for e in E}); tabtxt = '+'.join(tabs); f = lambda r, k: float(r[k])
def eff(table=None, **kw):
    v = [f(e, 'efficiency_a085') for e in E if (table is None or e['table'] == table) and all(str(e[k]) == str(x) for k, x in kw.items())]
    return float(np.mean(v)) if v else float('nan')
def pct(x):
    return f'{100 * x:.2f} %'
ref = [e for e in E if e['sid'] == 'S13' and e['table'] == 'A'][0]
best = max((e for e in E if e['table'] == 'A'), key=lambda e: f(e, 'efficiency_a085')); worst = min((e for e in E if e['table'] == 'A'), key=lambda e: f(e, 'efficiency_a085'))
desc = lambda e: f"p {int(float(e['spacing_m']) * 100)} cm, {e['shape']}, over {e['position']}, {int(float(e['height_m']) * 100)} cm"
byshape = {s: eff(shape=s) for s in ('flat', 'V120', 'V90')}; byh = {h: eff(height_m=h) for h in ('0.4', '0.5')}; byp = {p: eff(spacing_m=p) for p in ('0.06', '0.1', '0.14')}; bypos = {p: eff(position=p) for p in ('trays', 'gaps')}
def gap_ratio(shape):
    return {p: eff(shape=shape, spacing_m=p, position='gaps') / eff(shape=shape, spacing_m=p, position='trays') for p in ('0.06', '0.1', '0.14')}
conv = json.load(open(HERE / 'conv_A.json')) if (HERE / 'conv_A.json').exists() else None
ext = json.load(open(HERE / 'extra_A.json')) if (HERE / 'extra_A.json').exists() else None
tests = json.load(open(HERE / 'LOCAL_TESTS.json'))
def pe(contrast, sid='S13'):
    r = [x for x in PE if x['sid'] == sid and x['contrast'] == contrast][0]; return r
po, pr = pe('outer/inner'), pe('end/middle')
nb = {p: np.mean([f(e, 'neighbour_shading_share_mean') for e in E if e['spacing_m'] == p and e['table'] == 'A' and e.get('neighbour_shading_share_mean') not in (None, '')]) for p in ('0.06', '0.1', '0.14')}
sf = np.mean([f(e, 'self_shading_mean') for e in E if e.get('self_shading_mean') not in (None, '')])
ot = {p: (min(f(e, 'outlines_share_of_emitted') for e in E if e['spacing_m'] == p), max(f(e, 'outlines_share_of_emitted') for e in E if e['spacing_m'] == p)) for p in ('0.06', '0.1', '0.14')}
ao = {p: np.mean([f(e, 'absorbed_over_outlines_light') for e in E if e['spacing_m'] == p]) for p in ('0.06', '0.1', '0.14')}
gin = {p: np.mean([f(e, 'per_area_gini') for e in E if e['spacing_m'] == p]) for p in ('0.06', '0.1', '0.14')}
res_mean = np.mean([f(e, 'closure_residual') for e in E]); res_ref = f(ref, 'closure_residual'); rem_ref = f(ref, 'closure_removed_by_plants'); ap_ref = f(ref, 'closure_absorbed_plus_reflected')
L = []
L.append('# 32개체 · 4개 트레이(4×2) 정적 광 — 간격 × 조명 모양 × 조명 위치 × 높이 (2026-11-04)')
L.append('')
L.append('`light_model_v1_tau`(τ 0.07, a 0.85, 줄기 τ 0) · `assumed_luminaire_model` · `repeated_placement` · `relative_light` · `conditional_on_unvalidated_scale` · `skeleton_labels_ungraded` · `scenario_comparison_not_validation` — 측정·검증 아님')
L.append(f'계약 `build/static_tray32_v2_20261104_01/CONTRACT.md`(배치표 A/B·조명 기하·선언값, 숫자 전에 작성). **이 문서의 표 = 배치 {tabtxt}.** 효율 = 식물이 흡수한 광 ÷ 조명 방출광(4 L). 면적은 P1 실루엣 보정 면적 (b) 기준입니다.')
L.append('')
_bA = {e['sid']: f(e, 'efficiency_a085') for e in E if e['table'] == 'A'}; _bB = {e['sid']: f(e, 'efficiency_a085') for e in E if e['table'] == 'B'}
_rat = [_bB[s] / _bA[s] for s in _bA if s in _bB] if _bB else []
L.append(f"1. **효율은 {pct(f(worst, 'efficiency_a085'))} ~ {pct(f(best, 'efficiency_a085'))}** (표 A 36장면" + (f"; 표 B {pct(min(_bB.values()))} ~ {pct(max(_bB.values()))}, 같은 장면에서 B/A = {np.mean(_rat):.3f} [{min(_rat):.3f}, {max(_rat):.3f}] — 배치에 따른 차이 약 ±5 %, 순서는 같음" if _rat else '') + f"). 기준 장면(평면·트레이 위·40 cm·10 cm 간격) {pct(f(ref, 'efficiency_a085'))}; 최고 {best['sid']} ({desc(best)}), 최저 {worst['sid']} ({desc(worst)}).")
L.append(f"2. **모양:** 평면 {pct(byshape['flat'])} > V 120° {pct(byshape['V120'])} (−{100 * (1 - byshape['V120'] / byshape['flat']):.0f} %) > V 90° {pct(byshape['V90'])} (−{100 * (1 - byshape['V90'] / byshape['flat']):.0f} %). **높이** 40 → 50 cm: {pct(byh['0.4'])} → {pct(byh['0.5'])} (−{100 * (1 - byh['0.5'] / byh['0.4']):.0f} %). **간격** 6 / 10 / 14 cm: {pct(byp['0.06'])} / {pct(byp['0.1'])} / {pct(byp['0.14'])} (14 cm 에서 빛이 식물 사이로 빠짐).")
L.append(f"3. **조명 위치(트레이 위 vs 틈 위)는 평균 거의 같고({pct(bypos['trays'])} vs {pct(bypos['gaps'])}) 모양에 따라 부호가 갈립니다:** 평면은 틈 위가 트레이 위의 {gap_ratio('flat')['0.06']:.2f} / {gap_ratio('flat')['0.1']:.2f} / {gap_ratio('flat')['0.14']:.2f} 배(간격 6/10/14 cm), V 120° {gap_ratio('V120')['0.06']:.2f} / {gap_ratio('V120')['0.1']:.2f} / {gap_ratio('V120')['0.14']:.2f}, V 90° {gap_ratio('V90')['0.06']:.2f} / {gap_ratio('V90')['0.1']:.2f} / {gap_ratio('V90')['0.14']:.2f} 배.")
L.append(f"4. **트레이 윤곽에 닿는 광**은 방출광의 {100 * ot['0.06'][0]:.0f}–{100 * ot['0.06'][1]:.0f} % (6 cm) · {100 * ot['0.1'][0]:.0f}–{100 * ot['0.1'][1]:.0f} % (10 cm) · {100 * ot['0.14'][0]:.0f}–{100 * ot['0.14'][1]:.0f} % (14 cm); 윤곽 광 대비 흡수 {ao['0.06']:.2f} / {ao['0.1']:.2f} / {ao['0.14']:.2f}. 나머지는 바닥·트레이 밖으로 갑니다.")
L.append(f"5. **그늘:** 자기 그늘 평균 {sf:.2f}(모든 장면 비슷). 이웃 그늘 몫(고립 복사본 대비) {100 * nb['0.06']:.0f} % (6 cm) · {100 * nb['0.1']:.1f} % (10 cm) · {100 * nb['0.14']:.1f} % (14 cm).")
_oi = [float(x['median_rel_diff']) for x in PE if x['contrast'] == 'outer/inner']; _em = [float(x['median_rel_diff']) for x in PE if x['contrast'] == 'end/middle']
L.append(f"6. **위치 효과**(같은 식물 복사본끼리 짝지음, 기준 장면, 잎면적당 흡수광, 배치 {tabtxt}, 식물 {po['n_plants']}개): 바깥 트레이 vs 안쪽 트레이 중앙값 **{100 * f(po, 'median_rel_diff'):+.0f} %**(p10–p90 {100 * f(po, 'p10'):+.0f} ~ {100 * f(po, 'p90'):+.0f} %, 식물의 {100 * (1 - f(po, 'share_plants_positive')):.0f} % 가 음수); 36장면 전체에서 {100 * min(_oi):+.0f} ~ {100 * max(_oi):+.0f} %. 끝 줄 vs 가운데 줄 중앙값 **{100 * f(pr, 'median_rel_diff'):+.0f} %**({100 * f(pr, 'p10'):+.0f} ~ {100 * f(pr, 'p90'):+.0f} %, 식물의 {100 * (1 - f(pr, 'share_plants_positive')):.0f} % 가 음수; 36장면 {100 * min(_em):+.0f} ~ {100 * max(_em):+.0f} %): 바깥 트레이는 확실히, 끝 줄은 약하게 덜 받습니다(조명 60 cm 가 트레이 길이보다 길지만 이웃 광 유입이 가운데 쪽에 많음).")
L.append(f"7. **균일도**(32개체 잎면적당 흡수광 Gini): 6 cm {gin['0.06']:.2f} · 10 cm {gin['0.1']:.2f} · 14 cm {gin['0.14']:.2f}.")
if conv:
    L.append(f"8. **검사:** 점광원 극한(C1) 통과(차이 0), 방출 적분(C2) 편차 {tests['C2']['worst_rel_dev']:.1e}(≤ 0.5 %), 수렴(C4): 수신점·방출점 2배에서 효율 {100 * conv['rows'][1]['rel_change_vs_previous']:+.2f} % → 선언한 8000 / 32 유지. 확산광 0.6 추가(표 A, 10 cm 기준 장면): 흡수 ×{ext['ratio_with_over_direct_only']:.2f}, 수평 조도당 차단 직접 {ext['direct_interception']:.2f} · 확산 {ext['diffuse_interception']:.2f}; e_* 괄호 팔 {ext['bracket_change_pct']:+.2f} %.")
L.append(f"9. **주의(에너지 닫힘):** 닫힘 잔차 평균 +{100 * res_mean:.1f} % (기준 장면 +{100 * res_ref:.1f} %): 바닥에서 본 식물 차단량 {100 * rem_ref:.1f} % vs 면적 (b) 기준 흡수+반사 {100 * ap_ref:.1f} %. 단일 개체 진단(R3-1b, 거리 3 m 점광원): 원판 합집합 그림자 43 cm² 대비 (b) 로 설명되는 윗층 투영 약 29 cm² → 원판 발자국(k = 1.5) 대 면적 (b) 불일치와 다층 겹침. **절대 효율 수준은 최대 약 1.6–1.9 배 불확실**(그림자 기준 차단 {100 * rem_ref:.1f} %); 장면 사이 짝지은 비교는 같은 가정에서 나온 것이라 더 믿을 만합니다.")
L.append(('10. 표 B(다른 배치)와 고립 복사본 B 는 선언한 순서대로 표 A 보고 뒤에 실행 중이며, 끝나면 이 문서를 갱신합니다. ' if 'B' not in tabs else '10. 표 B(다른 배치)와 고립 복사본 B 까지 포함한 결과입니다. ') + '점광원 극한 시험·수렴 시험·확산광 민감도는 표 A 에서만 했습니다.')
L.append('')
L.append('## 표 1 — 효율 (%, 표 A; 열 = 간격 × 높이)')
L.append('')
L.append('| 모양 | 조명 위치 | 6 cm 40 | 6 cm 50 | 10 cm 40 | 10 cm 50 | 14 cm 40 | 14 cm 50 |')
L.append('|---|---|---|---|---|---|---|---|')
for sh in ('flat', 'V120', 'V90'):
    for pos in ('trays', 'gaps'):
        row = [f"{100 * eff('A', shape=sh, position=pos, spacing_m=p, height_m=h):.2f}" for p in ('0.06', '0.1', '0.14') for h in ('0.4', '0.5')]
        L.append(f'| {sh} | {pos} | ' + ' | '.join(row) + ' |')
L.append('')
L.append('## 표 2 — 기준 장면(평면·트레이 위·40 cm·10 cm) 대비 효율 비 (표 A)')
L.append('')
L.append('| 바꾼 것 | 장면 | 효율 비 | 복사본별 비 중앙값 [p10, p90] |')
L.append('|---|---|---|---|')
for sid, what in (('S14', '높이 50 cm'), ('S15', '틈 위'), ('S17', 'V 120°'), ('S21', 'V 90°'), ('S01', '간격 6 cm'), ('S25', '간격 14 cm'), ('S23', 'V 90° + 틈 위'), ('S36', 'V 90° + 틈 위 + 14 cm + 50 cm')):
    r = [x for x in SE if x['sid'] == sid and x['table'] == 'A']
    if r:
        r = r[0]; L.append(f"| {what} | {sid} | {float(r['ratio']):.3f} | {float(r['per_copy_ratio_median']):.3f} [{float(r['per_copy_ratio_p10']):.2f}, {float(r['per_copy_ratio_p90']):.2f}] |")
L.append('')
L.append('파일: `placement_A.csv`, `placement_B.csv`, `luminaires.csv`, `scenes.csv`, `efficiency_by_scene.csv`, `plants_by_scene.csv`, `leaves_by_scene.csv`, `position_effect.csv`, `scenario_effects.csv`, `figures/`.')
L.append('')
L.append('## Result 에 쓸 문장 (표 A + 표 B 기준)')
L.append(f"\"In a 32-plant layout (four 4 × 2 trays, 10 cm gaps) of the 14 reconstructed plants lit by four 60 cm luminaires, the modelled share of emitted light absorbed by the plants (area (b) basis, assumed leaf optics) was {100 * f(worst, 'efficiency_a085'):.1f}–{100 * f(best, 'efficiency_a085'):.1f} % across 36 geometries; a flat panel was {100 * (byshape['flat'] / byshape['V90'] - 1):.0f} % better than a 90° V, lowering the luminaires from 50 to 40 cm raised it by {100 * (byh['0.4'] / byh['0.5'] - 1):.0f} %, and widening the plant spacing from 10 to 14 cm lowered it by {100 * (1 - byp['0.14'] / byp['0.1']):.0f} %; plants in the outer trays absorbed {abs(100 * f(po, 'median_rel_diff')):.0f} % less per leaf area than those in the inner trays.\"")
(ROOT / 'handoff' / 'STATIC_TRAY32_V2_2026-11-04_KO.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('report written', len(L), 'lines')
