"""Markdown tables from results/run*.json for the STATIC report. python tables.py [run2|run3|run4|run5] -> stdout"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; R = HERE / 'results'


def m(d, k, f='%.3f', scale=1.0):
    v = d.get(k)
    if isinstance(v, dict) and 'mean' in v:
        return (f % (v['mean'] * scale)) + ' ± ' + (f % (v['sigma'] * scale))
    if v is None:
        return 'n/a'
    return f % (v * scale) if isinstance(v, float) else str(v)


def run2():
    d = json.load(open(R / 'run2.json', encoding='utf-8'))
    print('| plant | leaf area b cm² | b/a | LAI pot top | LAI 60 mm cell | shaded fraction | mean incident r.u. | absorbed r.u.·cm² | T rim layer | k diag | LUE gC/mol (cond.) | outside support | area above pot |')
    print('|' + '---|' * 13)
    for pid, p in d['plants'].items():
        r = p['reference']
        print(f"| {pid} | {m(r, 'leaf_area_b_m2', '%.1f', 1e4)} | {m(r, 'ratio_b_over_a', '%.3f')} | {m(r, 'lai_pot_top', '%.2f')} | {m(r, 'lai_tray_cell', '%.2f')} | "
              f"{m(r, 'shaded_fraction_leaf')} | {m(r, 'mean_incident_with_extension_ru')} | {m(r, 'absorbed_with_extension_ru_m2', '%.2f', 1e4)} | "
              f"{m(r, 'transmittance_at_rim_layer')} | {m(r, 'k_extinction_diagnostic', '%.2f')} | {m(r, 'lue_gC_mol_conditional_on_meter_scale')} | "
              f"{m(r, 'area_fraction_outside_support', '%.2f')} | {m(r, 'leaf_area_fraction_above_pot_square', '%.2f')} |")
    print('\nSensitivity (seed 101): relative change of absorbed r.u.·m² vs reference seed 101')
    print('| plant | footprint ×0.8 | ×1.2 | δ 3 mm | δ 8 mm | diffuse 0.5 | 0.7 | leaf area ×0.8 | ×1.2 |')
    print('|' + '---|' * 9)
    for pid, p in d['plants'].items():
        ref = p['reference']['absorbed_with_extension_ru_m2']['mean']; s = p['sensitivity']
        refa = p['reference']['leaf_area_b_m2']['mean']
        cells = [f"{s[k]['absorbed_with_extension_ru_m2'] / ref - 1:+.3f}" for k in ('footprint_0p8', 'footprint_1p2', 'delta_3mm', 'delta_8mm', 'diffuse_0p5', 'diffuse_0p7')]
        cells += [f"{s[k]['leaf_area_b_m2'] / refa - 1:+.3f}" for k in ('footprint_0p8', 'footprint_1p2')]
        print(f"| {pid} | " + ' | '.join(cells) + ' |')
    r = d['plants']['R3-1b']['reference']['per_leaf']
    print('\nR3-1b per leaf (usable 1, 7; other = all remaining leaves incl. viewing-only, aggregated)')
    print('| leaf | area b cm² | z mean mm | shaded fraction | mean incident r.u. | absorbed r.u.·cm² | LUE (cond.) |')
    print('|---|---|---|---|---|---|---|')
    for k, v in r.items():
        print(f"| {k} | {m(v, 'area_b_m2', '%.1f', 1e4)} | {m(v, 'z_mean_mm', '%.1f')} | {m(v, 'shaded_fraction')} | {m(v, 'mean_incident_ru')} | "
              f"{m(v, 'absorbed_with_extension_ru_m2', '%.2f', 1e4)} | {m(v, 'lue_gC_mol_conditional_on_meter_scale')} |")


def run3():
    d = json.load(open(R / 'run3.json', encoding='utf-8'))
    print('| plant | leaf area cm² | LAI pot top | shaded fraction | mean incident r.u. | absorbed r.u.·cm² | T rim layer | k diag | LUE (cond.) | top z mm |')
    print('|' + '---|' * 10)
    for pid, r in d['plants'].items():
        top = max((p['z_mm'][1]['mean'] if isinstance(p['z_mm'][1], dict) else p['z_mm'][1]) for p in r['profile']) + 60 if 'profile' in r else None
        print(f"| {pid} | {m(r, 'leaf_area_b_m2', '%.1f', 1e4)} | {m(r, 'lai_pot_top', '%.2f')} | {m(r, 'shaded_fraction_leaf')} | {m(r, 'mean_incident_with_extension_ru')} | "
              f"{m(r, 'absorbed_with_extension_ru_m2', '%.2f', 1e4)} | {m(r, 'transmittance_at_rim_layer')} | {m(r, 'k_extinction_diagnostic', '%.2f')} | "
              f"{m(r, 'lue_gC_mol_conditional_on_meter_scale')} | {top} |")
    r = d['plants']['model_p1']
    print(f"\nmodel_p1 coincident area fraction {m(r, 'coincident_area_fraction')}, unique leaf area {m(r, 'leaf_area_unique_m2', '%.1f', 1e4)} cm²")


def run4():
    d = json.load(open(R / 'run4.json', encoding='utf-8'))
    for pitch, t in d['trays'].items():
        print(f'\nTray pitch {pitch} mm (assumed_layout)')
        print('| plant | absorbed all r.u.·cm² | alone | none | self-shading loss | neighbour loss | total loss | LAI cell | outside support |')
        print('|' + '---|' * 9)
        for pid, p in t.items():
            a = p['attribution']
            print(f"| {pid} | {m(p['all'], 'absorbed_with_extension_ru_m2', '%.2f', 1e4)} | {m(p['alone'], 'absorbed_with_extension_ru_m2', '%.2f', 1e4)} | "
                  f"{m(p['none'], 'absorbed_with_extension_ru_m2', '%.2f', 1e4)} | {m(a, 'self_shading_loss')} | {m(a, 'neighbour_shading_loss')} | {m(a, 'total_loss')} | "
                  f"{m(p['all'], 'lai_tray_cell', '%.2f')} | {m(p['all'], 'area_fraction_outside_support', '%.2f')} |")


def run5():
    d = json.load(open(R / 'run5.json', encoding='utf-8'))
    print('Tray (60 mm reference): relative Δ absorbed (r.u.·m², model extension where outside support) ± σ; * = resolved (|Δ| > 2σ)')
    names = list(d['tray'])
    print('| plant | ' + ' | '.join(names) + ' |'); print('|' + '---|' * (len(names) + 1))
    for pid in d['tray'][names[0]]:
        cells = []
        for n in names:
            x = d['tray'][n][pid]['absorbed_with_extension_ru_m2']
            cells.append(f"{x['rel_delta']:+.3f} ± {x['sigma'] / x['reference']:.3f}{'*' if x['resolved'] else ''}")
        print(f'| {pid} | ' + ' | '.join(cells) + ' |')
    print('\nR3-1b alone, lamp scenarios: relative Δ absorbed ± σ (* resolved)')
    names = list(d['r31b_single'])
    print('| unit | ' + ' | '.join(names) + ' |'); print('|' + '---|' * (len(names) + 1))
    for unit in ('plant', 'leaf_1', 'leaf_7'):
        cells = []
        for n in names:
            x = d['r31b_single'][n][unit]['absorbed_with_extension_ru_m2']
            cells.append(f"{x['rel_delta']:+.3f} ± {x['sigma'] / x['reference']:.3f}{'*' if x['resolved'] else ''}")
        print(f'| {unit} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    globals()[sys.argv[1]]()
