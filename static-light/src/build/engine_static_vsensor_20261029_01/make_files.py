"""VSENSOR tray 1: CSV from the user's xlsx (copied unchanged to handoff/), measured shading ratios vs the stated unshaded 282, flags.
No model value is computed here (Step 3 is blocked on user inputs). python make_files.py"""
import csv, json
from pathlib import Path
import openpyxl
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
X = REPO / 'handoff/Vsensor20261029110933.xlsx'
IND = ['e_01', 'e_02', 'e_03', 'e_04', 'e_05', 'e_06']; FACES = ['yellow', 'green', 'red', 'blue']   # user-confirmed order
REF = 282.0; LOW = 10.0
ws = openpyxl.load_workbook(X, data_only=True).active
vals = [[float(c) for c in r] for r in ws.iter_rows(values_only=True)]
assert len(vals) == 6 and all(len(r) == 4 for r in vals)
rows = []
for i, ind in enumerate(IND):
    for j, fc in enumerate(FACES):
        rows.append(dict(individual=ind, face_color=fc, ppfd_measured=vals[i][j]))
with (REPO / 'handoff/VSENSOR_TRAY1_2026-10-29.csv').open('w', encoding='utf-8', newline='') as fh:
    fh.write('# source: handoff/Vsensor20261029110933.xlsx (copied unchanged; cells stored as text; no header row in the sheet), measured 2026-10-29 ~11:09\n')
    fh.write('# rows e_01..e_06, columns yellow, green, red, blue (order user-confirmed); unit_unconfirmed (assumed umol m-2 s-1)\n')
    fh.write('# ppfd_unshaded_ref = 282 (same unit; user statement; position, height, one point or tray mean, plants present or removed: unknown)\n')
    fh.write('# validation_comparison: no lamp, optics, scale or label parameter may be changed after seeing these values\n')
    w = csv.DictWriter(fh, fieldnames=['individual', 'face_color', 'ppfd_measured', 'repeat_no', 'height_mm', 'sensor_normal', 'sensor_xyz_note', 'condition', 'note'])
    w.writeheader()
    for r in rows:
        w.writerow(dict(r, ppfd_measured=f"{r['ppfd_measured']:.3f}"))
tab = []
for r in rows:
    rm = r['ppfd_measured'] / REF
    tab.append(dict(r, r_meas=rm, low_signal=r['ppfd_measured'] < LOW, above_ref=rm > 1.0))
json.dump(dict(ref=REF, low_signal_threshold=LOW, rows=tab, above_ref=[(t['individual'], t['face_color'], round(t['r_meas'], 3)) for t in tab if t['above_ref']],
               low_signal=[(t['individual'], t['face_color'], t['ppfd_measured']) for t in tab if t['low_signal']]), open(HERE / 'measured_table.json', 'w'), indent=1)
print('above 282:', [(t['individual'], t['face_color'], round(t['r_meas'], 3)) for t in tab if t['above_ref']])
print('low_signal:', [(t['individual'], t['face_color']) for t in tab if t['low_signal']])
