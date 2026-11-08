"""microtom_parameters_v2.csv = tools/microtom_parameters.csv (v1 rows, unchanged, v2_status 'v1_unchanged' or 'replaced_in_v2') + the v2 rows with a
source column (hand_measured / skeleton_traits_ungraded / user_stated / user_set_range / literature_range / fitted_to_reconstruction_area / user_assumed).
python -X utf8 make_params_v2.py"""
import csv, json
from pathlib import Path
import v2_patch as P
H = Path(__file__).resolve().parent; v1 = list(csv.DictReader(open(H.parents[1] / 'tools' / 'microtom_parameters.csv', encoding='utf-8')))
REPLACED = {'PAIRS', 'RP_PAIRS', 'COMPOUND_INSERTION_DEG', 'RP_COMPOUND_INSERTION_DEG', 'TOTAL_BEND_DEG', 'RP_TOTAL_BEND_DEG', 'INTERNODE_CAP_M', 'RP_INTERNODE_CAP_M',
            'FIRST_LEAVES', 'CADENCE', 'FLOWERS', 'SLA_M2_G'}
v2 = [
    ('RP_LEAFLET_COUNT', json.dumps({str(k): v for k, v in P.LEAFLET_MAIN.items()}) + ' ; rank>9 ' + json.dumps(P.LEAFLET_FALLBACK) + ' ; side ' + json.dumps(P.LEAFLET_SIDE), 'count', 'hand_measured', 'C1; n=6 -> left leaflet of the distal pair omitted (user_assumed stand-in)'),
    ('RP_THETA0_DEG', json.dumps({str(k): v for k, v in P.THETA_MAIN.items()}) + f' ; rank>9 {P.THETA_FALLBACK} ; side {P.THETA_SIDE} ; clip {P.THETA_CLIP}', 'deg', 'hand_measured', 'C2; insertion = 90 - theta0'),
    ('RP_TOTAL_BEND_DEG', f'normal {P.BEND} clip {P.BEND_CLIP}', 'deg', 'skeleton_traits_ungraded', 'C3; replaces U(0, 40)'),
    ('RP_INTERNODE_MM', f"scaled Beta a {P.BETA['a']:.4f} b {P.BETA['b']:.4f} on [2, 4] mm (median 3.75)", 'mm', 'user_set_range', 'C4; per internode, exact ppf in the engine'),
    ('INTERNODE_SCALE_MAIN_RANK_GE_5', '1.25', 'factor', 'user_stated', 'C4U/C4V 2026-11-05 (main axis only)'),
    ('LOWEST_LEAF_SCALE', '0.7', 'factor on rank-1 main leaflet max mass', 'user_stated', 'C4V 2026-11-05 (C4U 0.5 kept on record)'),
    ('RP_DIVERGENCE_DEG', '119.3-154.7 uniform x handedness', 'deg', 'literature_range', 'unchanged from v1'),
    ('SIDE_AXES_PER_PLANT', '2', 'count', 'hand_measured', 'C4; new controllers S1, S2'),
    ('RP_SIDE_ATTACH_RANK', json.dumps(P.SIDE_ATTACH), 'main rank', 'hand_measured', 'C4; without replacement'),
    ('SIDE_AXIS_ELEVATION_DEG', str(P.SIDE_ELEV_DEG), 'deg above horizontal', 'hand_measured', 'C4; 2 values (sparse)'),
    ('SIDE_BUD_DELAY_DD', str(P.BUD_DELAY_DD), 'degree_day', 'user_assumed', 'C4; registry BUD_BREAK_DELAY_DD, eligibility after_parent_axis_end'),
    ('RP_MAIN_TRUSS_NODE', json.dumps(P.MAIN_TRUSS_NODE), 'node', 'user_stated', 'C4; template v2.1/v2.2'),
    ('RP_MAIN_TOTAL_LEAVES', json.dumps(P.MAIN_TOTAL), 'count', 'user_stated', 'C4; template v2.2 end rule, two terminal trusses'),
    ('SIDE_FIRST_TRUSS_NODE', str(P.SIDE_FIRST_TRUSS), 'node', 'user_stated', 'C4'),
    ('RP_SIDE_TOTAL_LEAVES', json.dumps(P.SIDE_TOTAL), 'count', 'user_stated', 'C4; template v2.2 end rule'),
    ('RP_FLOWERS_PER_TRUSS', json.dumps(P.FLOWERS), 'count', 'hand_measured', 'C4; replaces FLOWERS 4'),
    ('SLA_M2_G', str(P.SLA_BASE * 1.15), 'm2/gDM', 'fitted_to_reconstruction_area', 'C5; bound f = 1.15 of the random range; targets need f 1.57-1.83 (extrapolated, C5_FIT.json); SLA not measured'),
]
with open(H / 'microtom_parameters_v2.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['key', 'value', 'unit', 'source', 'note', 'model'])
    for k, v, u, s, n in v2: w.writerow([k, v, u, s, n, 'v2'])
    for r in v1: w.writerow([r['key'], r['value'], r['unit'], r['source_kind'], ('replaced_in_v2; ' if r['key'] in REPLACED else 'v1_unchanged; ') + r['note'], 'v1'])
(H / 'START_GUI_V2.bat').write_text('''@echo off
rem Open the frozen Engine v2 model (build/engine_v2_calib_20261104_01/work/Model_v2.rgg) in the GroIMP GUI. The v1 engine is untouched.
rem Working directory = work\\gui_v2 (Model.rgg = copy of Model_v2.rgg, leaflet_outline_uv.csv next to it; runs write their csv files here).
chcp 65001 >nul
cd /d "%~dp0work"
if not exist gui_v2 mkdir gui_v2
copy /y Model_v2.rgg gui_v2\\Model.rgg >nul
copy /y leaflet_outline_uv.csv gui_v2\\leaflet_outline_uv.csv >nul
cd gui_v2
set "JDK_JAVA_OPTIONS="
set "JAVA=C:\\Program Files\\Eclipse Adoptium\\jre-21.0.12.101-hotspot\\bin\\java.exe"
if not exist "%JAVA%" set "JAVA=java"
set "CORE=C:\\Program Files\\GroIMP 2.2.1\\core.jar"
echo.
echo  Engine v2 (동결) 모델을 GroIMP 로 엽니다. 원본 v1 은 그대로입니다.
echo  작업 폴더: %CD%
echo  GroIMP 가 열리면: File - Open - 이 폴더의 Model.rgg, 그다음 메서드 simulate 실행 (72일, 한 개체, 점광원)
echo  메모리 1 GB 이하로 실행합니다. 이 PC 는 여유 RAM 이 작아 긴 실행은 Kaggle 을 권합니다.
echo.
"%JAVA%" -Xmx1g -XX:ActiveProcessorCount=1 -Dfile.encoding=UTF-8 -jar "%CORE%"
''', encoding='utf-8')
print(len(v2), len(v1))
