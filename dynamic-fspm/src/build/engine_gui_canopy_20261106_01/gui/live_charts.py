"""Live chart window (fallback for the GroIMP chart panels; build/engine_gui_canopy_20261106_01 CONTRACT §3).
Reads the newest gui_runs/<timestamp>/daily.csv next to this file every 5 s and redraws six charts vs DAG:
leaf area, dry mass, height, trusses, fruits (canopy median + min-max band over plants), and canopy light % / mean neighbour shading % / index.
Labels: gui_scenario_tool, engine_v2_2, assumed_lamp_model, relative_light, not_validation. Read-only: it never writes into the run folder."""
import csv, math
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
H = Path(__file__).resolve().parent


def latest():
    runs = sorted((H / 'gui_runs').glob('*/daily.csv'), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def read(p):
    plants, canopy, settings = {}, {}, ''
    for line in open(p, encoding='utf-8'):
        if line.startswith('# settings'): settings = line[2:].strip()
    rows = [r for r in csv.reader(l for l in open(p, encoding='utf-8') if not l.startswith('#'))][1:]
    for r in rows:
        try: dag = float(r[1])
        except ValueError: continue
        if dag < 0: continue
        if r[2] == 'CANOPY':
            ex = dict(x.split('=', 1) for x in r[12:] if '=' in x)
            sh = float(r[11]) if r[11] not in ('', 'NaN') else math.nan
            canopy[dag] = dict(light=float(ex.get('light_pct', 'nan')), shade=100 * sh, index=float(ex['index']) if ex.get('index', '—') not in ('—', 'NaN') else math.nan)
        else:
            plants.setdefault(dag, []).append([float(x) for x in r[3:8]])
    return plants, canopy, settings


fig, axs = plt.subplots(2, 3, figsize=(15, 7.5)); plt.ion()
while True:
    p = latest()
    for ax in axs.flat: ax.clear()
    if p:
        plants, canopy, settings = read(p); days = sorted(plants)
        for k, (ax, lab) in enumerate(zip(axs.flat[:5], ['잎면적 (cm²)', '건물중 (g)', '키 (mm)', '화방 수', '열매 수'])):
            med = [sorted(v[k] for v in plants[d])[len(plants[d]) // 2] for d in days]
            lo = [min(v[k] for v in plants[d]) for d in days]; hi = [max(v[k] for v in plants[d]) for d in days]
            ax.fill_between(days, lo, hi, color='#2e7d32', alpha=.18); ax.plot(days, med, color='#2e7d32'); ax.set_title(lab); ax.set_xlabel('DAG')
        ax = axs.flat[5]; cd = sorted(canopy)
        ax.plot(cd, [canopy[d]['light'] for d in cd], label='캐노피 빛 (%)', color='#f9a825')
        ax.plot(cd, [canopy[d]['shade'] for d in cd], label='평균 이웃 그늘 (%)', color='#5d4037')
        ax2 = ax.twinx(); ax2.plot(cd, [canopy[d]['index'] for d in cd], label='지수 = 빛 ÷ 그늘', color='#1565c0', ls='--'); ax2.set_ylabel('지수 (그늘 < 0.5 %면 표시 안 함)')
        ax.set_title('캐노피 빛·이웃 그늘·지수'); ax.set_xlabel('DAG'); ax.legend(loc='upper left', fontsize=8)
        fig.suptitle(f'{p.parent.name} — {settings}\n시나리오 도구 (검증 아님, 상대 빛)', fontsize=9)
    else:
        fig.suptitle('아직 실행 결과가 없습니다 (gui_runs/ 비어 있음). GroIMP 에서 simulate 를 누르세요.')
    fig.tight_layout(); plt.pause(5)
    if not plt.fignum_exists(fig.number): break
