"""Korean per-individual notes, rolling status and the summary table from the kernel's small JSON outputs (no arrays are loaded locally).
python make_reports.py [--status-only <text>]"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
OUT = HERE / 'kaggle_out'; H = REPO / 'handoff'
ORDER = ['t2_02', 'n_04', 'n_01', 't2_05', 'n_06', 'n_05', 't2_04', 'n_02', 'n_03']
RIM = json.load(open(HERE / 'kaggle_data/rim.json', encoding='utf-8'))
TAGS = '`relative_light` · `conditional_on_unvalidated_scale` · `biological_validation = not_established` · `light_model=v1 d_ex=2.4 mm` · 조명 바 위치 가정 · 광학 가정(v1 검은 잎)'
PLATFORM = 'Kaggle CPU 커널 dolgfish/static-nine-trace-20261028 (버전 1)'


def scale_status(pid):
    return 'rim_fit_ok' if RIM[pid]['rim_visual'] == 'pass' else 'rim_fit_failed'


def load(pid):
    p = OUT / pid / 'plant_light.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else None


def f(x, n=3):
    return f'{x:.{n}f}'


def note(pid, d):
    pl = d['plant_level']; m = pl['2.4']; rng = [pl[k]['relative_light_per_area']['mean'] for k in ('1.5', '2', '3', '5')]
    lines = [f'# 정적 광 — {pid} (2026-10-28)', TAGS,
             f"label_source {d['label_source']} · scale_status {scale_status(pid)} (림 {RIM[pid]['rim_ratio_vs_59']}) · 도달 {d['tier_reached']}단계 · 면적 (b) 있음(P1 실루엣 보정, 합계 {d['plant_area_b_silhouette_mm2']:.0f} mm²)",
             '| 식물 평균 상대 광 | 자기 그늘 | 미배정 광 몫 | 흡수광 합 r.u.·mm² | 높이 p99.5 mm | 측정 범위 밖 면적 | 두 시드 CV |', '|---|---|---|---|---|---|---|',
             f"| {f(m['relative_light_per_area']['mean'])} (1.5–5 mm: {f(min(rng))}–{f(max(rng))}) | {f(m['self_shading']['mean'])} | {f(d['unassigned_light_share'])} | {m['absorbed']['mean']:.0f} | "
             f"{d['height_above_base_mm']['p995']:.0f} | {f(d['out_of_field_support_area_fraction'], 2)} | {100 * d['cv_two_seed_plant_absorbed_dex']:.2f} % |"]
    g = d.get('groups')
    if g:
        t = g['top10'][0]
        lines.append(f"소엽 묶음 {g['count']}개 중 {g['flagged_groups']}개가 검사 표시 보유(빛의 {100 * g['flagged_share_of_absorbed']:.0f} %). 빛 상위 10개 묶음 = 식물 빛의 {100 * g['top10_share_of_plant']:.0f} %, 1위 묶음 {t['group']} ({100 * t['share_of_plant']:.1f} %, 표시 {','.join(t['flags']) or '없음'}).")
    c = d.get('user_click_leaf1_lower_bound')
    if c:
        lines.append(f"사용자 클릭 잎 1(묶음 {len(c['groups'])}개): 식물 빛의 {100 * c['share_of_plant']:.1f} % 이상(하한, 클릭한 묶음만), 면적당 빛 {f(c['light_per_area'])}.")
    if scale_status(pid) == 'rim_fit_failed':
        lines.append('림 맞춤 실패: 절대 높이는 신뢰 불가, 상대 광은 사용 가능.')
    lines.append(f'한계: 미배정 광 몫은 줄기를 뺀 하한(P1 이 줄기를 분류하지 않음), 생물학 검증 없음, 라벨은 소엽 묶음 수준(잎 번호 아님), R8_pc 없음(1단계 대기) → 잎별 결과 없음. {PLATFORM}.')
    assert len(lines) <= 10, (pid, len(lines))
    (H / f'STATIC_NINE_{pid}_2026-10-28_KO.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def status(extra=''):
    L = ['# 정적 광 9개체 — 진행 상태 (자동 갱신)', '', f'플랫폼: {PLATFORM}. 로컬에서는 추적·집계를 하지 않음. 마감(동결) 2026-10-30 11:00, 이후는 not_run.', '',
         '| 개체 | 상태 | 이유·메모 |', '|---|---|---|']
    done = []
    blocked = (OUT / 'external_blocked.json')
    for pid in ['R3-1b'] + ORDER:
        d = load(pid)
        if blocked.exists() and pid != 'R3-1b':
            L.append(f'| {pid} | external_blocked | {json.load(open(blocked, encoding="utf-8")).get("external_blocked")} |')
        elif d:
            done.append(pid); L.append(f"| {pid} | done | {'재현 시험(10번째 값)' if pid == 'R3-1b' else '1단계(R8_pc) 대기라 잎별 결과 없음, 소엽 묶음까지'} |")
        else:
            L.append(f"| {pid} | {'running' if extra == 'running' else 'waiting'} | {'커널 실행 중' if extra == 'running' else 'R3-1b 재현 게이트 뒤 같은 커널에서 실행'} |")
    (H / 'STATIC_NINE_STATUS_KO.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def summary():
    ds = {pid: load(pid) for pid in ['R3-1b'] + ORDER}
    gate = json.load(open(OUT / 'R3-1b/reproduction_gate.json', encoding='utf-8'))
    S = json.load(open(OUT / 'SUMMARY.json', encoding='utf-8'))
    L = ['# 정적 광 — 나머지 아홉 개체 (2026-10-28)', TAGS, '',
         '| 개체 | label_source | scale_status | 식물 평균 상대 광 | 자기 그늘 | 미배정 광 몫 | 줄기(P1 분류) 광 몫 | 도달 단계 | 면적 (b) |', '|---|---|---|---|---|---|---|---|---|']
    for pid, d in ds.items():
        m = d['plant_level']['2.4']
        L.append(f"| {pid}{' (재현 시험)' if pid == 'R3-1b' else ''} | {d['label_source']} | {'rim_fit_ok' if pid == 'R3-1b' else scale_status(pid)} | {f(m['relative_light_per_area']['mean'])} | "
                 f"{f(m['self_shading']['mean'])} | {f(d['unassigned_light_share'])}{' (줄기·미배정 모두 포함)' if pid == 'R3-1b' else ''} | {f(d['p1_organ_split'].get('stem_surface', {}).get('share_of_absorbed', 0))} | {d['tier_reached']} | 있음 (P1 실루엣, {d['plant_area_b_silhouette_mm2']:.0f} mm²) |")
    L += ['', '열 정의: 식물 평균 상대 광 = 흡수광 합 ÷ 면적 (b) 합(U·K 면적 가중 평균, 줄기 포함 식물 전체). 자기 그늘 = 1 − 흡수광 ÷ 식물 없는 때의 흡수광(U 가중, 식물 전체; 이전 R3-1b 잎 전용 값과 정의가 달라 직접 비교 금지). '
          '미배정 광 몫: 아홉 개체 = 축 모양(axis_like) 서펠 + 묶음 없는 날 서펠의 빛(줄기를 가리키는 P1 분류 몫은 따로 옆 열), R3-1b 는 V-P 정리 라벨의 잎 아님 전체(줄기 포함)라 아홉 개체와 한 줄로 비교할 수 없음.',
          '', '## 기술 기록 (모든 보고 공통)',
          f"- 플랫폼 {PLATFORM}, 엔진 해시(정적 패키지 5개 파일 sha256) {json.dumps({k: v[:12] for k, v in S['engine_hash'].items()})}, 시드 101/202, 실행 시간 {S.get('wall_seconds', '?')} 초.",
          '- 추적기는 해석식이라 표본 광선이 없음(수신점마다 바 표본점 24개 + 돔 방향 72개). 시드 두 개 사이 식물 흡수광 CV: ' + ', '.join(f"{p} {100 * d['cv_two_seed_plant_absorbed_dex']:.2f} %" for p, d in ds.items()) + '. 5 %를 넘은 개체: ' + (', '.join(p for p, d in ds.items() if d['cv_two_seed_plant_absorbed_dex'] > 0.05) or '없음') + '.',
          f"- 재현 게이트(R3-1b, 로컬 결과와): 식물 합 {100 * gate['plant_total_d5_cube_vpcleanup']['rel_diff']:+.4f} %, 평균 상대 광 d 2.4 mm {100 * gate['mean_relative_light']['rel_diff']:+.4f} %, d 5 mm {100 * gate['mean_relative_light_d5mm']['rel_diff']:+.4f} % → {'통과(≤ 1 %)' if gate['pass'] else '불합격'}.",
          '', '## 묘사만 하는 비교 (순위·우열 판단 아님)']
    for pid, d in ds.items():
        h = d['height_above_base_mm']['p995']; m = d['plant_level']['2.4']
        L.append(f"- {pid}: 표면 서펠 {d['surfels']:,}, 높이 p99.5 {h:.0f} mm, 자기 그늘 {f(m['self_shading']['mean'])}, 평균 상대 광 {f(m['relative_light_per_area']['mean'])}, 측정 범위 밖 면적 {f(d['out_of_field_support_area_fraction'], 2)}")
    L += ['', '## 한계',
          '- 생물학 검증 없음. 조명 바 위치 가정, 광학 가정(v1 검은 잎; 요청서의 a 0.85·τ 0.07·ρ 0.08 은 v2 값이라 R3-1b 실행과 같게 적용하지 않음), 면적 (b)와 축척은 conditional_on_unvalidated_scale.',
          '- **중요(미배정 몫)**: 아홉 개체는 P1 이 줄기를 따로 분류하지 않았고(stem_surface 0 서펠) 3D-first 묶음의 axis_like 서펠도 매우 적습니다. 줄기 서펠 대부분이 "날 묶음"에 들어 있어, 위 미배정 몫(1.6–4 %)은 상한이 아니라 줄기를 뺀 하한입니다. R3-1b 는 같은 몫이 22 %(줄기)입니다. 계약서의 "라벨이 더 옮길 수 있는 상한" 표현은 아홉 개체에는 성립하지 않으며, 줄기 분류가 생기면 새 추적 없이 per_surfel_light.npz 로 다시 계산합니다.',
          '- 라벨은 R3-1b 단계 1(V-P 파생)과 독립이 아니며, 이 아홉 개체에는 R8_pc 가 아직 없음(STAGE1 waiting_stage1). 3D-first 묶음은 소엽 수준이며 거의 모두 검사 표시를 달고 있음 → 잎별 결과 없음.',
          '- 림 맞춤 실패(n_01, n_04, n_05, t2_04): 절대 높이 신뢰 불가, 상대 광은 사용 가능. length_scale_checked 는 척도 점검 보고서가 없어 부여하지 않음.',
          '- 높이 z 150 mm 위(측정 광장 밖)는 모델 확장(U_clamped × 모델 비)이며 개체별 면적 비율을 위 표 옆 개체 노트에 적음.',
          '- 트레이 이웃 효과는 이 작업의 범위가 아님(계획된 6+2 장면, B5, 트레이 배치 필요).',
          '- Tier 2(R8_pc 잎별 재집계)는 R8_pc 가 생기면 per_surfel_light.npz 로 외부 소형 커널에서 수행(새 추적 없음).']
    (H / 'STATIC_NINE_2026-10-28_KO.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--status-only':
        status(sys.argv[2] if len(sys.argv) > 2 else ''); sys.exit(0)
    for pid in ORDER:
        d = load(pid)
        if d:
            note(pid, d)
    status(); summary()
    print('reports written')
