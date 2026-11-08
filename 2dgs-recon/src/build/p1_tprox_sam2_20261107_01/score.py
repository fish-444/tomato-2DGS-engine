"""CONTRACT.md section 3: tuning-half choice per arm, then ONE report-half score per arm (the chosen setting only), paired bootstrap vs T0.
usage (project root): python -X utf8 build/p1_tprox_sam2_20261107_01/score.py <kernel output dir>"""
from pathlib import Path
import csv, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; D = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from tomato3d.organs import oversegment as OS, skeleton_io as SIO, skeleton_r31b_eval as EV
TUNE, REP = (1, 3, 6, 8), (2, 4, 5, 7); T0_FROZEN = 0.7503


def per_proj(z, Bs, si):
    pos = {int(r): i for i, r in enumerate(z["rows"])}; brow = Bs["pl"]["rows"][Bs["b"]]; lab_b = np.array([z["label"][pos[int(r)]] for r in brow]); return lab_b[si]


def acc(got, hl, si, leaves):
    m = np.isin(hl, leaves); n = int(m.sum()); k = int((got[m] == hl[m]).sum()); c = int((got[m] == 0).sum())
    return dict(n=n, correct=k, accuracy=round(k / n, 4), wilson95=[round(x, 4) for x in SIO.wilson(k, n)], contested_share=round(c / n, 4))


def boot_diff(g1, g0, hl, si, leaves, B=2000):
    m = np.isin(hl, leaves); s = si[m]; d = (g1[m] == hl[m]).astype(float) - (g0[m] == hl[m]).astype(float); us, inv = np.unique(s, return_inverse=True)
    num = np.bincount(inv, weights=d); cnt = np.bincount(inv); rng = np.random.default_rng(0); out = []
    for _ in range(B):
        pick = rng.integers(0, len(us), len(us)); out.append(num[pick].sum() / cnt[pick].sum())
    return dict(diff=round(float(d.mean()), 4), ci95=[round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)])


def main(kout):
    kout = Path(kout); Bs = OS.load_base(); si, hl = EV.human_projections(Bs); res = json.load(open(kout / "results.json"))
    tags = sorted(p.stem[len("labels_"):] for p in kout.glob("labels_*.npz")); got = {t: per_proj(np.load(kout / f"labels_{t}.npz"), Bs, si) for t in tags}
    t0 = acc(got["T0"], hl, si, REP); assert abs(t0["accuracy"] - T0_FROZEN) <= 0.002, ("T0 does not reproduce", t0)
    grid = {t: acc(got[t], hl, si, TUNE) for t in tags}; arms = dict(A=[t for t in tags if t.startswith("A_")], B=[t for t in tags if t.startswith("B_")], AB=[t for t in tags if t.startswith("AB_")])
    out = dict(T0=dict(tag="T0", tuning=grid["T0"], report=t0)); rows = []
    for arm, ts in arms.items():
        best = max(grid[t]["accuracy"] for t in ts); tied = [t for t in ts if grid[t]["accuracy"] >= best - 0.002]; ch = min(tied, key=lambda t: grid[t]["contested_share"])
        rep = acc(got[ch], hl, si, REP); bd = boot_diff(got[ch], got["T0"], hl, si, REP); improved = bool(rep["accuracy"] > t0["accuracy"] and bd["ci95"][0] > 0)
        out[arm] = dict(tag=ch, tuning=grid[ch], report=rep, paired_diff_vs_T0=bd, improved=improved, run=res["runs"].get(ch))
    fair = dict(T0="frozen", A="prompt_independent", B="prompt_from_scoring_labels", AB="prompt_from_scoring_labels")
    for arm, o in out.items():
        r_ = res["runs"].get(o["tag"], {}); rows.append(dict(arm=arm, setting=o["tag"], fairness=fair[arm], half="report", accuracy=o["report"]["accuracy"], ci95=o["report"]["wilson95"], contested_share=o["report"]["contested_share"],
                                                           tuning_accuracy=o["tuning"]["accuracy"], coverage_grouped_nodes=r_.get("grouped_nodes"), added_seeds=r_.get("added_seeds"), paired_diff=o.get("paired_diff_vs_T0"), improved=o.get("improved")))
    with open(D / "RESULT_P1_TPROX_SAM2.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(D / "tuning_grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["setting", "tuning_accuracy", "tuning_contested", "added_seeds", "grouped_nodes"])
        for t in tags: w.writerow([t, grid[t]["accuracy"], grid[t]["contested_share"], res["runs"].get(t, {}).get("added_seeds"), res["runs"].get(t, {}).get("grouped_nodes")])
    json.dump(dict(fairness=fair, T0_match_frozen_r8pc=res.get("T0_match_frozen_r8pc"), arms=out), open(D / "score.json", "w"), indent=1); print(json.dumps({k: dict(tag=v["tag"], tune=v["tuning"]["accuracy"], rep=v["report"]["accuracy"], ci=v["report"]["wilson95"], cont=v["report"]["contested_share"], diff=v.get("paired_diff_vs_T0"), improved=v.get("improved")) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
