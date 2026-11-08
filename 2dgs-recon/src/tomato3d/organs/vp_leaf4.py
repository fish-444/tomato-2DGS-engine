"""Revision R7: SAM 2 video propagation retried on crops around leaf 4 (same predictor, rings, seeds and seed maps as the original V-P; each frame = the leaf-4
region crop resized to 1024 px), new maps pasted inside the crop only; then the V-P 3D vote and the R6 chain rerun on the new maps.
Contract: build/p1_local_diffusion_20261027_01/contested_resolve/CONTRACT.md, Revision R7.

Revision R7b: leaf-4-only paste (old maps + pixels the crop calls leaf 4), same evaluation.

usage: python -m tomato3d.organs.vp_leaf4 --step propagate | evaluate | paste4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import local_diffusion as LD
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2

OUT = LD.OUT / "vp_leaf4"; R6 = LD.OUT / "r6_chain"; STATUS = "R3-1b_unaccepted"
R7 = dict(region_mm=25.0, margin=0.20, res=1024, cap_minutes=120, leaf=4)


def region_boxes(pl, frames):
    """Square box (H2 grid) per frame around the projection of R6's leaf-4 surfels + blade surfels within 25 mm."""
    z = np.load(R6 / "labels.npz"); lab = z["label"]; b = np.nonzero(pl["blade"])[0]; xyz = pl["xyz"]
    l4 = np.nonzero(lab == R7["leaf"])[0]; d, _ = cKDTree(xyz[l4]).query(xyz[b]); reg = np.union1d(l4, b[d <= R7["region_mm"] / 1000]); boxes = {}
    for v in frames:
        pv = V2.project_view(xyz[:1], pl["sc"][:1], pl["po"], v, V2.VISP["res"]); u, w, zc = V2.project_points(xyz[reg], pv); s = UL.H2 / V2.VISP["res"]; ok = zc > 0
        x, y = u[ok] * s, w[ok] * s; x0, x1, y0, y1 = np.percentile(x, 1), np.percentile(x, 99), np.percentile(y, 1), np.percentile(y, 99)
        side = int(min(UL.H2, max(x1 - x0, y1 - y0) * (1 + 2 * R7["margin"]))); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bx = int(np.clip(cx - side / 2, 0, UL.H2 - side)); by = int(np.clip(cy - side / 2, 0, UL.H2 - side)); boxes[v] = (bx, by, side)
    return boxes, int(len(reg))


def propagate():
    from tomato3d.leafseg import run as r2
    import torch
    cfg = json.load(open(r2.SAMCFG))
    for k in ("runtime", "repository"):
        if cfg.get(k): sys.path.insert(0, str(cfg[k]))
    from sam2.build_sam import build_sam2_video_predictor
    torch.set_num_threads(int(cfg.get("threads", 4))); t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    vp = build_sam2_video_predictor(cfg["model_config"], str(cfg["checkpoint"]), device="cpu")
    pl = UL.load_plant(); views = pl["views"]; H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]
    nmap = json.load(open(UL.OUT / "labels" / "number_map_v3_charuco.json", encoding="utf-8")); nmap = nmap["map"] if "map" in nmap else nmap["latest"]["map"]
    raw, _ = UL.label_maps(UL.OUT, chosen + [held])
    for v in chosen: UL.STATE_PPM[v] = UL.plant_px_per_mm(pl, AP.view_file(v), H2id[:, views.index(AP.view_file(v))])[0]
    topo_rows, _ = UL.topology_report(pl, raw, nmap, chosen + [held], H2id)
    seed_maps, _, _ = UL.build_vote_maps(pl, raw, nmap, chosen, [r for r in topo_rows if r["photo"] in chosen], "V-H+fill+topo")
    rings, elaz = UL.rings_of(pl, views, {held}); boxes, nreg = region_boxes(pl, [v for rg in rings for v in rg]); new = {}; log = []; res = R7["res"]
    json.dump({UL.vid_of(v): list(map(int, bx)) for v, bx in boxes.items()}, open(OUT / "crop_boxes.json", "w"), indent=0)
    with torch.inference_mode():
        for rg in rings:
            seeds = [AP.view_file(v) for v in chosen if AP.view_file(v) in rg]
            if not seeds: log.append(dict(ring_elev=round(elaz[rg[0]][0], 1), frames=len(rg), note="no seed")); continue
            for s, order in UL.segments(rg, seeds).items():
                k0 = order.index(rg.index(s)); frames = [rg[i] for i in order]; tmp = Path_tmp = tempfile.mkdtemp(prefix="vp4_")
                for i, v in enumerate(frames):
                    bx, by, side = boxes[v]; img = Image.open(UL.PHOTOS / v).convert("RGB"); f = img.size[0] / UL.H2
                    img.crop((int(bx * f), int(by * f), int((bx + side) * f), int((by + side) * f))).resize((res, res), Image.BILINEAR).save(f"{tmp}/{i:05d}.jpg", quality=95)
                sm = seed_maps[UL.vid_of(s)]; bx, by, side = boxes[s]; smc = sm[by:by + side, bx:bx + side]; leaves = sorted(set(smc[smc > 0].tolist()))
                if R7["leaf"] not in leaves: log.append(dict(seed=UL.vid_of(s), frames=len(frames), note="leaf 4 not in seed crop", leaves=leaves))
                st = vp.init_state(video_path=tmp)
                for L in leaves:
                    vp.add_new_mask(st, frame_idx=k0, obj_id=int(L), mask=np.asarray(Image.fromarray((smc == L).astype(np.uint8)).resize((res, res), Image.NEAREST)).astype(bool))
                logits = {}
                for rev in (False, True):
                    for fi, ids, lg in vp.propagate_in_video(st, start_frame_idx=k0, reverse=rev): logits[fi] = (list(ids), lg[:, 0].float().cpu().numpy())
                vp.reset_state(st); shutil.rmtree(tmp, ignore_errors=True)
                for fi, (ids, lg) in logits.items():
                    v = frames[fi]
                    if v == s: continue
                    best = lg.argmax(0); pos = lg.max(0) > 0; m = np.where(pos, np.array(ids)[best], 0).astype(np.int16); bx, by, side = boxes[v]
                    new[UL.vid_of(v)] = (bx, by, side, np.asarray(Image.fromarray(m).resize((side, side), Image.NEAREST)).astype(np.int16))
                log.append(dict(seed=UL.vid_of(s), frames=len(frames), leaves=leaves, seconds=round(time.time() - t0))); print("R7 V-P", log[-1], flush=True)
                if time.time() - t0 > R7["cap_minutes"] * 60: log.append(dict(stopped="cap reached")); break
    old = np.load(UL.OUT / "map_v3_charuco" / "user_labelled_propagated" / "propagated_maps.npz"); maps = {}
    for k in old.files:
        m = old[k].copy(); vid = k[1:]
        if vid in new: bx, by, side, crop = new[vid]; m[by:by + side, bx:bx + side] = crop
        maps[k] = m
    np.savez_compressed(OUT / "propagated_maps_leaf4crop.npz", **maps)
    json.dump(dict(status=STATUS, params=R7, region_surfels=nreg, photos_replaced=sorted(new), log=log, seconds=round(time.time() - t0)), open(OUT / "propagate.json", "w"), indent=1)
    print("done", len(new), "photos", round(time.time() - t0), "s", flush=True)


def paste4():
    """R7b maps: the original propagated maps, with every pixel the R7 crop map calls leaf 4 set to 4 (additive only)."""
    old = np.load(UL.OUT / "map_v3_charuco" / "user_labelled_propagated" / "propagated_maps.npz"); crop = np.load(OUT / "propagated_maps_leaf4crop.npz"); out = OUT / "leaf4_only"; out.mkdir(parents=True, exist_ok=True); maps = {}; n4 = {}
    for k in old.files:
        m = old[k].copy(); add = (crop[k] == R7["leaf"]) & (m != R7["leaf"]); m[add] = R7["leaf"]; maps[k] = m; n4[k[1:]] = int(add.sum())
    np.savez_compressed(out / "propagated_maps_leaf4only.npz", **maps); json.dump(dict(status=STATUS, rule="old map + crop leaf-4 pixels (additive)", pixels_added=n4, total=int(sum(n4.values()))), open(out / "paste4.json", "w"), indent=1)
    print("R7b paste", sum(n4.values()), "px over", sum(1 for v in n4.values() if v), "photos", flush=True)


def evaluate(maps_file=None, out=None, grouping="compound_chain_R7_vp_leaf4_crop"):
    from tomato3d.organs import chain_r6 as C6, oversegment_second as O2, sheet_retry as SR, trace_reconcile as TR
    maps_file = maps_file or OUT / "propagated_maps_leaf4crop.npz"; out = out or OUT
    t0 = time.time(); B = LD.load("vp"); pl, b, xyz, S = B["pl"], B["b"], B["xyz"], B["S"]; H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; B["chosen"] = chosen; B["held"] = held
    nmap = json.load(open(TR.MAP_FILE, encoding="utf-8"))["map"]; raw, _ = UL.label_maps(UL.OUT, chosen + [held])
    for v in chosen: UL.STATE_PPM[v] = UL.plant_px_per_mm(pl, AP.view_file(v), H2id[:, pl["views"].index(AP.view_file(v))])[0]
    topo_rows, _ = UL.topology_report(pl, raw, nmap, chosen + [held], H2id)
    seed_maps, _, _ = UL.build_vote_maps(pl, raw, nmap, chosen, [r for r in topo_rows if r["photo"] in chosen], "V-H+fill+topo")
    old = np.load(TR.VP_DIR / "propagated_maps.npz"); newz = np.load(maps_file)
    def vote(prop):
        allm = dict(seed_maps, **{k[1:]: prop[k] for k in prop.files}); vids = chosen + sorted(k[1:] for k in prop.files); lab, _ = UL.vote_views(pl, allm, vids, None, H2id); return np.where(lab > 0, lab, 0)
    base_old = vote(old); vp_file = np.where(B["base"] > 0, B["base"], 0); repro = bool(np.array_equal(base_old, vp_file)); print("V-P vote reproduced:", repro, flush=True)
    base_new = vote(newz)
    truth = UL.remap({held: raw[held]}, nmap)[held]; truth = np.where(truth > 0, truth, 0).astype(np.int16); hv = AP.view_file(held); col = H2id[:, pl["views"].index(hv)]
    def score_full(f):
        ren = UL.render_ids(pl, f.astype(np.int16), hv, col); rws, g = UL.score_heldout(truth, ren); cov = float(((ren > 0) & (truth > 0)).sum() / max((truth > 0).sum(), 1))
        return dict(rows={str(k): {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in rws.items()}, summary=g, coverage=round(cov, 4))
    s_old, s_new = score_full(base_old), score_full(base_new)
    # R6 chain on the new base and maps
    B["base"] = base_new.astype(np.int16); base_b = B["base"][b]
    G, Gp, _ = LD.sheet_graph(xyz, S); seeds, _, _ = LD.seeds_and_reach(B, Gp); leaves, C = LD.costs(G, seeds)
    dense_all = np.vstack([LD.CC.densify(t["pts"]) for t in B["traces"] if t["leaf"] > 0]); dnear, _ = cKDTree(np.vstack([xyz[base_b > 0], dense_all])).query(xyz)
    cand = (base_b == 0) & (B["axis"][b] == 0) & (dnear <= LD.D["candidate_mm"] / 1000)
    vis_views = {vid: v[0] for vid, v in O2.vote_views(B).items()}; vpmaps = {k[1:]: newz[k] for k in newz.files if k[1:] != held and k[1:] not in chosen}
    steps, tg, log = C6.run_chain(B, G, Gp, leaves, C, cand, C6.U_MM, vis_views, vpmaps)
    um = UL.remap({v: raw[v] for v in chosen}, nmap); um = {v: np.where(um[v] > 0, um[v], 0) for v in chosen}; prj = SR.projections(pl, b, chosen)
    def purity(lb, mask, per_leaf=False):
        hit = tot = 0; pl_ = Counter(); pn = Counter()
        for vid, p in prj.items():
            ul = um[vid][p["wi"], p["ui"]]; m = B["VISb"][:, p["k"]] & p["inside"] & mask & (ul > 0); eq = lb[m] == ul[m]; hit += int(eq.sum()); tot += int(m.sum())
            for L, e in zip(lb[m], eq): pl_[int(L)] += int(e); pn[int(L)] += 1
        out = ((round(hit / tot, 4) if tot else None), tot)
        return (out, {str(L): (round(pl_[L] / pn[L], 3), pn[L]) for L in sorted(pn)}) if per_leaf else out
    def score_b(lb):
        f = np.zeros(len(pl["xyz"]), np.int16); f[b] = lb; return score_full(f)
    per_step = {}; prev = base_b
    for name, lb in steps.items():
        added = (lb != prev) & (lb > 0); (p_, n_), pleaf = purity(lb, added, True); sc = score_b(lb)
        per_step[name] = dict(added=int(added.sum()), purity_added=p_, projections=n_, per_leaf=pleaf, held_out=[round(sc["summary"]["median_iou"], 4), round(sc["summary"]["max_confusion"], 4)])
        prev = lb
    final = steps["4_bridge"]; (pall, nall) = purity(final, (final != base_b) & (final > 0)); sf = score_b(final); g = sf["summary"]; gv = s_new["summary"]
    r6m = json.load(open(R6 / "metrics.json", encoding="utf-8"))
    res = dict(input_status=STATUS, scale=LO_SCALE(), grouping=grouping, params=R7, maps_file=str(maps_file), vp_vote_reproduced=repro,
               vp_old=s_old, vp_new=s_new, leaf4_surfels=dict(VP_old=int((base_old[b] == 4).sum()), VP_new=int((base_b == 4).sum()), R6=int((np.load(R6 / "labels.npz")["label"][b] == 4).sum()), R7=int((final == 4).sum())),
               per_step=per_step, pooled_purity_all_added=pall, projections=nall, final_held_out=sf, R6_reference=dict(held_out=r6m["held_out"]["final"]["summary"], pooled_purity=r6m["pooled_purity_all_added"]),
               reference=dict(V_P=[0.546, 0.438], A_B_C_Dprime=[0.517, 0.393]),
               passed=bool((pall or 0) >= 0.85 and g["median_iou"] >= gv["median_iou"] and g["max_confusion"] <= gv["max_confusion"]), beats_A_B_C_Dprime_on_both=bool(g["median_iou"] > 0.517 and g["max_confusion"] <= 0.393),
               tag=("gate_passed" if g["gate_pass"] else "gate_failed_viewing_only"), seconds=round(time.time() - t0))
    json.dump(res, open(out / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    fl = np.zeros(len(pl["xyz"]), np.int16); fl[b] = final; ft = np.zeros(len(pl["xyz"]), np.int8); ft[b] = tg
    np.savez_compressed(out / "labels.npz", rows=pl["rows"], label=fl, source=ft, source_names=np.array(C6.TAGS), base_vp_new=base_new.astype(np.int16), **{f"step_{k}": v for k, v in steps.items()})
    from tomato3d.organs import contested_resolve as CR
    CR.images(B, base_b, final, np.where((final != base_b) & (final > 0), 4, np.where(final == 0, 5, 1)).astype(np.int8), out / "images")
    print(json.dumps({k: v for k, v in res.items() if k not in ("vp_old", "vp_new", "final_held_out")}, indent=1, default=float, ensure_ascii=False))
    print("V-P old", s_old["summary"], "| V-P new", s_new["summary"], {k: (v["iou"], v["confusion"]) for k, v in s_new["rows"].items()})
    print("final", g, {k: (v["iou"], v["confusion"]) for k, v in sf["rows"].items()})


def LO_SCALE():
    from tomato3d.organs import light_organs as LO
    return LO.SCALE_TAG


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", choices=("propagate", "evaluate", "paste4"), required=True); a = ap.parse_args(argv)
    if a.step == "propagate": propagate()
    elif a.step == "evaluate": evaluate()
    else: paste4(); evaluate(OUT / "leaf4_only" / "propagated_maps_leaf4only.npz", OUT / "leaf4_only", "compound_chain_R7b_leaf4_only_paste")


if __name__ == "__main__":
    main()
