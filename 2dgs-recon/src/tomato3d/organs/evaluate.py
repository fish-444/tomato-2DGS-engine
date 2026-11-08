"""Step 6 metrics on the synthetic bench: nodes, leaf membership, leaflet F1, side axes, ambiguous / unattached shares."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def node_metrics(pred_nodes: int, truth_ranks: int) -> dict:
    return dict(pred=int(pred_nodes), truth=int(truth_ranks), exact=bool(pred_nodes == truth_ranks), abs_err=abs(int(pred_nodes) - int(truth_ranks)))


def _match(overlap: np.ndarray):
    """Hungarian match on an overlap matrix (rows = truth, cols = pred); returns {truth_row: pred_col}."""
    if overlap.size == 0:
        return {}
    r, c = linear_sum_assignment(-overlap)
    return {int(i): int(j) for i, j in zip(r, c) if overlap[i, j] > 0}


def membership(truth_leaf: np.ndarray, pred_leaf: np.ndarray, is_blade_truth: np.ndarray, truth_obj: np.ndarray) -> dict:
    """truth_leaf / pred_leaf: per point leaf id strings ('' / 'unattached' = none). Point- and leaflet-weighted shares."""
    m = is_blade_truth & (truth_leaf != "")
    tl = np.unique(truth_leaf[m]); pl = np.unique(pred_leaf[m]); pl = pl[(pl != "") & (pl != "unattached")]
    ov = np.array([[int(((truth_leaf == t) & (pred_leaf == p) & m).sum()) for p in pl] for t in tl]).reshape(len(tl), len(pl))
    mt = _match(ov); matched = {tl[i]: pl[j] for i, j in mt.items()}
    pw = float(np.mean([matched.get(t) == p for t, p in zip(truth_leaf[m], pred_leaf[m])])) if m.any() else float("nan")
    per_leaflet = []
    for o in np.unique(truth_obj[m]):
        mm = m & (truth_obj == o); t = truth_leaf[mm][0]
        vals, cnt = np.unique(pred_leaf[mm], return_counts=True); per_leaflet.append(vals[np.argmax(cnt)] == matched.get(t))
    return dict(point=pw, leaflet=float(np.mean(per_leaflet)) if per_leaflet else float("nan"),
                matched={str(k): str(v) for k, v in matched.items()}, truth_leaves=int(len(tl)), pred_leaves=int(len(pl)))


def leaflet_f1(truth_obj: np.ndarray, is_blade_truth: np.ndarray, cl: np.ndarray, iou_min=0.5) -> dict:
    """Truth leaflet instances (blade objects) vs predicted clusters, matched at point-IoU >= iou_min."""
    m = is_blade_truth; tobjs = np.unique(truth_obj[m]); ncl = int(cl.max()) + 1
    if not len(tobjs) or ncl == 0:
        return dict(f1=0.0, precision=0.0, recall=0.0, tp=0, fp=int(ncl), fn=int(len(tobjs)))
    tid = {o: i for i, o in enumerate(tobjs)}; T = np.full(len(truth_obj), -1); T[m] = [tid[o] for o in truth_obj[m]]
    inter = np.zeros((len(tobjs), ncl)); tsz = np.bincount(T[T >= 0], minlength=len(tobjs)); csz = np.bincount(cl[cl >= 0], minlength=ncl)
    both = (T >= 0) & (cl >= 0); np.add.at(inter, (T[both], cl[both]), 1)
    iou = inter / np.maximum(tsz[:, None] + csz[None, :] - inter, 1)
    mt = _match(np.where(iou >= iou_min, iou, 0)); tp = len(mt)
    return dict(f1=2 * tp / max(2 * tp + (ncl - tp) + (len(tobjs) - tp), 1), precision=tp / ncl, recall=tp / len(tobjs), tp=tp, fp=ncl - tp, fn=len(tobjs) - tp)


def side_axes(pred: int, truth: int) -> dict:
    tp = min(pred, truth)
    return dict(pred=pred, truth=truth, precision=(tp / pred) if pred else float("nan"), recall=(tp / truth) if truth else float("nan"))


def pooled(rows: list[dict]) -> dict:
    g = lambda k: [r[k] for r in rows if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]  # noqa: E731
    sp = [r["side"] for r in rows]; tp = sum(min(s["pred"], s["truth"]) for s in sp); pp = sum(s["pred"] for s in sp); tt = sum(s["truth"] for s in sp)
    return dict(n=len(rows), node_exact=float(np.mean([r["node"]["exact"] for r in rows])), node_mae=float(np.mean([r["node"]["abs_err"] for r in rows])),
                membership_point=float(np.mean(g("membership_point"))), membership_leaflet=float(np.mean(g("membership_leaflet"))),
                leaflet_f1=float(np.mean([r["leaflet"]["f1"] for r in rows])),
                side_precision=(tp / pp) if pp else float("nan"), side_recall=(tp / tt) if tt else float("nan"),
                ambiguous=float(np.mean(g("ambiguous"))), unattached=float(np.mean(g("unattached"))),
                unattached_blade_truth=float(np.mean(g("unattached_blade_truth"))))
