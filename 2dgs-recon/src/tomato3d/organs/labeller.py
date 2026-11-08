"""Step 2 labeller: class-balanced logistic regression on the 14 features, trained on the synthetic TRAIN half only.

usage: python -m tomato3d.organs.labeller            (trains, reports TRAIN/TEST, writes labeller.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tomato3d.organs import contract as C
from tomato3d.organs import features as FE


def load_bench(ind: str, recompute=False):
    z = np.load(C.RUN / "bench" / f"{ind}.npz"); fp = C.RUN / "features" / f"{ind}.npz"
    if recompute or not fp.exists():
        F, nrm = FE.compute(z["xyz"]); np.savez_compressed(fp, F=F, normal_pca=nrm)
    else:
        F = np.load(fp)["F"]
    return z, F


class Labeller:
    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale, self.coef, self.intercept = (np.asarray(x, np.float64) for x in (mean, scale, coef, intercept))

    def p_axis(self, F: np.ndarray) -> np.ndarray:
        z = ((F - self.mean) / self.scale) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-z))

    @classmethod
    def load(cls, path=None):
        d = json.load(open(path or C.RUN / "labeller.json"))
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"])

    @classmethod
    def train(cls, F: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        mean, scale = F.mean(0), F.std(0) + 1e-9
        m = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0).fit((F - mean) / scale, y)
        return cls(mean, scale, m.coef_[0], m.intercept_[0])

    def to_json(self, trained_on):
        return dict(**C.TAG, features=C.FEATURES, ks=list(C.FEAT_KS), smooth_mm=C.LABEL_SMOOTH_MM, thr=C.LABEL_THR,
                    mean=self.mean.tolist(), scale=self.scale.tolist(), coef=self.coef.tolist(), intercept=float(self.intercept),
                    trained_on=trained_on, revision=4)


def report(lab: Labeller, inds) -> dict:
    tp = fp = fn = tn = 0; n = 0
    for ind in inds:
        z, F = load_bench(ind); y = z["axis_label"]; m = y >= 0
        pred = FE.smooth(z["xyz"], lab.p_axis(F) >= C.LABEL_THR)
        p, t = pred[m], y[m] == 1
        tp += int((p & t).sum()); fp += int((p & ~t).sum()); fn += int((~p & t).sum()); tn += int((~p & ~t).sum()); n += int(m.sum())
    return dict(n=n, accuracy=(tp + tn) / n, axis_precision=tp / max(tp + fp, 1), axis_recall=tp / max(tp + fn, 1), blade_recall=tn / max(tn + fp, 1))


def main():
    Fs, ys = [], []
    for ind in C.BENCH["train"]:
        z, F = load_bench(ind, recompute=True); m = z["axis_label"] >= 0; Fs.append(F[m]); ys.append(z["axis_label"][m] == 1)
    for ind in C.BENCH["test"]:
        load_bench(ind, recompute=True)
    lab = Labeller.train(np.vstack(Fs), np.concatenate(ys))
    json.dump(lab.to_json(C.BENCH["train"]), open(C.RUN / "labeller.json", "w"), indent=1)
    rep = dict(**C.TAG, revision=4, thr=C.LABEL_THR, train=report(lab, C.BENCH["train"]), test=report(lab, C.BENCH["test"]),
               coef={k: round(float(v), 3) for k, v in zip(C.FEATURES, lab.coef)})
    json.dump(rep, open(C.RUN / "labeller_report.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("train", "test")}, indent=1))


if __name__ == "__main__":
    main()
