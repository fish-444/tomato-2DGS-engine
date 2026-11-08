"""Amendment 3: record the skeleton label files chosen via latest.json (file name + sha256) before any number."""
import hashlib, json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
B = R / "build/p1_all_3dfirst_20261027_01"; D = Path(__file__).resolve().parent
PL = ["e_01", "e_04", "e_05", "e_07", "n_01", "n_02", "n_03", "n_05", "n_06"]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
rec = {}
for p in PL:
    j = json.loads((B / p / "skeleton/latest.json").read_text(encoding="utf-8")); f = B / p / "skeleton" / j["file"]
    rec[p] = dict(file=f.relative_to(R).as_posix(), sha256=sha(f), latest_json_sha256=sha(B / p / "skeleton/latest.json"),
                  groups=(B / p / "merge/groups.npz").relative_to(R).as_posix(), groups_sha256=sha(B / p / "merge/groups.npz"), branches=j["branches"], at=j["at"])
(D / "skeleton_inputs_amendment3.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
for p, v in rec.items():
    print(p, v["file"].split("/")[-1], v["sha256"][:16], v["branches"])
