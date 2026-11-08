"""FINAL_HASHES.json over every output of this folder (+ the two handoff notes) and the lock rename (CONTRACT §6). No deletion."""
import hashlib, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = {}
for p in sorted(HERE.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts and p.name not in ("FINAL_HASHES.json",) and not p.name.startswith("RUNNING.lock"):
        files[str(p.relative_to(HERE)).replace("\\", "/")] = sha(p)
for h in ("handoff/MTG_BUILD_TO_BMODEL_ENGINE_2026-11-02.md", "handoff/MTG_BUILD_2026-11-02_KO.md"):
    if (ROOT / h).exists():
        files[h] = sha(ROOT / h)
out = dict(written=time.strftime("%Y-%m-%d %H:%M"), n=len(files), files=files, inputs=json.load(open(HERE / "inputs/INPUT_HASHES.json", encoding="utf-8"))["files"])
(HERE / "FINAL_HASHES.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("hashed", len(files), "files; FINAL_HASHES.json sha256", sha(HERE / "FINAL_HASHES.json")[:16])
if "--release" in sys.argv:
    lock = HERE / "RUNNING.lock"
    if lock.exists():
        lock.rename(HERE / f"RUNNING.lock.done_{time.strftime('%Y%m%d_%H%M')}"); print("lock released")
