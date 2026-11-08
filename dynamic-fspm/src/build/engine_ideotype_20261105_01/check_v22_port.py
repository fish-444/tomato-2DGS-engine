"""Check of the v2.2 port of ideo_patch.py: on the same raw prepared Model.rgg, the ported apply_multi with the default (v2.1) profile must give
the same text as the v2.1-only copy (ideo_patch_v21only.py) — G1-style identity; then the v2.2 A profile must apply (all patch points found).
python -X utf8 check_v22_port.py  (Python only, no JVM; writes into a scratch build dir under C:/fsp_build)"""
import hashlib, shutil, sys
from pathlib import Path
import prepare_ideo as PI
import ideo_patch as NEW
import ideo_patch_v21only as OLD

captured = {}
real = PI.IP.apply_multi


def grab(text, n, ideo=None, lum=None):
    captured['raw'] = text; captured['args'] = (n, ideo, lum); return real(text, n, ideo, lum)


PI.IP.apply_multi = grab
D = Path('C:/fsp_build/engine_ideotype_20261105_01/scratch_port_check')
if D.exists(): shutil.rmtree(D)
PI.build(D, PI.parse('IN150@s10h40@r1'))
raw = captured['raw']; n, ideo, lum = captured['args']
a = OLD.apply_multi(raw, n, ideo, lum); PI.IP.apply_multi = real; b = NEW.apply_multi(raw, n, ideo, lum)
h = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]
print('v2.1 default identical:', a == b, h(a), h(b))
try:
    c = NEW.apply_multi(raw, n, ideo, lum, profile=NEW.profile_v22('A', 5.85))
    print('v2.2 A applies on the basipetal raw text (unexpected):', h(c))
except AssertionError as e:
    print('v2.2 A on a basipetal raw text refused as expected (needs the simultaneous request):', str(e)[:120])
sys.exit(0 if a == b else 1)
