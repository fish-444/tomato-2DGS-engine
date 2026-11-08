"""Ideotype validator = the Engine v2 validator (v2_validate.py, unchanged file) + one more replay replacement:
rachis / petiole length = Model_v2 law × the plant's LS / PE multiplier (CONTRACT §3-4). Internode lengths already replay the recorded
(scaled) INTERNODE_MM draw. install(cfg) with cfg = v2_config.json of the case (cfg['ideo'] = None or dict(plants, IN, LS, PE, beta))."""
import v2_validate as VV

OLD = "close(f(s,'length_m'),(par['PETIOLE_M'] if kind==7 else par['RACHIS_M']/m)*fraction,'leaf-pair funded rachis/petiole length')"
NEW = "close(f(s,'length_m'),(par['PETIOLE_M']*V2.k(pid,'PE') if kind==7 else par['RACHIS_M']/m*V2.k(pid,'LS'))*fraction,'leaf-pair funded rachis/petiole length')"


def install(cfg: dict) -> dict:
    ideo = cfg.get('ideo') or {}
    plants = {f'p{i + 1}' for i in ideo.get('plants', [])}

    def k(pid, key):
        return float(ideo.get(key, 1.0)) if pid in plants else 1.0
    VV.V2.k = staticmethod(k)
    if (OLD, NEW) not in VV.REPLACEMENTS:
        VV.REPLACEMENTS.append((OLD, NEW))
    return VV.install(cfg)
