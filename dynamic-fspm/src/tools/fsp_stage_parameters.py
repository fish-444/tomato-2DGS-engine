"""Default-off S1 contracts, native injection and independent numerical references."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tools.vtc_growth_io import require, number, read_json

SWITCHES = {'organ_maintenance_q10', 'maint_eq7', 'fruit_sink', 'stage_sla',
            'process_base_temperatures', 'rank_morphology', 'side_axis_phytomer_cap'}
MAINT = {'MAINT_LEAF_GC': 'gC/gDM/day', 'MAINT_STEM_GC': 'gC/gDM/day',
         'MAINT_ROOT_GC': 'gC/gDM/day', 'MAINT_FRUIT_GC': 'gC/gDM/day',
         'MAINT_Q10': 'dimensionless', 'MAINT_TREF_C': 'degC', 'CH2O_TO_GC': 'gC/gCH2O'}
FRUIT = {'FRUIT_FORM': 'categorical', 'FRUIT_MAX_DM': 'gDM', 'ANTHESIS_DELAY_DD': 'degree_day'}
FRUIT_EXTRA = {'FRUIT_MID_DD': 'degree_day', 'FRUIT_K': '1/degree_day',
               'FRUIT_GOMPERTZ_K': '1/degree_day', 'FRUIT_GOMPERTZ_XM_DD': 'degree_day',
               'FRUIT_INTERVAL_DD': 'degree_day', 'FRUITS_PER_TRUSS_MAX': 'count',
               'FRUIT_RIPENING_DD': 'degree_day', 'TRUSS_MAX_DM': 'gDM',
               'TRUSS_MID_DD': 'degree_day', 'TRUSS_TAU95_DD': 'degree_day'}
SLA = {'SLA_PRE': 'm2/gDM', 'SLA_POST': 'm2/gDM', 'SLA_TRANSITION_DD': 'degree_day',
       'SLA_FIRST_ANTHESIS_DD': 'degree_day'}
BASES = {'TB_DEVELOPMENT_C': 'degC', 'TB_ORGAN_GROWTH_C': 'degC', 'TB_GERMINATION_C': 'degC'}
UNITS = {**MAINT, **FRUIT, **FRUIT_EXTRA, **SLA, **BASES,
         'MAINT_ACTIVITY_F': 'day', 'RANK1_LEAF_LW': 'length/width', 'SIDE_AXIS_PHYTOMER_CAP': 'count'}
PARAMETERS = set(UNITS)
SOURCES = {'literature_T1', 'literature_T3', 'literature_T4', 'literature_secondary',
           'derived_low_confidence', 'user_assumed', 'literature_verified_fulltext',
           'literature_abstract_only'}
GROUP = {'Root': 'ROOT', 'Internode': 'STEM', 'Petiole': 'STEM', 'RachisSegment': 'STEM',
         'TrussAxis': 'STEM', 'Leaflet': 'LEAF', 'Flower': 'FRUIT', 'Fruit': 'FRUIT'}


def enabled(req: dict, key: str) -> bool:
    return req.get('engine_hooks', {}).get(key, False) is True


def active(req: dict) -> bool:
    return any(enabled(req, k) for k in SWITCHES)


def values(req: dict) -> dict:
    return {k: p['value'] for k, p in req.get('engine_hooks', {}).get('parameters', {}).items() if k in PARAMETERS}


def validate(req: dict) -> None:
    for switch in SWITCHES:
        require(type(req.get('engine_hooks',{}).get(switch,False)) is bool,'S1 switch must be boolean')
    p = req.get('engine_hooks', {}).get('parameters', {})
    required = set(); allowed = set()
    if enabled(req, 'organ_maintenance_q10') or enabled(req, 'maint_eq7'):
        required.update(MAINT)
    require(not (enabled(req, 'organ_maintenance_q10') and enabled(req, 'maint_eq7')),
            'select reduced maintenance OR maint_eq7, not both')
    if enabled(req, 'maint_eq7'): required.add('MAINT_ACTIVITY_F')
    if enabled(req, 'fruit_sink'):
        required.update(FRUIT); allowed.update(FRUIT_EXTRA)
        form = p.get('FRUIT_FORM', {}).get('value')
        require(form in ('logistic', 'gompertz'), 'FRUIT_FORM must be logistic or gompertz')
        if form == 'logistic':
            required.update({'FRUIT_MID_DD', 'FRUIT_K', 'FRUIT_INTERVAL_DD', 'FRUITS_PER_TRUSS_MAX',
                             'FRUIT_RIPENING_DD', 'TRUSS_MAX_DM', 'TRUSS_MID_DD', 'TRUSS_TAU95_DD'})
            require(not {'FRUIT_GOMPERTZ_K', 'FRUIT_GOMPERTZ_XM_DD'} & set(p), 'Gompertz parameters on logistic fruit')
        else:
            required.update({'FRUIT_GOMPERTZ_K', 'FRUIT_GOMPERTZ_XM_DD'})
            require(not {'FRUIT_MID_DD', 'FRUIT_K'} & set(p), 'logistic parameters on Gompertz fruit')
        truss = set(p) & {'TRUSS_MAX_DM', 'TRUSS_MID_DD', 'TRUSS_TAU95_DD'}
        require(not truss or len(truss) == 3, 'truss override needs mass, MID and tau95')
    if enabled(req, 'stage_sla'): required.update(SLA)
    if enabled(req, 'process_base_temperatures'): required.update(BASES)
    if enabled(req, 'rank_morphology'): required.add('RANK1_LEAF_LW')
    if enabled(req, 'side_axis_phytomer_cap'): required.add('SIDE_AXIS_PHYTOMER_CAP')
    require(required <= set(p), 'missing S1 parameters: ' + ', '.join(sorted(required - set(p))))
    require((set(p) & PARAMETERS) <= required | allowed, 'S1 parameter supplied for disabled switch')
    if not active(req): return
    require(not req.get('light_only'), 'S1 growth hooks incompatible with light_only')
    for key in set(p) & PARAMETERS:
        rec = p[key]
        require(rec.get('unit') == UNITS[key] and rec.get('source_kind') in SOURCES
                and bool(rec.get('source_ref')), 'S1 provenance/unit invalid: ' + key)
        if key == 'FRUIT_FORM': continue
        value = number(rec['value'], key, None if key.endswith('_C') else 0)
        if key not in ('ANTHESIS_DELAY_DD', 'FRUIT_INTERVAL_DD', 'SLA_FIRST_ANTHESIS_DD') and not key.endswith('_C'):
            require(value > 0, 'S1 positive parameter required: ' + key)
        if key in ('FRUITS_PER_TRUSS_MAX', 'SIDE_AXIS_PHYTOMER_CAP'):
            require(type(rec['value']) is int, 'S1 count must be integer')
    v = values(req)
    if 'CH2O_TO_GC' in v: require(v['CH2O_TO_GC'] == .4, 'CH2O carbon fraction must be 12/30, not CF')
    if 'TRUSS_TAU95_DD' in v:
        require(v['TRUSS_TAU95_DD'] > v['TRUSS_MID_DD'], 'truss tau95 must exceed MID')
    if enabled(req, 'side_axis_phytomer_cap'):
        require(req.get('axis_contract') == 'fsp-axes-1' and enabled(req, 'planned_axes'),
                'side cap requires declared planned new axes')


def logistic(age: float, mid: float, k: float) -> float:
    s0 = 1 / (1 + math.exp(k * mid))
    s = 1 / (1 + math.exp(-k * (age - mid)))
    return max(0., min(1., (s - s0) / (1 - s0)))


def logistic_k(mid: float, tau95: float) -> float:
    lo, hi = 0., 1.
    for _ in range(100):
        k = (lo + hi) / 2
        if logistic(tau95, mid, k) < .95: lo = k
        else: hi = k
    return (lo + hi) / 2


def gompertz(age: float, k: float, xm: float) -> float:
    # Offset normalization preserves zero mass at anthesis and the supplied asymptote.
    initial = math.exp(-math.exp(k * xm))
    return max(0., min(1., (math.exp(-math.exp(-k * (age - xm))) - initial) / (1 - initial)))


def thermal(req: dict, env: dict, dt: float, process: str = 'organ_growth') -> float:
    if enabled(req, 'development_temperature_curve'):
        from tools.fsp_temperature import thermal as temperature_thermal
        return temperature_thermal(req, env, dt, process)
    v = values(req)
    base = v['TB_' + process.upper() + '_C'] if enabled(req, 'process_base_temperatures') else req['parameters']['BASE_C']['value']
    h = float(env['light_hours'])
    mean = (float(env['temp_day_c']) * h + float(env['temp_night_c']) * (24 * dt - h)) / (24 * dt)
    return max(0., mean - base) * dt


def q10_factor(req: dict, env: dict, dt: float, *, periods: bool) -> float:
    v = values(req); h = float(env['light_hours']); td = float(env['temp_day_c']); tn = float(env['temp_night_c'])
    q, ref = v['MAINT_Q10'], v['MAINT_TREF_C']
    if periods: return (h * q ** ((td-ref)/10) + (24*dt-h) * q ** ((tn-ref)/10)) / (24*dt)
    return q ** (((td*h + tn*(24*dt-h))/(24*dt)-ref)/10)


def rgr5(history: list[tuple[float, float]], now: float) -> tuple[float, float, str]:
    """Only actual positive-mass boundary samples; no log(0), imputation or future data."""
    positive = [(t,m) for t,m in history if now-5 <= t <= now and m > 0]
    if len(positive) < 2: return 0., 0., 'insufficient_positive_history'
    t0, m0 = positive[0]; t1, m1 = positive[-1]
    if t1 <= t0: return 0., 0., 'insufficient_positive_history'
    require(m1 >= m0, 'negative RGR is unsupported: structural mass loss needs a reviewed contract')
    return math.log(m1/m0)/(t1-t0), t1-t0, 'full_window' if t1-t0 >= 5 else 'available_window'


def maintenance(req: dict, kind: str, mass: float, env: dict, dt: float,
                history: list[tuple[float, float]], now: float) -> dict:
    v = values(req); group = GROUP.get(kind)
    factor = q10_factor(req, env, dt, periods=enabled(req, 'maint_eq7') or enabled(req, 'maintenance_temperature_periods'))
    reduced = mass * v['MAINT_' + group + '_GC'] * factor * dt if group else 0.
    rgr, window, status = rgr5(history, now) if enabled(req, 'maint_eq7') else (0., 0., 'not_used')
    activity = -math.expm1(-v['MAINT_ACTIVITY_F']*rgr) if enabled(req, 'maint_eq7') else 1.
    return dict(maintenance_gC=reduced*activity, reduced_gC=reduced, activity=activity,
                rgr_per_day=rgr, window_days=window, history_status=status, q10_factor=factor)


def advance_seed_clock(clock: float, germ: float, dev: float, germ_increment: float, threshold: float) -> tuple[float, float]:
    if germ < threshold:
        fraction = max(0., 1-(threshold-germ)/germ_increment) if germ_increment > 0 else 0.
        germ += germ_increment
        return (threshold + dev*fraction if germ >= threshold else 0.), germ
    return clock + dev, germ + germ_increment


def retime_bud_host(host: dict, axis_end_day: dict, source: str, pkg: dict) -> dict:
    """Run-owner adapter: preserve candidates/draw keys, retime eligibility AND delay."""
    from tools.microtom_budbreak import retime_host
    from tools.fsp_extensions import interval_seconds
    req=pkg['request'];require(enabled(req,'process_base_temperatures'),'process clock switch required')
    env=pkg['tables']['environment.csv'];n=host['days']
    require(len(env)>=n and all(interval_seconds(req,i)==86400 for i in range(n)),
            'bud planner uses daily boundaries; explicit daily environment required')
    out=retime_host(host,axis_end_day,source)
    process='bud_delay' if enabled(req,'development_temperature_curve') else 'development'
    out['thermal_per_day']=[thermal(req,env[i],1.,process) for i in range(n)]
    out['retimed']['thermal_basis']=dict(process='dd_development',base_c=values(req)['TB_DEVELOPMENT_C'],
        source='S1 frozen environment; candidates and random draw keys unchanged',previous_thermal_per_day=host['thermal_per_day'])
    if enabled(req,'development_temperature_curve'):
        out['retimed']['thermal_basis'].update(process='bud_delay',
            source='S2-T period-integrated bud-delay clock; candidates and random draw keys unchanged')
    return out


BUTTURINI_LOGISTIC_NOTE = "Butturini et al. 2026 (cv. Heartbreakers F1 Twiggy Red, explicitly dwarf, not Micro-Tom, tier P2) uses a beta growth function (Eq. 12); this scenario places Butturini's numbers into our logistic form - an approximation, not source-equivalent"


def fruit_form_source_note(req: dict) -> str | None:
    """Provenance relabel 2026-10-20: a logistic fruit form whose FRUIT_/TRUSS_ records cite Butturini is an approximation of Butturini's beta function.
    Identified by declared provenance + form, never by coefficient values; generic logistic requests and the Ke Gompertz return None."""
    recs = req.get('engine_hooks', {}).get('parameters', {})
    cites = any('butturini' in str(p.get('source_ref', '')).lower() for n, p in recs.items() if n.startswith(('FRUIT_', 'TRUSS_')))
    return BUTTURINI_LOGISTIC_NOTE if cites and values(req).get('FRUIT_FORM') == 'logistic' else None


def metadata(req: dict) -> dict:
    v = values(req)
    from tools.fsp_engine_hooks import tau95
    return dict(switches={k:enabled(req,k) for k in sorted(SWITCHES)},
                parameters={k:p for k,p in req['engine_hooks'].get('parameters',{}).items() if k in PARAMETERS},
                model_form='full_eq7' if enabled(req,'maint_eq7') else 'reduced_no_activity_factor' if enabled(req,'organ_maintenance_q10') else 'legacy',
                maintenance_mapping=GROUP, mapping_source_kind='user_assumed',
                CH2O_TO_GC=v.get('CH2O_TO_GC'), CARBON_FRACTION=req['parameters']['CARBON_FRACTION']['value'],
                maintenance_mass_basis='structural organ mass; plant reserve is a separate pool, never assigned to organs',
                rgr_policy='log ratio over <=5 elapsed days of positive boundary history; <2 positive samples => activity 0; user approved 2026-10-17',
                maintenance_temperature=('period-weighted under S2-T guard' if enabled(req,'maintenance_temperature_periods') else 'period-weighted for Eq7; mean-temperature reduced form as declared in S1'),
                sla_policy='freeze SLA at leaf birth from cumulative host organ-growth DD and declared first anthesis; imported leaves unchanged',
                fruit_tau95_dd=tau95(v['FRUIT_MID_DD'],v['FRUIT_K']) if v.get('FRUIT_FORM')=='logistic' else None,
                truss_k=logistic_k(v['TRUSS_MID_DD'],v['TRUSS_TAU95_DD']) if 'TRUSS_MAX_DM' in v else None,
                truss_mapping='MID retained, normalized tau95 solved; user approved 2026-10-17',
                fruit_geometry='existing Flower proxy retained; no fruit size/shape or optical calibration supplied',
                fruit_clock=('separate flowering and fruit clocks; user-approved remaining interval fraction at anthesis; identity and funded mass retained' if enabled(req,'development_temperature_curve') else 'Flower age minus anthesis delay and within-truss interval; same identity and funded mass retained'),
                gompertz_mapping='offset normalized to zero at anthesis, same k/xm/asymptote; prevents unfunded birth mass',
                fruit_form_source_note=fruit_form_source_note(req),
                omitted_fruit_options='existing simultaneous Flower schedule, registry FLOWERS count, no new ripening event, unchanged TrussAxis',
                biological_validation='not_established')


class Reference:
    """Recompute S1 terms from frozen environment and returned organ mass histories."""
    def __init__(self, pkg: dict, days: dict):
        from tools.fsp_extensions import interval_seconds
        self.req=pkg['request']; self.v=values(self.req); self.days=days; self.env=pkg['tables']['environment.csv']
        self.dt=[interval_seconds(self.req,i)/86400 for i in range(len(self.env))]
        self.times=[0.]; self.growth=[0.]; self.dev=[0.]; self.germ=[0.]
        for i,e in enumerate(self.env):
            self.times.append(self.times[-1]+self.dt[i])
            for name,series in (('organ_growth',self.growth),('development',self.dev),('germination',self.germ)):
                series.append(series[-1]+thermal(self.req,e,self.dt[i],name))
        self.stagger=[0.]; self.fruit=[0.]
        for i,e in enumerate(self.env):
            if enabled(self.req,'development_temperature_curve'):
                self.stagger.append(self.stagger[-1]+thermal(self.req,e,self.dt[i],'stagger'))
                self.fruit.append(self.fruit[-1]+thermal(self.req,e,self.dt[i],'fruit') if enabled(self.req,'fruit_sink') else 0.)
            else:
                self.stagger.append(self.dev[i+1]); self.fruit.append(self.growth[i+1])
        self.initial={s['organ_id']:s for s in pkg['tables'].get('initial_state.csv',[])}
        self.rpd=pkg.get('_random_draws')   # random_parameters draws (None = constants)

    def history(self, oid: str, interval: int) -> list[tuple[float,float]]:
        return [(self.times[d],float(self.days[d][oid]['mass_gDM'])) for d in range(interval+1)
                if oid in self.days[d] and self.times[d]>=self.times[interval]-5]

    def maintenance(self, s: dict, previous: dict | None, interval: int) -> dict:
        mass=float(previous['mass_gDM']) if previous else 0.
        if enabled(self.req,'maint_eq7') or enabled(self.req,'organ_maintenance_q10'):
            return maintenance(self.req,s['class'],mass,self.env[interval],self.dt[interval],
                               self.history(s['organ_id'],interval),self.times[interval])
        from tools.vtc_development_io import KINDS
        kind=11 if s['class']=='Fruit' else KINDS.index(s['class'])
        return dict(maintenance_gC=mass*self.req['parameters']['MAINT']['value'][kind]*self.dt[interval])

    def sla(self,s: dict) -> float:
        oid=s['organ_id']; base=self.req['parameters']['SLA_M2_G']['value']
        if oid in self.initial:
            old=self.initial[oid]
            return float(old['area_m2'])/float(old['mass_gDM']) if float(old['mass_gDM'])>0 else base
        if not enabled(self.req,'stage_sla') or int(s['birth_day'])<0:return base
        t=max(0.,min(1.,(self.growth[int(s['birth_day'])]-self.v['SLA_FIRST_ANTHESIS_DD'])/self.v['SLA_TRANSITION_DD']))
        return self.v['SLA_PRE']+(self.v['SLA_POST']-self.v['SLA_PRE'])*t

    def leaf_age(self,s: dict,day: int) -> tuple[float,float]:
        from tools.fsp_engine_hooks import offset_dd
        born=int(s['birth_day']); start=max(0,born)
        if s['organ_id'] in self.initial:
            initial=float(self.initial[s['organ_id']]['thermal_age_dd'])
            return initial+self.dev[day], initial+self.growth[day]
        pairs=self.rpd.pairs(s['plant_instance_id'],False) if self.rpd is not None else self.req['parameters']['PAIRS']['value']; rank=int(s['true_leaf_order'])
        count=pairs[min(max(0,rank-1),len(pairs)-1)]
        delay=offset_dd(self.req,s['class'],int(s['pair_index']),count,int(s['side']))
        dev=0.; stagger=0.; growth=0.
        for i in range(start,day):
            delta=self.dev[i+1]-self.dev[i]
            stagger_delta=self.stagger[i+1]-self.stagger[i]
            fraction=max(0.,min(1.,(stagger+stagger_delta-delay)/stagger_delta)) if stagger_delta>0 else float(stagger>=delay)
            growth+=(self.growth[i+1]-self.growth[i])*fraction;dev+=delta;stagger+=stagger_delta
        return dev,growth

    def fruit_age(self,s: dict) -> float:
        if enabled(self.req,'development_temperature_curve'):
            # Independent reconstruction from birth; never trust the returned fruit-age column.
            start=max(0,int(s['birth_day'])); day=int(s['day'])
            onset=self.v['ANTHESIS_DELAY_DD']+max(0,int(s['pair_index'])-1)*self.v.get('FRUIT_INTERVAL_DD',0.)
            flowering=0.; age=0.
            for i in range(start,day):
                dev=self.dev[i+1]-self.dev[i]
                fraction=max(0.,min(1.,(flowering+dev-onset)/dev)) if dev>0 else float(flowering>=onset)
                age+=(self.fruit[i+1]-self.fruit[i])*fraction;flowering+=dev
            return age
        return max(0.,float(s['thermal_age_dd'])-self.v['ANTHESIS_DELAY_DD']-max(0,int(s['pair_index'])-1)*self.v.get('FRUIT_INTERVAL_DD',0.))

    def target(self,s: dict,states: dict) -> float:
        from tools.fsp_engine_hooks import target_age
        from tools.vtc_development_io import KINDS
        v=self.v;mass=float(s['max_mass_gDM'])
        if s['class']=='Fruit':
            age=self.fruit_age(s)
            fraction=logistic(age,v['FRUIT_MID_DD'],v['FRUIT_K']) if v['FRUIT_FORM']=='logistic' else gompertz(age,v['FRUIT_GOMPERTZ_K'],v['FRUIT_GOMPERTZ_XM_DD'])
            return mass*fraction
        if enabled(self.req,'fruit_sink') and 'TRUSS_MAX_DM' in v and s['class']=='TrussAxis' and s['organ_id'] not in self.initial:
            return mass*logistic(float(s['thermal_age_dd']),v['TRUSS_MID_DD'],logistic_k(v['TRUSS_MID_DD'],v['TRUSS_TAU95_DD']))
        kind=KINDS.index(s['class']);par=self.req['parameters']
        return mass*logistic(target_age(self.req,s,states),par['MID_DD']['value'][kind],par['K']['value'][kind])

    def validate_state(self,s: dict,day: int,events: dict) -> None:
        def eq(a: float,b: float,label: str) -> None:
            require(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),'S1 '+label+f': {a} != {b}')
        oid=s['organ_id'];v=self.v
        if enabled(self.req,'process_base_temperatures') and day>0:
            dev,growth=self.leaf_age(s,day)
            eq(float(s['s1_development_age_dd']),dev,'development clock')
            eq(float(s['s1_leaf_growth_age_dd']),growth,'post-stagger growth clock')
            if enabled(self.req,'development_temperature_curve'):
                origin=float(self.initial[oid]['thermal_age_dd']) if oid in self.initial else 0.
                start=0 if oid in self.initial else max(0,int(s['birth_day']))
                eq(float(s['s2_stagger_age_dd']),origin+self.stagger[day]-self.stagger[start],'S2 stagger clock')
        if active(self.req) and s['class']=='Leaflet' and oid not in self.initial:
            eq(float(s['s1_sla_m2_g']),self.sla(s),'birth-frozen SLA')
            if enabled(self.req,'rank_morphology') and not enabled(self.req,'leaflet_outline') and float(s['width_m'])>0:
                expected=v['RANK1_LEAF_LW'] if int(s['true_leaf_order'])==1 else self.req['parameters']['LEAF_LW']['value']
                eq(float(s['length_m'])/float(s['width_m']),expected,'node-indexed leaf L/W')
        if enabled(self.req,'fruit_sink') and s['class'] in ('Flower','Fruit') and oid not in self.initial:
            onset=v['ANTHESIS_DELAY_DD']+max(0,int(s['pair_index'])-1)*v.get('FRUIT_INTERVAL_DD',0.)
            flower_age=self.leaf_age(s,day)[0] if enabled(self.req,'development_temperature_curve') else float(s['thermal_age_dd'])
            expected=(not 'FRUITS_PER_TRUSS_MAX' in v or int(s['pair_index'])<=v['FRUITS_PER_TRUSS_MAX']) and flower_age>=onset and day>0
            require((s['class']=='Fruit')==expected,'S1 fruit transition time/cap mismatch')
            if expected:
                eq(float(s['max_mass_gDM']),v['FRUIT_MAX_DM'],'fruit potential mass')
                eq(float(s['s1_fruit_age_dd']),self.fruit_age(s),'fruit age')
                event=events.get((oid,'anthesis_fruit_set'));require(event is not None and int(event['day'])<=day,'S1 fruit set event missing')
                eq(float(event['structural_gC']),0.,'fruit transition must not manufacture carbon')
                if 'FRUIT_RIPENING_DD' in v and self.fruit_age(s)>=v['FRUIT_RIPENING_DD']:
                    require((oid,'fruit_ripening') in events,'S1 ripening event missing')
        if enabled(self.req,'fruit_sink') and 'TRUSS_MAX_DM' in v and s['class']=='TrussAxis' and oid not in self.initial:
            eq(float(s['max_mass_gDM']),v['TRUSS_MAX_DM'],'truss potential mass')


def export_diagnostics(source: Path, output: Path, pkg: dict) -> None:
    from collections import defaultdict
    from tools.vtc_growth_io import rows,write_csv,write_json
    req=pkg['request']
    from tools import fsp_temperature
    if not (active(req) or fsp_temperature.active(req)):return
    days=defaultdict(dict)
    for s in rows(source/'states.csv'):days[int(s['day'])][s['organ_id']]=s
    from tools import fsp_random_parameters as rp
    pkg['_random_draws']=rp.Draws(req,source)
    ref=Reference(pkg,days);organ=[];kind_sums=defaultdict(lambda:[0.,0.])
    for f in rows(source/'flux.csv'):
        i=int(f['interval']);oid=f['organ_id'];s=days[i+1][oid];before=days[i].get(oid)
        calc=ref.maintenance(s,before,i)
        require(math.isclose(float(f['maintenance_gC']),calc['maintenance_gC'],rel_tol=1e-12,abs_tol=1e-12),'S1 maintenance diagnostic mismatch')
        organ.append(dict(interval=i,plant_instance_id=s['plant_instance_id'],organ_id=oid,organ_kind=s['class'],**calc))
        key=(i,s['plant_instance_id'],s['class']);kind_sums[key][0]+=float(f['maintenance_gC']);kind_sums[key][1]+=calc.get('reduced_gC',0.)
    write_csv(output/'s1_maintenance.csv',organ)
    write_csv(output/'s1_maintenance_by_kind.csv',[dict(interval=k[0],plant_instance_id=k[1],organ_kind=k[2],maintenance_gC=v[0],same_period_reduced_gC=v[1],eq7_to_same_period_reduced=v[0]/v[1] if v[1] and enabled(req,'maint_eq7') else 'unavailable') for k,v in sorted(kind_sums.items())])
    write_csv(output/'s1_process_clocks.csv',[dict(boundary=i,elapsed_days=ref.times[i],dd_development=ref.dev[i],dd_organ_growth=ref.growth[i],dd_germination=ref.germ[i],germination_scope='seed_only') for i in range(max(days)+1)])
    if fsp_temperature.active(req):
        write_json(output/'s2t_parameters.json',fsp_temperature.metadata(req))
        write_csv(output/'s2t_process_clocks.csv',[dict(boundary=i,elapsed_days=ref.times[i],dd_development=ref.dev[i],dd_stagger=ref.stagger[i],dd_organ_growth=ref.growth[i],dd_fruit=ref.fruit[i],dd_germination=ref.germ[i]) for i in range(max(days)+1)])
    reserves={(0,p['plant_instance_id']):float(p['reserve_gC']) for p in pkg['tables']['plants.csv']}
    reserves.update({(int(r['interval'])+1,r['plant_instance_id']):float(r['reserve_end_gC']) for r in rows(source/'ledger.csv')})
    cf=req['parameters']['CARBON_FRACTION']['value']
    summaries=[]
    for day,ss in sorted(days.items()):
        for pid in sorted({s['plant_instance_id'] for s in ss.values()}):
            group=[s for s in ss.values() if s['plant_instance_id']==pid];dm=sum(float(s['mass_gDM']) for s in group)
            fruit=sum(float(s['mass_gDM']) for s in group if s['class']=='Fruit')
            reserve=reserves[day,pid];total=dm+reserve/cf
            summaries.append(dict(day=day,plant_instance_id=pid,structural_dm_g=dm,reserve_gC=reserve,total_dm_g=total,
                reserve_fraction_total_dm=(reserve/cf)/total if total else 'unavailable',fruit_dm_g=fruit,
                fruit_fraction_structural_dm=fruit/dm if dm else 'unavailable',fruit_fraction_total_dm=fruit/total if total else 'unavailable',
                leaf_area_m2=sum(float(s['area_m2']) for s in group if s['class']=='Leaflet')))
    write_csv(output/'s1_plant_summary.csv',summaries);write_json(output/'s1_parameters.json',metadata(req))


def transform_native(template: str, req: dict, records: list[dict]) -> str:
    if not active(req): return template
    validate(req); v = values(req)
    def replace(old: str, new: str) -> None:
        nonlocal template
        require(template.count(old)==1, 'S1 insertion point changed: '+old[:90])
        template=template.replace(old,new)
    code=Path(__file__).with_name('fsp_stage_parameters_native.rgg').read_text(encoding='utf-8')
    constants = dict(v)
    constants.update({k:enabled(req,k) for k in SWITCHES})
    # Disabled branches still compile; inert literals are not effective scientific defaults.
    for k in PARAMETERS: constants.setdefault(k, 'logistic' if k=='FRUIT_FORM' else 0.)
    constants['TRUSS_K']=logistic_k(v['TRUSS_MID_DD'],v['TRUSS_TAU95_DD']) if 'TRUSS_MAX_DM' in v else 0.
    constants['HAS_TRUSS']='TRUSS_MAX_DM' in v
    constants['HAS_RIPENING']='FRUIT_RIPENING_DD' in v
    constants['HAS_FRUIT_CAP']='FRUITS_PER_TRUSS_MAX' in v
    hp=req.get('engine_hooks',{}).get('parameters',{})
    constants['STAGGER']=hp.get('LEAFLET_STAGGER_DD',{}).get('value',0.) if enabled(req,'leaf_development') and hp.get('LEAFLET_GROWTH_ORDER',{}).get('value')=='basipetal' else 0.
    constants['NEW_AXES']='new boolean[]{'+','.join(str(a.get('origin')=='axillary' and a.get('provenance')=='model_generated').lower() for a in records)+'}'
    if not records: constants['NEW_AXES']='new boolean[@NA@]'
    for key,value in constants.items(): code=code.replace('@S1_'+key+'@',value if key=='NEW_AXES' else json.dumps(value))
    replace('@AXIS_CODE@','@AXIS_CODE@\n'+code)
    replace('float[] shape=null;', 'float[] shape=null; boolean s1Initial=false,s1Fruit=false,s1Ripe=false; double s1Sla=-1,s1Dev=0,s1LeafAge=0,s1FruitAge=0,s1Rgr=0,s1Window=0,s1Activity=1,s1Reduced=0; ArrayList s1History=new ArrayList();')
    replace('initialize();','initialize();if(observationMode)for(int j=0;j<organs.size();j++)org(j).s1Initial=true;')
    replace('double targetMass(Organ o) {','double targetMass(Organ o) {\n    if(s1FruitEnabled && o.s1Fruit)return s1FruitTarget(o);\n    if(s1FruitEnabled && s1HasTruss && o.kind==10 && !o.s1Initial)return o.maxMass*s1Logistic(o.thermal,s1TrussMid,s1TrussK);')
    replace('o.maxMass=MAX_MASS[kind];','o.maxMass=MAX_MASS[kind];if(s1FruitEnabled && s1HasTruss && kind==10)o.maxMass=s1TrussMass;')
    replace('o.maint=o.mass*MAINT[o.kind]*dtDays;', 'o.maint=s1Maintenance(o);')
    replace('o.area=o.mass*SLA_M2_G;', 'o.area=o.mass*s1LeafSla(o);')
    replace('Math.sqrt(2*o.area*LEAF_LW);o.wid=o.len/LEAF_LW;', 'Math.sqrt(2*o.area*s1LeafLW(o));o.wid=o.len/s1LeafLW(o);')
    replace('double gross=0,maintenance=0,demand=0;', 'if(s1FruitEnabled)s1FruitTransitions(p,step+1);\n            double gross=0,maintenance=0,demand=0;')
    replace('int rank=++leafOrders[controller];','if(s1Cap && s1NewAxes[controller] && leafOrders[controller]>=s1CapCount){org(activeApex[controller]).stopped=true;event(org(activeApex[controller]),day,"phytomer_cap","new_axis_leaf_bearing_node_cap",0);break;}\n            int rank=++leafOrders[controller];')
    replace('kinds[o.kind]+","+o.rank', '(o.s1Fruit?"Fruit":kinds[o.kind])+","+o.rank')
    # Extra columns are bound by the existing state/flux hashes and checkpoint manifests.
    state_header='provenance,parent_rank,developmental_axis_id,segment_index'+(',shape_source' if enabled(req,'leaflet_outline') else '')
    replace(state_header+'\");',state_header+',s1_development_age_dd,s1_leaf_growth_age_dd,s1_sla_m2_g,s1_fruit_age_dd\");')
    tail='+segmentLabel(o)'+('+","+shapeSource(o)' if enabled(req,'leaflet_outline') else '')+');\n        if(o.kind==2)'
    replace(tail,tail.replace(');\n', '+","+o.s1Dev+","+o.s1LeafAge+","+(o.kind==9?s1LeafSla(o):0)+","+o.s1FruitAge);\n'))
    # Sample history at the actual interval-start boundary, before allocation.
    replace('extensionInterval();','extensionInterval();s1StartInterval();')
    if enabled(req,'process_base_temperatures'):
        replace('double dd=Math.max(0,(tday[step]*hours[step]+tnight[step]*(24*dtDays-hours[step]))/(24*dtDays)-BASE_C)*dtDays;',
                'double dd=s1DD(s1GrowthBase),devDD=s1DD(s1DevBase),germDD=s1DD(s1GermBase);')
        replace('clocks[c]+=dd;', 's1AdvanceClock(c,devDD,germDD);')
        replace('for(int j=0;j<organs.size();j++)if(org(j).owner==p)org(j).thermal+=dd;',
                'for(int j=0;j<organs.size();j++)if(org(j).owner==p)s1AdvanceOrgan(org(j),dd,devDD);')
        old='double age=Math.max(0,o.thermal-position*'
        if old in template:
            start=template.index(old);end=template.index(';',start)
            template=template[:start]+'double age=o.s1LeafAge'+template[end:]
    return template


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    a=p.parse_args();req=read_json(a.request);validate(req)
    print(json.dumps(metadata(req) if active(req) else {'active':False},indent=2))


if __name__=='__main__': main()
