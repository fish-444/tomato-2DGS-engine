"""Vein graph review bridge for the existing preprocessing lab.

python -m tools.vein_correspondence prepare --run build/leaf_veins_dwt01_20260930
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from tools.leaf_preprocess_lab import Lab, read_json, write_json, sha, data_url
from tools.midrib_review import propose
from tools.vein_landmarks import graph, propose_matches
from tools.plant_structure import Cameras
from tools.xview_geometry import essential, epipolar_distance_px
from tools.leaf_light.reconstruction import triangulate, project
from tools.leaf_light.common import StudyError
from tools.vein_boundary import boundary_candidates, point_role

ROOT=Path(__file__).resolve().parents[1]
CODE_HASHES={n:sha(Path(__file__).with_name(n)) for n in ('vein_correspondence.py','vein_landmarks.py','midrib_review.py','midrib_flow.py','vein_boundary.py')}


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def settings(value):
    if not isinstance(value,dict) or set(value)-{'sigma_px','threshold'}:
        raise ValueError('Unknown graph settings')
    s=value.get('sigma_px',2)
    t=value.get('threshold')
    if isinstance(s,bool) or s not in (1,2,3):
        raise ValueError('Graph sigma must be 1,2,3')
    if t is not None and (isinstance(t,bool) or not isinstance(t,(float,int)) or not np.isfinite(t) or not 0<=t<=1):
        raise ValueError('Threshold must be null or 0..1')
    return dict(sigma_px=s,threshold=t)


def prepare(root,refresh=False):
    from tools.cross_view_identity_import import load_decisions,group_decisions,plan_groups
    from tools.leaflet_mask_batch import Batch
    lab=Lab(root)
    folder=lab.root/'landmarks'; folder.mkdir(exist_ok=refresh)
    if refresh and (folder/'context.json').exists():
        archive=folder/'contexts';archive.mkdir(exist_ok=True)
        (archive/f'{time.time_ns()}.json').write_bytes((folder/'context.json').read_bytes())
    plant_ids={o['plant_id'] for o in lab.samples.values()}
    if len(plant_ids)!=1:
        raise ValueError('Prepare one plant per run')
    plant=next(iter(plant_ids))
    cfgpath=ROOT/'tools/plant_pipeline_plants.json'; cfg=read_json(cfgpath); pc=cfg['plants'][plant]
    capture=ROOT/pc['capture']/'edit_session/capture.json'
    paths=[ROOT/p for p in pc['decisions']]
    paths.extend(sorted(Path(cfg['downloads']).glob(f'cross_view_decisions_{plant}_*.json')))
    decisions=load_decisions(paths)
    grouped=group_decisions(decisions,plant)
    plans,skipped=plan_groups(grouped,Batch(ROOT/pc['batch']))
    membership={o['mask_id']:p['group_id'] for p in plans for o in p['observations']}
    different=[]; different_sets=[]
    for d in decisions:
        detail=d['decision_detail']
        if d['decision']=='different' and detail['kind'] in ('match','pair','ext','trip'):
            panels=[p['mask_ids'] for p in detail['masks']]
            # A rejected extension/triple means "not all the same", not every pair different.
            if len(panels)==2:
                different.extend([x,y] for x in panels[0] for y in panels[1] if x!=y)
            elif len(panels)>2:
                different_sets.append(dict(decision=d['object_id'],masks=sorted({m for p in panels for m in p})))
    sources=dict(lab.doc['sources'])
    sources[str(capture)]=sha(capture)
    sources[str(cfgpath)]=sha(cfgpath)
    sources.update({d['_file']:d['_sha256'] for d in decisions})
    # Freeze discovery as well as file content; new human decisions invalidate this context.
    discovery=dict(roots=[str(ROOT/p) for p in pc['decisions']],downloads=cfg['downloads'],pattern=f'cross_view_decisions_{plant}_*.json')
    discovered=sorted(str(p.resolve()) for r in discovery['roots'] for p in Path(r).glob('*.json'))
    discovered+=sorted(str(p.resolve()) for p in Path(discovery['downloads']).glob(discovery['pattern']))
    doc=dict(schema='vein-context-1',plant_id=plant,capture_path=str(capture),capture=read_json(capture),
             membership=membership,different=different,different_sets=different_sets,excluded=sorted(set(grouped['merged'])|
               {m for s in skipped for m in s['masks']}),identity_plans=plans,skipped=skipped,
             sources=sources,discovery=discovery,discovered=sorted(set(discovered)),dataset_sha256=lab.dataset_hash)
    write_json(folder/'context.json',doc)
    return dict(plant=plant,groups=len(plans),observations=len(lab.samples),excluded=len(doc['excluded']))


class Workflow:
    def __init__(self,lab):
        self.lab=lab; self.root=lab.root/'landmarks'
        self.context_path=self.root/'context.json'; self.context=read_json(self.context_path)
        self.context_hash=sha(self.context_path)
        self.cams=Cameras(self.context['capture']); self.cache={}

    def verify(self):
        if any(sha(Path(__file__).with_name(n))!=h for n,h in CODE_HASHES.items()):
            raise ValueError('Source code changed; restart this server before continuing')
        if sha(self.context_path)!=self.context_hash or sha(self.lab.dataset_path)!=self.context['dataset_sha256']:
            raise ValueError('stale_context_or_dataset')
        for p,h in self.context['sources'].items():
            if sha(Path(p))!=h:
                raise ValueError('stale_source: '+p)
        d=self.context['discovery']
        discovered=[str(p.resolve()) for r in d['roots'] for p in Path(r).glob('*.json')]
        discovered += [str(p.resolve()) for p in Path(d['downloads']).glob(d['pattern'])]
        if sorted(set(discovered))!=self.context['discovered']:
            raise ValueError('new_identity_decisions_require_new_context')

    def relation(self,a,b):
        oa,ob=self.lab.samples[a],self.lab.samples[b]
        if oa['image_file']==ob['image_file']:
            return 'same_photo'
        ma,mb=oa['observation_id'],ob['observation_id']; c=self.context
        if ma in c['excluded'] or mb in c['excluded']:
            return 'identity_conflict_or_excluded'
        ga,gb=c['membership'].get(ma),c['membership'].get(mb)
        for x,y in c['different']:
            ax=x==ma or (ga and ga==c['membership'].get(x))
            by=y==mb or (gb and gb==c['membership'].get(y))
            bx=x==mb or (gb and gb==c['membership'].get(x))
            ay=y==ma or (ga and ga==c['membership'].get(y))
            if (ax and by) or (bx and ay): return 'different'
        return 'same' if ga and ga==gb else 'unknown'

    def extract(self,sid,cfg):
        cfg=settings(cfg); obs=self.lab.samples[sid]
        key=digest([obs,cfg,self.context_hash,CODE_HASHES])
        if key in self.cache: return self.cache[key]
        rgb,mask=self.lab.load(sid)
        r=propose(rgb,mask,rgb,[],obs['crop_xyxy_px'],dict(sigma_px=cfg['sigma_px'],channel='original_l',polarity='bright'))
        g=graph(r['evidence'],r['valid'],rgb,obs['crop_xyxy_px'],cfg['threshold'])
        g['boundary_candidates']=boundary_candidates(mask,obs['crop_xyxy_px'][:2])
        skeleton=g.pop('skeleton')
        overlay=rgb.copy(); overlay[skeleton]=[255,150,0]
        overlay[mask&~r['valid']]=np.rint(.6*overlay[mask&~r['valid']]+.4*np.array([110,70,160])).astype(np.uint8)
        g.update(sample_id=sid,revision=key,settings=cfg,observation=obs,usable_percent=r['usable_percent'],
                 images=dict(original=data_url(rgb),evidence=data_url(overlay)),review='unreviewed')
        self.cache[key]=g
        return g

    def candidates(self,sid,cfg,image_file=None):
        self.verify(); a=self.extract(sid,cfg); rows=[]
        owners=self.reviewed_owners()
        negatives={frozenset((r['a'],r['b'])) for r in self.latest_reviews() if r['identity']=='different'}
        for other in self.lab.samples:
            if image_file is not None and self.lab.samples[other]['image_file']!=image_file: continue
            relation=self.relation(sid,other)
            if relation in ('same_photo','different','identity_conflict_or_excluded') or frozenset((sid,other)) in negatives: continue
            b=self.extract(other,cfg)
            matches=propose_matches(a,b)
            best=defaultdict(list)
            for m in matches: best[m['source']].append(m['score'])
            score=float(np.mean(sorted(min(v) for v in best.values())[:5])) if best else None
            reviewed_same=bool(owners.get(sid,set()) & owners.get(other,set()))
            rows.append(dict(sample_id=other,relation=relation,reviewed_same=reviewed_same,score=score,
                             annotation_id=self.lab.samples[other]['source_annotation_id'],image_file=self.lab.samples[other]['image_file']))
        return dict(candidates=sorted(rows,key=lambda r:(not(r['relation']=='same' or r['reviewed_same']),r['score'] is None,r['score'] or 0,r['sample_id'])),left=a)

    def pair_key(self,a,b): return digest(sorted([a,b]))[:20]

    def latest_reviews(self):
        return [read_json(sorted(f.glob('*.json'))[-1]) for f in (self.root/'reviews').glob('*') if list(f.glob('*.json'))]

    def expected_signature(self,r):
        return digest([self.extract(s,r['graph_settings'])['revision'] for s in (r['a'],r['b'])]+[self.context_hash,self.relation(r['a'],r['b'])])

    def review_is_current(self,r):
        path=self.root/'current'/f"{self.pair_key(r['a'],r['b'])}.json"
        return (path.is_file() and read_json(path)['signature']==r['signature'] and
                r['signature']==self.expected_signature(r))

    def reviewed_owners(self,exclude_pair=None):
        """Suggest existing IDs without confirming an unseen pair or merging identities."""
        owners=defaultdict(set)
        for sid,obs in self.lab.samples.items():
            gid=self.context['membership'].get(obs['observation_id'])
            if gid: owners[sid].add(gid)
        for r in self.latest_reviews():
            if (r['identity']=='same' and self.pair_key(r['a'],r['b'])!=exclude_pair and r['context_sha256']==self.context_hash):
                for sid in (r['a'],r['b']):owners[sid].add(r['leaflet_id'])
        return owners

    def pair(self,a,b,cfg):
        self.verify(); left,right=self.extract(a,cfg),self.extract(b,cfg)
        relation=self.relation(a,b)
        if relation in ('same_photo','different','identity_conflict_or_excluded'):
            raise ValueError('Blocked pair: '+relation)
        matches=propose_matches(left,right)
        ka,kb=(self.lab.samples[s]['image_file'].lower() for s in (a,b))
        posed=self.cams.posed(ka) and self.cams.posed(kb)
        if posed:
            for m in matches:
                pa=next(n['xy_original_px'] for n in left['nodes'] if n['id']==m['source'])
                pb=next(n['xy_original_px'] for n in right['nodes'] if n['id']==m['target'])
                v=float(epipolar_distance_px(self.cams,ka,kb,np.array([pa]),np.array([pb]))[0])
                m['epipolar_distance_px']=v if np.isfinite(v) else None
        sig=digest([left['revision'],right['revision'],self.context_hash,relation])
        key=self.pair_key(a,b)
        proposals=self.root/'proposals';proposals.mkdir(exist_ok=True)
        proposal=proposals/f'{sig}.json'
        if not proposal.exists():
            write_json(proposal,dict(schema='vein-pair-proposal-1',left={k:v for k,v in left.items() if k!='images'},
                right={k:v for k,v in right.items() if k!='images'},matches=matches,relation=relation,
                context_sha256=self.context_hash,score_semantics='Uncalibrated ranking, not probability or accuracy',
                code_hashes={n:sha(Path(__file__).with_name(n)) for n in ('vein_landmarks.py','midrib_review.py','midrib_flow.py','vein_correspondence.py')}))
        current=self.root/'current'; current.mkdir(exist_ok=True)
        write_json(current/f'{key}.json',dict(signature=sig,a=a,b=b))
        owners=self.reviewed_owners(); ids=owners.get(a,set())|owners.get(b,set())
        leaflet=next(iter(ids)) if len(ids)==1 else f'vein_{key[:10]}'
        history=sorted((self.root/'reviews'/key).glob('*.json'))
        known=[]
        for r in self.latest_reviews():
            if r['identity']=='same' and r['leaflet_id']==leaflet and self.review_is_current(r):
                for p in r['points']:
                    if p['state']=='confirmed':
                        known.append(dict(point_id=p['point_id'],samples=[r['a'],r['b']],xy=p['xy'],kind=p.get('kind','interior'),boundary_index=p.get('boundary_index',-1)))
        return dict(left=left,right=right,matches=matches,signature=sig,relation=relation,leaflet_id=leaflet,
                    known_points=known,identity_id_conflict=len(ids)>1,suggested_leaflet_ids=sorted(ids),
                    geometry_available=posed,poses_pass_gate=posed and all(self.cams.frames[k].get('passes_pose_gate') is True for k in (ka,kb)),
                    saved=read_json(history[-1]) if history else None,
                    stability={str(s):{'left':self.extract(a,dict(sigma_px=s,threshold=cfg.get('threshold')))['nodes'],
                                       'right':self.extract(b,dict(sigma_px=s,threshold=cfg.get('threshold')))['nodes']} for s in (1,2,3)})

    def epiline(self,a,b,xy):
        self.verify()
        ka,kb=(self.lab.samples[s]['image_file'].lower() for s in (a,b))
        if not(self.cams.posed(ka) and self.cams.posed(kb)): return []
        self.check_xy(a,xy)
        p=cv2.undistortPoints(np.array([[xy]],float),self.cams.K,self.cams.D).reshape(2)
        line=essential(self.cams,ka,kb)@np.r_[p,1.]
        if not np.isfinite(line).all() or np.linalg.norm(line[:2])<1e-12: return []
        obs=self.lab.samples[b]; x0,y0,x1,y1=obs['crop_xyxy_px']
        corners=cv2.undistortPoints(np.array([[[x0,y0]],[[x1,y1]]],float),self.cams.K,self.cams.D).reshape(-1,2)
        if abs(line[1])>abs(line[0]):
            x=np.linspace(corners[:,0].min()-.05,corners[:,0].max()+.05,100); y=-(line[0]*x+line[2])/line[1]
        else:
            y=np.linspace(corners[:,1].min()-.05,corners[:,1].max()+.05,100); x=-(line[1]*y+line[2])/line[0]
        pts=cv2.projectPoints(np.c_[x,y,np.ones(len(x))],np.zeros(3),np.zeros(3),self.cams.K,self.cams.D)[0].reshape(-1,2)
        return pts.tolist() if np.isfinite(pts).all() else []

    def check_xy(self,sid,xy):
        if not isinstance(xy,list) or len(xy)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in xy):
            raise ValueError('Invalid original-pixel point')
        if not np.isfinite(xy).all(): raise ValueError('Nonfinite point')
        _,mask=self.lab.load(sid); x0,y0,_,_=self.lab.samples[sid]['crop_xyxy_px']
        x,y=int(round(xy[0]-x0)),int(round(xy[1]-y0))
        if not (0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]): raise ValueError('Point outside visible leaf mask')

    def save(self,body):
        self.verify(); a,b=body['a'],body['b']; cfg=settings(body['graph_settings'])
        sig=digest([self.extract(a,cfg)['revision'],self.extract(b,cfg)['revision'],self.context_hash,self.relation(a,b)])
        if sig!=body['signature']: raise ValueError('stale_pair_preview')
        current=read_json(self.root/'current'/f'{self.pair_key(a,b)}.json')
        if current['signature']!=sig: raise ValueError('stale_pair_settings')
        relation=self.relation(a,b)
        if relation not in ('same','unknown'): raise ValueError('Blocked identity relation')
        identity=body.get('identity','unresolved')
        if identity not in ('same','different','unresolved'): raise ValueError('Invalid identity state')
        if relation=='same' and identity=='different':
            # Save the disagreement, but no landmarks or geometry can be exported.
            identity='different'
        leaflet=body['leaflet_id']; rows=body.get('points',[])
        if not isinstance(leaflet,str) or not leaflet or len(leaflet)>100 or not isinstance(rows,list) or len(rows)>200:
            raise ValueError('Invalid leaflet ID or point list')
        existing={self.context['membership'].get(self.lab.samples[s]['observation_id']) for s in (a,b)}-{None}
        if existing and leaflet not in existing: raise ValueError('Use existing confirmed leaflet ID')
        if len(existing)>1: raise ValueError('Existing identity groups require separate conflict review')
        if identity=='same':
            owners=self.reviewed_owners(exclude_pair=self.pair_key(a,b))
            local_ids=owners.get(a,set())|owners.get(b,set())
            if len(local_ids)>1 or (local_ids and leaflet not in local_ids):
                raise ValueError('Reuse the current physical leaflet ID; conflicting groups require separate review')
        seen=set(); locations=[set(),set()]
        duration=body.get('review_seconds')
        if duration is not None and (isinstance(duration,bool) or not isinstance(duration,(int,float)) or not np.isfinite(duration) or duration<0):
            raise ValueError('Invalid review duration')
        if not isinstance(body.get('history',[]),list): raise ValueError('Invalid edit history')
        correction=body.get('correction_seconds')
        if correction is not None and (isinstance(correction,bool) or not isinstance(correction,(float,int)) or not np.isfinite(correction) or correction<0):
            raise ValueError('Invalid manually reported correction time')
        for row in rows:
            kind,index=point_role(row)
            row.update(kind=kind,boundary_index=index)
            if row.get('state') not in ('confirmed','rejected','pending'): raise ValueError('Invalid point state')
            pid=row.get('point_id')
            if not isinstance(pid,str) or not pid or len(pid)>100 or pid in seen: raise ValueError('Duplicate or invalid physical point ID')
            seen.add(pid)
            if not isinstance(row.get('xy'),list) or len(row['xy'])!=2: raise ValueError('Two observed coordinates required')
            for side,sid in enumerate((a,b)):
                xy=row['xy'][side]
                if xy is None:
                    if row['state']=='confirmed': raise ValueError('Confirmed point must be observed in both views')
                    continue
                self.check_xy(sid,xy)
                if row['state']=='confirmed':
                    k=tuple(xy)
                    if k in locations[side]: raise ValueError('Many-to-one confirmed point conflict')
                    locations[side].add(k)
            row['visibility']=['observed' if xy is not None else 'unobserved' for xy in row['xy']]
        boundary_orders=[r['boundary_index'] for r in rows if r['kind']=='boundary' and r['state']=='confirmed']
        if len(boundary_orders)!=len(set(boundary_orders)): raise ValueError('Duplicate boundary order')
        for prior in self.latest_reviews():
            if self.pair_key(prior['a'],prior['b'])==self.pair_key(a,b) or prior['leaflet_id']!=leaflet or prior['identity']!='same': continue
            for old in prior['points']:
                if old['state']!='confirmed': continue
                for row in rows:
                    if row['state']!='confirmed': continue
                    if row['point_id']==old['point_id'] and point_role(row)!=point_role(old):
                        raise ValueError('Physical point kind/order conflicts across views')
                    if row['point_id']!=old['point_id'] and point_role(row)[0]=='boundary' and point_role(row)==point_role(old):
                        raise ValueError('Boundary order belongs to another physical point')
        key=self.pair_key(a,b); folder=self.root/'reviews'/key; folder.mkdir(parents=True,exist_ok=True)
        record=dict(schema='vein-pair-review-1',a=a,b=b,signature=sig,graph_settings=cfg,
                    proposal_path=str(self.root/'proposals'/f'{sig}.json'),proposal_sha256=sha(self.root/'proposals'/f'{sig}.json'),
                    context_sha256=self.context_hash,leaflet_id=leaflet,identity=identity,
                    points=rows,history=body.get('history',[]),review_seconds=body.get('review_seconds'),
                    correction_seconds=correction,correction_time_source='human_reported' if correction is not None else 'unavailable',
                    parent=sorted(p.name for p in folder.glob('*.json'))[-1:] or None,
                    observations=[self.lab.samples[a],self.lab.samples[b]],validated_metric=False,suitable_for_ppfd=False)
        path=folder/f'{time.time_ns()}_{uuid.uuid4().hex[:6]}.json'; write_json(path,record)
        return dict(saved=str(path.relative_to(self.lab.root)),record=record)

    def status(self):
        """Old exports remain on disk but cannot masquerade as current geometry."""
        self.verify()
        latest={str(sorted(f.glob('*.json'))[-1]):sha(sorted(f.glob('*.json'))[-1])
                for f in (self.root/'reviews').glob('*') if list(f.glob('*.json'))}
        results=[]
        for path in sorted((self.root/'exports').glob('*/result.json')):
            r=read_json(path)
            stale={x['path']:x['sha256'] for x in r['source_reviews']}!=latest
            for x in r['source_reviews']:
                review=read_json(Path(x['path']))
                current=read_json(self.root/'current'/f"{self.pair_key(review['a'],review['b'])}.json")
                stale |= current['signature']!=review['signature'] or self.expected_signature(review)!=review['signature']
            stale |= any(sha(Path(__file__).with_name(n))!=h for n,h in r.get('settings_sources',{}).items())
            results.append(dict(path=str(path.relative_to(self.lab.root)),stale=bool(stale),mode=r['mode']))
        counts=dict(leaflet_pair_verdicts={k:0 for k in ('same','different','unresolved')},
                    point_pair_verdicts={k:0 for k in ('confirmed','rejected','pending')},stale_pairs=0)
        for review in self.latest_reviews():
            if not self.review_is_current(review):counts['stale_pairs']+=1;continue
            counts['leaflet_pair_verdicts'][review['identity']]+=1
            for p in review['points']:counts['point_pair_verdicts'][p['state']]+=1
        times=[read_json(p).get('correction_seconds') for p in (self.root/'reviews').glob('*/*.json')]
        reported=[t for t in times if t is not None]
        counts.update(correction_seconds=sum(reported) if reported else None,correction_reports=len(reported),
                      timing_scope='All saved review revisions with explicit human-entered time; UI elapsed time excluded',
                      leaflet_identity_accuracy=None,point_correspondence_accuracy=None)
        return dict(exports=results,review_summary=counts)

    def export(self,mode='strict'):
        self.verify()
        if mode not in ('strict','diagnostic'): raise ValueError('Unknown geometry mode')
        out=self.root/'exports'/f'{time.time_ns()}_{uuid.uuid4().hex[:6]}'
        out.mkdir(parents=True)
        grouped=defaultdict(dict); excluded=[]; identities=defaultdict(set); reviews=[]; roles={}
        for folder in sorted((self.root/'reviews').glob('*')):
            paths=sorted(folder.glob('*.json'))
            if not paths: continue
            r=read_json(paths[-1]); reviews.append(dict(path=str(paths[-1]),sha256=sha(paths[-1])))
            current=read_json(self.root/'current'/f'{folder.name}.json')
            if r['signature']!=current['signature'] or r['signature']!=self.expected_signature(r):
                excluded.append(dict(pair=folder.name,reason='stale_settings')); continue
            if r['identity']!='same':
                excluded.append(dict(pair=folder.name,reason='identity_not_confirmed_same')); continue
            a,b=r['a'],r['b']
            if self.relation(a,b) not in ('same','unknown'): raise ValueError('Blocked saved relation')
            identities[r['leaflet_id']].update([a,b])
            for p in r['points']:
                if p['state']!='confirmed': continue
                identity=(r['leaflet_id'],p['point_id'])
                role=point_role(p)
                if identity in roles and roles[identity]!=role: raise ValueError('Conflicting physical point kind/order')
                roles[identity]=role
                for sid,xy in zip((a,b),p['xy']):
                    self.check_xy(sid,xy)
                    obs=self.lab.samples[sid]; photo=obs['image_file']
                    row=dict(image_file=photo,organ_id=obs['plant_id'],leaflet_id=identity[0],point_id=identity[1],
                             x_px=xy[0],y_px=xy[1],boundary_index=role[1],boundary_count='',source_annotation_id=obs['source_annotation_id'])
                    if photo in grouped[identity] and grouped[identity][photo]!=row: raise ValueError('Conflicting point observations')
                    grouped[identity][photo]=row
        # Enforce negative constraints through the complete saved identity group, not only pairwise.
        negative_pairs=[(r['a'],r['b']) for r in self.latest_reviews() if r['identity']=='different']
        observation_owner={}
        for leaf,sids in identities.items():
            if any(a in sids and b in sids for a,b in negative_pairs): raise ValueError('Human different decision conflicts with saved identity group')
            for sid in sids:
                if sid in observation_owner and observation_owner[sid]!=leaf: raise ValueError('Observation assigned to multiple physical leaflet IDs')
                observation_owner[sid]=leaf
            known_masks={self.lab.samples[s]['observation_id'] for s in sids}
            group_ids={self.context['membership'].get(m) for m in known_masks}-{None}
            known_masks|={m for m,g in self.context['membership'].items() if g in group_ids}
            if any(set(c['masks'])<=known_masks for c in self.context.get('different_sets',[])):
                raise ValueError('Rejected multi-view group would become entirely identical: '+leaf)
            for a in sids:
                for b in sids:
                    if a!=b and self.relation(a,b) in ('different','identity_conflict_or_excluded','same_photo'):
                        raise ValueError('Transitive leaflet identity conflict: '+leaf)
        # Distinct physical IDs cannot claim the same observed pixel, even across pair reviews.
        occupied={}
        for identity,byphoto in grouped.items():
            for photo,row in byphoto.items():
                key=(photo,row['x_px'],row['y_px'])
                if key in occupied and occupied[key]!=identity: raise ValueError('Cross-pair many-to-one point conflict')
                occupied[key]=identity
        orders=defaultdict(dict)
        for (leaf,pid),(kind,index) in roles.items():
            if kind=='boundary':
                if index in orders[leaf] and orders[leaf][index]!=pid: raise ValueError('Duplicate boundary order across views')
                orders[leaf][index]=pid
        allrows=[]; depths=[]; results=[]; frames={f['image_file']:f for f in self.context['capture']['frames']}
        policy=dict(self.context['capture']['quality'])
        for (leaf,pid),byphoto in grouped.items():
            rows=list(byphoto.values())
            for row in rows: row['boundary_count']=len(orders[leaf]) if orders[leaf] else ''
            allrows.extend(rows)
            usable=[]
            for row in rows:
                f=frames[row['image_file']]
                if f.get('pose_status')!='solved' or (mode=='strict' and f.get('passes_pose_gate') is not True): continue
                usable.append((f,np.array([row['x_px'],row['y_px']],float)))
            result=dict(leaflet_id=leaf,point_id=pid,kind=roles[(leaf,pid)][0],boundary_index=roles[(leaf,pid)][1],status='unavailable',reason='',observed_views=len(rows),eligible_views=len(usable),xyz_mm=None)
            try:
                if len(usable)<2: raise StudyError('fewer_than_two_eligible_poses')
                # Null gate stays inactive, never replaced by a stricter guessed threshold.
                p=dict(policy)
                if p['max_point_error_px'] is None: p['max_point_error_px']=float('inf')
                if p['min_ray_angle_deg'] is None: p['min_ray_angle_deg']=0
                xyz,error=triangulate(usable,self.context['capture']['intrinsics'],p)
                result.update(status='diagnostic_point' if mode=='diagnostic' else 'qualified_point',xyz_mm=xyz.tolist(),rms_px=error,reprojections=[])
                for row in rows:
                    if frames[row['image_file']]['pose_status']=='solved':
                        q=project(xyz[None],frames[row['image_file']],self.context['capture']['intrinsics'])[0]
                        result['reprojections'].append(dict(image_file=row['image_file'],xy_original_px=q.tolist(),observed_xy=[row['x_px'],row['y_px']]))
                        depths.append(dict(row,x_board_mm=xyz[0],y_board_mm=xyz[1],z_board_mm=xyz[2],
                            reprojection_error_px=float(np.linalg.norm(q-[row['x_px'],row['y_px']]))))
            except (StudyError,ValueError) as exc: result['reason']=str(exc)
            results.append(result)
        columns=['image_file','organ_id','leaflet_id','point_id','x_px','y_px','boundary_index','boundary_count','source_annotation_id']
        for name,rows,cols in [('points.interior.csv',[r for r in allrows if r['boundary_index']<0],columns),('points.boundary.csv',[r for r in allrows if r['boundary_index']>=0],columns),('depths.csv',depths,columns+['x_board_mm','y_board_mm','z_board_mm','reprojection_error_px'])]:
            with (out/name).open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
        record=dict(schema='vein-landmark-export-1',mode=mode,points=results,excluded=excluded,source_reviews=reviews,
                    context_sha256=self.context_hash,policy=policy,surface_status='needs_confirmed_boundary_points',
                    note='Boundary/interior CSVs share physical point IDs and reviewed boundary counts; an empty count means no reviewed boundary set exists.',
                    coordinate_frame=self.context['capture'].get('coordinate_frame'),validated_metric=False,suitable_for_ppfd=False)
        record['settings_sources']=dict(CODE_HASHES)
        write_json(out/'result.json',record)
        return dict(path=str(out.relative_to(self.lab.root)),**record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','extract']); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--refresh-context',action='store_true',help='Archive the old context before re-reading human decisions')
    a=p.parse_args()
    if a.command=='prepare': result=prepare(a.run,a.refresh_context)
    else:
        w=Workflow(Lab(a.run)); w.verify(); rows=[]
        started=time.perf_counter()
        folder=w.root/f'graphs_{time.time_ns()}'; folder.mkdir(exist_ok=False)
        for sid in w.lab.samples:
            g={k:v for k,v in w.extract(sid,dict(sigma_px=2,threshold=None)).items() if k!='images'}
            write_json(folder/f'{sid}.json',g)
            rows.append(dict(sample_id=sid,junctions=sum(n['kind']=='junction' for n in g['nodes']),status=g['status']))
        w.verify()
        with (folder/'summary.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['sample_id','junctions','status']); writer.writeheader(); writer.writerows(rows)
        result=dict(observations=len(rows),junctions=sum(r['junctions'] for r in rows),seconds=time.perf_counter()-started,
                    context_sha256=w.context_hash,output=str(folder),code_sha256={n:sha(Path(__file__).with_name(n)) for n in ('vein_landmarks.py','midrib_review.py','midrib_flow.py')})
        write_json(folder/'manifest.json',result)
    print(json.dumps(result))


if __name__=='__main__': main()
