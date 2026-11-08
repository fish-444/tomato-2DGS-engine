"""Classical vein graphs and correspondence proposals; no anatomical or identity decisions."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, zero-padded, without connecting disconnected components."""
    a = np.pad(binary.astype(np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            c = a[1:-1, 1:-1]
            p = [a[:-2,1:-1], a[:-2,2:], a[1:-1,2:], a[2:,2:],
                 a[2:,1:-1], a[2:,:-2], a[1:-1,:-2], a[:-2,:-2]]
            count = sum(p)
            transitions = sum(((p[i] == 0) & (p[(i+1)%8] == 1)).astype(np.uint8) for i in range(8))
            if phase == 0:
                constraint = (p[0]*p[2]*p[4] == 0) & (p[2]*p[4]*p[6] == 0)
            else:
                constraint = (p[0]*p[2]*p[6] == 0) & (p[0]*p[4]*p[6] == 0)
            remove = (c != 0) & (count >= 2) & (count <= 6) & (transitions == 1) & constraint
            if remove.any():
                changed = True
                c[remove] = 0
        if not changed:
            return a[1:-1,1:-1].astype(bool)


def descriptor(rgb: np.ndarray, x: int, y: int) -> list:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    patch = cv2.getRectSubPix(gray, (17,17), (float(x),float(y)))
    patch = cv2.resize(patch, (9,9), interpolation=cv2.INTER_AREA)
    patch = (patch-patch.mean()) / max(float(patch.std()), 1.0)
    return patch.ravel().tolist()


def graph(evidence: np.ndarray, valid: np.ndarray, rgb: np.ndarray, crop: list,
          threshold: float | None = None) -> dict:
    if evidence.shape != valid.shape or rgb.shape[:2] != valid.shape:
        raise ValueError('Graph input shape mismatch')
    if threshold is not None and (isinstance(threshold,bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Threshold must be null (Otsu) or 0..1')
    if not valid.any() or float(evidence[valid].max()) <= 1e-8:
        return dict(nodes=[], edges=[], threshold=threshold, status='no_evidence', skeleton=np.zeros(valid.shape,bool))
    if threshold is None:
        t, _ = cv2.threshold(np.rint(evidence[valid]*255).astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        threshold = float(t/255)
    sk = skeletonize(valid & (evidence > threshold))
    pixels = set(zip(*np.nonzero(sk)))
    adjacency = {}
    for y,x in pixels:
        adjacency[(y,x)] = [(y+dy,x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
                            if (dy or dx) and (y+dy,x+dx) in pixels and
                            (not (dy and dx) or ((y,x+dx) not in pixels and (y+dy,x) not in pixels))]
    branching = np.zeros(sk.shape,bool)
    for p, ns in adjacency.items():
        if len(ns) >= 3:
            branching[p] = True
    labels, count = ndimage.label(branching, np.ones((3,3)))
    groups = [list(zip(*np.where(labels==i))) for i in range(1,count+1)]
    groups += [[p] for p in sorted(pixels) if len(adjacency[p]) <= 1]
    node_at, nodes = {}, []
    distance = ndimage.distance_transform_edt(valid)
    for i, group in enumerate(groups):
        pts = np.array(group); mean = pts.mean(axis=0)
        y,x = pts[np.argmin(np.square(pts-mean).sum(axis=1))].tolist()
        nid = f'n{i:04d}'
        for p in group:
            node_at[p] = nid
        nodes.append(dict(id=nid, xy_original_px=[x+crop[0],y+crop[1]],
                          kind='junction' if i<count else 'endpoint', degree=0, neighbours=[],
                          near_valid_boundary=bool(distance[y,x]<=2), descriptor=descriptor(rgb,x,y)))
    edges, seen = [], set()
    for p, nid in node_at.items():
        for q in adjacency[p]:
            if node_at.get(q)==nid or (p,q) in seen:
                continue
            path=[p,q]; seen.add((p,q)); seen.add((q,p))
            previous, current = p,q
            while current not in node_at:
                ns=[n for n in adjacency[current] if n!=previous]
                if len(ns)!=1 or (current,ns[0]) in seen:
                    break
                nxt=ns[0]; seen.add((current,nxt)); seen.add((nxt,current))
                path.append(nxt); previous,current=current,nxt
            other=node_at.get(current)
            if other and other!=nid:
                edges.append(dict(a=nid,b=other,length_px=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                                  xy_original_px=[[int(x+crop[0]),int(y+crop[1])] for y,x in path]))
    by_id={n['id']:n for n in nodes}
    for e in edges:
        for a,b in ((e['a'],e['b']),(e['b'],e['a'])):
            by_id[a]['neighbours'].append(b); by_id[a]['degree']+=1
    for n in nodes:
        n['neighbour_degrees']=sorted(by_id[k]['degree'] for k in n['neighbours'])
        centre=np.array(n['xy_original_px'])
        ordered=sorted(n['neighbours'],key=lambda k:float(np.arctan2(*(np.array(by_id[k]['xy_original_px'])-centre)[::-1])))
        n['ordered_neighbours']=ordered
        n['branch_signature']=[[by_id[k]['degree'],sum(by_id[j]['degree'] for j in by_id[k]['neighbours'] if j!=n['id'])] for k in ordered]
    return dict(nodes=nodes,edges=edges,threshold=threshold,status='candidates' if nodes else 'no_nodes',
                closed_components_without_nodes=bool(pixels and not nodes),skeleton=sk)


def propose_matches(left: dict, right: dict) -> list:
    """Top three candidates; dihedral patch comparison includes reflection. No auto acceptance."""
    output=[]
    targets=[n for n in right['nodes'] if n['kind']=='junction']
    if not targets: return []
    patches=np.array([n['descriptor'] for n in targets],float)
    degrees=np.array([n['degree'] for n in targets])
    for a in (n for n in left['nodes'] if n['kind']=='junction'):
        pa=np.array(a['descriptor']).reshape(9,9)
        transforms=np.array([np.rot90(p,k).ravel() for p in (pa,pa[:,::-1]) for k in range(4)])
        distances=np.square(transforms[:,None,:]-patches[None,:,:]).mean(axis=2)
        orientations=distances.argmin(axis=0); patch=distances.min(axis=0)
        degree=np.abs(degrees-a['degree'])
        width=max(len(a['neighbour_degrees']),max(len(b['neighbour_degrees']) for b in targets),1)
        left_topology=np.pad(a['neighbour_degrees'],(0,width-len(a['neighbour_degrees'])))
        right_topology=np.array([np.pad(b['neighbour_degrees'],(0,width-len(b['neighbour_degrees']))) for b in targets])
        topology=np.abs(right_topology-left_topology).sum(axis=1)/np.maximum(1,np.maximum(len(a['neighbour_degrees']),[len(b['neighbour_degrees']) for b in targets]))
        sa=np.array(a.get('branch_signature',[]),float); order=np.zeros(len(targets))
        indices=[i for i,b in enumerate(targets) if len(sa)>0 and np.array(b.get('branch_signature',[])).shape==sa.shape]
        if indices:
            sb=np.array([targets[i]['branch_signature'] for i in indices])
            options=np.array([np.roll(v,k,axis=0) for v in (sa,sa[::-1]) for k in range(len(sa))])
            order[indices]=np.abs(options[:,None,:,:]-sb[None,:,:,:]).mean(axis=(2,3)).min(axis=0)
        score=patch+degree+topology+order
        for i in sorted(range(len(targets)),key=lambda i:(score[i],targets[i]['id']))[:3]:
            output.append(dict(source=a['id'],target=targets[i]['id'],score=float(score[i]),
                patch_distance=float(patch[i]),degree_difference=int(degree[i]),topology_difference=float(topology[i]),
                branch_order_distance=float(order[i]),reflected_patch=bool(orientations[i]>=4),status='unreviewed'))
    uses={}
    for c in output:uses.setdefault(c['target'],set()).add(c['source'])
    for c in output:c['ambiguous_target']=len(uses[c['target']])>1
    return output
