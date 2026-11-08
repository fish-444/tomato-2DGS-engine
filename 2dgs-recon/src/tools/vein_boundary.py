"""Mask-derived boundary proposals, never physical cross-view correspondences."""
import cv2
import numpy as np


def boundary_candidates(mask, offset):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    nodes = []
    for component, contour in enumerate(contours):
        if hierarchy[0,component,3] >= 0: continue  # Holes are not outer leaf margins.
        polygon = cv2.approxPolyDP(contour, 2.0, True).reshape(-1,2)
        if len(polygon) < 3: continue
        for i, point in enumerate(polygon):
            nodes.append(dict(id=f'b{component}_{i}',kind='boundary',component=component,
                xy_original_px=(point+np.asarray(offset)).tolist(),review='unreviewed',
                source='mask_turn_proposal',boundary_semantics='May be an occlusion edge; human review required'))
    return nodes


def point_role(row):
    kind = row.get('kind','interior')
    index = row.get('boundary_index',-1)
    if kind not in ('interior','boundary'): raise ValueError('Unknown point kind')
    if kind == 'boundary':
        if isinstance(index,bool) or not isinstance(index,int) or not 0 <= index < 200:
            raise ValueError('Boundary order must be an integer from 0 to 199')
        if row.get('state') == 'confirmed' and row.get('boundary_reviewed') is not True:
            raise ValueError('Confirm the actual visible leaf boundary separately')
    elif index != -1: raise ValueError('Interior point cannot have a boundary order')
    return kind, index


def check_boundary_polygon(xyz):
    """Reject crossed/repeated order in the same local plane used by triangulation."""
    _,_,vt=np.linalg.svd(xyz-xyz.mean(axis=0))
    points=(xyz-xyz.mean(axis=0))@vt[:2].T
    def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])
    def on_segment(a,b,p):return abs(cross(b-a,p-a))<1e-9 and np.dot(p-a,p-b)<=1e-9
    for i,a in enumerate(points):
        b=points[(i+1)%len(points)]
        if np.linalg.norm(b-a)<1e-9:raise ValueError('repeated_boundary_position')
        for j in range(i+1,len(points)):
            if j in ((i+1)%len(points),) or (j+1)%len(points)==i:continue
            c,d=points[j],points[(j+1)%len(points)]
            if (cross(b-a,c-a)*cross(b-a,d-a)<0 and cross(d-c,a-c)*cross(d-c,b-c)<0) or any((on_segment(a,b,c),on_segment(a,b,d),on_segment(c,d,a),on_segment(c,d,b))):
                raise ValueError('self_intersecting_boundary_order')
