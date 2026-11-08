"""P2-only boundary-constrained meshes and explicit triangle intersection gates.

Triangle is an optional, pinned execution dependency, imported only on use.
No source contour is repaired, simplified or redrawn here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
import numpy as np
from shapely import LineString, Point, Polygon, union_all

from tomato3d.light.geometry import triangle_areas_mm2
from tomato3d.state.fixture_truth import _orient_adaxial
from tomato3d.synth.leaflet_surface import surface_points
from tools.p2_d3_meshes import export_vertices


def _cross2(a: tuple, b: tuple, p: tuple) -> Fraction:
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def _exact_area(points: list[tuple]) -> Fraction:
    if not points: return Fraction(0)
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(points,points[1:]+points[:1])))/2


def _clipped_area(subject: list[tuple], triangle: np.ndarray) -> Fraction:
    """Exact Sutherland-Hodgman area, including disconnected clipped components.

    Oppositely traversed connector edges cancel in the signed shoelace sum.
    Fractions represent the supplied binary64 coordinates exactly; no snapping,
    buffering, contour repair or tolerance widening is performed.
    """
    clip = [tuple(Fraction(float(x)) for x in p) for p in triangle]
    if _cross2(clip[0],clip[1],clip[2]) < 0: clip.reverse()
    output = subject
    for a,b in zip(clip,clip[1:]+clip[:1]):
        incoming = output; output = []
        if not incoming: break
        previous = incoming[-1]; before = _cross2(a,b,previous)
        for current in incoming:
            after = _cross2(a,b,current)
            if (after>=0) != (before>=0):
                ratio = before/(before-after)
                output.append(tuple(previous[k]+ratio*(current[k]-previous[k]) for k in range(2)))
            if after>=0: output.append(current)
            previous,before = current,after
    return _exact_area(output)


def boundary_outside_area(triangles: np.ndarray, contour: np.ndarray) -> tuple[float,int]:
    """Exact area on every boundary candidate; isolated interiors use point location.

    Raw-coordinate AABBs conservatively select all triangles that can touch a
    contour segment. For other triangles the boundary cannot cross the triangle,
    so its centroid determines the whole face's inside/outside classification.
    """
    c = np.asarray(contour,float); polygon = Polygon(c)
    following = np.roll(c,-1,axis=0); low = np.minimum(c,following); high = np.maximum(c,following)
    subject = [tuple(Fraction(float(x)) for x in p) for p in c]
    outside = Fraction(0); candidates = 0
    for tri in triangles:
        crossing = np.any(np.all(low<=tri.max(axis=0),axis=1) & np.all(high>=tri.min(axis=0),axis=1))
        if not crossing and polygon.contains(Point(tri.mean(axis=0))): continue
        exact_tri = [tuple(Fraction(float(x)) for x in p) for p in tri]
        area = _exact_area(exact_tri)
        if crossing:
            candidates += 1
            intersection = _clipped_area(subject,tri)
            if intersection > area: raise ValueError('exact polygon intersection exceeds triangle area')
            outside += area-intersection
        else: outside += area
    return float(outside),candidates


@dataclass(frozen=True)
class Settings:
    """Numerical defaults; serialized in every run, never crop parameters."""
    minimum_angle_deg: float = 3.0
    refinement_angle_deg: float = 20.0
    angle_atol_deg: float = 1e-7
    max_steiner: int = 20000
    min_triangle_area_mm2: float = 1e-6
    outside_atol_mm2: float = 1e-9
    intersection_atol_mm: float = 1e-8
    max_candidate_pairs: int = 2000000
    max_intersection_examples: int = 20
    area_rtol: float = .005
    solver_rtol: float = 1e-5
    solver_iterations: int = 12

    def validate(self) -> None:
        if not 0 < self.minimum_angle_deg <= self.refinement_angle_deg <= 20.7:
            raise ValueError('require 0 < minimum angle <= refinement target <= 20.7 degrees')
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('positive finite numerical settings required')


def constrained(contour: np.ndarray, grid_n: int, settings: Settings) -> tuple[np.ndarray, np.ndarray, dict]:
    """PSLG + quality refinement; preserve every supplied boundary segment."""
    import triangle
    settings.validate()
    c = np.asarray(contour, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) < 3 or not np.isfinite(c).all():
        raise ValueError('finite contour Nx2 required')
    if c[:, 0].min() < 0 or c[:, 0].max() > 1:
        raise ValueError('contour outside normalized domain')
    polygon = Polygon(c)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError('self_intersecting_or_empty_contour')
    if len(np.unique(c, axis=0)) != len(c):
        raise ValueError('duplicate contour vertex; no repair allowed')
    if grid_n < 2:
        raise ValueError('grid_n >= 2 required')
    # A doubled grid quarters the maximum parameter-plane triangle area.
    maximum_area = float(np.prod(np.ptp(c, axis=0)) / grid_n**2)
    # Existing acute boundary corners cannot be widened without changing outline.
    # Refine toward 20 degrees; enforce the explicit common hard floor separately.
    opts = f'pq{settings.refinement_angle_deg:.10g}a{maximum_area:.17f}S{settings.max_steiner}Q'
    result = triangle.triangulate(dict(vertices=c, segments=np.column_stack((np.arange(len(c)), np.roll(np.arange(len(c)), -1)))), opts)
    if 'triangles' not in result:
        raise ValueError('constrained triangulation produced no faces')
    uv = np.asarray(result['vertices']); faces = np.asarray(result['triangles'], dtype=int)
    tri = uv[faces]
    sides = np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2)
    angles = []
    for k in range(3):
        a, b, opposite = sides[:, k], sides[:, (k+1)%3], sides[:, (k+2)%3]
        angles.append(np.degrees(np.arccos(np.clip((a*a+b*b-opposite*opposite)/(2*a*b), -1, 1))))
    measured_angle = float(np.min(angles))
    pieces = [Polygon(t) for t in tri]
    overlay_outside_uv2 = sum(p.difference(polygon).area for p in pieces)
    outside_uv2,exact_candidates = boundary_outside_area(tri,c)
    missing_uv2 = polygon.difference(union_all(pieces)).area
    actual_max_area = max(p.area for p in pieces)
    audit = dict(triangulator='triangle', triangle_version=triangle.__version__, options=opts,
        parameter_minimum_angle_deg=measured_angle, parameter_max_triangle_area=actual_max_area,
        requested_parameter_max_triangle_area=maximum_area, outside_uv2=outside_uv2, missing_uv2=missing_uv2,
        geos_overlay_outside_uv2=overlay_outside_uv2,exact_boundary_candidates=exact_candidates,
        outside_predicate='exact_rational_clipping_at_boundary; unchanged input coordinates',
        added_vertices=len(uv)-len(c), angle_bound_passed=measured_angle+settings.angle_atol_deg >= settings.minimum_angle_deg,
        refinement_bound_passed=actual_max_area <= maximum_area*(1+settings.angle_atol_deg))
    return uv, faces, audit


def _section(tri: np.ndarray, distances: np.ndarray, tolerance_mm: float) -> np.ndarray:
    """Intersection of triangle with another triangle's plane."""
    points = [tri[k] for k in range(3) if abs(distances[k]) <= tolerance_mm]
    for k in range(3):
        j = (k+1)%3
        if distances[k]*distances[j] < 0:
            points.append(tri[k]+distances[k]/(distances[k]-distances[j])*(tri[j]-tri[k]))
    return np.asarray(points)


def _legal_contact(points: np.ndarray, shared: np.ndarray, tolerance_mm: float) -> bool:
    if len(shared) == 0:
        return False
    if len(shared) == 1:
        return bool(np.all(np.linalg.norm(points-shared[0], axis=1) <= tolerance_mm))
    edge = shared[1]-shared[0]; length = np.linalg.norm(edge); direction = edge/length
    along = (points-shared[0])@direction
    residual = points-shared[0]-along[:, None]*direction
    return bool(np.all(np.linalg.norm(residual, axis=1) <= tolerance_mm) and
                np.all(along >= -tolerance_mm) and np.all(along <= length+tolerance_mm))


def _pair_intersects(a: np.ndarray, b: np.ndarray, na: np.ndarray, nb: np.ndarray,
                     da: np.ndarray, db: np.ndarray, tolerance_mm: float) -> bool:
    shared = np.asarray([p for p in a if np.any(np.all(p == b, axis=1))]).reshape(-1, 3)
    direction = np.cross(na, nb); magnitude = np.linalg.norm(direction)
    coplanar = np.max(abs(da)) <= tolerance_mm and np.max(abs(db)) <= tolerance_mm
    if coplanar:
        keep = [k for k in range(3) if k != int(np.argmax(abs(na)))]
        hit = Polygon(a[:, keep]).intersection(Polygon(b[:, keep]))
        if hit.is_empty:
            return False
        if len(shared) == 0 or len(shared) == 3:
            return True
        legal = Point(shared[0, keep]) if len(shared) == 1 else LineString(shared[:, keep])
        # A buffer in the projection cannot exceed the declared 3D distance.
        return not hit.difference(legal.buffer(tolerance_mm)).is_empty
    if magnitude <= np.finfo(float).eps:
        return False  # Parallel distinct planes, already separated by distance.
    if len(shared) >= 2:
        return False  # Noncoplanar triangles sharing an edge intersect only on that edge.
    direction /= magnitude
    pa = _section(a, da, tolerance_mm); pb = _section(b, db, tolerance_mm)
    if not len(pa) or not len(pb):
        return False
    aa, bb = pa@direction, pb@direction
    lower, upper = max(aa.min(), bb.min()), min(aa.max(), bb.max())
    if lower > upper+tolerance_mm:
        return False
    anchor = pa[0]; hits = anchor + (np.array([lower, upper])-anchor@direction)[:, None]*direction
    return not _legal_contact(hits, shared, tolerance_mm)


def _exact_noncoplanar_intersection(a: np.ndarray, b: np.ndarray, tolerance_mm: float) -> bool:
    """Confirm a floating positive using exact signs on exported coordinates.

    Near-plane vertices must not be projected onto the plane. Coplanar pairs
    retain the existing predicate; shared contact still uses the same mm bound.
    This only filters floating positives, so previously passed meshes stay passed.
    """
    aa, bb = [[tuple(Fraction(float(v)) for v in p) for p in t] for t in (a, b)]

    def sub(x, y): return tuple(u-v for u, v in zip(x, y))
    def dot(x, y): return sum(u*v for u, v in zip(x, y))
    def cross(x, y):
        return (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0])
    def normal(t): return cross(sub(t[1], t[0]), sub(t[2], t[0]))
    na, nb = normal(aa), normal(bb)
    if all(dot(sub(p, bb[0]), nb) == 0 for p in aa):
        return True  # Exact coplanarity: keep the existing coplanar result.
    shared = np.asarray([p for p in a if np.any(np.all(p == b, axis=1))]).reshape(-1, 3)
    if len(shared) >= 2:
        return False
    hits = set()
    for edge_tri, face, n in ((aa, bb, nb), (bb, aa, na)):
        distances = [dot(sub(p, face[0]), n) for p in edge_tri]
        points = [p for p, d in zip(edge_tri, distances) if d == 0]
        for k in range(3):
            j = (k+1)%3
            if distances[k]*distances[j] < 0:
                fraction = distances[k]/(distances[k]-distances[j])
                points.append(tuple(edge_tri[k][v]+fraction*(edge_tri[j][v]-edge_tri[k][v]) for v in range(3)))
        for p in points:
            if all(dot(cross(sub(face[(k+1)%3], face[k]), sub(p, face[k])), n) >= 0 for k in range(3)):
                hits.add(p)
    if not hits:
        return False
    return not _legal_contact(np.asarray([[float(v) for v in p] for p in hits]), shared, tolerance_mm)


def intersections(triangles_mm: np.ndarray, settings: Settings) -> dict:
    """Sweep AABBs, reject plane-separated pairs, then exact triangle sections.

    All adjacent pairs are inspected too: shared vertices/edges authorize contact
    only on that simplex. Coplanar overlaps and duplicate faces are rejected.
    Floating predicates use the recorded absolute mm tolerance.
    """
    t = np.asarray(triangles_mm, float)
    if t.ndim != 3 or t.shape[1:] != (3, 3) or not np.isfinite(t).all():
        raise ValueError('finite triangle Nx3x3 required')
    normals = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]); norms = np.linalg.norm(normals, axis=1)
    if np.any(norms == 0):
        raise ValueError('3D intersection check cannot accept zero-area faces')
    normals /= norms[:, None]
    low, high = t.min(axis=1), t.max(axis=1); tol = settings.intersection_atol_mm
    order = np.argsort(low[:, 0], kind='stable'); sorted_x = low[order, 0]
    candidates = 0; count = 0; examples = []; narrow = 0
    for position, i in enumerate(order):
        end = np.searchsorted(sorted_x, high[i, 0]+tol, side='right')
        js = order[position+1:end]
        js = js[np.all(low[js, 1:] <= high[i, 1:]+tol, axis=1) & np.all(high[js, 1:] >= low[i, 1:]-tol, axis=1)]
        candidates += len(js)
        if candidates > settings.max_candidate_pairs:
            raise ValueError('3D intersection candidate resource limit exceeded')
        db = (t[js]-t[i, 0])@normals[i]
        da = np.einsum('jkl,jl->jk', t[i][None]-t[js, 0, None], normals[js])
        separate = np.all(db > tol, axis=1) | np.all(db < -tol, axis=1) | np.all(da > tol, axis=1) | np.all(da < -tol, axis=1)
        js,da,db = js[~separate],da[~separate],db[~separate]
        equal = np.all(t[i][None,:,None,:] == t[js,None,:,:], axis=3)
        shared_a,shared_b = equal.any(axis=2),equal.any(axis=1)
        shared_count = shared_a.sum(axis=1)
        coplanar = np.all(abs(da)<=tol,axis=1) & np.all(abs(db)<=tol,axis=1)
        # A noncoplanar shared edge is the entire plane intersection. For a
        # shared vertex, one triangle touching the other plane ONLY there is
        # likewise legal. Do not skip adjacent faces solely by topology.
        legal = (shared_count==2) & ~coplanar
        one_a = np.all(da>=-tol,axis=1) | np.all(da<=tol,axis=1)
        one_b = np.all(db>=-tol,axis=1) | np.all(db<=tol,axis=1)
        only_shared_a = np.all((abs(da)>tol)|shared_a,axis=1)
        only_shared_b = np.all((abs(db)>tol)|shared_b,axis=1)
        legal |= (shared_count==1) & ((one_a & only_shared_a)|(one_b & only_shared_b))
        # Coplanar shared-edge triangles are disjoint in their interiors iff
        # their third vertices lie on opposite sides of the common edge.
        edge_rows = np.flatnonzero(coplanar & (shared_count==2))
        if len(edge_rows):
            third_a = np.argmin(shared_a[edge_rows],axis=1)
            third_b = np.argmin(shared_b[edge_rows],axis=1)
            edge_start = t[i][(third_a+1)%3]; edge_end = t[i][(third_a+2)%3]
            edge = edge_end-edge_start
            side_a = np.cross(edge,t[i][third_a]-edge_start)@normals[i]
            side_b = np.cross(edge,t[js[edge_rows],third_b]-edge_start)@normals[i]
            legal[edge_rows] |= side_a*side_b<0
        for j, distances_a, distances_b in zip(js[~legal], da[~legal], db[~legal]):
            narrow += 1
            if _pair_intersects(t[i], t[j], normals[i], normals[j], distances_a, distances_b, tol) and _exact_noncoplanar_intersection(t[i], t[j], tol):
                count += 1
                if len(examples) < settings.max_intersection_examples:
                    examples.append([int(i), int(j)])
    return dict(self_intersections=count, intersection_examples=examples, candidate_pairs=candidates, narrow_pairs=narrow,
                intersection_atol_mm=tol, self_intersection_status='passed' if count == 0 else 'failed')


def mapped(record: dict, contour: np.ndarray, grid_n: int, settings: Settings,
           *, scale: float = 1., match_area: bool = True) -> tuple[np.ndarray, dict]:
    uv, faces, quality = constrained(contour, grid_n, settings)
    params = deepcopy(record['parameters']); target = record['source_area_mm2']
    matrix = np.asarray(record['canopy_transform']); original_length_mm = params['length_mm']
    for iteration in range(settings.solver_iterations):
        params['length_mm'] = original_length_mm*scale
        vertices = surface_points(uv, params)
        oriented = _orient_adaxial(vertices, faces, np.asarray(params['up']))
        placed = export_vertices(vertices, matrix); tri = placed[oriented]
        areas = triangle_areas_mm2(tri); area_mm2 = float(areas.sum())
        error = abs(area_mm2/target-1)
        if not match_area or error <= settings.solver_rtol:
            break
        scale *= np.sqrt(target/area_mm2)
    else:
        # fmt(4) creates an area staircase. The tighter solver target is an
        # optimization goal, not the contract's acceptance tolerance.
        if error > settings.area_rtol:
            raise ValueError('area matching did not satisfy contract tolerance')
        # The last loop update proposed a scale that has not been evaluated.
        # Retain the scale of the actual accepted exported mesh.
        scale = params['length_mm']/original_length_mm
    raw = (vertices@matrix[:3, :3].T+matrix[:3, 3])[oriented]
    before = np.cross(raw[:, 1]-raw[:, 0], raw[:, 2]-raw[:, 0])
    after = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    up = matrix[:3, :3]@np.asarray(params['up'])
    factor_mm2 = params['length_mm']**2*params['width_scale']
    quality.update(triangles=len(tri), scale=scale, area_mm2=area_mm2, area_relative_error=error,
        iterations=iteration+1, min_triangle_area_mm2=float(areas.min()),
        degenerate_triangles=int((areas < settings.min_triangle_area_mm2).sum()),
        normal_flips=int(((np.einsum('ij,ij->i', before, after) <= 0) | (after@up <= 0)).sum()),
        outside_area_mm2=quality['outside_uv2']*factor_mm2, missing_area_mm2=quality['missing_uv2']*factor_mm2,
        curvature_source='fixture_generator (user_assumed roll/bend)')
    if np.all(areas > 0):
        quality.update(intersections(tri, settings))
    else:
        quality.update(self_intersection_status='unavailable_degenerate_mesh', self_intersections=None)
    failures = []
    for flag in ('angle_bound_passed', 'refinement_bound_passed'):
        if not quality[flag]: failures.append(flag)
    for field in ('degenerate_triangles', 'normal_flips'):
        if quality[field]: failures.append(field)
    if quality['self_intersection_status'] != 'passed': failures.append('3D_self_intersection')
    for field in ('outside_area_mm2', 'missing_area_mm2'):
        if quality[field] > settings.outside_atol_mm2: failures.append(field)
    if match_area and error > settings.area_rtol: failures.append('area_match')
    quality.update(status='supported' if not failures else 'unavailable', reason=';'.join(failures))
    return tri, quality
