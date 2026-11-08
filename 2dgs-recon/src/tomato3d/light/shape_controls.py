"""Report-only leaflet shape controls. Undefined curvature cannot change geometry."""
from dataclasses import asdict, dataclass
import numpy as np
from tomato3d.light.geometry import triangle_areas_mm2


@dataclass(frozen=True)
class CurvatureInput:
    coefficient: float = 0.0
    source_kind: str = 'unavailable'
    definition_ref: str | None = None
    sign_convention: str | None = None
    displacement_model: str = 'quadratic_z_pending_definition'
    area_rescaling: str = 'preserve_full_outline_surface_area'
    area_basis: str = 'full_outline'

    def validate(self) -> None:
        if self != CurvatureInput():
            raise ValueError('O7 curvature definition pending: only zero/unavailable is executable')

    def record(self) -> dict:
        self.validate()
        return asdict(self)


def flat_geometry(triangles_mm: np.ndarray, curvature: CurvatureInput) -> np.ndarray:
    curvature.validate()
    t = np.asarray(triangles_mm, dtype=float)
    if t.ndim != 3 or t.shape[1:] != (3, 3) or not len(t) or not np.isfinite(t).all():
        raise ValueError('finite nonempty triangle mesh required')
    if np.any(triangle_areas_mm2(t) <= 0):
        raise ValueError('degenerate triangle')
    if not np.allclose(t[:, :, 2], 0., rtol=0., atol=1e-10):
        raise ValueError('shape control requires declared flat local XY geometry')
    return t.copy()


def ellipse_same_area_aspect(triangles_mm: np.ndarray, segments: int) -> np.ndarray:
    """Preserve full triangle area and local X/Y extent ratio, not both absolute extents.

    Finite polygon area is normalized exactly. This is an experimental copy only.
    """
    t = flat_geometry(triangles_mm, CurvatureInput())
    if type(segments) is not int or segments < 8 or segments % 4:
        raise ValueError('ellipse segments must be >=8 and divisible by four')
    points = t.reshape(-1, 3)
    span = np.ptp(points, axis=0)
    if min(span[:2]) <= 0:
        raise ValueError('positive length and width required')
    center = (points.min(axis=0)+points.max(axis=0))/2
    angles = np.arange(segments)*2*np.pi/segments
    ring = np.column_stack((span[0]/2*np.cos(angles), span[1]/2*np.sin(angles), np.zeros(segments)))
    fan = np.stack((np.zeros_like(ring), ring, np.roll(ring, -1, axis=0)), axis=1)
    fan *= np.sqrt(triangle_areas_mm2(t).sum()/triangle_areas_mm2(fan).sum())
    # Preserve winding of the supplied local surface.
    if np.cross(t[0, 1]-t[0, 0], t[0, 2]-t[0, 0])[2] < 0:
        fan = fan[:, [0, 2, 1], :]
    return fan+center


def normalize_outline(triangles_mm: np.ndarray, area_mm2: float, length_width_ratio: float) -> np.ndarray:
    """Explicit full-outline area and local X-length/Y-width control; bbox centre anchor."""
    t=flat_geometry(triangles_mm,CurvatureInput())
    if not np.isfinite([area_mm2,length_width_ratio]).all() or min(area_mm2,length_width_ratio)<=0:
        raise ValueError('positive finite area/aspect controls required')
    points=t.reshape(-1,3);span=np.ptp(points,axis=0)
    if min(span[:2])<=0: raise ValueError('positive outline spans required')
    t-=(points.min(axis=0)+points.max(axis=0))/2
    t[:,:,0]*=length_width_ratio/(span[0]/span[1])
    t*=np.sqrt(area_mm2/triangle_areas_mm2(t).sum())
    return t


def triangulate_full_outline(contour_mm: list) -> np.ndarray:
    """Constrained polygon triangulation; refuse invalid outlines instead of repairing them."""
    from shapely import Polygon, constrained_delaunay_triangles
    xy = np.asarray(contour_mm,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.isfinite(xy).all():
        raise ValueError('finite XY outline required')
    polygon = Polygon(xy)
    if not polygon.is_valid or polygon.area<=0 or polygon.interiors:
        raise ValueError('invalid or holed full outline; source review required')
    faces = constrained_delaunay_triangles(polygon)
    result = np.array([np.column_stack((np.asarray(p.exterior.coords)[:3],np.zeros(3))) for p in faces.geoms])
    if not np.isclose(triangle_areas_mm2(result).sum(),polygon.area,atol=1e-9,rtol=1e-10):
        raise ValueError('full outline triangulation area mismatch')
    reverse=np.cross(result[:,1]-result[:,0],result[:,2]-result[:,0])[:,2]<0
    result[reverse]=result[reverse][:,[0,2,1],:]
    return result
