"""Explicit optical experiments around the existing direct-light transport.

Grow coordinates are mm, one-sided geometric areas are m2, photon rates umol/s.
No production caller opts in implicitly. Reflected exchange uses bounded centroid
patch quadrature; the hemisphere background is a declared lumped approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np

from tomato3d.light.raytrace import Receivers, Occluders, direct_light, count_crossings
from tomato3d.light.scene import Emitters
from tomato3d.io.schemas import LightSimulationSettings
from tomato3d.light.geometry import sample_triangles, triangle_areas_mm2, triangle_unit_normals


def finite(value: object, name: str, low: float = 0., high: float = math.inf) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(name + ': explicit finite value required')
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(name + ': value outside bounds')
    return number


@dataclass(frozen=True)
class OpticalOptions:
    optics_model: str = 'black'
    leaf_tau: float | None = None
    leaf_rho: float | None = None
    surface_reflectance: dict | None = None
    indirect_model: str = 'off'
    diffuse_ppfd_umol_m2_s: float | None = None
    ceiling: dict | None = None
    source_kind: str | None = None
    source_ref: str | None = None

    def validate(self) -> None:
        if self.optics_model not in ('black', 'transmissive'):
            raise ValueError('unsupported optics_model')
        if self.indirect_model not in ('off', 'uniform_diffuse', 'single_bounce_ceiling'):
            raise ValueError('unsupported indirect_model')
        enabled = self.optics_model != 'black' or self.indirect_model != 'off'
        if enabled and (self.source_kind not in ('user_assumed', 'literature_generic', 'measured') or not self.source_ref):
            raise ValueError('explicit optical provenance required')
        if self.optics_model == 'transmissive':
            tau = finite(self.leaf_tau, 'leaf_tau', 0, 1)
            rho = finite(self.leaf_rho, 'leaf_rho', 0, 1)
            if tau + rho > 1:
                raise ValueError('tau + rho exceeds one')
        elif self.leaf_tau is not None or self.leaf_rho is not None:
            raise ValueError('black options must not hide supplied leaf optics')
        for kind, value in (self.surface_reflectance or {}).items():
            if kind not in ('soil', 'tray', 'wall', 'ceiling'):
                raise ValueError('unknown surface material')
            if value is not None:
                finite(value, kind + '_reflectance', 0, 1)
                if value and self.optics_model == 'black' and not (kind == 'ceiling' and self.indirect_model == 'single_bounce_ceiling'):
                    raise ValueError('nonblack material requires transmissive optics_model')
        if self.indirect_model == 'uniform_diffuse':
            finite(self.diffuse_ppfd_umol_m2_s, 'diffuse_ppfd')
        elif self.diffuse_ppfd_umol_m2_s is not None:
            raise ValueError('inactive diffuse value supplied')
        if self.indirect_model == 'single_bounce_ceiling':
            if self.ceiling is None:
                raise ValueError('ceiling geometry unavailable')
            for name in ('height_mm', 'width_mm', 'depth_mm'):
                finite(self.ceiling.get(name), name, 1e-12)
            c = np.asarray(self.ceiling.get('center_xy_mm'), float)
            if c.shape != (2,) or not np.isfinite(c).all():
                raise ValueError('ceiling center unavailable')
            finite((self.surface_reflectance or {}).get('ceiling'), 'ceiling_reflectance', 0, 1)
        elif self.ceiling is not None:
            raise ValueError('inactive ceiling geometry supplied')

    @property
    def tau(self) -> float:
        self.validate()
        return 0. if self.optics_model == 'black' else float(self.leaf_tau)

    @property
    def rho(self) -> float:
        self.validate()
        return 0. if self.optics_model == 'black' else float(self.leaf_rho)


def default_emitters() -> list[dict]:
    return [dict(fixture_id='L1', position_grow_mm=[0.,100.,700.],
                 photon_flux_umol_s=1200., distribution='isotropic',
                 source_kind='user_assumed', source_ref='explicit_hypothetical_point_LED')]


def expand_emitters(items: list[dict] | None) -> list[dict]:
    """Expand declared bars without changing total PPF or inventing optical inputs."""
    items = default_emitters() if items is None else items
    if not items:
        return []  # explicitly lamp-free, useful for background-only controls
    result = []
    ids = set()
    for item in items:
        fid = item['fixture_id']
        if not isinstance(fid, str) or not fid or fid in ids:
            raise ValueError('unique fixture IDs required')
        ids.add(fid)
        if not item.get('source_kind') or not item.get('source_ref'):
            raise ValueError('emitter provenance required')
        power = finite(item.get('photon_flux_umol_s'), 'photon_flux_umol_s')
        position = np.asarray(item.get('position_grow_mm'), float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('emitter position unavailable')
        distribution = item.get('distribution')
        if distribution not in ('isotropic', 'cosine_power'):
            raise ValueError('explicit supported emitter distribution required')
        copied = dict(item, photon_flux_umol_s=power)
        if distribution == 'cosine_power':
            aim = np.asarray(item.get('aim'), float)
            if aim.shape != (3,) or not np.isfinite(aim).all() or np.linalg.norm(aim) == 0:
                raise ValueError('emitter aim unavailable')
            half = finite(item.get('half_intensity_half_angle_deg'), 'half angle', 1e-9, 89.999999)
            cutoff = finite(item.get('cutoff_half_angle_deg'), 'cutoff', half, 90.)
            copied.update(aim=(aim / np.linalg.norm(aim)).tolist(),
                          cosine_exponent=-math.log(2)/math.log(math.cos(math.radians(half))),
                          cos_cutoff=math.cos(math.radians(cutoff)))
        if item.get('kind', 'point') == 'point':
            result.append(copied)
        elif item['kind'] == 'bar':
            length = finite(item.get('length_mm'), 'length_mm', 1e-12)
            axis = np.asarray(item.get('long_axis'), float)
            count = item.get('point_count')
            if type(count) is not int or not 1 <= count <= 4096 or axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
                raise ValueError('explicit bar discretization and axis required')
            axis /= np.linalg.norm(axis)
            for i in range(count):
                result.append(dict(copied, fixture_id=f'{fid}:{i}', parent_fixture_id=fid, kind='point',
                    position_grow_mm=(position + axis * length * ((i+.5)/count-.5)).tolist(),
                    photon_flux_umol_s=power/count))
        else:
            raise ValueError('unknown emitter shape')
    return result


def emitter_states(items: list[dict] | None) -> tuple:
    states = []
    for item in expand_emitters(items):
        p = item['photon_flux_umol_s']
        if item['distribution'] == 'isotropic':
            # direct_light's legacy hemisphere gate is unchanged. Two disjoint
            # hemispheres implement a complete isotropic fixture in this adapter.
            components = [([0,0,1],0.,0.,p/(4*math.pi),'up'), ([0,0,-1],0.,0.,p/(4*math.pi),'down')]
        else:
            m, cutoff = item['cosine_exponent'], item['cos_cutoff']
            components = [(item['aim'],m,cutoff,p*(m+1)/(2*math.pi*(1-cutoff**(m+1))), 'cos')]
        for aim,m,cutoff,intensity,suffix in components:
            fid = item['fixture_id'] + '/' + suffix
            em = Emitters(fid,np.array([item['position_grow_mm']],float),np.array(aim,float),intensity,m,cutoff)
            states.append(SimpleNamespace(fixture_id=fid,emitters=em))
    return tuple(states)


def hemisphere(n_mu: int, n_phi: int) -> tuple[np.ndarray,np.ndarray]:
    if type(n_mu) is not int or type(n_phi) is not int or min(n_mu,n_phi) < 2:
        raise ValueError('positive explicit hemisphere quadrature >=2 required')
    mu,w = np.polynomial.legendre.leggauss(n_mu); mu=(mu+1)/2; w=w/2
    phi=2*math.pi*(np.arange(n_phi)+.5)/n_phi
    z=np.repeat(mu,n_phi); a=np.tile(phi,n_mu)
    directions=np.column_stack((np.sqrt(1-z*z)*np.cos(a),np.sqrt(1-z*z)*np.sin(a),z))
    return directions, np.repeat(w,n_phi)*2/n_phi  # radiance D/pi times dOmega


def uniform_background(receivers: Receivers, occluders: Occluders, options: OpticalOptions,
                       settings: LightSimulationSettings, n_mu: int, n_phi: int) -> tuple[np.ndarray,np.ndarray]:
    options.validate()
    value=finite(options.diffuse_ppfd_umol_m2_s,'diffuse_ppfd')
    total=np.zeros(len(receivers.points_grow_mm)); upper=total.copy()
    if value == 0:
        return total,upper
    dirs,weights=hemisphere(n_mu,n_phi)
    all_points=np.concatenate((receivers.points_grow_mm,occluders.vertices_grow_mm.reshape(-1,3)))
    escape_mm=max(1.,float(np.linalg.norm(np.ptp(all_points,axis=0))*2+1))
    for direction,weight in zip(dirs,weights):
        segments=np.broadcast_to(direction*escape_mm,receivers.points_grow_mm.shape)
        origins=receivers.points_grow_mm+direction*settings.ray_offset_mm
        n,blocked=count_crossings(origins,segments,receivers.own_triangle,occluders,settings.max_ray_triangle_pairs_per_chunk)
        transmission=np.where(blocked,0.,np.power(options.tau,n))
        cosine=receivers.unit_normals@direction
        front=np.maximum(cosine,0.)*value*weight*transmission
        upper+=front
        total+=(np.abs(cosine) if receivers.two_sided else np.maximum(cosine,0.))*value*weight*transmission
    return total,upper


def incident_fields(total: np.ndarray, upper: np.ndarray, area_m2: np.ndarray, absorptance: float) -> dict:
    """A is one-sided area; two-sided PPFD is (upper+lower flux)/A, never /2A."""
    total,upper,area_m2=np.broadcast_arrays(total,upper,area_m2)
    if np.any(area_m2 <= 0) or not np.isfinite(area_m2).all():
        raise ValueError('positive supported one-sided area required')
    return dict(incident_upper_ppfd_umol_m2_s=upper,incident_lower_ppfd_umol_m2_s=total-upper,
        incident_two_sided_ppfd_umol_m2_s=total,intercepted_umol_s=total*area_m2,
        absorbed_umol_s=total*area_m2*absorptance)


def surface_receivers(triangles_mm: np.ndarray, subdivisions: int) -> tuple[Receivers,np.ndarray]:
    points,indices=sample_triangles(triangles_mm,subdivisions)
    normals=triangle_unit_normals(triangles_mm)[indices]
    weights=triangle_areas_mm2(triangles_mm)[indices]*1e-6/subdivisions**2
    return Receivers(points,normals,True,indices),weights


def direct_observation(receivers: Receivers, fixtures: tuple, occluders: Occluders,
                       leaf_transmittance: float | None, settings: LightSimulationSettings,
                       area_weights_m2: np.ndarray, absorptance: float | None = None) -> dict:
    """Observe existing direct transport exactly, retaining unknowns and originals."""
    legacy=direct_light(receivers,fixtures,occluders,leaf_transmittance,settings)
    total=legacy.ppfd_umol_m2_s.sum(axis=0)
    upper=legacy.upper_side_ppfd_umol_m2_s.sum(axis=0)
    fields=incident_fields(total,upper,area_weights_m2,np.nan if absorptance is None else absorptance)
    return dict(legacy=legacy,fields=fields)


def ceiling_sources(items: list[dict] | None, occluders: Occluders, options: OpticalOptions,
                    settings: LightSimulationSettings, grid: int) -> tuple[list[dict],dict]:
    """One lamp->ceiling->receiver bounce, Lambertian patch quadrature.

    Only direct lamp irradiance powers the ceiling. No fitted diffuse fraction,
    repeated ceiling bounce or illumination of the ceiling by reflected leaves.
    """
    options.validate()
    if options.indirect_model!='single_bounce_ceiling' or type(grid) is not int or grid<2:
        raise ValueError('explicit ceiling model and patch grid required')
    c=options.ceiling;rho=options.surface_reflectance['ceiling']
    u=(np.arange(grid)+.5)/grid-.5;x,y=np.meshgrid(u,u)
    points=np.column_stack((c['center_xy_mm'][0]+x.ravel()*c['width_mm'],
                           c['center_xy_mm'][1]+y.ravel()*c['depth_mm'],
                           np.full(grid*grid,c['height_mm'])))
    if occluders.vertices_grow_mm.size and np.max(occluders.vertices_grow_mm[:,:,2])>=c['height_mm']:
        raise ValueError('ceiling must lie above all supplied plant surfaces')
    rec=Receivers(points,np.tile([0,0,-1],(len(points),1)),False,np.full(len(points),-1))
    light=direct_light(rec,emitter_states(items),occluders,options.tau,settings).ppfd_umol_m2_s.sum(axis=0)
    area=c['width_mm']*c['depth_mm']*1e-6/(grid*grid)
    sources=[dict(fixture_id=f'ceiling:{i}',position_grow_mm=(p+[0,0,-settings.ray_offset_mm]).tolist(),
        photon_flux_umol_s=float(q*area*rho),distribution='cosine_power',aim=[0,0,-1],
        half_intensity_half_angle_deg=60.,cutoff_half_angle_deg=90.,source_kind=options.source_kind,
        source_ref=options.source_ref) for i,(p,q) in enumerate(zip(points,light))]
    return sources,dict(model='single_bounce_lambertian_ceiling_patch_quadrature',grid=grid,
        direct_incident_ceiling_umol_s=float(light.sum()*area),reflected_umol_s=float(light.sum()*area*rho),
        repeated_ceiling_reflection=False)


def surface_light(triangles_mm: np.ndarray, surface_codes: np.ndarray, items: list[dict] | None,
                  options: OpticalOptions, settings: LightSimulationSettings, *, subdivisions: int = 1,
                  hemisphere_grid: tuple[int,int] = (8,16), ceiling_grid: int = 8,
                  max_patch_pairs: int = 1_000_000, reflection_iterations: int = 64,
                  reflection_rtol: float = 1e-9, background: tuple[dict,...] = ()) -> dict:
    """Direct transport plus bounded diffuse-reflection exchange on the same mesh.

    Reflecting patches use existing segment tests, ballistic leaf transmission and
    centroid Lambertian quadrature. It is an opt-in approximation, not a new tracer.
    No geometry is flattened or discarded. Refuse a dense matrix over its budget.
    """
    options.validate()
    tri=np.asarray(triangles_mm,float)
    if tri.ndim!=3 or tri.shape[1:]!=(3,3) or not np.isfinite(tri).all():raise ValueError('finite triangles required')
    supplied_codes=np.asarray(surface_codes)
    if supplied_codes.shape!=(len(tri),) or not np.issubdtype(supplied_codes.dtype,np.integer):
        raise ValueError('one integer physical-surface code per triangle required')
    leaf_count=len(tri); codes=list(surface_codes);rho_values=[options.rho]*leaf_count
    used_materials=set()
    for i,surface in enumerate(background):
        kind=surface['material'];used_materials.add(kind)
        vertices=np.asarray(surface['vertices_grow_mm'],float).reshape(-1,3,3)
        if not np.isfinite(vertices).all():raise ValueError('invalid background geometry')
        rho=0. if options.optics_model=='black' or kind=='probe' else finite((options.surface_reflectance or {}).get(kind),kind+'_reflectance',0,1)
        tri=np.concatenate((tri,vertices));codes.extend([max(codes,default=-1)+1]*len(vertices));rho_values.extend([rho]*len(vertices))
    unused={k for k,v in (options.surface_reflectance or {}).items() if v not in (None,0) and k not in used_materials}
    if options.indirect_model=='single_bounce_ceiling':unused.discard('ceiling')
    if unused:raise ValueError('nonzero material has no geometry: '+','.join(sorted(unused)))
    if any(rho_values) and (2*len(tri))**2>max_patch_pairs:
        raise ValueError('reflection patch-pair budget exceeded before transport')
    areas=triangle_areas_mm2(tri)*1e-6
    if np.any(areas<=0):raise ValueError('nonpositive triangle area')
    leaf=np.arange(len(tri))<leaf_count
    occ=Occluders(tri,leaf,np.asarray(codes,int),np.zeros(len(tri),int))
    rec,weights=surface_receivers(tri,subdivisions)
    fixtures=emitter_states(items)
    bounce={}
    if options.indirect_model=='single_bounce_ceiling':
        sources,bounce=ceiling_sources(items,occ,options,settings,ceiling_grid)
        fixtures+=emitter_states(sources)
    light=direct_light(rec,fixtures,occ,options.tau,settings)
    total=light.ppfd_umol_m2_s.sum(axis=0);upper=light.upper_side_ppfd_umol_m2_s.sum(axis=0)
    if options.indirect_model=='uniform_diffuse':
        d,u=uniform_background(rec,occ,options,settings,*hemisphere_grid);total+=d;upper+=u
    # Reflection uses one patch per side of each triangle. Quadrature resolution
    # must be tested independently of Monte Carlo ray-count convergence.
    rho=np.asarray(rho_values)
    reflection=dict(model='off',iterations=0)
    if np.any(rho):
        n=len(tri); count=2*n
        if count*count>max_patch_pairs:raise ValueError('reflection patch-pair budget exceeded')
        normal=triangle_unit_normals(tri);center=tri.mean(axis=1)
        patch_rec=Receivers(np.concatenate((center,center)),np.concatenate((normal,-normal)),False,np.tile(np.arange(n),2))
        base=np.concatenate((upper.reshape(n,-1).mean(1),(total-upper).reshape(n,-1).mean(1)))
        matrix=np.zeros((count,count)); patch_area=np.tile(areas,2)
        for j in range(count):
            fid=f'reflect:{j}';aim=patch_rec.unit_normals[j];p=patch_rec.points_grow_mm[j]+aim*settings.ray_offset_mm
            em=Emitters(fid,np.array([p]),aim,1/math.pi,1.,0.)
            matrix[:,j]=direct_light(patch_rec,(SimpleNamespace(fixture_id=fid,emitters=em),),occ,options.tau,settings).ppfd_umol_m2_s[0]*patch_area[j]*rho[j%n]
        returned_power=(matrix*patch_area[:,None]).sum(axis=0)
        emitted_power=patch_area*np.tile(rho,2)
        if np.any(returned_power>emitted_power*(1+1e-6)+1e-12):
            raise ValueError('centroid reflection quadrature violates energy budget; refine mesh')
        value=base.copy()
        for step in range(reflection_iterations):
            new=base+matrix@value
            residual=float(np.max(np.abs(new-value)))
            value=new
            if residual<=reflection_rtol*max(1.,float(np.max(value))):break
        else:raise ValueError('reflection exchange did not converge')
        correction=value-base
        upper+=np.repeat(correction[:n],subdivisions**2)
        total+=np.repeat(correction[:n]+correction[n:],subdivisions**2)
        reflection=dict(model='centroid_lambertian_exchange',iterations=step+1,residual_ppfd=residual)
    alpha=np.repeat(1-options.tau*leaf-rho,subdivisions**2)
    fields=incident_fields(total,upper,weights,alpha)
    return dict(fields=fields,weights_m2=weights,receiver_triangle=rec.own_triangle,
                receiver_is_leaf=leaf[rec.own_triangle],reflection=reflection,ceiling=bounce)


def spectral_bounds(rows: list[dict]) -> list[dict]:
    keys=('r_600_700','g_500_600','b_400_500')
    opened=[r for r in rows if r['condition']=='open']
    if len(opened)!=1: raise ValueError('one open spectrum required')
    baseline=np.array([finite(opened[0][k],k,1e-12) for k in keys])
    out=[]
    for row in rows:
        if row['condition']=='open':continue
        covered=np.array([finite(row[k],k,1e-12) for k in keys])
        ratio=covered/baseline
        if np.any(ratio>1) or not ratio[1]>ratio[2]:raise ValueError('inconsistent two-path band ordering')
        gmax=float(ratio.min()); total=float(covered.sum()/baseline.sum())
        denominator=float(ratio[2]-gmax)
        out.append(dict(pair=row['pair'],gap_fraction_min=0.,gap_fraction_max=gmax,
            transmitted_fraction_min=1-gmax/total,transmitted_fraction_max=1.,
            tau_g_over_tau_b_min=float(ratio[1]/ratio[2]),
            tau_g_over_tau_b_max=None if denominator==0 else float((ratio[1]-gmax)/denominator),
            upper_bound_status='unbounded' if denominator==0 else 'finite',
            measurement_uncertainty='unavailable',source_kind='conditional_two_path_bounds'))
    return out


def horizontal_probe(position_mm: list, diameter_mm: float, segments: int = 64) -> tuple[np.ndarray, float]:
    """Area-matched disk mesh, +Z front, grow mm; no inferred meter dimensions."""
    p = np.asarray(position_mm, float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('probe xyz unavailable')
    diameter = finite(diameter_mm, 'probe diameter', 1e-12)
    if type(segments) is not int or not 8<=segments<=4096:
        raise ValueError('disk segments must be in [8,4096]')
    radius=diameter/2
    ring_radius=radius*math.sqrt(2*math.pi/(segments*math.sin(2*math.pi/segments)))
    angle=np.arange(segments+1)*2*math.pi/segments
    ring=p+np.column_stack((ring_radius*np.cos(angle),ring_radius*np.sin(angle),np.zeros_like(angle)))
    return np.array([[p,ring[k],ring[k+1]] for k in range(segments)]),math.pi*radius**2*1e-6


def probe_light(position_mm: list, diameter_mm: float, items: list[dict] | None,
                occluders: Occluders, options: OpticalOptions, settings: LightSimulationSettings,
                *, segments: int = 64, subdivisions: int = 4, hemisphere_grid: tuple = (8,16),
                ceiling_grid: int = 8, background: tuple[dict,...] = ()) -> dict:
    """Separate cosine probe observation: does not shade the plant calculation.

    The reflective case observes a separate scene containing the black sensor.
    Rig materials require explicit background geometry; untyped nonleaf
    occluders cannot silently be assigned reflective material properties.
    """
    options.validate()
    tri, area = horizontal_probe(position_mm, diameter_mm, segments)
    if options.rho or background or any(v for k,v in (options.surface_reflectance or {}).items() if k != 'ceiling'):
        if not np.all(occluders.is_leaf):
            raise ValueError('reflective probe requires explicit material-tagged background geometry')
        result=surface_light(occluders.vertices_grow_mm,occluders.surface_code,items,options,settings,
            subdivisions=subdivisions,hemisphere_grid=hemisphere_grid,ceiling_grid=ceiling_grid,
            background=tuple(background)+({'material':'probe','vertices_grow_mm':tri},))
        first_probe=len(occluders.vertices_grow_mm)+sum(len(np.asarray(b['vertices_grow_mm']).reshape(-1,3,3)) for b in background)
        select=result['receiver_triangle']>=first_probe
        ppfd=float(result['fields']['incident_upper_ppfd_umol_m2_s'][select] @ result['weights_m2'][select]/area)
        return dict(incident_ppfd_umol_m2_s=ppfd,area_m2=area,position_grow_mm=list(map(float,position_mm)),
                    diameter_mm=float(diameter_mm),response='horizontal_cosine',segments=segments,subdivisions=subdivisions,
                    reflection=result['reflection'])
    rec, weights = surface_receivers(tri, subdivisions)
    rec = Receivers(rec.points_grow_mm, rec.unit_normals, False, np.full(len(weights), -1))
    fixtures = emitter_states(items)
    if options.indirect_model == 'single_bounce_ceiling':
        bounce, _ = ceiling_sources(items, occluders, options, settings, ceiling_grid)
        fixtures += emitter_states(bounce)
    flux = direct_light(rec, fixtures, occluders, options.tau, settings).ppfd_umol_m2_s.sum(axis=0)
    if options.indirect_model == 'uniform_diffuse':
        diffuse, _ = uniform_background(rec, occluders, options, settings, *hemisphere_grid)
        flux += diffuse
    return dict(incident_ppfd_umol_m2_s=float(flux @ weights / area), area_m2=area,
                position_grow_mm=list(map(float,position_mm)), diameter_mm=float(diameter_mm),
                response='horizontal_cosine', segments=segments, subdivisions=subdivisions)
