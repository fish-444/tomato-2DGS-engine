"""Default-preserving GroIMP adapter for explicit P2 optical experiments."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import csv
import json
import math
import re
import numpy as np

from tomato3d.light.revision import OpticalOptions, expand_emitters, hemisphere, default_emitters, ceiling_sources, horizontal_probe
from tomato3d.light.raytrace import Occluders
from tomato3d.io.schemas import LightSimulationSettings
from tools.p2_d3_optical import model as legacy_model
from tools.ppfd_validation import PHONG

OBSERVATION = r'''
import de.grogra.ray.physics.Spectrum;
import de.grogra.ray.physics.Spectrum3d;
import de.grogra.ray.physics.Collector;
import de.grogra.ray.physics.Collector3d;
import de.grogra.ray.physics.Environment;
import de.grogra.ray.util.RayList;
import javax.vecmath.Vector3f;
import javax.vecmath.Tuple3d;
import java.util.Random;

class FaceTag {
    static ThreadLocal side = new ThreadLocal();
    static ThreadLocal ratio = new ThreadLocal();
}
class FaceSpectrum extends Collector3d {
    double front=0, back=0;
    double incomingFront=0, incomingBack=0;
    public FaceSpectrum newInstance() {return new FaceSpectrum();}
    public FaceSpectrum clone() {
        FaceSpectrum c=new FaceSpectrum(); c.x=x; c.y=y; c.z=z;
        c.front=front; c.back=back; c.incomingFront=incomingFront; c.incomingBack=incomingBack;
        c.asCollector=asCollector; return c;
    }
    public void setZero() {super.setZero();front=0;back=0;incomingFront=0;incomingBack=0;}
    public void add(Spectrum s) {
        Spectrum3d c=(Spectrum3d)s; x+=c.x;y+=c.y;z+=c.z;
        if(s instanceof FaceSpectrum && ((FaceSpectrum)s).isCollector()) {
            front+=((FaceSpectrum)s).front;back+=((FaceSpectrum)s).back;
            incomingFront+=((FaceSpectrum)s).incomingFront;incomingBack+=((FaceSpectrum)s).incomingBack;
        }
    }
    public void add(Collector s) {add((Spectrum)s);}
    public double frontPower() {return front+back==0 ? 0 : integrate()*front/(front+back);}
    public double backPower() {return front+back==0 ? 0 : integrate()*back/(front+back);}
    public double incomingFrontPower() {return front+back==0 ? 0 : integrate()*incomingFront/(front+back);}
    public double incomingBackPower() {return front+back==0 ? 0 : integrate()*incomingBack/(front+back);}
    public void addToStatistic(Tuple3d direction,Spectrum s,double factor,boolean primary) {
        double weight=s.integrate()*factor;
        if(Boolean.TRUE.equals(FaceTag.side.get()))front+=weight;else back+=weight;
        double ratio=FaceTag.ratio.get()==null ? 0 : ((Double)FaceTag.ratio.get()).doubleValue();
        if(Boolean.TRUE.equals(FaceTag.side.get()))incomingFront+=weight*ratio;else incomingBack+=weight*ratio;
    }
    public void addToStatistic(Tuple3d direction,Spectrum3d s,double factor,boolean primary) {
        addToStatistic(direction,(Spectrum)s,factor,primary);
    }
}
class ObservedShader extends Phong {
    de.grogra.imp3d.shading.Shader original;
    ObservedShader(de.grogra.imp3d.shading.Shader s) {original=s;}
    public int getFlags() {return original.getFlags() | de.grogra.ray.physics.Scattering.NEEDS_NORMAL;}
    public void generateRandomRays(Environment env,Vector3f out,Spectrum input,RayList rays,boolean adjoint,Random random) {
        FaceTag.side.set(Boolean.valueOf(env.normal.dot(out)>=0));
        double incoming=input.integrate();
        original.generateRandomRays(env,out,input,rays,adjoint,random);
        double absorbed=incoming-rays.lastRay().spectrum.integrate();
        FaceTag.ratio.set(Double.valueOf(absorbed>0 ? incoming/absorbed : 0));
    }
}
'''

COSINE_LIGHT = r'''
class CosinePointLight extends PointLight {
    double exponent, cutoff;
    CosinePointLight(double m,double c) {exponent=m;cutoff=c;}
    public void generateRandomRays(de.grogra.ray.physics.Environment env,javax.vecmath.Vector3f out,
        de.grogra.ray.physics.Spectrum input,de.grogra.ray.util.RayList rays,boolean adjoint,java.util.Random random) {
        for(int i=rays.getSize()-1;i>=0;i--) {
            double base=Math.pow(cutoff,exponent+1);
            double mu=Math.pow(base+(1-base)*random.nextDouble(),1/(exponent+1));
            double phi=2*Math.PI*random.nextDouble(); double sin=Math.sqrt(1-mu*mu);
            de.grogra.ray.util.Ray r=rays.rays[i];
            r.direction.set((float)(sin*Math.cos(phi)),(float)(sin*Math.sin(phi)),(float)mu);
            env.localToGlobal.transform(r.direction);r.direction.normalize();
            r.spectrum.set(input);
            r.directionDensity=(float)((exponent+1)*Math.pow(mu,exponent)/(2*Math.PI*(1-base)));
        }
    }
}
class HemisphericLight extends de.grogra.imp3d.objects.DirectionalLight {
    double irradiance;
    public HemisphericLight(double value) {irradiance=value;}
    public double getTotalPower(Environment env) {
        return 2*Math.PI*env.boundsRadius*env.boundsRadius*irradiance;
    }
    public void generateRandomOrigins(Environment env,RayList rays,Random random) {
        double power=getTotalPower(env);
        for(int i=0;i<rays.getSize();i++) {
            de.grogra.ray.util.Ray r=rays.rays[i];
            r.origin.set(env.boundsCenter);
            r.spectrum.set(new Vector3f(1,1,1));r.spectrum.scale(power/3);
            r.originDensity=(float)(1/(Math.PI*env.boundsRadius*env.boundsRadius));
        }
    }
    public void generateRandomRays(Environment env,Vector3f out,Spectrum input,RayList rays,boolean adjoint,Random random) {
        for(int i=0;i<rays.getSize();i++) {
            double mu=random.nextDouble(),phi=2*Math.PI*random.nextDouble();
            double sine=Math.sqrt(1-mu*mu);
            javax.vecmath.Vector3d sky=new javax.vecmath.Vector3d(sine*Math.cos(phi),sine*Math.sin(phi),mu);
            javax.vecmath.Vector3d x=new javax.vecmath.Vector3d();
            x.cross(new javax.vecmath.Vector3d(0,1,0),sky);x.normalize();
            javax.vecmath.Vector3d y=new javax.vecmath.Vector3d();y.cross(sky,x);
            double radius=env.boundsRadius*Math.sqrt(random.nextDouble()),angle=2*Math.PI*random.nextDouble();
            double a=radius*Math.cos(angle),b=radius*Math.sin(angle),height=env.boundsRadius+0.0001;
            de.grogra.ray.util.Ray r=rays.rays[i];
            r.origin.set((float)(env.boundsCenter.x+height*sky.x+a*x.x+b*y.x),
                         (float)(env.boundsCenter.y+height*sky.y+a*x.y+b*y.y),
                         (float)(env.boundsCenter.z+height*sky.z+a*x.z+b*y.z));
            r.direction.set((float)-sky.x,(float)-sky.y,(float)-sky.z);
            r.spectrum.set(input);r.directionDensity=(float)(1/(2*Math.PI));
        }
    }
}
class CeilingPatchLight extends CosinePointLight {
    double[] coordinates, cumulative;
    double total;
    public CeilingPatchLight(double[] xyz,double[] powers) {
        super(1,0);coordinates=xyz;cumulative=new double[powers.length];total=0;
        for(int i=0;i<powers.length;i++){total+=powers[i];cumulative[i]=total;}
    }
    public void generateRandomOrigins(Environment env,RayList rays,Random random) {
        for(int j=0;j<rays.getSize();j++) {
            double draw=random.nextDouble()*total;int k=0;
            while(k<cumulative.length-1 && draw>=cumulative[k])k++;
            de.grogra.ray.util.Ray r=rays.rays[j];
            r.origin.set((float)coordinates[3*k],(float)coordinates[3*k+1],(float)coordinates[3*k+2]);
            r.spectrum.set(new Vector3f(1,1,1));r.spectrum.scale(getPower()/3.0);
            r.originDensity=DELTA_FACTOR;
        }
    }
}
'''


def transform(position_mm: list, aim: list) -> str:
    z=np.array(aim,float);z/=np.linalg.norm(z)
    x=np.cross([0,1,0] if abs(z[1])<.9 else [1,0,0],z);x/=np.linalg.norm(x)
    y=np.cross(z,x);mat=np.eye(4);mat[:3,:3]=np.column_stack((x,y,z));mat[:3,3]=np.asarray(position_mm)/1000
    return 'Null().(setTransform(new javax.vecmath.Matrix4d('+','.join(format(v,'.17g') for v in mat.ravel())+')))'


def lamp_nodes(items: list[dict]) -> str:
    nodes=[]
    for item in expand_emitters(items):
        p=item['position_grow_mm'];power=item['photon_flux_umol_s']
        if item['distribution']=='isotropic':
            nodes.append(f'[ Translate({p[0]/1000},{p[1]/1000},{p[2]/1000}) Lamp().(setLight(new PointLight().(setPower({power}f)))) ]')
        else:
            nodes.append('[ '+transform(p,item['aim'])+f' Lamp().(setLight(new Revision.CosinePointLight({item["cosine_exponent"]},{item["cos_cutoff"]}).(setPower({power}f)))) ]')
    return ' '.join(nodes)


def ceiling_node(sources: list[dict]) -> str:
    """Flux-weighted patch mixture in one source, using native transport."""
    sources=[s for s in sources if s['photon_flux_umol_s']>0]
    if not sources:return ''
    positions=','.join(format(v/1000,'.17g') for s in sources for v in s['position_grow_mm'])
    powers=','.join(format(s['photon_flux_umol_s'],'.17g') for s in sources)
    total=sum(s['photon_flux_umol_s'] for s in sources)
    return '[ '+transform([0,0,0],[0,0,-1])+f' Lamp().(setLight(new Revision.CeilingPatchLight(new double[]{{{positions}}},new double[]{{{powers}}}).(setPower({total}f)))) ]'


def model(geometry: Path, rays: int, seed: int, *, options: OpticalOptions | None = None,
          diagnostics: bool = False, emitters: list[dict] | None = None,
          hemisphere_grid: tuple[int,int] = (8,16), ceiling_grid: int = 8,
          background: tuple[dict,...] = (), probe: bool = False,
          sensor: dict | None = None) -> str:
    """Disabled adapter returns the legacy model byte-for-byte.

    Incident diagnostics wrap the original shader without changing its sampling,
    and accumulate face tags in the existing LightModel spectrum reduction.
    Existing LEAF rows remain unchanged and retain their first-hit/absorbed meanings.
    """
    options=options or OpticalOptions();options.validate()
    legacy_lamps=[dict(x_m=0.,y_m=.1,z_m=.7,power_umol_s=1200.)]
    text=legacy_model(geometry,legacy_lamps,rays,seed)
    if options==OpticalOptions() and not diagnostics and emitters is None and not background and not probe and sensor is None:
        return text
    if options.indirect_model == 'single_bounce_ceiling':
        triangles=[];codes=[];leaf=[]
        for i,row in enumerate(csv.reader(geometry.open(encoding='utf-8'))):
            t=np.asarray(row[1:],float).reshape(-1,3,3)*1000
            triangles.extend(t);codes.extend([i]*len(t));leaf.extend([True]*len(t))
        for i,surface in enumerate(background):
            t=np.asarray(surface['vertices_grow_mm'],float).reshape(-1,3,3)
            triangles.extend(t);codes.extend([len(codes)+i]*len(t));leaf.extend([False]*len(t))
        occ=Occluders(np.asarray(triangles),np.asarray(leaf),np.asarray(codes),np.zeros(len(codes),int))
        bounce,_=ceiling_sources(emitters,occ,options,LightSimulationSettings(triangle_subdivisions=1, ray_offset_mm=.01, max_ray_triangle_pairs_per_chunk=1_000_000),ceiling_grid)
        text=text.replace('Axiom ==>','Axiom ==> '+ceiling_node(bounce))
    if emitters is not None:
        old='[ Translate(0.0,0.1,0.7) Lamp().(setLight(new PointLight().(setPower(1200.0f)))) ]'
        if old not in text:raise ValueError('legacy lamp insertion anchor changed')
        text=text.replace(old,lamp_nodes(emitters))
    if options.optics_model=='transmissive':
        # Straight-through tau (IOR1 in a surface without a medium), diffuse rho.
        material=PHONG.replace('@NAME@','revisionOptics').replace('@R@',repr(options.rho)).replace('@T@','0.0')
        material=material.replace('setTransparency(new Graytone(0))',f'setTransparency(new Graytone({options.tau}))')
        text=text.replace('protected void init()',material+'\nprotected void init()')
        text=text.replace('setShader(BLACK)','setShader(revisionOptics)')
        text=text.replace(f'new LightModel({rays}, 3)',f'new LightModel({rays}, 32)')
    used=set()
    for index,surface in enumerate(background):
        kind=surface['material'];used.add(kind)
        rho=0. if options.optics_model=='black' else (options.surface_reflectance or {}).get(kind)
        if rho is None:raise ValueError('background material reflectance unavailable: '+kind)
        vertices=np.asarray(surface['vertices_grow_mm'],float).reshape(-1)/1000
        if not np.isfinite(vertices).all() or len(vertices)%9:raise ValueError('invalid background geometry')
        name=f'rigOptics{index}'
        material=PHONG.replace('@NAME@',name).replace('@R@',repr(rho)).replace('@T@','0.0')
        text=text.replace('protected void init()',material+'\nprotected void init()')
        coords=','.join(format(x,'.9g')+'f' for x in vertices)
        node=f'[ RigSurface("rig{index}").(setPolygons(getMesh(new FloatList(new float[]{{{coords}}})).getPolygons()),setShader({name})) ]'
        text=text.replace('Axiom ==>', 'Axiom ==> '+node)
    unused={k for k,v in (options.surface_reflectance or {}).items() if v not in (0,None) and k not in used}
    if options.indirect_model=='single_bounce_ceiling':unused.discard('ceiling')
    if unused:raise ValueError('nonzero material has no geometry: '+','.join(sorted(unused)))
    if options.indirect_model=='uniform_diffuse':
        # One continuous hemispheric source avoids dropping weak directional
        # sources in GroIMP's finite 1000-ray bundle allocation. Origins lie on
        # upstream disks outside the scene bounds, never on a floor plane.
        if options.diffuse_ppfd_umol_m2_s:
            node=f'[ Lamp().(setLight(new Revision.HemisphericLight({options.diffuse_ppfd_umol_m2_s}))) ]'
            text=text.replace('Axiom ==>','Axiom ==> '+node)
    if probe:
        shader='revisionOptics' if options.optics_model=='transmissive' else 'BLACK'
        text=text.replace('setShader('+shader+')','setShader(new SideSwitchShader(BLACK,new RGBAShader(0,0,0,0)))')
    if sensor is not None:
        tri,area=horizontal_probe(sensor['position_grow_mm'],sensor['diameter_mm'],sensor['segments'])
        coords=','.join(format(x,'.9g')+'f' for x in tri.ravel()/1000)
        node=f'[ Probe("sensor").(setPolygons(getMesh(new FloatList(new float[]{{{coords}}})).getPolygons()),setShader(new SideSwitchShader(BLACK,new RGBAShader(0,0,0,0)))) ]'
        text=text.replace('Axiom ==>','Axiom ==> '+node)
        text=text.replace('    println("SANITY_DONE");',f'    for (Probe p : (* Probe *)) println("PROBE_PPFD\\t"+p.sensorId+"\\t"+lm.getAbsorbedPower(p).integrate()/{area});\n    println("SANITY_DONE");')
    if diagnostics:
        shader=('new SideSwitchShader(BLACK,new RGBAShader(0,0,0,0))' if probe else
                'revisionOptics' if options.optics_model=='transmissive' else 'BLACK')
        text=text.replace('setShader('+shader+')','setShader(new Revision.ObservedShader('+shader+'))')
        text=text.replace('lm.compute();','lm.compute(new Revision.FaceSpectrum());')
        # Never infer front/back from a lamp centre or total direction average.
        extra=r'''
    for (LeafSurface r : (* LeafSurface *)) {
        Revision.FaceSpectrum incident=(Revision.FaceSpectrum)lm.getReceivedPower(r);
        Revision.FaceSpectrum absorbed=(Revision.FaceSpectrum)lm.getAbsorbedPower(r);
        println("FACE\t"+r.surfaceId+"\t"+incident.frontPower()+"\t"+incident.backPower()+"\t"+absorbed.frontPower()+"\t"+absorbed.backPower());
        println("INCIDENT\t"+r.surfaceId+"\t"+absorbed.incomingFrontPower()+"\t"+absorbed.incomingBackPower());
    }
'''
        if not probe and 1-options.tau-options.rho<=0:
            extra=extra.replace('println("INCIDENT\\t"+r.surfaceId+"\\t"+absorbed.incomingFrontPower()+"\\t"+absorbed.incomingBackPower());',
                                'println("INCIDENT_UNAVAILABLE\\t"+r.surfaceId+"\\tzero_absorptance");')
        text=text.replace('    println("SANITY_DONE");',extra+'    println("SANITY_DONE");')
    imports=re.findall(r'^import .+?;',text,re.M)
    text=re.sub(r'^import .+?;\s*','',text,flags=re.M)
    return '\n'.join(dict.fromkeys(imports))+'\n'+text


def library_source() -> str:
    """Java observation/source classes, compiled by GroIMP with the RGG project."""
    text=OBSERVATION+COSINE_LIGHT
    imports=re.findall(r'^import .+?;',text,re.M)
    imports+=['import de.grogra.imp3d.shading.Phong;', 'import de.grogra.imp3d.objects.PointLight;']
    text=re.sub(r'^import .+?;\s*','',text,flags=re.M)
    text=re.sub(r'^class (\w+)',r'public static class \1',text,flags=re.M)
    text=text.replace('    double front=0, back=0;','    public double front=0, back=0;')
    text=text.replace('    ObservedShader(', '    public ObservedShader(').replace('    CosinePointLight(', '    public CosinePointLight(')
    return '\n'.join(dict.fromkeys(imports))+'\npublic class Revision {\n'+text+'\n}\n'


def run_model(folder: Path, timeout_s: int = 1800, *, helper_jar: Path | None = None,
              processor_count: int | None = None) -> None:
    """Compile the explicit helper in this project; no installed jar is modified."""
    import subprocess,time
    from tools.groimp_light_sanity import CORE
    from tools.groimp_static_verify import cli_path
    log=folder/'groimp.log'
    if processor_count is not None and (type(processor_count) is not int or processor_count<1):
        raise ValueError('positive explicit processor count required')
    command=['java']+([] if processor_count is None else ['-XX:ActiveProcessorCount='+str(processor_count)])
    command+=['-Dfile.encoding=UTF-8','-jar',str(CORE),'--headless','-a','cli']
    with log.open('wb') as stream:
        process=subprocess.Popen(command,
            stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        deadline=time.monotonic()+timeout_s
        peak_rss=None
        try:
            import psutil
            monitored=psutil.Process(process.pid)
        except ImportError:monitored=None
        def memory_sample():
            nonlocal peak_rss
            if monitored is not None:
                try:peak_rss=max(peak_rss or 0,monitored.memory_info().rss)
                except psutil.Error:pass
        def wait(predicate):
            while time.monotonic()<deadline:
                memory_sample()
                text=log.read_text(encoding='utf-8',errors='replace')
                if predicate(text):return text
                if process.poll() is not None:break
                time.sleep(.2)
            raise ValueError('native compile/run timeout: '+str(log))
        try:
            wait(lambda t:'[base]>>' in t)
            process.stdin.write('$cd '+cli_path(folder)+'\n$open Model.rgg\n');process.stdin.flush()
            wait(lambda t:'[Model]>>' in t)
            if helper_jar is not None:
                process.stdin.write('addFile '+cli_path(helper_jar)+'\n');process.stdin.flush();time.sleep(2)
                process.stdin.write('compile\n');process.stdin.flush()
            # Compilation/initialisation is asynchronous. A fixed sleep races
            # large meshes; ask for the published function instead of issuing
            # measure before it exists. No numerical job is retried here.
            ready_end=min(deadline,time.monotonic()+120)
            offset=len(log.read_text(encoding='utf-8',errors='replace'))
            while time.monotonic()<ready_end:
                memory_sample()
                process.stdin.write('listFunctions\n');process.stdin.flush();time.sleep(.5)
                tail=log.read_text(encoding='utf-8',errors='replace')[offset:]
                if re.search(r'^\s*- measure\s*$',tail,re.M):break
                if process.poll() is not None:break
            else:raise ValueError('native compile did not publish measure: '+str(log))
            process.stdin.write('measure\n');process.stdin.flush()
            text=wait(lambda t:'SANITY_DONE' in t or 'Command not found' in t)
            if 'SANITY_DONE' not in text:raise ValueError('native model did not compile: '+str(log))
        finally:
            memory_sample()
            if process.poll() is None:process.kill();process.wait()
            (folder/'runtime_resources.json').write_text(json.dumps(dict(
                sampled_peak_rss_bytes=peak_rss,memory_status='sampled_0_2s_not_OS_high_water' if peak_rss is not None else 'unavailable',
                processor_count_override=processor_count,timeout_s=timeout_s),indent=2)+'\n',encoding='utf-8')


def parse_faces(path: Path) -> dict:
    result={}
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        if line.startswith('FACE\t'):
            _,key,*values=line.split('\t')
            if key in result or len(values)!=4:raise ValueError('duplicate/incomplete face output')
            values=np.asarray(values,float)
            if not np.isfinite(values).all():raise ValueError('nonfinite face output')
            result[key]=values.tolist()
    return result


def compile_helper(folder: Path, compiler_jar: Path, *, scratch_dir: Path | None = None) -> Path:
    """Build the local helper with an explicitly supplied Eclipse ECJ compiler.

    No network/install side effects. ECJ is only a build dependency. The resulting
    jar is attached to this GroIMP project, never installed into GroIMP.
    """
    import subprocess, zipfile
    from tools.groimp_light_sanity import CORE
    folder.mkdir(parents=True,exist_ok=True)
    source=folder/'Revision.java';source.write_text(library_source(),encoding='utf-8')
    classes=(scratch_dir or folder/'trash')/'classes';classes.mkdir(parents=True,exist_ok=True)
    classpath=';'.join(str(p) for p in CORE.parent.rglob('*.jar'))
    command=['java','-jar',str(compiler_jar),'-17','-cp',classpath,'-d',str(classes),str(source)]
    run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (folder/'compile.log').write_bytes(run.stdout)
    if run.returncode:raise ValueError('helper compile failed: '+str(folder/'compile.log'))
    jar=folder/'Revision.jar'
    with zipfile.ZipFile(jar,'w',compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(classes.glob('*.class')):stream.write(path,path.name)
    return jar
