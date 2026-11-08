"""Export labeled leaf triangles to metre-scale Wavefront OBJ for GroIMP import.

This transfers geometry/identity only. It does not equate RGB radiometric power to PPFD,
execute GroIMP, fill leaf gaps or claim a validated GroIMP light model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

from tomato3d.light.geometry import load_geometry, place_in_grow_frame
from tomato3d.light.scene import load_scene, resolve_scenario
from tools.leaf_light.common import StudyError, file_hash, output_dir, write_json, write_rows


def mm_to_m(vertices_mm):
    """One shared numeric unit conversion for OBJ and static RGG transport."""
    return np.asarray(vertices_mm, dtype=float) / 1000.0


def export(geometry: Path, variant: str, out: Path, scene: Path | None = None, scenario: str = 'as_built') -> dict:
    variants=load_geometry([geometry])
    if variant not in variants: raise StudyError('Unknown geometry variant')
    surfaces=variants[variant]
    if scene is None and scenario != 'as_built': raise StudyError('A lighting scenario needs its scene')
    vertices=surfaces.vertices_board_mm
    if scene is not None:
        vertices=place_in_grow_frame(surfaces,resolve_scenario(load_scene(scene),scenario).placements)
    folder=output_dir(out)
    mapping=[]
    with (folder/'leaves_m.obj').open('w',encoding='utf-8') as stream:
        stream.write('# Units: metres. Open leaf surfaces; no watertightness repair.\n')
        for i,triangle in enumerate(mm_to_m(vertices)):
            group=f'leaf_triangle_{i:06d}'
            stream.write(f'o {group}\ng {group}\n')
            for xyz in triangle: stream.write('v '+' '.join(format(float(v),'.12g') for v in xyz)+'\n')
            stream.write(f'f {3*i+1} {3*i+2} {3*i+3}\n')
            mapping.append(dict(obj_group=group,plant_id=surfaces.plant_ids[i],session_id=surfaces.session_ids[i],
                organ_id=surfaces.organ_ids[i],leaflet_id=surfaces.leaflet_ids[i],triangle_id=surfaces.triangle_ids[i],
                normal_known=bool(surfaces.normal_known[i])))
    write_rows(folder/'identity_map.csv',mapping,('obj_group','plant_id','session_id','organ_id','leaflet_id','triangle_id','normal_known'))
    metadata=dict(geometry_path=str(geometry.resolve()),geometry_sha256=file_hash(geometry),variant_id=variant,
        coordinate_frame='grow' if scene else 'board',units='metres',source_units='millimetres',scale_factor=.001,
        scene_sha256=file_hash(scene) if scene else None,scenario_id=scenario if scene else None,
        n_triangles=surfaces.n_triangles,watertight_required=False,groimp_import_verified=False,
        radiometric_calibration='unavailable',
        # Only leaf-surface triangles of a geometry variant are transported. Diagnostic bbox cuboids
        # and unreviewed partial patches live in separate result files and never enter this export.
        geometry_kind='leaf_surface_triangles',contains_bbox_cuboid=False,
        validated_metric=False,suitable_for_ppfd=False,
        quality_note='Exporting a mesh is not evidence of metric or PPFD validity; check the source variant status.',note='Import OBJ with no extra scale. Preserve identity_map.csv. Review two-sided visibility and optical properties before radiation simulation.')
    write_json(folder/'export.json',metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--geometry',type=Path,required=True); cli.add_argument('--variant',required=True)
    cli.add_argument('--scene',type=Path); cli.add_argument('--scenario',default='as_built')
    cli.add_argument('--out',type=Path,required=True); args=cli.parse_args(argv)
    try: export(args.geometry,args.variant,args.out,args.scene,args.scenario)
    except (ValueError,OSError,KeyError) as exc:
        print(f'GroIMP export error: {exc}',file=sys.stderr); return 2
    print(args.out.resolve()/'leaves_m.obj'); return 0


if __name__=='__main__': raise SystemExit(main())
