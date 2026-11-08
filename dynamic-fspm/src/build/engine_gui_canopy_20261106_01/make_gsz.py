"""Package gui/Model_gui_canopy_sla<ARM>.rgg as a GroIMP project (adapted from build/engine_gui_edit_20261101_01/make_gsz.py; canopy GUI CONTRACT.md).
python make_gsz.py A|R -> gui/TomatoCanopyGUI_sla<ARM>.gsz (+ read-only _original copy). Layout: toolbar + 3D view (left, top), file explorer /
status (left, bottom), text editor with the model (its SETTINGS block at the top) and the attribute editor (right, top), log / console (right, bottom).
The chart panels are opened at run time by chart(); where GroIMP places them could not be checked on this PC (free RAM < 4 GB)."""
import hashlib, os, stat, zipfile
from pathlib import Path
import sys
GUI = Path(__file__).resolve().parent / 'gui'
ARM = sys.argv[1] if len(sys.argv) > 1 else 'A'
MODEL = f'Model_gui_canopy_sla{ARM}.rgg'

MANIFEST = ('\r\nName: project.gs\r\nContent-Type: application/x-grogra-registry+xml\r\n\r\n'
            'Name: ' + MODEL + '\r\nContent-Type: text/x-grogra-rgg; charset=UTF-8\r\n\r\n'
            'Name: graph.xml\r\nContent-Type: application/x-grogra-graph+xml\r\n\r\n')

GRAPH = '''<?xml version="1.0" encoding="UTF-8"?><graph xmlns="http://grogra.de/xmlpersistence" xmlns:gx="http://grogra.de/xmlpersistence">
 <node root="MainGraph" id="0" type="de.grogra.graph.impl.Node">
  <data gx:name="extentIndex" gx:value="7"/>
  <node id="21" type="de.grogra.rgg.RGGRoot" edges="+">
   <node id="23" type="de.grogra.rgg.Axiom" edges="-"/>
  </node>
 </node>
 <node root="MetaGraph" id="1" type="de.grogra.graph.impl.Node">
  <data gx:name="extentIndex" gx:value="7"/>
 </node>
</graph>
'''

VIEW = ('graphDescriptor=[de.grogra.imp.ProjectGraphDescriptor]visibleScales={true true true true true true true true true true true true true true true}'
        'visibleLayers={true true true true true true true true true true true true true true true true}epsilon=1.0E-6 visualEpsilon=0.01 magnitude=1.0 '
        'camera=(minZ=0.005 maxZ=50.0 projection=[de.grogra.imp3d.PerspectiveProjection aspect=1.0 fieldOfView=1.0471976]'
        'transformation=(1.0 0.0 0.0 0.0 0.0 0.2588190451 0.9659258263 -0.04 0.0 -0.9659258263 0.2588190451 -0.35 0.0 0.0 0.0 1.0))')

PROJECT = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<project xmlns="http://grogra.de/registry" graph="graph.xml">
 <import plugin="de.grogra.imp" version="2.1.3"/>
 <import plugin="de.grogra.pf" version="2.1.3"/>
 <import plugin="de.grogra.imp3d" version="2.1.3"/>
 <import plugin="de.grogra.rgg" version="2.1.3"/>
 <import plugin="de.grogra.math" version="2.1.3"/>
 <registry>
  <ref name="project">
   <ref name="objects">
    <ref name="files">
     <de.grogra.pf.ui.registry.SourceFile mimeType="text/x-grogra-rgg" name="{MODEL}" systemId="pfs:{MODEL}"/>
    </ref>
    <ref name="meta">
    </ref>
   </ref>
  </ref>
  <ref name="workbench">
   <ref name="state">
    <de.grogra.pf.ui.registry.Layout name="layout">
     <de.grogra.pf.ui.registry.MainWindow>
      <de.grogra.pf.ui.registry.Split location="0.55">
       <de.grogra.pf.ui.registry.Split location="0.82" orientation="0">
        <de.grogra.pf.ui.registry.Split orientation="0">
         <de.grogra.pf.registry.Link source="/ui/panels/rgg/toolbar"/>
         <de.grogra.pf.ui.registry.PanelFactory source="/ui/panels/3d/defaultview">
          <de.grogra.pf.registry.Option name="panelId" type="java.lang.String" value="/ui/panels/3d/defaultview"/>
          <de.grogra.pf.registry.Option name="panelTitle" type="java.lang.String" value="View"/>
          <de.grogra.pf.registry.Option name="view" type="de.grogra.imp3d.View3D" value="{VIEW}"/>
         </de.grogra.pf.ui.registry.PanelFactory>
        </de.grogra.pf.ui.registry.Split>
        <de.grogra.pf.ui.registry.Split orientation="0">
         <de.grogra.pf.ui.registry.Tab selectedIndex="0">
          <de.grogra.pf.registry.Link source="/ui/panels/fileexplorer"/>
          <de.grogra.pf.registry.Link source="/ui/panels/objects/meta"/>
         </de.grogra.pf.ui.registry.Tab>
         <de.grogra.pf.registry.Link source="/ui/panels/statusbar"/>
        </de.grogra.pf.ui.registry.Split>
       </de.grogra.pf.ui.registry.Split>
       <de.grogra.pf.ui.registry.Split location="0.70" orientation="0">
        <de.grogra.pf.ui.registry.Tab selectedIndex="0">
         <de.grogra.pf.ui.registry.PanelFactory source="/ui/panels/texteditor">
          <de.grogra.pf.registry.Option name="documents" type="java.lang.String" value="&quot;\\&quot;pfs:{MODEL}\\&quot;&quot;"/>
          <de.grogra.pf.registry.Option name="panelId" type="java.lang.String" value="/ui/panels/texteditor"/>
          <de.grogra.pf.registry.Option name="panelTitle" type="java.lang.String" value="jEdit - {MODEL}"/>
          <de.grogra.pf.registry.Option name="selected" type="java.lang.String" value="pfs:{MODEL}"/>
         </de.grogra.pf.ui.registry.PanelFactory>
         <de.grogra.pf.registry.Link source="/ui/panels/attributeeditor"/>
        </de.grogra.pf.ui.registry.Tab>
        <de.grogra.pf.ui.registry.Tab selectedIndex="0">
         <de.grogra.pf.registry.Link source="/ui/panels/log"/>
         <de.grogra.pf.registry.Link source="/ui/panels/rgg/console"/>
        </de.grogra.pf.ui.registry.Tab>
       </de.grogra.pf.ui.registry.Split>
      </de.grogra.pf.ui.registry.Split>
     </de.grogra.pf.ui.registry.MainWindow>
    </de.grogra.pf.ui.registry.Layout>
   </ref>
  </ref>
 </registry>
</project>
'''


def build(path):
    if path.exists():
        os.chmod(path, stat.S_IWRITE)
    fixed = (2026, 11, 6, 20, 0, 0)  # fixed timestamps -> reproducible archive bytes
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in (('META-INF/MANIFEST.MF', MANIFEST.encode('utf-8')),
                           (MODEL, (GUI / MODEL).read_bytes()),
                           ('graph.xml', GRAPH.encode('utf-8')),
                           ('project.gs', PROJECT.encode('utf-8'))):
            z.writestr(zipfile.ZipInfo(name, fixed), data, zipfile.ZIP_DEFLATED)
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    a = build(GUI / f'TomatoCanopyGUI_sla{ARM}.gsz'); b = build(GUI / f'TomatoCanopyGUI_sla{ARM}_original.gsz')
    os.chmod(GUI / f'TomatoCanopyGUI_sla{ARM}_original.gsz', stat.S_IREAD)
    assert a == b
    print(f'TomatoCanopyGUI_sla{ARM}.gsz', a)
