"""Configure PyCharm to use the OrientAL Python interpreter
This inserts the OrientAL Python interpreter into PyCharm's configuration in the home directory,
and it sets PyCharm's default interpreter to be OrientAL's interpreter.
Also, it tries to set the least recently used Python interpreter, such that upon creating a new project, that interpreter is already selected.
However, that does not work, while it would be necessary to use PyCharm in the students' exercises.
What seems to work is to setup a PyCharm-OrientAL-project, edit the project files to use the OrientAL interpreter
(i.e. editing the absolute path only, if already present and OrientAL has been moved), and tell PyCharm to open that project.
"""
import sys, os, platform, shutil
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom

#PyCharm default system/config path
# os.environ['USERPROFILE']
pyCharmConfigDir = sys.argv[1]
pyCharmOptionsDir = Path.home() / pyCharmConfigDir / 'config' / 'options'
pyCharmPyCfgPath = pyCharmOptionsDir / 'jdk.table.xml'
pyCharmDefaultsPath = pyCharmOptionsDir / 'project.default.xml'

pythonVersionString = 'Python ' + platform.python_version()
pyCharmPythonInterpreterId = '{} ({})'.format( pythonVersionString, sys.executable )
print( "Python interpreter id: {}".format(pyCharmPythonInterpreterId) )

pycharmProjPath = Path(__file__).parent.parent.parent / 'oriental' / '.idea'
developerMode = len(sys.argv) > 1 and sys.argv[1].lower() in ("1", "true")

#set OrientAL Python as default Python interpreter
pymisc_in = pymisc_out = pycharmProjPath / 'misc.xml'
pyiml_in  = pyiml_out  = pycharmProjPath / 'oriental.iml'
pyoriental = pycharmProjPath / 'modules.xml'

if developerMode:
    pymisc_in = add_base(pymisc_out)
    pyiml_in = add_base(pyiml_out)

def add_base(path):
    return path.parent / ( path.stem + '.base' + path.suffix )

def prettifyXml( node, indentStr='  '):
    def clearNode(node):
        node.tail = None
        node.text = None
        for child in node:
            clearNode(child)

    clearNode(node)
    uglyString = ET.tostring(node, 'utf-8')
    dom = minidom.parseString(uglyString)
    text = dom.toprettyxml( indent=indentStr )
    return text

def createOrientalPythonInterpreterNode(parentNode):
    pyDir = Path(sys.executable).parent
    orientalRoot = pyDir.parent / 'oriental'

    jdkNode = ET.SubElement(parentNode,"jdk", {"version" : "2"} )
    ET.SubElement( jdkNode, "type",    {"value" : "Python SDK"} )
    ET.SubElement( jdkNode, "version", {"value" : pythonVersionString} )
    ET.SubElement( jdkNode, "homePath",{"value" : sys.executable} )
    ET.SubElement( jdkNode, "name",    {"value" : pyCharmPythonInterpreterId} )
    rootsNode = ET.SubElement( jdkNode,   "roots" )
    classPathNode = ET.SubElement( rootsNode, "classPath" )
    rootNode = ET.SubElement( classPathNode, "root", { "type" : "composite"} )
    ET.SubElement( rootNode, "root", { "type" : "simple", "url" : pyDir.as_uri() } )
    ET.SubElement( rootNode, "root", { "type" : "simple", "url" : orientalRoot.as_uri() } )

    sourcePathNode = ET.SubElement( rootsNode, "sourcePath" )
    ET.SubElement( sourcePathNode, "root", { "type" : "composite"} )

def configPyCharmOrientalInterpreter():
    print('Ensure OrientAL Python interpreter in PyCharm configuration')
    try:
        tree = ET.parse(str(pyCharmPyCfgPath))
    except OSError:
        print('  PyCharm configuration does not exist, create from scratch')
        pyCharmPyCfgPath.parent.mkdir( parents=True, exist_ok=True )

        root = ET.Element("application")
        compNode = ET.SubElement( root, "component", {"name" : "ProjectJdkTable"} )
        createOrientalPythonInterpreterNode(compNode)
        pyCharmPyCfgPath.write_text( prettifyXml( root ), encoding='utf8' )
        return

    root = tree.getroot()
    for jdk in root.findall('component/jdk'):
        id = jdk.find("name").get("value")
        if id == pyCharmPythonInterpreterId:
            print("  PyCharm configuration already contains the OrientAL Python interpreter")
            break
    else:
        print("  Add OrientAL Python interpreter to PyCharm configuration")
        compNode = root.find("component")
        createOrientalPythonInterpreterNode( compNode )
        pyCharmPyCfgPath.write_text( prettifyXml( root ), encoding='utf8' )

def defaultToOrientalInterpreter():
    print('Ensure that the OrientAL Python interpreter is the default one')
    try:
        tree = ET.parse(str(pyCharmDefaultsPath))
        root = tree.getroot()
    except OSError:
        print('PyCharm default configuration does not exist, create from scratch')
        pyCharmDefaultsPath.parent.mkdir( parents=True, exist_ok=True )
        root = ET.Element("application")

    write = False
    component = root.find('component') \
                or ET.SubElement( root, 'component', {'name' : 'ProjectManager'} )
    defaultProject = component.find('defaultProject') \
                     or ET.SubElement( component, 'defaultProject' )
    compProjectRootManager = defaultProject.find("component[@name='ProjectRootManager']")
    if compProjectRootManager is None:
        print("  Create default PyCharm interpreter as OrientAL Python interpreter")
        ET.SubElement( defaultProject, 'component', { 'name' : 'ProjectRootManager', 'version' : '2', 'project-jdk-name' : pyCharmPythonInterpreterId, 'project-jdk-type' : 'Python SDK' } )
        write = True
    elif compProjectRootManager.get("project-jdk-name") != pyCharmPythonInterpreterId:
        print("  Set OrientAL Python interpreter as default PyCharm interpreter")
        compProjectRootManager.set('project-jdk-name',pyCharmPythonInterpreterId)
        write = True
    else:
        print("  OrientAL Python interpreter is already the default")

    print('Ensure that the OrientAL Python interpreter is the least recently used one')
    compPropertiesComponent = defaultProject.find("component[@name='PropertiesComponent']") \
                              or ET.SubElement( defaultProject, 'component' )
    property_ = compPropertiesComponent.find("property[@name='last_opened_file_path']")
    if property_ is None:
        print("  Create least recently used PyCharm interpreter as OrientAL Python interpreter")
        ET.SubElement( compPropertiesComponent, 'property', { 'name' : 'last_opened_file_path', 'value' : Path(sys.executable).as_posix() } )
        write = True
    elif property_.get("value") != Path(sys.executable).as_posix():
        print("  Set least recently used PyCharm interpreter as OrientAL Python interpreter")
        property_.set('value', Path(sys.executable).as_posix() )
        write = True
    else:
        print("  OrientAL Python interpreter is already the least recently used one")

    if write:
        pyCharmDefaultsPath.write_text( prettifyXml( root ), encoding='utf8' )

def setProjectInterpreter( pyproj_in, pyproj_out, xpath, attrstr ):
    print('Set current Interpreter to be used for OrientAL PyCharm project')
    try:
        tree = ET.parse(str(pyproj_in))
    except OSError:
        return False
    root = tree.getroot()
    nodes = root.findall(xpath)
    if len(nodes) != 1:
        raise Exception( "Unable to find unique corresponding Python interpreter entry for XPath {} in file '{}': found {} nodes.".format( xpath, pyproj_in, len(nodes) ) )

    jdk_name = nodes[0].get(attrstr)
    if pyCharmPythonInterpreterId == jdk_name:
        print("  Correct project Python interpreter already set")
        if pyproj_in != pyproj_out:
            shutil.copyfile( str(pyproj_in), str(pyproj_out) )
        return

    nodes[0].set( attrstr, pyCharmPythonInterpreterId )

    print("  {}writing OrientAL project file '{}'".format( 're-' if pyproj_in == pyproj_out else '', os.path.abspath(str(pyproj_out)) ) )
    pyproj_out.write_text( prettifyXml( root ), encoding='utf8' )
    return True

configPyCharmOrientalInterpreter()
defaultToOrientalInterpreter()

pycharmProjPath.mkdir( parents=True, exist_ok=True )

if not setProjectInterpreter( pymisc_in, pymisc_out, "component[@name='ProjectRootManager']", "project-jdk-name" ):
    pymisc_out.write_text(
"""<project version="4">
  <component name="ProjectRootManager" project-jdk-name="{}" project-jdk-type="Python SDK" version="2"/>
</project>
""".format(pyCharmPythonInterpreterId), encoding='utf8' )
if not setProjectInterpreter( pyiml_in , pyiml_out , "component/orderEntry[@type='jdk']"    , "jdkName" ):
    pyiml_out.write_text(
"""<module type="PYTHON_MODULE" version="4">
  <component name="NewModuleRootManager">
    <content url="file://$MODULE_DIR$">
      <sourceFolder isTestSource="false" url="file://$MODULE_DIR$"/>
    </content>
    <orderEntry jdkName="{}" jdkType="Python SDK" type="jdk"/>
    <orderEntry forTests="false" type="sourceFolder"/>
  </component>
</module>
""".format(pyCharmPythonInterpreterId), encoding='utf8' )

if not pyoriental.exists():
    pyoriental.write_text(
"""<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectModuleManager">
    <modules>
      <module fileurl="file://$PROJECT_DIR$/.idea/oriental.iml" filepath="$PROJECT_DIR$/.idea/oriental.iml" />
    </modules>
  </component>
</project>
""", encoding='utf8')
