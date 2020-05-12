# -*- coding: cp1252 -*-
"Orientation of Aerial Photographs"

__author__   = """VIAS - Vienna Institute for Archaeological Science
Interdisziplinäre Forschungsplattform Archäologie
Universität Wien
Franz-Klein-Gasse 1/III
1190 Vienna
Austria

Research Group Photogrammetry
Dept. of Geodesy and Geoinformation
TU Wien
Gusshausstrasse 27-29
1040 Vienna
Austria
"""

__url__      = "http://vias.univie.ac.at/"

import glob as _glob, os  as _os, sys as _sys, configparser as _configparser, traitlets as _traitlets
from traitlets import default as _default
from enum import IntEnum as _IntEnum

# load the conversion types for numpy.ndarray <-> {cv::Mat, Eigen::Matrix)
if _os.environ.get("ORIENTAL_DEBUG","0") != "0":
    from ._oriental_d import *
    from ._oriental_d import __date__, __license__
else:
    from ._oriental import *
    from ._oriental import __date__, __license__

__version__  = __date__

# support conversion of _IDE to bool, with _IDE.none evaluating to False
IDE = _IntEnum( 'IDE', 'none pyscripter pycharm ptvs kdevelop', start=0 )

# ---------------  config --------------------------

class Config(_traitlets.HasTraits):
    """Central configuration singleton"""

    #: Load debug-.pyd's or release-.pyd's?
    debug = _traitlets.Bool( _os.environ.get("ORIENTAL_DEBUG","0") != "0" )

    #: Explicitly set the GUI toolkit to use. If None, let the plotting packages choose it.
    #: Mayavi runs with wx and qt4 only.
    gui = _traitlets.Unicode( allow_none=True )
    @_default('gui')
    def _gui_default(self):
        return Config._defaultGuiToolkit()

    #: The running IDE
    ide = _traitlets.Enum( list(IDE) )
    @_default('ide')
    def _ide_default(self):
        # With the internal Python engine, the executable is PyScripter.exe
        # With the external Python engine, package 'rpyc' is loaded
        if 'rpyc' in _sys.modules or _os.path.basename( _sys.executable ).lower() == "pyscripter.exe":
            return IDE.pyscripter
        if 'pydevd' in _sys.modules:
            return IDE.pycharm
        if 'ptvsd' in _sys.modules: # This detects the PTVS Python debugger, but not its native code debugger
            return IDE.ptvs
        if 'kdevpdb' in _sys.modules:
            return IDE.kdevelop
        return IDE.none

    #: If True, then redirect plot commands to an oriental-IPython - subprocess,
    #: and immediately display plots in dedicated windows.
    #: If users choose to load oriental from an IPython shell without GUI loop integration, then we don't redirect (either).
    #: Note: even if the IPython shell was startet with gui and matplotlib integration, it can be turned off temporarily
    #: using %run e.g. for efficient execution of an oriental-script
    #: Set to False for speedup.
    #: PyCharm is the only tested IDE that supports interactive plotting in the debugger, without oriental's redirection.
    redirectedPlotting = _traitlets.Bool()
    @_default('redirectedPlotting')
    def _redirectedPlotting_default(self):
        return False#self.ide not in (IDE.none, IDE.pycharm) and not Config._ipythonInteractiveShell()

    #: Use dill for message serialization
    plotUseDill = _traitlets.Bool(False)
    # 2014-05-27: dill is unable to unserialize the return value of oriental.utils.pyplot.figure(..),
    # when run within the OrientALShell (outside an IDE) and from a network drive:
    # call stack:
    #   IPython\kernel\zmq\serialize.py", line 121, in unserialize_object
    #   pobj = bufs.pop(0)
    #   IndexError: pop from empty list

    pkgRoot = _traitlets.Unicode()
    @_default('pkgRoot')
    def _pkgRoot_default(self):
        return _os.path.dirname( _os.path.abspath( __file__ ) )

    #: :term:`$ORIENTAL`
    installDir = _traitlets.Unicode()
    @_default('installDir')
    def _installDir_default(self):
        return _os.path.dirname( self.pkgRoot ) 

    # HasTraits has a metaclass. As we derive from it, we cannot derive from _oriental.Installation, because it does not have that metaclass.
    isDvlp = Installation.isDvlp
    isGPL  = Installation.isGPL
    #: :oriental-root:`/data`
    dataDir = _traitlets.Unicode()
    @_default('dataDir')
    def _dataDir_default(self):
        return Installation.dataDir

    #: Optionally use APIS DB, to be set via :oriental-root:`/data/config.ini`
    dbAPIS = _traitlets.Unicode(allow_none=True)

    dbCameras = _traitlets.Unicode()
    @_default('dbCameras')
    def _dbCameras_default(self):
        return _os.path.join( self.dataDir, 'cameras.sqlite' ) 

    dot = _traitlets.Unicode()
    @_default('dot')
    def _dot_default(self):
        return "dot" if self.isDvlp or _os.name=='posix' else _os.path.join( self.installDir, "bin", "Graphviz", "dot" )

    threeJs = _traitlets.Unicode()
    @_default('threeJs')
    def _threeJs_default(self):
        if self.isDvlp:
            return sorted( _glob.glob( _os.path.join( self.installDir, "external", "three.js-r[0-9][0-9][0-9]" ) ) )[-1]
        return _os.path.join( self.installDir, "bin", "three.js" )

    #: Test data directory
    testDataDir = _traitlets.Unicode()
    @_default('testDataDir')
    def _testDataDir_default(self):
        return _os.path.join( self.pkgRoot, 'tests', 'data' ) 

    testDataDirNotInDistro = _traitlets.Unicode()
    @_default('testDataDirNotInDistro')
    def _testDataDirNotInDistro_default(self):
        return _os.path.join( self.pkgRoot, 'tests', 'data_not_in_distro' ) 

    def __init__( self ):
        # support replacement of ${ORIENTAL} in config.ini
        cfgPrs = _configparser.ConfigParser( defaults={ 'ORIENTAL' : self.installDir }, interpolation = _configparser.ExtendedInterpolation() )
        # values in the user's home directory overrule those in the installation's config.ini
        cfgPrs.read( [ _os.path.join( self.dataDir, 'config.ini' ),
                       _os.path.expanduser('~/OrientAL/config.ini') ]  )
        self.dbAPIS = cfgPrs['DEFAULT'].get('dbAPIS',None)
        if self.isDvlp:
            assert self.dbAPIS is None
            self.dbAPIS = _os.path.join( self.testDataDirNotInDistro , 'APIS.sqlite' )
        dontTrace = ('traitlets','contracts')
        if self.ide == IDE.pyscripter:
            # These modules are realized by function wrappers/decorators.
            # When jumping with the debugger into a function that has been decorated,
            # then the debugger will first enter the wrapper code, which is very distracting.
            # Thus, tell PyScripter to not de-bug these modules.
            # https://groups.google.com/forum/#!topic/PyScripter/euRULh5-BsA
            #import IPython.utils.traitlets
            #IPython.utils.traitlets.__traceable__ = 0
            #importlib.import_module('IPython.utils.traitlets').__traceable__ = 0
            import importlib
            for el in dontTrace:
                importlib.import_module(el).__traceable__ = 0
        #elif self.ide == IDE.pycharm:
        #    # DONT_TRACE has been moved to helpers\pydev\_pydevd_bundle\pydevd_dont_trace_files.py
        #    import pydevd
        #    for el in dontTrace:
        #        pydevd.DONT_TRACE[el] = pydevd.LIB_FILE

        # Better configure PTVS to not debug any standard library modules. This also affects site packages, it seems.
        #elif self.ide == IDE.ptvs:
        #    from ptvsd.debugger import DONT_DEBUG as _DONT_DEBUG
        #    import contracts.main as _main
        #    _DONT_DEBUG.append( _os.path.normcase(_main.__file__) )

        if self.debug:
            print("oriental.config:")
            for name in sorted( self.trait_names() ):
                print( name + ": {}".format( getattr( self, name ) ) )

    @staticmethod
    def _ipythonInteractiveShell() -> bool:
        if "IPython" not in _sys.modules:
            return False
        import IPython
        # get_ipython() returns None if no InteractiveShell instance is registered.
        return IPython.get_ipython() is not None

    @staticmethod
    def _ipythonMatplotlib() -> 'str|None':
        """returns the active matplotlib-backend if IPython has been started with --matplotlib or --pylab
        note: --pylab implies --matplotlib, and additionally imports pylab into the top-level namespace"""
        if not Config._ipythonInteractiveShell():
            return None
        import IPython
        return getattr( IPython.get_ipython(), 'pylab_gui_select', None )

    @staticmethod
    def _ipythonInputHook() -> bool:
        """returns True if oriental is loaded from a console with GUI integration (using PyOS_InputHook)
        However, that does not necessarily mean that matplotlib has been configured to work with IPython
        IPython --gui is not enough"""
        if not Config._ipythonInteractiveShell():
            return False
        # This tests if oriental is loaded from an IPython interactive shell with an installed PyOS_InputHook
        from IPython.lib import inputhook
        return inputhook.current_gui() is not None

    @staticmethod
    def _ipythonGuiKernel() -> bool:
        "returns True if oriental is loaded from an IPython kernel with GUI integration (e.g. IPython QtConsole)"
        if not Config._ipythonInteractiveShell():
            return False
        from IPython.config.application import Application
        if Application.initialized():
            kernel = getattr( Application.instance(), 'kernel', None )
            if kernel is not None:
                if hasattr( kernel, 'eventloop' ):
                    return True
        return False

    @staticmethod
    def _defaultGuiToolkit() -> 'str|None':
        if 'matplotlib.backends' in _sys.modules:
            import matplotlib
            return matplotlib.get_backend()[:-3] # trim trailing 'Agg'
        return Config._ipythonMatplotlib()
        # Tk is always available.
        # Note: 3D-plotting with PySide within an IPython-console seems to be broken
        # (3d-plotting with PySide within IPython qtconsole works)

#: The configuration singleton
config = Config()

def _summary():
    pass

def _setup_summary_docstring( _summary, moduleName ):
    """
    Generates the docstring of `_summary`.

    This must be done after the entire module is imported, so it is
    called from the end of this module.
    """
    import oriental, inspect, sys
    compiledModuleName = moduleName + '._' + moduleName.split('.')[-1]
    if oriental.config.debug:
        compiledModuleName += '_d'

    module = sys.modules.get(moduleName)
    compiledModule = sys.modules.get(compiledModuleName, None)

    def getAttributes(parent, baseName = None):
        baseName = baseName or []
        commands = []
        for name, obj in getattr(parent, '__dict__', {}).items():
            if name.startswith('_'):
                continue
            if not inspect.getmodule(obj) in (module, compiledModule):
                continue
            if inspect.isclass(obj) or \
               not baseName and inspect.isroutine(obj) and getattr(obj, '__self__', None) is None:
                # There seems to be a bug in boost.python: dir(x) lists __qualname__, but accessing it yields an AttributeError.
                # Hence, fall back to __name__
                qualNames = *baseName, name
                commands.append(('.'.join(qualNames), obj))
                commands.extend( getAttributes(obj, qualNames) )
        return commands

    attributes = getAttributes(module)
    attributes.sort()

    rows = [('Class','Description')]
    for qualname, obj in attributes:
        summary = ''
        if obj.__doc__:
            nonEmptyLines = [ el for el in obj.__doc__.splitlines() if len(el.strip()) ]
            if nonEmptyLines:
                summary = nonEmptyLines[0].strip()
        role = 'class' if inspect.isclass(obj) else 'func'
        theModName = inspect.getmodule(obj).__name__
        name = f':py:{role}:`{qualname} <{theModName}.{qualname}>`'
        rows.append((name, summary))

    maxNameLen = max( len(name) for name, summary in rows)
    maxSummaryLen = max( len(summary) for name, summary in rows)

    rows = [ '{:{}} {:{}}'.format(name, maxNameLen, summary, maxSummaryLen) for name, summary in rows ]
    sep = '=' * maxNameLen + ' ' + '=' * maxSummaryLen
    rows.insert(0,sep)
    rows.insert(2,sep)
    rows.append(sep)

    _summary.__doc__ = '\n'.join(rows)

_setup_summary_docstring( _summary, __name__ )

