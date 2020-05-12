# -*- coding: cp1252 -*-
from .. import config as _config
from os import environ as _environ

# Enthougt introduced the environment variable QT_API to distinguish between PyQt and PySide.
# Both IPython and matplotlib consider this environment variable when selecting one of those 2 Python-Qt-bindings
# Only if QT_API is not set, then matplotlib will load the Python-Qt-binding according to matplotlib.rcParams['backend.qt4'], which defaults to PyQt4
# However, instead of checking beforehand, which of PyQt and PySide are available/installed, matplotlib decides first which Qt-binding to import (even if that is only a default value),
# and then issues an error if that fails: 'no module named sip'!
# Thus, let's try to check if PySide can be imported. If so, then set QT_API accordingly.
# Setting QT_API instead of matplotlib.rcParams['backend.qt4'] has the advantage that QT_API is inherited by the IPython-subprocess (if redirected plotting is used),
# while re-setting matplotlib.rcParams['backend.qt4'] would only be considered in the current process / Interpreter session.
if _config.gui is not None and \
   _config.gui.lower() == 'qt4' and \
   'QT_API' not in _environ: # don't overrule user-settings.
    try:
        import PySide
        _environ['QT_API'] = "pyside"
    except ImportError:
        _environ['QT_API'] = "pyqt" # unnecessary, because matplotlib.backends.qt4_compat defaults to PyQt4 anyway.
        
if _config.gui is not None:
    _environ['ETS_TOOLKIT'] = _config.gui

# when ipython is started with --pylab=qt,
# then ipython asks matplotlib which Qt-Backend to use/integrate, if the environment variable QT_API is undefined.
# Drawback: the matplotlib installation must be configured appropriately (after installation and update!), before the PySide-Qt-Backend will be used.
##if 0:
##    import matplotlib
##    if( matplotlib.rcParams["backend.qt4"] != "PySide" ):
##        from os.path import dirname as _dirname, join as _join
##        raise Exception("""matplotlib.rcParams["backend.qt4"] has not been set to "PySide", and hence, qt will not work on the remote side.
##                           please edit {}""".format( _join( _dirname(matplotlib.__file__), 'mpl-data/matplotlibrc' ) ) )
##else:
##    # This decision can be overruled using the environment variable QT_API. The environment is inherited by the kernel (child-)process.
##    from os import environ as _environ
##    _environ['QT_API'] = "pyside"
##    # set the graphics toolkit to be used by mayavi.mlab
##    _environ['ETS_TOOLKIT'] = 'qt4' if _config.gui=='qt' else _config.gui
## 
##    # The child process will try to import the wrapped functions - e.g. in oriental.pyplot.
##    # Thus, oriental.pyplot must be on the module path!
##    #orientalParentDir = _dirname( _dirname( _dirname(__file__) ) )
##    #_environ['PYTHONPATH'] = _environ.get( 'PYTHONPATH', '' ) + ";" + orientalParentDir
