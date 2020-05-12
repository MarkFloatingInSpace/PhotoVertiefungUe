# -*- coding: cp1252 -*-
"""Provides a functional interface to mpl_toolkits.mplot3d.Axes3D,
just like matplotlib.pyplot for 2D-plots.
Redirect calls to the IPython-kernel, eventually.
Does not work with dill. Reason unclear."""

# mpl_toolkits.mplot3d seems to not work with PySide!
# check:
# C:\>ipython --matplotlib=qt4
##from matplotlib import pyplot as plt
##from mpl_toolkits.mplot3d import Axes3D
##import numpy as np
##imPts = np.random.rand( 100, 3 )*( 50, 80, 10 )
##fig = plt.figure()#gcf()
##axes = Axes3D( fig )
##axes.scatter3D( imPts[:,0], imPts[:,1], imPts[:,2] )
##plt.show()
# -> opens a window, but does not draw anything
# -> use with Tk

# pickle error: Can't pickle <class 'module'>: attribute lookup builtins.module failed
##import functools
##def _wrapMeth( meth ):
##
##    from .BlockingKernelManager import apply as _apply
##
##    def wrapped( show, *args, **kwargs ):
##        fig = kwargs.pop('fig',None)
##        fig = fig or plt.gcf()
##        axes = fig.gca( projection='3d' )
##        ret = meth( axes, *args, **kwargs )
##        if show:
##            plt.show()
##        return ret
##
##    f = functools.partial( wrapped, _config.redirectedPlotting )
##    if not _config.redirectedPlotting:
##        return f
##
##    return functools.partial( _apply, f )
##
##scatter3D = _wrapMeth( Axes3D.scatter3D )

# Solution without closures -> picklable; But:
# type object argument after ** must be a mapping, not NoneType
##if _config.redirectedPlotting:
##    from .BlockingKernelManager import apply as _apply
##
##import functools
##def _wrapMeth( meth, show, *args, **kwargs ):
##    import matplotlib.pyplot as plt
##    fig = kwargs.pop('fig',None) or plt.gcf()
##    axes = fig.gca( projection='3d' )
##    meth = getattr( axes, meth )
##    ret = meth( *args, **kwargs )
##    if show:
##        plt.show()
##    return ret
##
##scatter3D = functools.partial( _wrapMeth, 'scatter3D', False ) \
##                if not _config.redirectedPlotting else \
##            functools.partial( _apply, functools.partial( _wrapMeth, 'scatter3D', True  ) )

from .. import config as _config
from . import pyplot as _plt

# this registers the 3D-axis-class in the list of supported axis classes
from mpl_toolkits.mplot3d import Axes3D as _Axes3D

from inspect import isfunction as _isfunction

if _config.redirectedPlotting:
    from oriental.utils.BlockingKernelManager import apply as _apply
    from functools import ( partial        as _partial,
                            update_wrapper as _update_wrapper )

def _wrapFunc( key, prefix ):
    code = """def {0}{1}( *args, **kwargs ):
                  axes = _plt.gcf().gca( projection='3d' )
                  method = getattr( axes, '{1}' )
                  return method( *args, **kwargs )
           """.format( prefix, key )
    exec( code, globals() )

for key, value in _Axes3D.__dict__.items():

    # Only methods of class instances are classified as methods.
    # Methods of class types are functions.
    if key[0] is '_' or not _isfunction(value):
        continue

    if key in globals():
        raise Exception("Overwriting existing method '{}'!?".format(key))

    if not _config.redirectedPlotting:
        _wrapFunc( key, prefix='' )
    else:
        _wrapFunc( key, prefix='_' )

        globals()[key] = _partial( _apply, globals()['_' + key] )

        _update_wrapper( globals()[key], value )

### test with:
##from oriental.utils import mplot3d
##import numpy as np
##pts = np.random.rand(10,3)
##mplot3d.scatter3D( pts[:,0], pts[:,1], pts[:,2] )