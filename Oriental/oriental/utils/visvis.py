# -*- coding: cp1252 -*-
__traceable__ = 0 # prevent PyScripter from stepping into its code during debugging

from .. import config as _config

import visvis as _visvis
#_visvis.__traceable__ = 0 # prevent PyScripter from stepping into its code during debugging
# visvis tries to load pyside first anyway (=default), so this call is unnecessary
_visvis.use( 'pyside' if _config.gui == 'qt4' else _config.gui )

if not _config.redirectedPlotting:
    # best run in IPython!
    # set QT_API=pyside
    # IPython --pylab=qt test_visvis.py
    from visvis import *
else:
    from .BlockingKernelManager import apply as _apply
    from inspect import isfunction as _isfunction
    from functools import partial as _partial, update_wrapper as _update_wrapper
    
    # the function names that appear in the wrapped functions' code must be known on the remote side.
    # When started with '--pylab', the kernel already has matplotlib.pyplot imported.
    # So, either
    #   + use the full name, e.g. matplotlib.pyplot.plot,
    #   + or use a new name, e.g. _pyplot - however, with this latter approach, we need to make that new name known on the remote side:
    #kernelManager.shell_channel.execute("from sys import path\n"
    #                                    "path.append(r'D:\swdvlp64')\n"
    #                                    "from matplotlib import pyplot as _pyplot\n" )
    # note: must not import myself (oriental.pyplot) remotely, or otherwise, the remote side will start its own BlockingKernelManager,
    #   which will start IPython itself, starting another BlockingKernelManager ... in a infinite loop yielding an infinite number of Python processes!
    
    # works, but would be needed for every function in matplotlib.pyplot
    #def _figure( *args, **kwargs ):
    #    return _apply( _pyplot.figure, *args, **kwargs )
    
    # callable for all the functions in matplotlib.pyplot
    # matplotlib.pyplot.??? will only get called on the remote side!
    # the function definition is executed on the local side, thereby creating the function code.
    # IPython then serializes that code, passes it in a message to the remote side, and the remote side calls the function body, i.e. matplotlib
    def _wrapFunc( key, name ):
        code = """def _{0}( *args, **kwargs ):
                      return _visvis.{1}( *args, **kwargs )
               """.format( key, name )
        exec( code, globals() )
        
    for key, value in _visvis.__dict__.items():
        # funktions that start with '_' are private by convention
        if key[0] is '_' or \
           not _isfunction(value):
            continue
        
        if key in globals():
            raise Exception("Overwriting existing function '{}'!?".format(key))
         
        #def makeFigLambda( val = value ):
        #    return lambda *args, **kwargs : _mg.shell_channel.execute("pyplot.figure({})".format(args[0]) ) if len(args) else _mg.shell_channel.execute("pyplot.figure()") 
    
        # pyplot.figure is not picklable, because this function contains a nested function
        #if value == matplotlib.pyplot.figure:
        #    #globals()[key] = _partial( _apply, _figure )
        #    _wrapFunc( key, matplotlib.pyplot.figure.__name__ )
        #    globals()[key] = _partial( _apply, globals()['_figure'])
        #    globals()[key].__doc__ = value.__doc__
        #    continue
    
        # funzt: Funktionen werden richtig gemappt, aber eben nicht auf cmd
        #d[key] = value
    
        # funzt ned:
        #d[key] = lambda *args, **kwargs : cmd( value, *args, **kwargs  )
        # Grund: value referenziert während der Iteration immer das gleiche Objekt, nur dessen Wert wird verändert.
        # Daher bindet lambda bei allen Iterationen den gleichen Funktionszeiger.
        # Nach der letzten Iteration zeigen also alle lambdas auf die zuletzt zugewiesene matplotlib.pyplot-Funktion!
        # Abhilfe: eine factory-Funktion mit default-Argument verwenden, wie in http://mail.python.org/pipermail/tutor/2005-November/043360.html
        # Default-Werte werden zur Zeit der Definition der Funktion auf den aktuellen Wert gesetzt, welcher bei jeder Iteration unterschiedlich ist.
        #def makeLambda( val = value ):
        #    return lambda *args, **kwargs : _apply( val, *args, **kwargs )
        
        #globals()[key] = makeLambda()
        #globals()[key] = _partial( _apply, value )
        # this approach works for all functions including matplotlib.pyplot.figure, which has a nested function definition!
        # Attention: matplotlib.pyplot defines one alias: 
        # matplotlib.artis.get = matplotlib.artis.getp
        # Thus, for matplotlib.artist.get, the name of the function to define ('get'), and the function wrap (matplotlib.artist.getp) differ! 
        _wrapFunc( key, value.__name__ )
        
        # globals()[_plot] is the function that has been created with _wrapFunc!
        # _partial( _apply, _plot ) returns a _apply, with its first argument bound to _plot, which is picklable!
        # _apply gets called on the local side, which sends an 'apply_request' to the remote side,
        #   with the pickled, wrapped function. The code of the wrapped function is also sent,
        #   and therefore, we do not need to import the wrapped function definitions on the remote side!
        # only on the remote side, the wrapped function is executed!
        globals()[key] = _partial( _apply, globals()['_' + key])
        #globals()[key].__doc__ = value.__doc__
        # _update_wrapper should make the wrapper function 'look like' the wrapped function.
        # after calling _update_wrapper, globals()[key].__doc__ == value.__doc__
        # However, the help-system still spits out the doc of the partial-class:
        # help(pyplot.figure)
        # help on partial .... 
        _update_wrapper( globals()[key], value )

                    
if __name__ == "__main__":
    import numpy as np
    
    imPts = np.random.rand( 100, 2 )*( 50, 80 )
    l = vv.plot( imPts[:,1], imPts[:,0], markerStyle='.', markerColor='r', markerWidth='2', markerEdgeColor='w', markerEdgeWidth=1, lineStyle='' )
