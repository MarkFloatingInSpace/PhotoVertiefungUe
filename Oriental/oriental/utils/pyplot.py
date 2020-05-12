# -*- coding: cp1252 -*-
"""Remote plotting with matplotlib.pyplot.

Provides wrappers of all functions defined in :mod:`matplotlib.pyplot`.
Wrappers call their wrapped function in a child process running :mod:`IPython` ``--pylab``.
This makes interactive plotting possible while debugging."""

from .. import config as _config
from . import plot_environment
import matplotlib

if _config.ide:
    matplotlib.__traceable__ = 0

if _config.gui:
    # can only be called before importing pyplot
    matplotlib.use( "{}Agg".format( _config.gui ) )
    # not necessary, since we set the environment variable QT_API
    #matplotlib.rcParams['backend.qt4']='PySide'

if not _config.redirectedPlotting:
    # If we are in an IPython-session, then the following is executed.
    # This also holds true for a remote IPython-process!
    from matplotlib.pyplot import *
    #from mpl_toolkits.mplot3d.axes3d import Axes3D # this registers the 3D-axis-class in the list of supported axis classes
else:
    # It would be nice to skip the import of matplotlib.pyplot in the parent (==this) process,
    #   and import matplotlib.pyplot in the child process only.
    # For wrapping the functions, only their names are necessary, which can be stored (pickled) in a file once,
    #   and used for wrapping in all subsequent imports of oriental.pyplot.
    # This works for matplotlib, but it doesn't work in combination with mayavi.mlab!

    from .BlockingKernelManager import apply as _apply#, kernelManager
    import matplotlib.pyplot
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
                        return matplotlib.pyplot.{1}( *args, **kwargs )
                """.format( key, name )
        exec( code, globals() )

    for key, value in matplotlib.pyplot.__dict__.items():
        # Funktionen, die mit '_' beginnen, sind in Python meist 'interne' Funktionen (private)
        if key[0] is '_' or not _isfunction(value):
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

        if False:#not _config.plotUseDill:
            # This rather complicated approach is chosen, because some functions in matplotlib.pyplot contain nested function definitions,
            # most notably matplotlib.pyplot.figure()
            # Such functions cannot be serialized by IPython, which uses pickle for that.
            # As a work-around, we define another function using _wrapFunc that does not contain a nested function, but whose body consists only of the call to the original function.
            # This simple function can of course be serialized.

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

            globals()[key] = _partial( _apply, globals()['_' + key] )
        else:
            # If pickle could serialize functions with nested function definitions (closures),
            # then this simple approach would suffice!
            # dill can! supported by IPython since v.2.0.0
            # set IPython.utils.pickleutil.use_dill()
            # and use this branch!
            # Since Python 3.4, this seems possible even with pickle!
            globals()[key] = _partial( _apply, value )

        #globals()[key].__doc__ = value.__doc__
        # _update_wrapper should make the wrapper function 'look like' the wrapped function.
        # after calling _update_wrapper, globals()[key].__doc__ == value.__doc__
        # However, the help-system still spits out the doc of the partial-class:
        # help(pyplot.figure)
        # help on partial ....
        _update_wrapper( globals()[key], value )

    # Otherwise, python crashes at the end of a program when run in the PTVS-debugger.
    # Python still crashes if the PTVS-debugger is interrupted (Shift-F5).
    # The order in which atexit calls the registered functions is undefined.
    # Thus, register a single function that cleans up in the needed order.
    import atexit
    from .BlockingKernelManager import client, kernelManager
        
    @atexit.register
    def _shutDown():
        #print("shutting down kernel")
        if kernelManager.is_alive() and client.channels_running:
            close('all') # closes all the figure windows
            #client.shell_channel.execute("quit()")
            #client.stop_channels()
            kernelManager.shutdown_kernel( restart=False )


if __name__ == "__main__":
    import random
    x = [random.random() for var in range(10)]
    y = [random.random() for var in range(10)]
    fig = figure()
    #fig.add_subplot(1,2,1)
    scatter( x, y, marker='x')

    scatter( x, y, marker='D')

    axhline(y=.5,linewidth=4, color='r')

    figure(3)
    scatter( x, y, marker='x')

