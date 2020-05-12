# -*- coding: cp1252 -*-
"""
mlab documentation one-liner

mlab extended documentation
on 2 lines
"""

from .. import config as _config

from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = _config.gui

# it seems that mlab needs to be imported here, at file scope. Doing so only in the except-clause below,
# results in an exception saying 'global name "mayavi" not defined'
import mayavi.mlab

#if '__IPYTHON__' in globals() and globals()['__IPYTHON__'] == True:

if not _config.redirectedPlotting:
    # IPython is already running, thus no need to redirect calls into matplotlib to a separate IPython instance.
    if _config.gui == "Qt".lower():
        # optional: set toolkit to Qt/PySide
        # sonst: verwendet wx
        exec( "from traits.etsconfig.api import ETSConfig as _ETSConfig", globals() )
        exec( "_ETSConfig.toolkit = 'qt4'", globals() )
    elif _config.gui == "wx".lower():
        pass
    else:
        raise Exception("mayavi.mlab supports Wx and Qt4 only as GUI toolkits")
    
    from mayavi.mlab import *    

# this should be another option for oriental.config
if 0:# not _config.plotInterActive:
    from mayavi.mlab import *
    # http://docs.enthought.com/mayavi/mayavi/tips.html#off-screen-rendering
    mayavi.mlab.options.offscreen = True
    
else:
    try:
        #raise ImportError()
        from .BlockingKernelManager import apply as _apply, client, handle_msgs as _handle_msgs
        
        # superseded by _environ['ETS_TOOLKIT'] in BlockingKernelManager.py
##        if _config.gui == "qt":
##            # optional: set toolkit to Qt/PySide
##            # sonst: verwendet wx
##            client.shell_channel.execute("import traits.etsconfig.api")
##            client.shell_channel.execute("traits.etsconfig.api.ETSConfig.toolkit = 'qt4'")
##        elif _config.gui == "wx":
##            pass
##        else:
##            raise Exception("mayavi.mlab supports Wx and Qt4 only as GUI toolkits")
        
        # The remote side needs access to the vtk-DLLs, which are needed by mayavi.
        # However, the vtk-Python-module is not in the Python path, which we therefore need to extend.
        # Unlike matplotlib.pyplot, IPython does not import mayavi.mlab when started with --pylab.
        # Thus, we need to import it 'manually' on the remote side.
        client.shell_channel.execute("import oriental\n" # this sets the path to the vtk DLLs and the Python vtk-module correctly
                                     "import mayavi.mlab\n"
                                    )
                                            
        from inspect import isfunction as _isfunction
        
        from functools import partial as _partial, update_wrapper as _update_wrapper
        
        ##def _points3d(*args, **kwargs):
        ##    return mayavi.mlab.points3d(*args,**kwargs)
        ##
        ##_points3d.__doc__ = mayavi.mlab.points3d.__doc__
        
        def _wrapFunc( key ):
            code = "def _{0}( *args, **kwargs ):\n" \
                   "    return mayavi.mlab.{0}( *args, **kwargs )\n".format( key )
            exec( code, globals() )    
        
        # importing mayavi.mlab takes a long time. However, importing mayavi.mlab on the remote side only, doesn't work:
        # IPython then says that the global name 'mayavi' is not defined!
        # This probably means that together with the function code of the wrapped functions, the local environment is somehow transferred to the remote side, because:
        # if mayavi.mlab is ONLY imported locally, but not remotely, calls to oriental.mlab.figure() succeed!
        
        if 1: # no caching
            import mayavi.mlab
            
            for key, value in mayavi.mlab.__dict__.items():
                # Funktionen, deren Namen mit '_' beginnen, sind in Python meist 'interne' Funktionen (private)
                if key[0] is '_' or not _isfunction(value):
                    continue
                
                if key.startswith("test"):
                    continue
                
                if key in globals():
                    raise Exception("Overwriting existing function '{}'!?".format(key))
                
                #if value == mayavi.mlab.figure:
                #    globals()[key] = _partial( _apply, _figure ) 
                #    continue
            
                # Unlike matplotlib.pyplot, there are no aliases in mayavi.mlab,
                #   and hence there is no need for a distinction between the function name to create in globals() and the function object to wrap
                # Morever, this is not possible here, because the functions to be wrapped, don't have a __name__ attribute, as they are defined in an enclosing function (document_pypline.thefunction)!
                _wrapFunc( key )
                globals()[key] = _partial( _apply, globals()['_' + key]) 
                #globals()[key].__doc__ = value.__doc__
                _update_wrapper( globals()[key], value )
                
        else:     
            # Try to import mayavi on the remote side only.   
            from os.path import dirname as _dirname, join as _join
            import pickle as _pickle
            
            _reCache = True
            _funcFn = _join( _dirname(__file__), 'mayavi.mlab-funcnames' )
            import mayavi as _mayavi
            try:
                with open( _funcFn, 'rb' ) as _fin:
                    _cachedVersion = _pickle.load(_fin)
                    if _cachedVersion == _mayavi.__version__:
                        _reCache = False
            except IOError:
                pass
            
            if _reCache:
                from mayavi import mlab as _mlab
                with open( _funcFn, 'wb' ) as _fout:
                    _pickle.dump(_mayavi.__version__, _fout, -1)
                    
                    for key, value in _mlab.__dict__.items():
                        if key[0] is '_' or \
                           key.startswith("test_") or \
                           not _isfunction(value):
                            continue
                        
                        _pickle.dump( (key,value.__doc__), _fout, -1 )
                        
    
            
            try:
                with open( _funcFn, 'rb' ) as _fin:
                    cachedVersion = _pickle.load(_fin)
                    assert cachedVersion==_mayavi.__version__, "cachedVersion==_mayavi.__version__"
                    while 1:
                        key,doc = _pickle.load( _fin )
                        
                        if key in globals():
                            raise Exception("Overwriting existing function '{}'!?".format(key))
                        
                        _wrapFunc( key )
                        globals()[key] = _partial( _apply, globals()['_' + key]) 
                        globals()[key].__doc__ = doc            
                        
            except EOFError:
                pass
            
    except ImportError:
        from _mlab_threaded import *
            
    
if __name__ == "__main__":
  if 0:
    #aus: http://docs.enthought.com/mayavi/mayavi/mlab.html

    # Create the data.
    from numpy import pi, sin, cos, mgrid
    dphi, dtheta = pi/250.0, pi/250.0
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
    r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
    x = r*sin(phi)*cos(theta)
    y = r*cos(phi)
    z = r*sin(phi)*sin(theta)

    # View it.
    from mayavi import mlab
    s = mlab.mesh(x, y, z)
    mlab.show()

  elif 0:
    # diese Beispiel von http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#mayavi.mlab.imshow
    # funzt nicht -> bug!
    import numpy
    s = numpy.random.random((10,10))
    from mayavi import mlab
    mlab.imshow(s, colormap='gist_earth')

  else:
    import numpy
    #from mayavi import mlab
    t = numpy.linspace(0, 4*numpy.pi, 20)
    cos = numpy.cos
    sin = numpy.sin

    x = sin(2*t)
    y = cos(t)
    z = cos(2*t)
    s = 2+sin(t)

    #mlab.points3d(x, y, z, s, colormap="copper", scale_factor=.25)

    # Punkte werden mit je einem Pixel dargestellt
    #mlab.points3d(x,y,z,s,mode='point')

    # Punkte werden immer noch mit nur je einem Pixel Größe dargestellt
    #mlab.points3d(x,y,z,s,mode='point',scale_factor='10')

    # Punkte haben vernünftige Größe, denn
    # scale_mode='none' schaltet Skalierung an Hand von s ab
    # scale_factor='.1' setzt den Durchmesser der Kugeln auf .1 in Einheiten der Koordinaten (x,y,z)
    points3d(x,y,z,s,mode='sphere',scale_factor='.1',scale_mode='none')

    # gibt man s gleich gar nicht an, dann kann man sich scale_mode='none' sparen
    # gibt man s an, und scale_mode aber nicht, dann gibt scale_factor den maximalen Durchmesser der Kugeln an

    points3d(x,y,z,mode='sphere',scale_factor='.1')
    dummy = 1