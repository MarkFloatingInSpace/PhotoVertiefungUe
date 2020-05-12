# -*- coding: cp1252 -*-
from . import config as _config
from os.path import dirname as _dirname, join as _join

# the only clean solution for installing dummy-functions: import the module, and store the function names to be used.
# When the module shall not be imported, read the list of function names from that file.
_funcFn = _join( _dirname(__file__), 'mayavi.mlab-funcnames.pickle' )

_d = globals()

def _do_nothing():
    pass


if _config.plotActive:
    from GuiThread import cmd as _cmd
    
    if _config.gui == "Qt".lower():
        # optional: set toolkit to Qt/PySide
        # sonst: verwendet wx
        from traits.etsconfig.api import ETSConfig as _ETSConfig
        _ETSConfig.toolkit = 'qt4'
    elif _config.gui == "WXAgg".lower():
        pass
    else:
        raise Exception("mayavi.mlab supports Wx and Qt4 only as GUI toolkits")
    
    
    from inspect import isfunction as _isfunction
    
    from mayavi import mlab as _mlab
    
    # funzt, aber für alle Funktionen in mayavi.mlab wäre das umständlich und wartungsintensiv
    #def points3d(*args,**kwargs):
    #  return _cmd(_mlab.points3d,*args,**kwargs)
    
    
    
    # funzt, aber für alle Funktionen in mayavi.mlab wäre das umständlich und wartungsintensiv:
    #d["scatter"] = lambda *args, **kwargs : _mlab.scatter( *args, **kwargs )
    
    if _config.isDvlp:
        _fout = open( _funcFn, 'wt' )
    
    try:
        for key, value in _mlab.__dict__.items():
            # Funktionen, deren Namen mit '_' beginnen, sind in Python meist 'interne' Funktionen (private)
            if key[0] is '_' or not _isfunction(value):
                continue
            
            if key in _d:
                raise Exception("Overwriting existing function '{}'!?".format(key))
            
            if _config.isDvlp: 
                _fout.write("{}\n".format(key))
            
            # show() blocks and is not necessary. Thus, map it to an empty function
            if value == _mlab.show:
                _d[key] = lambda : _do_nothing
                continue
            
            # funzt: Funktionen werden richtig gemappt, aber eben nicht auf cmd
            #d[key] = value
        
            # funzt ned:
            #d[key] = lambda *args, **kwargs : cmd( value, *args, **kwargs  )
            # Grund: value referenziert während der Iteration immer das gleiche Objekt, nur dessen Wert wird verändert.
            # Daher bindet lambda bei allen Iterationen den gleichen Funktionszeiger.
            # Nach der letzten Iteration zeigen also alle lambdas auf die zuletzt zugewiesene matplotlib.pyplot-Funktion!
            # Abhilfe: eine factory-Funktion mit default-Argument verwenden, wie in http://mail.python.org/pipermail/tutor/2005-November/043360.html
            # Default-Werte werden zur Zeit der Definition der Funktion auf den aktuellen Wert gesetzt, welcher bei jeder Iteration unterschiedlich ist.
            def _makeLambda( val = value ):
                return lambda *args, **kwargs : _cmd( val, *args, **kwargs )
            
            _d[key] = _makeLambda()
            _d[key].__doc__ = value.__doc__
        
          #else:
          #  print("not exported",key, value)
        
        del key, value

    finally:
        if _config.isDvlp:
            _fout.close()
        
else: # !plotActive
    with open( _funcFn, 'rt' ) as _fin:
        for line in _fin:
            _d[line.strip()] = lambda *args, **kwargs : _do_nothing
    

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
    #points3d(x,y,z,s,mode='sphere',scale_factor='.1',scale_mode='none')

    # gibt man s gleich gar nicht an, dann kann man sich scale_mode='none' sparen
    # gibt man s an, und scale_mode aber nicht, dann gibt scale_factor den maximalen Durchmesser der Kugeln an

    #points3d(x,y,z,mode='sphere',scale_factor='.1')
    dummy = 1