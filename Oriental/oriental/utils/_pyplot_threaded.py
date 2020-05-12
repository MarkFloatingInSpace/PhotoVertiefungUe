# -*- coding: cp1252 -*-
from . import config as _config
from os.path import dirname as _dirname, join as _join

# the only clean solution for installing dummy-functions: import the module, and store the function names to be used.
# When the module shall not be imported, read the list of function names from that file.
_funcFn = _join( _dirname(__file__), 'matplotlib.pyplot-funcnames.pickle' )

_d = globals()

def _do_nothing():
    pass

if _config.plotActive:
    import matplotlib as _matplotlib
    if _config.gui == 'Wx'.lower():
        _matplotlib.use("WxAgg")
    elif _config.gui == 'Qt'.lower():
        _matplotlib.use("Qt4Agg")
        _matplotlib.rcParams['backend.qt4'] = "PySide"
    from GuiThread import cmd as _cmd
    from matplotlib import pyplot as _pyplot
    from inspect import isfunction as _isfunction
    
    
    # funzt, aber für alle Funktionen in matplotlib.pyplot wäre das umständlich und wartungsintensiv
    #def scatter(*args,**kwargs):
    #  return _cmd(matplotlib.pyplot.scatter,*args,**kwargs)
    
    #def figure(*args,**kwargs):
    #  return _cmd(matplotlib.pyplot.figure,*args,**kwargs)
    
    #_d = globals()
    # funzt, aber für alle Funktionen in matplotlib.pyplot wäre das umständlich und wartungsintensiv
    #d["scatter"] = lambda *args, **kwargs : matplotlib.pyplot.scatter( *args, **kwargs )
    if _config.isDvlp:
        _fout = open( _funcFn, 'wt' )
        
    try:
        for key, value in _pyplot.__dict__.items():
            # Funktionen, die mit '_' sind in Python meist 'interne' Funktionen (private)
            if key[0] is '_' or not _isfunction(value):
                continue
            
            if key in _d:
                raise Exception("Overwriting existing function '{}'!?".format(key))
            
            if _config.isDvlp: 
                _fout.write("{}\n".format(key))
                
            # show() blocks and is not necessary. Thus, map it to an empty function
            if value == _pyplot.show:
                _d[key] = _do_nothing
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
            def makeLambda( val = value ):
                return lambda *args, **kwargs : _cmd( val, *args, **kwargs )
            
            _d[key] = makeLambda()
            _d[key].__doc__ = value.__doc__
    
      #else:
      #  print("not exported",key, value)
    
    finally:
        if _config.isDvlp:
            _fout.close()
            
else: # !plotActive

    # must not import matplotlib if debugging .pyd's in MSVC shall work!
    # there seems to be not way to query the functions that a module defines without importing it first,
    # so parse the file, looking for
    # def xxx():

##    #from imp import find_module as _find_module
##    from re import compile as _compile
##    from os.path import join as _join, dirname as _dirname
##    #from sys import path as _path
##    #from sys import executable as _executable
##    
##
##    regex = _compile(r"^def\W+(\w+)")
##    
##    # even a call to imp.find_module("matplotlib") inhibits debugging !?
##    # even the import of imp.find_module inhibits debugging !?
##    #(file, pathname, description) = _find_module("matplotlib")
##
##    # importing sys.executable is okay, but not using it?
##    #pathname = _join( _dirname(_executable), r"Lib\site-packages\matplotlib" )
##    
##    # that seems to work, at least:
##    pathname = r"C:\Python27-x64\Lib\site-packages\matplotlib"
##    with open( _join( pathname, "pyplot.py" ), "rt" ) as fin:
##        for line in fin:
##            match = regex.search( line )
##            if match is None:
##                continue
##            key = match.group(1)
##            if key.startswith("_"):
##                continue
##            _d[key] = lambda *args, **kwargs : _do_nothing
  
    with open( _funcFn, 'rt' ) as _fin:
        for line in _fin:
            _d[line.strip()] = lambda *args, **kwargs : _do_nothing
                    
if __name__ == "__main__":
    import random
    scatter = globals()['scatter'] # make Cython happy
    axhline = globals()['axhline']
    figure = globals()['figure']
    x = [random.random() for var in range(10)]
    y = [random.random() for var in range(10)]
    fig = figure()
    #fig.add_subplot(1,2,1)
    scatter( x, y, marker='x')
    
    scatter( x, y, marker='D')
    
    axhline(y=.5,linewidth=4, color='r')
    
    figure(3)
    scatter( x, y, marker='x')

### funzt!
### aber: code muss als string übergeben werden :(
##from IPython.zmq.blockingkernelmanager import BlockingKernelManager
##import matplotlib
##import matplotlib.pyplot
###matplotlib.use('Qt4Agg')
###from matplotlib import pyplot
##from inspect import isfunction
##
##km = BlockingKernelManager()
##km.start_kernel(extra_arguments=["--pylab"])
##km.start_channels()
##
###from IPython.zmq.eventloops import enable_gui
###enable_gui('tk',km.kernel)
##
##def run_cell(code):
##    # now we can run code.  This is done on the shell channel
##    shell = km.shell_channel
##    #print( "running: {}".format(code) )
##
##    # execution is immediate and async, returning a UUID
##    msg_id = shell.execute("res="+code,user_variables=["res"])
##    # get_msg can block for a reply
##    reply = shell.get_msg()
##
##    status = reply['content']['status']
##    if status == 'ok':
##        pass
##        #return reply['content']["user_variables"]["res"]
##    elif status == 'error':
##        for line in reply['content']['traceback']:
##            print line
##        raise Exception(reply['content']['evalue'])
##
##def cmd( func, *args, **kwargs ):
##    cmd = func.__name__ + "("
##    sargs = [ '"' + arg + '"' if isinstance(arg, basestring) else str(arg) for arg in args]
##    cmd += ",".join(sargs)
##    skwargs = [ key + '=' + ('"' + val + '"' if isinstance(val, basestring) else str(val)) for key,val in kwargs.items() ]
##    if len(skwargs):
##       if len(sargs):
##          cmd += ','
##       cmd += ",".join(skwargs)
##    cmd += ")"
##    res = run_cell( cmd )
##    #return res
##
### funzt, aber für alle Funktionen in matplotlib.pyplot wäre das umständlich und wartungsintensiv
###def scatter(*args,**kwargs):
###    cmd(pyplot.scatter,*args,**kwargs)
##
