# -*- coding: cp1252 -*-
from . import plot_environment

"""2016-04-01 The following is supposed to be a solution that avoids the now deprecated functions in IPython
after the 'big split' into ipyparallel and jupyter. However, there is still a ton of deprecations warnings.
Maybe try again with the next versions of IPython, etc."""
#import subprocess
#ipclusterProc = subprocess.Popen( ['ipcluster', 'start', '-n', '1'], shell=True )
#from ipyparallel import Client
#rc = Client()
#dview = rc[0]
#dview.block = True
#dview.execute('%matplotlib tk')
#dview.execute('import matplotlib.pyplot as plt')
#import matplotlib.pyplot as plt
##dview.apply(plt.ion)
#res = dview.apply(plt.figure)
#import numpy as np
#img = np.random.randn(100,150)
#res = dview.apply(plt.imshow,img)
#dview.execute('plt.gcf().canvas.draw()')

# C:\>IPython kernel
# starts IPython.kernel.zmq.kernelapp.IpKernelApp
# whose kernel is of type
# IPython.kernel.zmq.ipkernel.Kernel
# However, we cannot directly start IpKernelApp here, as that would transfer control to that app (it starts the event loop).
# IPython.kernel.manager.KernelManager starts its kernel with e.g.:
# C:\>C:\\Python33-x64\\python.exe -c' from IPython.kernel.zmq.kernelapp import main; main()' -f c:\\users\\wk\\appdata\\local\\temp\\tmpowp94y.json' --pylab tk
# i.e. it simply starts IpKernelApp in a sub-process.
# But how to communicate with that sub-process?
# KernelManager's client provides no convenient functionality to execute functions remotely,
#   but only to send raw Python code to the kernel, using its shell_channels's base-class method IPython.kernel.channels.ShellChannel.execute
#   Seemingly, the IPython frontends use this cliend to directly send all input as strings to their kernel,
#   instead of importing Python-function-handles, serializing them, and then sending serialized versions, including function arguments to the kernel.
# Only IPython.parallel seems to provide the functionality to import functions locally (in the frontend/client) and execute them remotely (on the kernel).
# However, IPython.parallel.client connects to IPython clusters only, whose kernels/engines seem to not be able to integrate GUI event loops
# - according to the idea of parallel processing, that wouldn't make sense either: 1 GUI in the frontend to display results from computations executed in remote engines! 
# Thus, launch an IPython.kernel.manager.KernelManager, start its kernel, and retrieve a client
# using the client, execute functions remotely in a similar way as IPython.parallel.client does.
#from IPython.kernel.manager import KernelManager as _BlockingKernelManager
from jupyter_client.manager import KernelManager as _BlockingKernelManager, start_new_kernel as _start_new_kernel
#from IPython.kernel.inprocess.manager import InProcessKernelManager as _BlockingKernelManager

#from IPython.kernel.zmq import serialize as _serialize
import ipykernel.serialize as _serialize

#from IPython.parallel.error import RemoteError, unwrap_exception
#from ipyparallel.error import unwrap_exception as unwrap_exception # better use our own function here instead of introducing the dependency on ipyparallel

from ipython_genutils.text import strip_ansi

from .. import config as _config

from os.path import dirname as _dirname

if _config.plotUseDill: # this requires dill to be available. And it doesn't work - error: 'module' object has no attribute '_plot'
    import IPython.utils.pickleutil
    IPython.utils.pickleutil.use_dill()
else:
    # -------------------------
    # Class methods are not picklable by default!
    # However, we need them to be, e.g. for creating 3D-plots:
    # fig = plt.figure()
    # fig.add_subplot( 1, 1, 1, projection='3d' )
    import pickle
    import types
    import copyreg
    
    class MethodProxy(object):
        def __init__( self, obj, name ):
            self.obj = obj
            self.methodName = name
        
        def __call__(self, *args, **kwargs):
            return getattr(self.obj, self.methodName)(*args, **kwargs)
    
    def _pickle_method( method ):
        return ( MethodProxy, ( method.__self__, method.__name__ ) )
    
    copyreg.pickle( types.MethodType, _pickle_method )
# -------------------------

extra_arguments = [ "--pylab" ]
if _config.gui:
    extra_arguments.append( _config.gui )

kernelManager, client = _start_new_kernel( extra_arguments=extra_arguments )
#kernelManager = _BlockingKernelManager()
#kernelManager.start_kernel( extra_arguments=extra_arguments )
##kernelManager.start_kernel(extra_arguments=["--pylab=" + _config.gui, "-c import oriental\nimport mayavi.mlab\n"])
#assert kernelManager.is_alive()
#
#client = kernelManager.client()
#client.start_channels()
#assert client.channels_running

if _config.plotUseDill:
    client.shell_channel.execute("""
        import IPython.utils.pickleutil
        IPython.utils.pickleutil.use_dill()""")

def unwrap_exception(content):
    # TODO: if stdout is connected to an IPython-shell, then the ANSI color codes are interpreted as such.
    # If not, the output seems to be garbage.
    # If we knew how to determine if stdout supports ANSI color codes, then we could introduce a switch that strips the color codes using `strip_ansi``only if stdout does not support them.

    #import sys
    #strip_ansi = lambda x: x if 'pyreadline' in sys.modules else strip_ansi
    #import IPython.utils.rlineimpl as readline
    #readline.have_readline

    return Exception( '{}:\n'
                      '{}\n'
                      'Traceback:\n'
                      '{}\n'.format( content['ename'],
                                     content['evalue'],
                                     strip_ansi( '\n'.join(content['traceback']) ) ) )


# If func defines a nested function, then pack_apply_message fails!
# In that case, define another function that calls func
# and use functools.partial( apply, func2 )
# test:
def apply(func, *args, **kwargs ):
    # func.__module__ must be in oriental-namespace. otherwise, func's code may not be picklable by IPython/pickle (IPython registers pickle/unpickle functions for function code objects, which cannot pickle functions with closures)
    #assert func.__module__.startswith("oriental")
    global client
    if not kernelManager.is_alive():
        kernelManager.restart_kernel()
        assert kernelManager.is_alive()
        client = kernelManager.client()
        client.start_channels()
        assert client.channels_running
    
    # ----------------
    # see IPython.parallel.client.client.send_apply_request
    bufs = _serialize.pack_apply_message( func, args, kwargs )
    if 0:
        f2,a2,kw2 = _serialize.unpack_apply_message( bufs )
        f2( *a2, **kw2 )
    
    msgSent = client.session.send( client.shell_channel.socket, "apply_request", buffers=bufs )
    msg_id = msgSent['msg_id']
    # ----------------
        
    # ----------------
    # see IPython.parallel.client.client._handle_apply_reply
    while True:#client.shell_channel.msg_ready():
        #answerShell = client.shell_channel.get_msg(block=True)
        # or:
        answerShell = client.shell_channel.get_msg(block=True)
        if answerShell.get('parent_header',{}).get('msg_id') ==  msg_id:
            break

    # trying to overcome the memory leak. Doesn't help!
    while client.iopub_channel.msg_ready():
        answerIOPub = client.iopub_channel.get_msg(block=True)
        if answerIOPub.get('parent_header',{}).get('msg_id') ==  msg_id:
            break
   
    if 'ename' in answerShell.get('content',{}):
        # Generally, exceptions are only returned when they should be (e.g. import error on the remote side).
        # However, when e.g. matplotlib.pyplot.title("text") or matplotlib.pyplot.ylabel are used,
        # then that command succeeds (title is displayed),
        # but a TypeError('expected string or Unicode object, NoneType found') is returned,
        # even though the same command does not generate an exception when executed in a (local) IPython session.
        # As a work-around, return None instead of re-raising the exception.
        
        if answerShell['content']['ename'] == 'PicklingError':
            return None
        
        # e.g. missing argument, etc.; inform user about it!
        raise unwrap_exception( answerShell['content'] )
    
    # For whatever reason, unserialization may fail. However, we often don't need the return value anyway.
    try:
        return _serialize.deserialize_object( answerShell['buffers'] )[0]
    except:
        return None
    finally:
        # trying to overcome the memory leak. Doesn't help!
        handle_msgs()
    # ----------------
    
    # will the message queues overflow, if not emtied?
    #handle_msgs() 

def _transfer( variableName, variableValue ):
    globals()[variableName] = variableValue

def transfer( variableName, variableValue ):
    apply( _transfer, variableName, variableValue )
    
    #handle_msgs()
    
def handle_msgs():
    while client.stdin_channel.msg_ready():
        msg = client.stdin_channel.get_msg(block=True, timeout=2)
    while client.shell_channel.msg_ready():
        msg = client.shell_channel.get_msg(block=True, timeout=2)
        #if 'traceback' in msg['content']:
        #    for l in msg['content']['traceback']:
        #        print( str(l) )
            
            # IPython tries to pickle the return types, which fails and raises an exception, so don't re-raise them here.
            #raise Exception("""Remote exception
            #                   type: {}
            #                   message: {}
            #                   traceback: {}
            #                """.format( msg['content']['ename'], msg['content']['evalue'], msg['content']['traceback'] ) )
    while client.iopub_channel.msg_ready():
        msg = client.iopub_channel.get_msg(block=True, timeout=2)

# wait for kernel to be ready
#kernelManager.shell_channel.execute("pass")
#kernelManager.shell_channel.get_msg(block=True)
#kernelManager.shell_channel.get_msg(block=True, timeout=5)
#handle_msgs()