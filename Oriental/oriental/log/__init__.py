# -*- coding: cp1252 -*-
"""
Send log messages to screen and/or file.

Emit log messages using an instance of :class:`~_log.Logger`. 

All instances of :class:`~_log.Logger` send their messages to an application-wide logging kernel singleton. 2 :class:`sinks <_log.Sink>` are attached to this kernel:
   - the screen, and
   - the XML log file.

Call :func:`~_log.selectSinks` to choose a subset of these sinks, to which the kernel will forward its messages.

Set the XML log file's path using :func:`~_log.setLogFileName`.

:func:`~_log.setFileMinSeverity` and :func:`~_log.setScreenMinSeverity` set the application-wide minimum :class:`~_log.Severity`\\ s of log messages. Only log messages whose severity is higher or equal to this threshold will actually be forwared to sreen and/or file.
"""

from .. import config as _config
import functools as _functools, string as _string
#import sys as _sys

if _config.debug:
    from ._log_d import *
    from ._log_d import __date__
else:
    from ._log import *
    from ._log import __date__

def _partialmethod( func, *args1, **kwargs1 ):
    """It's impossible to set a docstring on a partialmethod object, because partialmethod
    is a class written in Python, and instances of classes get their docstring
    from the class's __doc__, not from a __doc__ attribute on the instances.
    Functions behave differently, with the __doc__ attribute of the function object being looked at.
    Since assigning __doc__ to the partial method fails anyway, let's not try at all,
    since it leads to a crash for an unknown reason when applied to Logger.raw"""
    @_functools.wraps(func, assigned = set(_functools.WRAPPER_ASSIGNMENTS) -  {'__doc__'} )  # copy attributes to start, they can be overwritten later
    def method(self, *args2, **kwargs2):
        return func(self, *args1, *args2, **kwargs1, **kwargs2)
    return method

# Even in the console, it may be preferable to log via PySys_WriteStdout:
# e.g. otherwise, the output of unittests is partly overwritten with log messages.
# However, the output won't be colored.
if _config.ide:  
    logToRedirectedStdout()    

# isatty( _fileno( stdout ) ) 
# returns true for the debugger consoles of PyScripter and PyCharm.
#         false for the debugger console of PTVS and, of course, for cmd.exe
# If we do not re-set the screen sink to use PySys_WriteStdout, then
# - in PyScripter, logs will not be visible at all on screen,
# - in PyCharm, logs will be visible on screen (without our font colors), but with 1 blank after each character: t e x t
# - in PTVS, logs will be visible in the console that opens in a separate window, but not in the "Python debug interactive" window within Visual Studio,
#   which may be an advantage!
#try:
#    _stdoutIsRedirected = not _sys.stdout.isatty()
#except AttributeError: # PyScripter
#    _stdoutIsRedirected = True
#if _stdoutIsRedirected:
#    logToRedirectedStdout()

class _Formatter(_string.Formatter):
    "Raise if too many positional or unused keyword arguments are supplied."
    def check_unused_args(self, used_args, args, kwargs):
        maxIdx = max( (el for el in used_args if isinstance(el,int)), default=-1 )
        usedKwargs = set( el for el in used_args if isinstance(el,str))
        if len(args) > maxIdx+1:
            raise IndexError(f'Unused positional argument with index {maxIdx+1}: "{args[maxIdx+1]}".')
        unusedKwargs = set(kwargs) - usedKwargs
        if unusedKwargs:
            raise KeyError('Unused keyword argument(s) {}: "{}"'.format( ', '.join(unusedKwargs),
                                                                         ', '.join( str(kwargs[key]) for key in unusedKwargs ) ) )

def _multiRecordPumpCall( self, msg, *args, **kwargs ):
    if not args and not kwargs:
        # If no formatting arguments are given, then msg is most probably not to be formatted.
        # Avoid errors due to msg containing curly braces that are not meant to be format placeholders.
        self.call( str(msg) )
    else:
        self.call( _Formatter().format( str(msg), *args, **kwargs ) )

MultiRecordPump.__call__ = _multiRecordPumpCall

def _record( self, severity, sink = Sink.all, tag = "" ):
    recPump = MultiRecordPump( self, severity, sink, tag )
    if not recPump:
        recPump.close()
        return
    yield recPump
    recPump.close()

Logger.record = _record
Logger.recDebug   = _partialmethod( _record, severity=Severity.debug   )
Logger.recVerbose = _partialmethod( _record, severity=Severity.verbose )
Logger.recInfo    = _partialmethod( _record, severity=Severity.info    )
Logger.recWarning = _partialmethod( _record, severity=Severity.warning )
Logger.recError   = _partialmethod( _record, severity=Severity.error   )

def _log( self, severity, sink, msg, *args, **kwargs ):
    recPump = MultiRecordPump( self, severity, sink, kwargs.pop( 'tag', '' ) )
    if not recPump:
        recPump.close()
        return
    recPump( msg, *args, **kwargs )
    recPump.close()

def _logSink( self, sink, severity, msg, *args, **kwargs ):
    _log( self, severity, sink, msg, *args, **kwargs )

Logger.log = _log
Logger.log.__doc__ = "emit a log message with specified severity to specified sink"

Logger.screen = _partialmethod( _logSink, Sink.screen )
Logger.file   = _partialmethod( _logSink, Sink.file )
Logger.screen.__doc__ = "emit a log message with specified severity to screen only"
Logger.file  .__doc__ = "emit a log message with specified severity to file only"

for _name, _value in Severity.names.items():
    _method = _partialmethod( _log, _value, Sink.all )
    _method.__doc__ = f"{_name}( (Logger)arg1, (str)msg, *args, **kwargs [, (str)tag='']) -> None :\n    emit a log message with severity '{_name}' to screen and file."
    setattr( Logger, _name, _method )

    _method = _partialmethod( _log, _value, Sink.screen )
    _method.__doc__ = f"{_name}Screen( (Logger)arg1, (str)msg, *args, **kwargs [, (str)tag='']) -> None :\n    emit a log message with severity '{_name}' to screen only"
    setattr( Logger, _name + 'Screen', _method )

    _method = _partialmethod( _log, _value, Sink.file )
    _method.__doc__ = f"{_name}File( (Logger)arg1, (str)msg, *args, **kwargs [, (str)tag='']) -> None :\n    emit a log message with severity '{_name}' to file only"
    setattr( Logger, _name + 'File', _method )

    _method = _partialmethod( Logger.raw, _value )
    _method.__doc__ = f"{_name}Raw( (Logger)arg1, (str)msg [, (str)tag='']) -> None :\n    emit a log message with severity '{_name}' unmodified to file only"
    setattr( Logger, _name + 'Raw', _method )

def clearLogFileName():
    """useful for multiprocessing, if child processes' file logging sink shall be deactivated.

       Deactivation of logging to file in sub-processes is generally needed, because :mod:`~oriental.log` does not support access to the same XML-log by multiple processes!
       There are 2 work-arounds:
       - suppress output to file by setting an empty log file name
       - log to arbitrary, temporary files.
       - cleanest would be to collect and return the log messages from sub-processes, and then log them in the main process.
        
       There is no robust way to determine if a process is a child process created by multiprocessing. However, if child processes are created with no explicit name, then their names match the following::

          if re.match(r'Process-\d+', multiprocessing.current_process().name):
              log.setLogFileName( '' )
       
       Even though this not-so-robust check could be done in every oriental function (mainly plotting), we better initialise multiprocessing.Pool with a process-initializer-function as argument
       
       multiprocessing.Pool tries to pickle its initializer and initargs, which is impossible for boost.Python functions. Thus, we cannot init Pool as Pool( initializer=log.setLogFileName, initargs="" ), 
       but define here a function implemented in Python to do the job.
       
       usage::

          with multiprocessing.Pool( initializer=log.clearLogFileName ) as pool:
              pass
    """
    setLogFileName("")

def suppressCustomerClearLogFileName():
    "multiprocessing.Pool expects its initializer function to be pickleable"
    suppressCustomer()
    clearLogFileName()

#def _infoFrame( self, *args, **kwargs ):
#    import inspect
#    frame,filename,line_number,function_name,lines,index = inspect.stack(context=0)[1]
#    uuid = filename + str(line_number)
#    self.info( *args, **kwargs )
#
#Logger.infoFrame = _infoFrame
