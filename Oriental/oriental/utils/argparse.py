# -*- coding: cp1252 -*-

import oriental
from oriental import log

import ctypes, os, platform, socket, sys, struct
import argparse, traceback
from pathlib import Path
from enum import Enum, EnumMeta
from itertools import zip_longest
import inspect
from datetime import datetime

import contracts
from contracts import contract, new_contract

new_contract('Path',Path)
new_contract('ArgumentParser',argparse.ArgumentParser)

logger = log.Logger("utils.argparse")

def exitCode(script):
    tProcessing = datetime.now()
    try:
        script()
    except SystemExit:
        # SystemExit is e.g. raised when the --help text of a parser has been requested, or unrecognized arguments were given.
        # We do not print a traceback then, especially not to the docs.
        # Note that we neither want to print licence information then (distracting, esp. in docs),
        # which would be triggered e.g. by logging something else (e.g. processing time).
        raise
    except: # catch all other exceptions, including KeyboardInterrupt
        # Make the output better readable by filtering out calls into pycontracts and the preceding decorator-generator calls
        def filterStack(msgs):
            if contracts.all_disabled():
                for msg in msgs:
                    yield msg
                return
            for curr, next in zip_longest( msgs, msgs[1:] ):
                if curr[0].startswith( '<decorator-gen-' ) and next is not None and next[0].startswith( contracts.__path__[0] ):
                    continue
                if curr[0].startswith( contracts.__path__[0] ) and next is not None:
                    continue
                yield curr

        klass, exc, tb = sys.exc_info()
        msgs = traceback.extract_tb(tb)
        msg = ''.join( ( 'Traceback (most recent call last):\n',
                         *traceback.format_list( filterStack(msgs) ),
                         *traceback.format_exception_only(klass,exc) ) )
        logger.error( msg )

        # If only after a long processing time an exception has been thrown,
        # the total processing time may still be of interest, even though unsuccessful.
        logger.info( datetime.now() - tProcessing, tag="Total processing time" )

        # Use a common exit code for Python exceptions, such that scriptLauncher can distinguish them from general OS errors that happen during initialization of the Python interpreter, or when searching for the script file to be executed.
        # Thus, this exit code should be different from all exit codes used by Windows and Python.
        # Windows exit codes: https://msdn.microsoft.com/en-us/library/windows/desktop/ms681381%28v=vs.85%29.aspx
        # Python exit codes: https://docs.python.org/3.4/library/errno.html
        # Do not print a traceback again.
        sys.tracebacklimit = 0
        # Avoid implicit exception chaining. Otherwise, the already printed exception would be printed another time.
        raise SystemExit(-1) from None
        #sys.exit(-1)
        #raise SystemExit(-1).with_traceback(None) # without the already printed traceback
    else:
        logger.info( datetime.now() - tProcessing, tag="Total processing time" )
    finally:
        log.flush()
        log.closeLogFile()

@contract
def ArgParseEnum(name : str, names : str ):
    '''return an Enum whose enumerators have auto-generated values of type str
       This is necessary for str being convertible to that enum, which is done by argparse, if an argument type is of the returned enum-type:
       e.g.: FeatureType('sift') 
       
       Use the functional API of Enum, but pass a list of pairs instead of a simple string of names, where each pair defines an enumerator and its value
       
       equivalent to:
       @unique
       class MyEnum( Enum ):
           sift = 'sift'
           surf = 'surf'
       '''
    #return Enum(name,[(el,el) for el in names.replace(',', ' ').split()])
    res = Enum(name,[(el,el) for el in names.replace(',', ' ').split()])
    res.__str__ = lambda x: x._name_ # unlike enum.Enum's __str__, omit the class name (distracting in help screens)
    res.__repr__ = res.__str__ # also overwrite __repr__, because we may need to print lists of ArgParseEnum objects. list.__str__ calls el.__repr__ for each of its items.
    return res

class Formatter( argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter ):
    # for enums, display the list of possible values without prefixing them with their enum-class'es name: 'sift' instead of 'FeatureType.sift'.
    # This seemed to work, but did not work for ortho.py's DenseMatcher, for no obvious reason.
    # Thus, replace ArgParseEnum's __str__ with a respective implementation, which does the right job, always!
    #def _metavar_formatter(self, action, default_metavar):
    #    # inspect action.type instead of action.choices
    #    # -> for enums, parser.add_argument(.) does not need to be called with choices=FeatureType
    #    # because: validation of input is done anyway during conversion of the cmdline-argument to the enum (e.g. 'sift'->FeatureType.sift)
    #    # our function here makes the choices be displayed on the help screen.
    #    if issubclass(type(action.type),EnumMeta):
    #        action.choices = [ choice.name for choice in action.type ]
    #    return super()._metavar_formatter( action, default_metavar )

    # print the argument type (if specified) instead of the argument name in upper case, e.g.
    # --outDir Path
    # instead of
    # --outDir OUTDIR
    # don't derive from MetavarTypeHelpFormatter, because:
    # - it crashes if no type has been specified for an option
    # - also for positional options, it outputs the type instead of the name i.e. the name of the optional argument will not be displayed at all

    # argparse.HelpFormatter doesn't try to find out the console width on Windows, but it simply queries _os.environ['COLUMNS'], which is defined on Linux only.
    def __init__( self, **kwargs ):
        if platform.system() == 'Windows':
            try:
                # stdin handle is -10
                # stdout handle is -11
                # stderr handle is -12
                h = ctypes.windll.kernel32.GetStdHandle(-12)
                csbi = ctypes.create_string_buffer(22)
                res = ctypes.windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
                if res:
                    (bufx, bufy, curx, cury, wattr,
                     left, top, right, bottom,
                     maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
                    kwargs['width'] = right - left + 1
                    kwargs['width'] -= 2 # argparse.HelpFormatter.__init__ does that, too.
            except:
                pass
        super().__init__( **kwargs )

    def _get_default_metavar_for_optional(self, action):
        if hasattr( action.type, '__name__' ):
            return action.type.__name__
        return super()._get_default_metavar_for_optional( action )

@contract
def addLoggingGroup( parser : argparse.ArgumentParser, logFn : 'Path|str', defaults : 'dict(str: *)|None' = None ):
    defaults = defaults or {}
    loggingGroup = parser.add_argument_group('Logging')
    loggingGroup.add_argument( '--logFile', type=Path, default=defaults.get( 'logFile', Path(logFn) ),
                               help='Log file path. If path is not absolute, then in <outDir>' )
    loggingGroup.add_argument( '--logScreenMinSeverity', type=str, default=defaults.get('logScreenMinSeverity','info'), choices=log.Severity.names.keys(),
                               help='Minimum severity above which log messages are logged to screen' )
    loggingGroup.add_argument( '--logFileMinSeverity', type=str, default=defaults.get('logFileMinSeverity','verbose'), choices=log.Severity.names.keys(),
                               help='Minimum severity above which log messages are logged to file' )
    loggingGroup.add_argument( '--logUnbuffered', action='store_true',
                               help="Use unbuffered streams for log output. Even in case of application crash, all log messages will be logged. Useful for debugging." )
    return loggingGroup

@contract
def applyLoggingGroup( args : argparse.Namespace, outDir : 'Path|str', logger : log.Logger, cmdLine : 'list(str)' ):
    if not args.logFile.is_absolute():
        args.logFile = Path(outDir) / args.logFile
    log.buffered( not args.logUnbuffered )
    log.setLogFileName( str(args.logFile) )
    log.setScreenMinSeverity( log.Severity.names[args.logScreenMinSeverity] )
    log.setFileMinSeverity  ( log.Severity.names[args.logFileMinSeverity  ] )

    logger.infoScreen( "Saving log to file '{}'", args.logFile )
    logger.verboseFile( 'Startup environment\n'
                        'name\tvalue\n'
                        '{}',
                        '\n'.join( '\t'.join( els ) for els in (
                            ( 'buildDate', oriental.__date__  ),
                            ( 'PC'       , socket.getfqdn()   ), # fully qualified domain name for localhost: PC-name + domain name, if present.
                            ( 'platform' , platform.platform() ),
                            ( 'PATH'     , os.environ.get('PATH','') ),
                            ( 'cwd'      , str(Path.cwd())    ) ) ) )
    if cmdLine:
        logger.infoFile( ' '.join(cmdLine), tag='commandline' )

@contract
def logScriptParameters( args : argparse.Namespace, logger : log.Logger, parser : 'ArgumentParser|None' = None ):
    if parser is None:
        logger.infoFile(  'script parameters\n' +
                          'name\tvalue\n' +
                          '\n'.join(( '{}\t{}'.format(*item) for item in sorted( vars(args).items(), key= lambda x: x[0] ) )) )
        return

    # Output `args` in the grouping and order defined by parser. Unfortunately, we must use argparse's private definitions for that.
    def getTables():
        groupName = None
        rows = []
        for action in parser._actions:
            if action.dest not in args:
                continue
            if groupName != action.container.title:
                if groupName is not None:
                    yield groupName, rows
                groupName = action.container.title
                rows = []
            rows.append(( action.dest, getattr( args, action.dest ) ))

    logger.infoFile( 'Script parameters\n'
                     '{}',
                     '\n'.join( ('group\tname\tvalue',) + tuple( '\t'.join( str(el) for el in (groupName,*row) )
                                                                  for groupName,rows in getTables() for row in rows ) ) )





