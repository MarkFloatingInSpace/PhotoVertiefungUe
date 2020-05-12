# -*- coding: utf-8 -*-
import environment
import oriental
from oriental import config, log
import os
import shutil
import unittest

class Custom(object):
    def __init__( self, name ):
        self.name = name
    def __str__(self):
        return self.name

class TestLog(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        log.setScreenMinSeverity( log.Severity.debug )
        cls.outDir = os.path.join( os.path.dirname(__file__), "out" )
        if not os.path.exists( cls.outDir ):
            os.mkdir( cls.outDir )
        logFn = os.path.join( cls.outDir, "test_log.xml" )
        if os.path.exists( logFn ):
            os.remove( logFn )
        print('logFn: {}'.format(logFn))
        log.setLogFileName( logFn )

    @classmethod
    def tearDownClass(cls):
        log.setScreenMinSeverity( log.Severity.info )
        log.setLogFileName("orientalLog.xml")
        #shutil.rmtree( cls.outDir )
        pass

    def setUp( self ):
        self.logger = log.Logger("test_logging")

    def tearDown( self ):
        fn = log.getLogFileName()
        #log.closeLogFile()
        #shutil.rmtree( self.outDir ) # removing the directory for every test leads to access/permission errors. Thus, delete only the log file.
        #os.remove( fn )

    def test_log( self ):
        self.logger.debug( "debug {name}", name="message" )
        self.logger.verbose( "verbose number {}", 1 )
        self.logger.info( "info {0} {1}", 'text', 1 )
        self.logger.warning( "warning {0} {1}", 'foo"bar"', 1 )
        self.logger.error( "error {0} {1}", 'foo"bar"', 1 )
    
        c = Custom("customer")
        print(c)
        self.logger.info( "printing Custom instance: {}", c )
    
        self.logger.info( "no args" )
    
        #logger.info( )
    
        #logger.emit( "hello from Python" )
    
        #logger.emit( "verbose", log.Severity.verbose )
        #logger.emit( "Logging to file: {}".format( log.getLogFileName() ) )
    
        #logger.emit( "debug", log.Severity.debug )
        #help(pro)
    
        # Is a percent-sign okay?
        self.logger.info( "Log a percentage formatted beforehand: ({:.0%})".format( float(3)/5 ) )
        self.logger.info( "Log a percentage formatted by oriental.log: ({:.0%})", float(3)/5 )
    
    def test_log_generic( self ):
        self.logger.log( log.Severity.warning, log.Sink.screen, "1 arg {}", 20 )
    
        self.logger.screen( log.Severity.warning, "1 arg {}", 20 )

    def test_log_raw( self ):
          self.logger.infoRaw( "<svg></svg>" )

          self.logger.infoRaw( "<!-- comment --><svg></svg>" )

          if config.debug: # XML well-formedness is only checked for _DEBUG-builds
              self.assertRaises( oriental.Exception, self.logger.infoRaw, "<svg></html>" )
              self.assertRaises( oriental.Exception, self.logger.infoRaw, "no valid XML" )
              self.assertRaises( oriental.Exception, self.logger.infoRaw, "<?xml version='1.0' encoding='ISO-8859-1' ?><svg></svg>" )
              self.assertRaises( oriental.Exception, self.logger.infoRaw, "<?xml-stylesheet type='text/xsl' href='#'?>" )

    def test_log_tag( self ):
        self.logger.info( "tagged hello", tag="tag" )
        self.logger.info( "hello {}", "onePlaceHolder", tag="tag" )
        self.logger.info( "hello {world}", world="namedKeyword", tag="tag" )
    
        self.logger.log( log.Severity.info, log.Sink.all, "hello {}", "generic", tag="tag" )

    @unittest.skip("Unit-tests that use mpl have never really worked")
    def test_log_mpl_svg( self ):
        import numpy as np
        from oriental.utils import pyplot as plt
        from oriental.utils import pyplot_utils as plt_utils
        from oriental.utils.BlockingKernelManager import client
        client.shell_channel.send("matplotlib.rcParams['svg.fonttype'] = 'none'") # don't convert text to paths! Fonts must be installed on viewing machine.

        fig = plt.figure(1, figsize=(4,2) )
        a = np.array([1.,5.,2.])
        plt.plot(a)
        xml = plt_utils.embedSVG()
        #xml = '<br/>'
        self.logger.infoRaw(xml)

        import tempfile, os, base64, shutil, urllib

        plt.figure(10); plt.clf()
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        plt.hist2d( x=x, y=y, bins=50 )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        nFigs = 1000
        if 1:
            # embed
            # Drawback: browsers do not offer 'Save image as...' in the context menu.
            # People recommend to inspect the element (e.g. Firebug), select outer HTML -> save that.
            # Handier might be: a button that links to embedded data (untested):
            # http://en.wikipedia.org/wiki/Data_URI_scheme
            # "In browsers that fully support Data URIs for "navigation", JavaScript-generated content can be provided as file "download" to the user, simply by setting window.location.href to a Data URI. One example is the conversion of HTML tables to downloadable CSV using a Data URI like this: 'data:text/csv;charset=UTF-8,' + encodeURIComponent(csv), where "csv" has been generated by JavaScript."
            if 0:
                # Firefox crashes.
                xml = plt_utils.embedSVG()
                for i in range(nFigs):
                    self.logger.infoRaw(xml)
            else:
                # Firefox loads within a few seconds! Total file size: 20MB
                msg = plt_utils.embedPNG()
                #for i in range(nFigs):
                #    self.logger.infoRaw( msg )
                self.logger.infoRaw( msg * nFigs )

        else:
            # reference external files
            if 1:
                # Firefox okay. Total file size incl. svg's: 87MB
                fn = 'test.svg'
            else:
                # Firefox takes hours to load. Total file size incl. png's: 15MB
                fn = 'test.png'

            plt.savefig( os.path.join( self.outDir, fn ), transparent=True, bbox_inches='tight' )
            for i in range(nFigs):
                p,ext = os.path.splitext( fn )
                fn2 = p + str(i) + ext
                shutil.copyfile( os.path.join( self.outDir, fn ), os.path.join( self.outDir, fn2 ) )
                self.logger.infoRaw( '<img src="{}" />'.format(fn2) )

    def test_log_nonASCII( self ):
        self.logger.info( "microns: [µm]" )
        self.logger.info( "sigma naught: σ_0" )
        # note: we use φ i.e. \N{GREEK SMALL LETTER PHI} throughout OrientAL
        # instead of ϕ i.e. \N{GREEK PHI SYMBOL}, because the latter looks bold and weird in the Ubuntu shell.
        self.logger.info( "angles: ωφκαζΔ" )
        self.logger.info( "Euro: €" )
    
    def test_log_utf8( self ):
        self.logger.info( "Arbitrary non-ASCII European: èéøÞǽлљΣæča" )
        self.logger.info( "Some Chinese: 陈亚男" ) # Chen Yanan
    
    def test_log_table( self ):
        prefix = 'hallo' + '\v'
        self.logger.info( prefix +
                 'Image point statistics\n'
                    '\t#imgPts per image\t#imgPts per objPt\n'
                 'min\t{nImgPtsPerImgMin}\t{nImgPtsPerObjPtMin}\n'
                 'med\t{nImgPtsPerImgMed}\t{nImgPtsPerObjPtMed}\n'
                 'max\t{nImgPtsPerImgMax}\t{nImgPtsPerObjPtMax}',
                 nImgPtsPerImgMin   = 0,
                 nImgPtsPerImgMed   = 1,
                 nImgPtsPerImgMax   = 2,
                 nImgPtsPerObjPtMin = 3,
                 nImgPtsPerObjPtMed = 4,
                 nImgPtsPerObjPtMax = 5 )

    def test_conditional_open( self ):
        for rec in self.logger.record( log.Severity.info ):
            rec( "conditional info message" )
            rec( "conditional info table\n"
                 "idx\tx\ty\n" +
                 "\n".join( "{}\t{}\t{}".format( idx, idx, idx+2 ) for idx in range(4) ) )

        for rec in self.logger.record( log.Severity.debug ):
            rec( "conditional debug message" )

        for rec in self.logger.recInfo():
            rec( "conditional info message" )

        for rec in self.logger.recInfo():
            rec( "conditional info message {}", "formatted" )
        
        dummy=0

if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestLog.test_log_utf8',
                       exit=False )
