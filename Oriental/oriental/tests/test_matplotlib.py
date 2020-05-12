# -*- coding: cp1252 -*-
import environment
from oriental import config
from oriental.utils import ( pyplot as plt,
                             mplot3d )
import numpy as np
import time
import unittest

@unittest.skip("Crashes upon shutdown when called from commandline.")
class TestMatplotlib(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        plt.close('all')

    def tearDown(self):
        time.sleep(2)

    def test_2d(self):
        y = np.array([ 2., 5., 1. ])
        res = plt.figure(2)
        res = plt.plot(y)
        #print(res)

    @unittest.skip("2015-05-20: This crashes the interpreter. We don't use 3d-plotting anyway.")
    def test_3d(self):
        imPts = np.random.rand( 100, 3 )*( 50, 80, 10 )
        plt.figure()
        mplot3d.scatter3D( imPts[:,0], imPts[:,1], imPts[:,2] )
        plt.show()

    @unittest.skipUnless( config.isDvlp and config.ide,
                          "memory leak is only present with oriental.config.redirectedPlotting=True")
    def test_memoryLeak(self):
        """2014-10-28: if redirectedPlotting is on (i.e. plotting over IPython-kernel), then the local process consumes more and more RAM, while the remote (IPython-) process'es RAM consumption is stable.
        TODO: check with different debugger!
        20150324 this memory leak seems to have gone! (Now tested with Python 3.4, and current Python package versions)
        """

        a = np.arange(7000**2).reshape( (7000,7000) )
        for idx in range(10):
            res = plt.figure(1)
            res = plt.clf()
            b = a.copy()
            res = plt.imshow(b)
            # watch the RAM consumption grow ...

if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest=None,#'TestMatplotlib.test_memoryLeak',
                       exit=False )
