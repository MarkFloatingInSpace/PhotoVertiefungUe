# -*- coding: utf-8 -*-
import environment
from oriental import config, Progress
import oriental
if config.debug:
    from oriental._oriental_d import _test
else:
    from oriental._oriental import _test

from pathlib import Path
import numpy as np
import unittest

class TestConverters(unittest.TestCase):

    #def test_matx(self):
    #    arr1 = np.arange(6, dtype=float).reshape((2,-1))
    #    arr2 = np.ones(6, dtype=float).reshape((2,-1))
    #    arr = _test.matxD23Add( arr1, arr2 )
    #    np.testing.assert_allclose( arr, arr1+arr2 )

    def test_matx_member(self):
        "check class member variable access"
        obj = _test.WithMatxD23member()
        matxd23 = obj.mat
        np.testing.assert_array_equal( matxd23, [[0,0,0],[ 0, 1, 2 ]], "WithMatxD23member initializes its 'mat' to that" )

        arr = np.arange(6, dtype=float).reshape((2,-1)) + 10
        obj.mat = arr
        arr2 = obj.mat
        np.testing.assert_array_equal( arr, arr2 )
        arr2[0,0]=-3
        np.testing.assert_array_equal( arr2, obj.mat, "matxd23 does not reference WithMatxD23member.mat" )
        dummy=0

    def test_vec3d_member(self):
        obj = _test.WithVec3dMember()
        vec3d = obj.vec
        np.testing.assert_array_equal( vec3d, [ 0, 1, 2 ], "WithVec3dMember initializes its 'vec' to that" )
        vec3d += 1
        np.testing.assert_array_equal( vec3d, obj.vec, "vec3d does not reference WithVec3dMember.vec" )
        obj.vec = np.array([3.,4.,5.])

    def test_path(self):
        p = Path(r'D:\data')
        p2 = _test.handlePath(p)
        self.assertEqual( str(p), p2 )

    def test_pathList(self):
        p = [ Path(r'D:\data'), Path(r'D:\data2') ]
        p2 = _test.handlePathVector(p)
        self.assertEqual( [str(el) for el in p], p2 )

    def test_sharedVoidPointer(self):
        data = ( 1, 'hallo' )
        data2 = _test.acceptAnyData( data )
        self.assertEqual( data, data2 )

class TestExceptions(unittest.TestCase):
    
    def test_printNonAscii(self):
        print('èé')

    def test_exception(self):
        self.assertRaises( oriental.Exception, _test.throwException, "Let's raise." )

    def test_exceptionWithScope(self):
        self.assertRaises( oriental.Exception, _test.throwExceptionWithScope, "Let's raise." )

    def test_exceptionWithScopes(self):
        self.assertRaises( oriental.Exception, _test.throwExceptionWithScopes, "Let's raise." )

@unittest.skip("Screen buffer overflow is an unresolved problem. So far, we just increase the buffer size to a value that is hopefully large enough.")
class TestProgress(unittest.TestCase):
    def test_windowBufferOverflowBefore(self):
        for idx in range(500): # this value should be larger than the number of lines that the console's screen buffer size, such that we print beyond that.
            print(idx)
        progress = Progress(100)
        for idx in range(100):
            progress += 1

    def test_windowBufferOverflowAfter(self):
        progress = Progress(100)
        for idx in range(500): # this value should be larger than the number of lines that the console's screen buffer size, such that we print beyond that.
            print(idx)
        for idx in range(100):
            progress += 1

class TestOpaque(unittest.TestCase):
    def get_opaque(self, problem):
        op1 = problem.getOpaque()
        s = str(op1)
        r = repr(op1)
        op2 = problem.getOpaque()
        op3 = problem.getOpaque()
        self.assertIsNot( op1, op2 )
        self.assertEqual( op1, op2 )
        self.assertEqual( hash(op1), hash(op2) )
        dummy=0

    def get_opaques(self,problem):
        problem.getOpaques()
        dummy=0

    def test_oqaque(self):
        problem = _test.Problem()
        self.get_opaques(problem)
        self.get_opaque(problem)
        dummy=1

if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestOpaque.test_oqaque',
                       exit=False )