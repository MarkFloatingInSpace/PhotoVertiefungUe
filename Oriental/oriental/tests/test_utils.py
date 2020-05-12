# -*- coding: cp1252 -*-
import environment
from oriental import config, utils, ori
from oriental.utils import db, stats, lsm, dsm, zip_equal, IterableLengthMismatch

import os, sys, unittest, tempfile, shutil
import urllib.request
from sqlite3 import dbapi2
import numpy as np
import cv2

from osgeo import ogr    

class TestDB(unittest.TestCase):   

    @classmethod
    def setUpClass(cls):     
        cls.dbFns = [ os.path.join( config.testDataDirNotInDistro, el )
                      for el in ( 'MonoScope-2015-06-25.sqlite',
                                  'MonoScope-2015-11-04.sqlite',
                                  'MonoScope-2016-03-15.sqlite' ) ]

    def test_rtree_available(self):
        with dbapi2.connect(':memory:') as conn:
            compOpts = [ row[0] for row in conn.execute('PRAGMA compile_options') ]
        self.assertIn( 'ENABLE_RTREE', compOpts, 'sqlite3 has been compiled without RTree support. This is the case for the SQlite-DLL that comes with Python, but e.g. not with the one that comes with SpatiaLite.' )

    @unittest.skipUnless(config.isDvlp, "Needed data not available in distro")
    def test_updateSchemaReadOnly(self):
        for dbFn in self.dbFns:
            with self.subTest(dbFn=dbFn):
                mtime = os.path.getmtime(dbFn)
                self.assertRaises( Exception, lambda: db.createUpdateSchema( dbFn, db.Access.readOnly ), 'Read-only update of old schema data base must raise.' )
                self.assertEqual( mtime, os.path.getmtime(dbFn), 'Read-only access to data base must not change its modification time.' )

    @unittest.skipUnless(config.isDvlp, "Needed data not available in distro")
    def test_updateSchemaReadWrite(self):
        for dbFn in self.dbFns:
            with self.subTest(dbFn=dbFn):
                with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as fout:
                    dbCopyFn = fout.name
                try:
                    shutil.copyfile( dbFn, dbCopyFn )
                    # Copy the original file modification time. Otherwise (with the new modification time), the test fails. Why?
                    shutil.copystat( dbFn, dbCopyFn )
                    mtime = os.path.getmtime(dbCopyFn)
                    db.createUpdateSchema( dbCopyFn, db.Access.readWrite )
                    self.assertEqual( mtime, os.path.getmtime(dbCopyFn), 'Read-write update of old schema data base must leave its modification time unchanged.' )
                finally:
                    try:
                        os.remove( dbCopyFn )
                    except OSError:
                        pass

class TestDsm(unittest.TestCase):
    def test_downLoad_urlDhmLamb10mZip(self):
        "Check if the URL is still valid. It already happened that it was changed."
        with urllib.request.urlopen(dsm.urlDhmLamb10mZip) as response:
            pass

class TestStats(unittest.TestCase):    

    def checkGeometricMedianIsLocalOptimum( self, data, med, atol ):
        # simple check if we have found the minimum. Only works with an absolute change as stopping criterion
        sumOfL2Norms = np.sum( np.sum( ( data - med )**2, axis=1 )**.5 )
        shift = atol*10
        for iCoo in range(data.shape[1]):
            for sign in (-1,1):
                medLoc = med.copy()
                medLoc[iCoo] += sign*shift
                sumOfL2NormsLoc = np.sum( np.sum( ( data - medLoc )**2, axis=1 )**.5 )
                self.assertLessEqual( sumOfL2Norms, sumOfL2NormsLoc, 'result is not the true median' )

    def test_geometricMedian3Pt(self):
        data=np.array([[ 0.        ,  0.99275087, -0.80981103],
                       [ 0.57081919, -0.77697838,  0.66240392],
                       [-0.39318025,  0.        ,  0.        ]])
        med = stats.geometricMedian(data)
        dummy=0

    def test_geometricMedian(self):
        """
        Check results by comparison with function 'L1median' of the R-package 'robustX': http://cran.r-project.org/web/packages/robustX/robustX.pdf
        library(robustX)
        data = matrix( c(2,3,1,2,4,4), nrow=2, ncol=3)
        L1median(data, method='VardiZhang', trace=1, pscale=c(1,1,1))
        """
        atol=1.e-6
        data = np.array([[ 0, 0 ],
                         [ 0, 1 ],
                         [ 1, 0 ]], float )
        med = stats.geometricMedian(data,atol)
        self.checkGeometricMedianIsLocalOptimum( data, med, atol )

        data = np.array([[ 0.       ,  0.        ,  0.        ],
                         [-0.9958467,  0.07688382, -0.04876708],
                         [ 1.2553453,  0.22045266, -0.16851064]])
        med = stats.geometricMedian(data)
        #rResult = np.array([ -2.105311e-17, 2.693027e-16, -1.950056e-16 ]) # result of R robustX's L1median(data, method='VardiZhang')
        #self.assertLess( np.max( np.abs( med - rResult ) ), 1.e-6 )

        # this may be surprising, but correct: for triangles with one of their angles > 120°, the geometric median is the vertex of that angle!
        data = np.array([[ 661948.99429854,  473506.48941083, 0. ],
                         [ 661887.18779521,  473848.21667201, 0. ],
                         [ 661838.22032397,  473771.77964373, 0. ]])
        med = stats.geometricMedian(data)
        #import oriental.utils.pyplot as plt
        #plt.axis('equal')
        #plt.scatter( data[:,0], data[:,1], c='k', marker='+' )
        #plt.scatter( med[0], med[1], c='r', marker='x' )

        self.checkGeometricMedianIsLocalOptimum( data, med, 1.e-3 )

        data = np.array([[ 1.46309316, -0.73561628, -3.79646293],
                         [-1.3840068 ,  1.04940002, -5.07546461],
                         [ 1.20991574, -0.81106673, -3.75421443],
                         [-1.67658197,  0.32831912, -4.63540901],
                         [ 0.559117  , -1.09042591, -3.60286767],
                         [ 1.26302913,  1.19772948, -5.03171689],
                         [ 0.43471286, -1.10880015, -3.59651757],
                         [ 0.42917188, -1.09657034, -3.60346781],
                         [-1.25741143,  1.22127812, -5.17105718],
                         [-1.45590128,  0.89485925, -4.98723542],
                         [-1.8430102 ,  0.19787363, -4.54903815],
                         [ 1.90595708, -0.59316477, -3.87267544],
                         [-1.75931364,  0.67107377, -4.89046811],
                         [-1.62504355,  0.86256551, -4.97249136],
                         [ 1.98014976,  1.26713985, -5.07195037],
                         [-0.86661951,  1.17166242, -5.13040938],
                         [-1.17468256, -0.62154617, -3.97408119],
                         [-1.41386897,  0.24861875, -4.5385238 ],
                         [ 0.58119176, -1.06695612, -3.61920171],
                         [ 1.08264065,  1.00440948, -4.91239311],
                         [-0.34202268, -0.53521811, -3.99840708],
                         [-1.43472904,  0.79165713, -4.94137498],
                         [-1.44895336,  0.52650144, -4.77023583],
                         [-1.2197861 ,  0.53110408, -4.71600759],
                         [ 1.56655914,  1.26268085, -5.06566719],
                         [-1.53758252,  0.67518689, -4.8714179 ],
                         [-1.73370072,  0.74722985, -4.93909505],
                         [-1.3228104 ,  0.35199088, -4.58581103],
                         [-1.38966686,  0.94192042, -5.02731703],
                         [-1.47340918,  0.49285855, -4.76217499],
                         [ 0.89147678, -0.74594902, -3.80968748],
                         [ 1.61865833,  0.39346868, -4.51239722],
                         [-1.5068838 ,  0.98971418, -5.00665953],
                         [ 1.71715371, -0.01744127, -4.24298084],
                         [-1.7797631 , -0.18560242, -4.29651508],
                         [-1.21095262,  0.5410663 , -4.71246167],
                         [ 0.07812978,  1.18669625, -5.09595114],
                         [-1.47987107,  1.09609517, -5.08583886],
                         [-1.62012635,  0.57348129, -4.8075013 ],
                         [ 1.72725026, -0.02600715, -4.23230193],
                         [-1.56140128,  0.39769306, -4.69191158],
                         [-0.26621915, -0.96718767, -3.71766354],
                         [-1.7656157 ,  0.28666094, -4.59637509],
                         [-0.90843949,  0.77187426, -4.86014008],
                         [ 2.04528177,  1.14569387, -4.99554362],
                         [-0.90494966,  1.58977436, -5.39561502],
                         [-1.14880058, -0.13877706, -4.29083613],
                         [-1.09046207,  0.80762257, -4.86926046],
                         [-0.49558609,  1.67372582, -5.43160352],
                         [ 1.34022752, -0.76647383, -3.77966257],
                         [ 1.77345915, -0.62067496, -3.85419745],
                         [-1.2816123 ,  0.4952305 , -4.71652218],
                         [ 0.58280425, -1.06098723, -3.62119027],
                         [-1.24713163,  0.54895663, -4.72913879],
                         [-1.16030643,  0.44943933, -4.63327507],
                         [ 0.57907173, -1.11591567, -3.58589122],
                         [ 0.96768825, -0.86569192, -3.72972621]])
        med = stats.geometricMedian(data)
        rResult = np.array([ -1.0828324, 0.4901572, -4.6931780 ]) # result of R robustX's L1median(data, method='VardiZhang')

        # Differences are considerable!
        self.assertLess( np.max( np.abs( med - rResult ) ), 1.e-4 )
        dummy=0

    def test_circularMedian( self ):
        alpha_deg = np.array([13, 15, 21, 26, 28, 30, 35, 36, 41,  60,  92, 103, 165, 199, 210, 250, 301, 320, 343, 359], float)
        beta_deg  = np.array([ 1, 13, 41, 56, 67, 71, 81, 85, 99, 110, 119, 131, 145, 177, 199, 220, 291, 320, 340, 355], float)
        alpha_rad, beta_rad = [ a / 180. * np.pi for a in (alpha_deg,beta_deg) ]
        alpha_med = stats.circular_median( alpha_rad )
        beta_med = stats.circular_median( beta_rad )
        np.testing.assert_almost_equal( alpha_med,  0.471238898038469 )
        np.testing.assert_almost_equal( beta_med,  1.326450231515691 )
        alpha_odd_med = stats.circular_median( alpha_rad[:-1] )
        np.testing.assert_almost_equal( alpha_odd_med,  0.488692190558412 )
        dummy=0

class TestZipEqual(unittest.TestCase):
    def test_sameLength(self):
        for el1,el2 in zip_equal([1,2],[10,20]):
            pass
    def test_sameLengthArray(self):
        import numpy as np
        a = np.array([10,20])
        for el1,el2 in zip_equal([1,2],a):
            pass
    def test_sameLengthArray2d(self):
        import numpy as np
        a = np.array([ [10,10],
                       [20,20] ])
        for el1,el2 in zip_equal([1,2],a):
            pass
    def test_sameLengthContainsSentinel(self):
        for el1,el2 in zip_equal([1,2],[10,object()]):
            pass # okay: zip_equal's sentinel does not compare equal with our object()
    def test_sameLength3iterables(self):
        for el1,el2,el3 in zip_equal([1,2],[10,20],[100,200]):
            pass
    def test_firstLonger(self):
        with self.assertRaises(IterableLengthMismatch):
            for el1,el2 in zip_equal([1,2,3],[10,20]):
                pass
    def test_secondLonger(self):
        with self.assertRaises(IterableLengthMismatch):
            for el1,el2 in zip_equal([1,2],[10,20,30]):
                pass
    def test_thirdLonger(self):
        with self.assertRaises(IterableLengthMismatch):
            for el1,el2,el3 in zip_equal([1,2],[10,20],[100,200,300]):
                pass
    def test_firstEmpty(self):
        with self.assertRaises(IterableLengthMismatch):
            for el1,el2 in zip_equal([],[10]):
                pass
    def test_secondEmpty(self):
        with self.assertRaises(IterableLengthMismatch):
            for el1,el2 in zip_equal([1],[]):
                pass

class TestLsm(unittest.TestCase):
    def test_types(self):
        trafo = lsm.TrafoAffine2D()
        A = trafo.A
        tPre = trafo.tPre
        A = np.array([[1.,2.],[3.,4.]])
        tPre = np.array([3.,4.])
        trafo.A = A
        trafo.tPre = tPre
        np.testing.assert_array_equal( trafo.A, A )
        np.testing.assert_array_equal( trafo.tPre, tPre )

        ring = ogr.Geometry(ogr.wkbLinearRing)
        for idx in range(4):
            ring.AddPoint( idx, idx*2 )
        ring.FlattenTo2D()
        ring.CloseRings()
        polyg = ogr.Geometry(ogr.wkbPolygon)
        polyg.AddGeometry(ring)
        wkt = polyg.ExportToWkt()
        self.assertTrue( len(wkt)>0 )

        img = lsm.Image()
        id_ = img.id
        path = img.dataSetPath
        mask = img.mask
        self.assertIsNone( mask )
        master2slave = img.master2slave
        brightness = img.brightness
        contrast = img.contrast

        img.mask = polyg
        self.assertEqual( img.mask.ExportToWkt(), wkt )

        polyg2 = img.mask
        self.assertEqual( polyg2.ExportToWkt(), wkt )

    def test_lsm(self):
        masterWindow = np.array([ -100., 100., 100., -100 ])

        img1 = lsm.Image()
        img1.id = 0
        img1.dataSetPath = os.path.join( config.testDataDir, '0120110503_047.JPG' )
        img1.master2slave.tPost = np.array([2808., -1292.])

        img2 = lsm.Image()
        img2.id = 1
        img2.dataSetPath = os.path.join( config.testDataDir, '0120110503_048.JPG' )
        angle = 30./200.*np.pi
        sAng = np.sin(angle)
        cAng = np.cos(angle)
        img2.master2slave.A = np.array([[cAng,-sAng],
                                        [sAng, cAng]])
        img2.master2slave.tPost = np.array([2870., -1070.])

        solveOpts = lsm.SolveOptions()
        solveOpts.storeFinalCuts = True
        lsmObj = lsm.Lsm( solveOpts )
        images = [img1, img2]
        summary = lsm.SolveSummary()
        lsmObj( masterWindow, images, summary )
        nResLvls = 0
        for resLvl in summary.resolutionLevels:
            nResLvls += 1
        resLvls = summary.resolutionLevels
        resLvl = resLvls[-1]
        nIter = 0
        for iter in resLvl.iterations:
            nIter += 1
        nCuts = 0
        for cut in resLvl.cuts:
            shape = cut.shape
            nCuts += 1
        dummy=0

    @unittest.skip("Not functional yet")
    def test_py_lsm(self):
        hWid = 100  # template half width
        picExp = 100  # expand the image w.r.t the template in every direction
        image = np.zeros( (picExp*2+hWid*2+1,)*2, np.uint8 )
        image = cv2.ellipse( img=image, center=(picExp+hWid,)*2, axes=(20,10),
                             angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1, lineType=cv2.LINE_8, shift=0 )
        image = cv2.ellipse( img=image, center=(picExp+hWid+20,)*2, axes=(10,20),
                             angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1, lineType=cv2.LINE_8, shift=0 )
        template = image[ picExp : -picExp,
                          picExp : -picExp ]
        # TODO with +10 instead of +5, LSM diverges, because the signal 'hills' won't overlap
        shift = 5
        template2image = ori.transform.AffineTransform2D( A = np.eye(2),
                                                          t = np.array([picExp+shift,-picExp+shift]) )
        lsm.py_lsm.lsm( template=template,
             image=image,
             template2image=template2image,
             geomTrafo=lsm.GeomTrafo.affine,
             radiometricTrafo=lsm.RadiometricTrafo.linear,
             solver = lsm.Solver.cholesky,
             plot=2 )

if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestDsm.test_downLoad_urlDhmLamb10mZip',
                       exit=False )
