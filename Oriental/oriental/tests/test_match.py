# -*- coding: cp1252 -*-
import environment, os, shutil, pickle, glob
from math import factorial
from pathlib import Path
from oriental import config, log, match, utils
import cv2, h5py
import numpy as np

import unittest
    
def nChooseK(n,k):
    return   factorial(n) / ( factorial(k) * factorial(n-k) )
    
class TestMatch(unittest.TestCase):    
    
    def setUp( self ):
        self.fns = glob.glob( os.path.join( os.path.dirname(__file__), "data", "*.jpg") )
        self.outDir = os.path.join( os.path.dirname(__file__), "out2" )
        if not os.path.isdir( self.outDir ):
            os.mkdir( self.outDir )
        
    def tearDown( self ):
        shutil.rmtree( self.outDir )        
        
    @unittest.skip("not functional")
    def test_absMatch(self):
        fnPickle = r"D:\swdvlp64\oriental\Scripts\matchTest.pickle"
        with open(fnPickle,"rb") as fin:
            imagePaths = pickle.load( fin )
            masks = pickle.load( fin )
        featureDetectorOptions = match.SurfOptions()
        featureDetectorOptions.hessianThreshold = 200.

        allKeyPts, edgeMatches = match.match(
            imagePaths = imagePaths,
            masks = masks,
            featureDetectorOptions=featureDetectorOptions,
            plotKeypoints=True,
            plotMatches=True#,
            #useSurf=True,
            #maxRadiusRatio = 2.,
            #nCandidates=20
        )


    @unittest.skipUnless( config.isDvlp, "Needed data only available on dvlp machines" )
    def test_match_pleiades( self ):
        featureDetectOpts = match.SiftOptions()
        matchingOpts = match.MatchingOpts()
        rootDir = r'P:\Projects\16_PleiAlps\07_Work_Data\downloads_Airbus\Austria_Groglockner_208_SO17013362-9-01_DS_PHR1A_201609251017273_FR1_PX_E012N47_0804_00974\TPP1600674671'
        fns = [ os.path.join( rootDir, r'IMG_PHR1A_PMS-N_001\IMG_PHR1A_PMS-N_201609251017273_SEN_2394355101-001_R1C1.JP2' ),
                os.path.join( rootDir, r'IMG_PHR1A_PMS-N_002\IMG_PHR1A_PMS-N_201609251018034_SEN_2394355101-002_R1C1.JP2' ) ]
        allKeypoints, edge2matches = match.match(
            imagePaths = fns,
            featureDetectOpts=featureDetectOpts,
            matchingOpts=matchingOpts )
        self.assertEqual( len(allKeypoints), len(fns), "Not for every image, keypoints have been found" )
        self.assertEqual( len(edge2matches), nChooseK( len(fns), 2 ), "Not for every image pair, matches have been found" )

    @unittest.skipUnless( config.isDvlp, "Needed data only available on dvlp machines" )
    def test_match_anker( self ):
        detectionOptions = match.SiftOptions()
        detectionOptions.nAffineTilts = 6
        filterOptions = match.FeatureFiltOpts()
        filterOptions.nRowCol = 3, 4
        matchingOptions = match.MatchingOpts()
        match.match( imagePaths = sorted( glob.glob( r'E:\Pho-Vertiefer\Anker\*.jpg' ) ),
                     featureDetectOpts = detectionOptions,
                     featureFiltOpts = filterOptions,
                     matchingOpts = matchingOptions )

    @unittest.skipUnless( config.isDvlp, "Needed data only available on dvlp machines" )
    def test_match_wuppertal( self ):
        log.setScreenMinSeverity( log.Severity.debug )
        preprocessingOpts = match.PreprocessingOpts()
        preprocessingOpts.histogramEqualization=True
        featureDetectOpts = match.SiftOptions()
        featureDetectOpts.nAffineTilts = 0
        featureFiltOpts = match.FeatureFiltOpts()
        featureFiltOpts.plotFeatures = True
        matchingOpts = match.MatchingOpts()
        matchingOpts.plotMatches = True
        allKeypoints, edge2matches = match.match(
            #imagePaths = self.fns,
            imagePaths = [r'P:\Studies\17_AdV_HistOrthoBenchmark\07_Work_Data\Wuppertal_1969\Tif_16bit\370_2_66.tif', r'P:\Studies\17_AdV_HistOrthoBenchmark\07_Work_Data\Wuppertal_1969\Tif_16bit\370_2_69.tif' ],
            masks = [r'P:\Studies\17_AdV_HistOrthoBenchmark\07_Work_Data\Wuppertal_1969\relOri\masks\370_2_66.tif', r'P:\Studies\17_AdV_HistOrthoBenchmark\07_Work_Data\Wuppertal_1969\relOri\masks\370_2_69.tif' ],
            featureDetectOpts=featureDetectOpts,
            featureFiltOpts=featureFiltOpts,
            matchingOpts=matchingOpts,
            preprocessingOpts=preprocessingOpts )
        self.assertEqual( len(allKeypoints), 2, "Not for every image, keypoints have been found" )
        self.assertEqual( len(edge2matches), nChooseK( 2, 2 ), "Not for every image pair, matches have been found" )

    def test_match( self ):
        allKeypoints, edge2matches = match.match(
            imagePaths = self.fns,
            featureDetectOpts=match.SiftOptions() )
        self.assertEqual( len(allKeypoints), len(self.fns), "Not for every image, keypoints have been found" )
        self.assertEqual( len(edge2matches), nChooseK( len(self.fns), 2 ), "Not for every image pair, matches have been found" )

    def test_match_akaze( self ):
        allKeypoints, edge2matches = match.match(
            imagePaths = self.fns,
            featureDetectOpts=match.AkazeOptions() )
        self.assertEqual( len(allKeypoints), len(self.fns), "Not for every image, keypoints have been found" )
        self.assertEqual( len(edge2matches), nChooseK( len(self.fns), 2 ), "Not for every image pair, matches have been found" )

    def test_match_candidates( self ):
        allKeypoints, edge2matches = match.match(
            imagePaths = self.fns,
            candidatePairs = np.array( [[0,1],[0,2]], np.uintp ) )
        self.assertEqual( len(allKeypoints), len(self.fns), "Not for every image, keypoints have been found" )
        self.assertEqual( len(edge2matches), 2, "Not for every image pair, matches have been found" )

    def test_match_gpu( self ):
        featureDetectOpts = match.SiftOptions()
        matchingOpts = match.MatchingOpts()
        matchingOpts.matchingPrecision = 1
        allKeypoints, edge2matches = match.match(
            imagePaths = self.fns,
            featureDetectOpts=featureDetectOpts,
            matchingOpts=matchingOpts)
        self.assertEqual( len(allKeypoints), len(self.fns), "Not for every image, keypoints have been found" )
        self.assertEqual( len(edge2matches), nChooseK( len(self.fns), 2 ), "Not for every image pair, matches have been found" )

    @unittest.skipUnless( config.isDvlp, "Needed data only available on dvlp machines" )
    def test_match_gpu2( self ):
        baseDir = r'P:\Projects\16_VOLTA\07_Work_Data\GCC_Sabina_Vajs_Montenegro\Pictures'
        
        featureDetectOpts = match.AkazeOptions()
        featureDetectOpts.nAffineTilts = 6

        featureFiltOpts = match.FeatureFiltOpts()
        featureFiltOpts.nMaxFeaturesPerImg = 100000
        featureFiltOpts.nRowCol = 4, 4

        matchingOpts = match.MatchingOpts()
        matchingOpts.matchingPrecision = 1

        fns = [os.path.join(baseDir, el) for el in ('501_0016_00002221.tif', '501_0016_00002222.tif')]
        allKeypoints, edge2matches = match.match(
            imagePaths = fns,
            featureDetectOpts=featureDetectOpts,
            featureFiltOpts=featureFiltOpts,
            matchingOpts=matchingOpts,
            extractAndDescribe = True)
        self.assertEqual( len(allKeypoints), len(fns), "Not for every image, keypoints have been found" )
        self.assertEqual( len(edge2matches), nChooseK( len(fns), 2 ), "Not for every image pair, matches have been found" )

    @unittest.skipUnless( config.isDvlp, "Needed data only available on dvlp machines" )
    def test_plotMatches( self ):
        baseDir = Path(r'D:\OrientAL\16_ReKlaSat3D\relOri')
        with h5py.File( baseDir / 'features.h5' ) as h5:
            imgFn2Path = { name : path for name, path in utils.zip_equal(sorted(h5['keypts']), sorted(h5.attrs['imagePaths']))}
            for name, matches in h5['matches'].items():
                imgs = []
                for fn in name.split('?'):
                    img = match.ImageMaskKeypts()
                    img.imgFn = imgFn2Path[fn]
                    img.keyPts = np.array(h5['keypts'][fn])
                    imgs.append(img)
                match.plotMatches(matches=np.array(matches),
                                  imagesMasksKeyPts=imgs,
                                  outDir=str(baseDir / 'matches'))
                break


if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestMatch.test_match_wuppertal',
                       exit=False )