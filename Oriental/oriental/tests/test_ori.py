# -*- coding: cp1252 -*-
import environment
from oriental import config, ori, adjust
import oriental.ori.transform
import oriental.adjust.parameters    
from oriental.relOri.SfMManager import SfMManager, IORgrouping
from oriental.utils.db import initDataBase

import pickle
import os
import glob
import unittest

import numpy as np
from scipy import linalg
import cv2

class TestOri(unittest.TestCase):    

    def test_PnP(self):
        nPts = 50
        inlierRatio = .25
        nModelPoints = 5
        assert nPts*inlierRatio > nModelPoints, 'how should we find a model free of ouliers, otherwise?'
        maxResidualNorm = 5.
        confidenceLevel = .999
        objPts = np.c_[ np.random.rand(nPts,2), np.zeros( nPts ) ]
        ior = np.array([50., -50., 75.])
        prc = np.array([.5, .5, .5])
        omfika = np.array([0.,0.,0.])
        imgPts = ori.projection( objPts, prc, omfika, ior )
        iOutliers = np.random.choice(nPts, int(nPts*(1.-inlierRatio)), replace=False)
        for iCoo in range(2):
            imgPts[iOutliers[0::2],iCoo] += maxResidualNorm + 1. + np.random.random(len(iOutliers[0::2]))*30
            imgPts[iOutliers[1::2],iCoo] -= maxResidualNorm + 1. + np.random.random(len(iOutliers[1::2]))*30
        maxNumIters = ori.maxNumItersRANSAC( nModelPoints=nModelPoints, nDataPoints=nPts, inlierRatio=inlierRatio, confidence=confidenceLevel )

        K = ori.cameraMatrix( ior )

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=( objPts * (1.,-1.,-1.) ).reshape((-1,1,3)),
            imagePoints= ( imgPts * (1.,-1.) ).reshape((-1,1,2)),
            cameraMatrix=K,
            distCoeffs=np.empty(0),
            useExtrinsicGuess=False,
            iterationsCount=maxNumIters,
            reprojectionError=maxResidualNorm,
            confidence=confidenceLevel,
            flags=cv2.SOLVEPNP_EPNP # is said to handle planar and non-planar RefSys!
        )
        inliers = inliers.squeeze()
        self.assertTrue( success, "solvePnPRansac was unsuccessful" )
        #np.testing.assert_array_equal( iOutliers, inliers )

        Rcv,_ = cv2.Rodrigues( rvec )
        R,t = ori.projectionMat2oriRotTrans( np.column_stack((Rcv,tvec)) ) 
        #np.testing.assert_array_almost_equal( ori.omfika(R), omfika )
        #np.testing.assert_array_almost_equal( t, prc )

        inliers = set(inliers)
        imgPts2 = ori.projection( objPts, t, ori.omfika(R), ior )
        for iPt,(imgPt,imgPt2) in enumerate(zip(imgPts,imgPts2)):
            resNorm = linalg.norm(imgPt - imgPt2)
            try:
                self.assertEqual( resNorm <= maxResidualNorm, iPt in inliers )
            except:
                raise
            
    def test_findEssentialMat(self):
        nPts = 50
        inlierRatio = .45
        nModelPoints = 5
        assert nPts*inlierRatio > nModelPoints, 'How should we find a model free of ouliers, otherwise?'
        maxResidualNorm = 1.
        confidenceLevel = .999
        maxNumIters = ori.maxNumItersRANSAC( nModelPoints=nModelPoints, nDataPoints=nPts, inlierRatio=inlierRatio, confidence=confidenceLevel )

        objPts = np.c_[ np.random.rand(nPts,2), np.random.rand( nPts )*.5 - .25 ]
        #y,x = np.mgrid[:2,:2]
        #objPts = np.column_stack(( x.flat, y.flat, np.zeros(4) ))
        ior = np.array([50., -50., 60.])
        prc1 = np.array([0., .5, .5])
        prc2 = np.array([1., .5, .5])
        omfika1 = np.array([0.,-40.,0.])
        omfika2 = np.array([0., 40.,0.])
        imgPts1 = ori.projection( objPts, prc1, omfika1, ior )
        imgPts2 = ori.projection( objPts, prc2, omfika2, ior )
        if 0:
            import oriental.utils.pyplot as plt
            plt.figure(1); plt.plot(np.atleast_2d(imgPts1[:,0]), np.atleast_2d(imgPts1[:,1]), 'x' ); plt.title('left');   plt.xlabel('x'); plt.ylabel('y')
            plt.figure(2); plt.plot(np.atleast_2d(imgPts2[:,0]), np.atleast_2d(imgPts2[:,1]), 'x' ); plt.title('right');  plt.xlabel('x'); plt.ylabel('y')
            plt.figure(3); plt.plot(np.atleast_2d( objPts[:,0]), np.atleast_2d( objPts[:,1]), 'x' ); plt.title('object'); plt.xlabel('X'); plt.ylabel('Y')

        K = ori.cameraMatrix( ior )
        Kinv = linalg.inv(K)
        R1 = ori.omfika(omfika1)
        R2 = ori.omfika(omfika2)
        tTrue = R1.T.dot( prc2 - prc1 ) * (1,-1,-1)
        R = R2.T.dot( R1 )
        Rx200 = np.diag([1.,-1.,-1.])
        Rocv = Rx200.dot( R ).dot( Rx200 )

        # E = R [t]_x
        # with:
        # x' = R( x - t )
        # [t]_x : cross product matrix of t
        ETrue = Rocv.dot( ori.crossProductMatrix( tTrue ) )

        imgPts1_h = np.c_[ imgPts1 * (1,-1), np.ones(nPts) ]
        imgPts2_h = np.c_[ imgPts2 * (1,-1), np.ones(nPts) ]

        imgPts1_hn = imgPts1_h.dot(Kinv.T) # <-> K^-1 dot imgPts1_h;  left-multiply with K^-T, so we can compute the whole batch of pts.
        imgPts2_hn = imgPts2_h.dot(Kinv.T)

        E, inliers = cv2.findEssentialMat(
                        points1 = imgPts1_hn[:,:2],
                        points2 = imgPts2_hn[:,:2],
                        focal = 1.,
                        pp = (0.,0.),
                        method = cv2.RANSAC,
                        prob = confidenceLevel,
                        threshold = maxResidualNorm / ior[2],
                        maxNumIters=maxNumIters
                        )
        self.assertEqual( inliers.sum(), nPts )
        # those two expressions should be similar, but their difference varies notably, depending on the randomly chosen observations.
        # Probably a matter of numerical stability. Mind that the final E computed by cv2.findEssentialMat is based on only 5 point pairs!
        #np.testing.assert_array_almost_equal( ETrue / linalg.norm(ETrue), E )
        nGood,Rrecon,tRecon,_ = cv2.recoverPose( ETrue, ( imgPts1_hn[:,:2].T / imgPts1_hn[:,2] ).T,
                                                        ( imgPts2_hn[:,:2].T / imgPts2_hn[:,2] ).T )
        self.assertEqual(nGood,nPts)
        # x' = [R | t ] x
        np.testing.assert_array_almost_equal( Rocv, Rrecon )
        np.testing.assert_array_almost_equal( -Rocv.dot(tTrue), tRecon.squeeze() )

        def plotEpiPolars(pts1,pts2,E,inliers):
            import oriental.utils.pyplot as plt
            import itertools
            # E = K2.T F K
            #F = linalg.inv(K.T).dot(E).dot(Kinv)
            colors = 'bgrcmy'
            for idx in range(2):
                plt.figure(idx); plt.clf()
                if idx>0:
                    pts1, pts2 = pts2, pts1
                    E = E.T
                for pt1,pt2,inlier,col in zip(pts1,pts2,inliers,itertools.cycle(colors)):
                    col = col if inlier else 'k'
                    plt.plot(pt2[0], pt2[1], col+'x' )
                    a,b,c = E.dot(pt1)
                    # ax + by + cw = 0
                    x = np.array([-50/ior[2], 50/ior[2] ])
                    y = x.copy()
                    for idxXY in [0,1]:
                        if abs(b) > abs(a):
                            y[idxXY] = ( a * x[idxXY] + c ) / -b
                        else:
                            x[idxXY] = ( b * y[idxXY] + c ) / -a
                    plt.plot( x, y, col+'-' )
                plt.axis('image')
                plt.ylim(plt.ylim()[::-1])
                plt.title( 'right' if idx==0 else 'left' )
                plt.xlabel('x')
                plt.ylabel('y')
        #plotEpiPolars(imgPts1_hn,imgPts2_hn,E,np.ones(nPts,np.bool))

        def geometricErrorsInImg2( pts1, pts2, E ):
            EPt1 = pts1.dot(E.T)
            EPt1n = ( EPt1.T / ( np.sum( EPt1[:,:2]**2, axis=1 )**.5 ) ).T # unit-normalize the normal vector of the epipolar line
            # this is the geometric, perpendicular distance of pt2 to the epipolar line of pt1
            return np.abs( np.sum( pts2 * EPt1n, axis=1 ) )

        def sampsonErrors( pts1, pts2, E ):
            EPts1 = pts1.dot(E.T)
            EtPts2 = pts2.dot(E)
            pts2Epts1 = np.sum( pts2 * EPts1, axis=1 )
            squaredSampsonErrors = pts2Epts1**2 / ( EPts1 [:,0]**2 + EPts1 [:,1]**2 +
                                                    EtPts2[:,0]**2 + EtPts2[:,1]**2  )
            return squaredSampsonErrors**.5

        np.testing.assert_array_almost_equal( geometricErrorsInImg2( imgPts1_hn, imgPts2_hn, ETrue ), 0. )
        np.testing.assert_array_almost_equal( sampsonErrors( imgPts1_hn, imgPts2_hn, ETrue ), 0. )

        epipolarLineNormalsImg1 = imgPts2_hn.dot(ETrue)[:,:2]
        epipolarLineNormalsImg1 = ( epipolarLineNormalsImg1.T / np.sum( epipolarLineNormalsImg1**2, axis=1 )**.5 ).T
        
        epipolarLineNormalsImg2 = imgPts1_hn.dot(ETrue.T)[:,:2]
        epipolarLineNormalsImg2 = ( epipolarLineNormalsImg2.T / np.sum( epipolarLineNormalsImg2**2, axis=1 )**.5 ).T
        
        iOutliers = np.random.choice(nPts, int(nPts*(1.-inlierRatio)), replace=False)
        self.assertEqual( len(set(iOutliers)), len(iOutliers) ) # replace=False -> no duplicates!
        bOutliers = np.zeros( nPts, np.bool )
        bOutliers[iOutliers] = True
        bInliers = bOutliers == False

        #offsetSigns = (-1)**np.random.randint(0,2,len(iOutliers))
        #offsetLengths = offsetSigns * ( maxResidualNorm * 5 + np.random.random(len(iOutliers))*100 )
        #imgPts1[bOutliers] += ( ( epipolarLineNormalsImg1[bOutliers] * (1,-1) ).T * offsetLengths ).T
        #imgPts1_h = np.c_[ imgPts1 * (1,-1), np.ones(nPts) ]
        #imgPts1_hn = imgPts1_h.dot(Kinv.T)
        if 1:
            offsetSigns = (-1)**np.random.randint(0,2,len(iOutliers))
            offsetLengths = offsetSigns * ( maxResidualNorm * 5 + np.random.random(len(iOutliers))*100 )
            imgPts2[bOutliers] += ( ( epipolarLineNormalsImg2[bOutliers] * (1,-1) ).T * offsetLengths ).T
        else:
            imgPts2[bOutliers] = np.random.permutation(imgPts2[bOutliers])
        imgPts2_h = np.c_[ imgPts2 * (1,-1), np.ones(nPts) ]
        imgPts2_hn = imgPts2_h.dot(Kinv.T)

        # plotEpiPolars(imgPts1_hn,imgPts2_hn,E,bInliers)

        # cv2.findEssentialMat uses Sampson's first-order approximation to the geometric error to distinguish inliers from outliers.
        # The Sampson error may be considerably smaller than the geometric error
        thresh = maxResidualNorm / ior[2]
        
        geomErrs2 = geometricErrorsInImg2( imgPts1_hn, imgPts2_hn, ETrue )
        np.testing.assert_array_less( geomErrs2[bInliers], 1.e-14 )
        #np.testing.assert_array_less( thresh * 5 - 1.e-14, geomErrs2[bOutliers] )
        sampsErrs = sampsonErrors( imgPts1_hn, imgPts2_hn, ETrue )
        np.testing.assert_array_less( sampsErrs[bInliers], 1.e-14 )
        #np.testing.assert_array_less( thresh, sampsErrs[bOutliers] )

        if 1:
            # pass unit normalized image coordinates with according focal, pp, and threshold
            E, inliers = cv2.findEssentialMat(
                            points1 = imgPts1_hn[:,:2],
                            points2 = imgPts2_hn[:,:2],
                            focal = 1.,
                            pp = (0.,0.),
                            method = cv2.RANSAC,
                            prob = confidenceLevel,
                            threshold = thresh, # unlike with cv2.solvePnPRansac, we pass the threshold as it should be, because EMEstimatorCallback::computeError correctly computes the squared (Sampson) error.
                            maxNumIters=maxNumIters
                         )
        else:
            E, inliers = cv2.findEssentialMat(
                            points1 = ( imgPts1 * (1.,-1.) ),
                            points2 = ( imgPts2 * (1.,-1.) ),
                            focal = ior[2],
                            pp = (ior[0], -ior[1]),
                            method = cv2.RANSAC,
                            prob = confidenceLevel,
                            threshold = thresh * ior[2], # unlike with cv2.solvePnPRansac, we pass the threshold as it should be, because EMEstimatorCallback::computeError correctly computes the squared (Sampson) error.
                            maxNumIters=maxNumIters
                         )


        inliers = inliers.squeeze() > 0

        sampsErrs = sampsonErrors( imgPts1_hn, imgPts2_hn, E )
        np.testing.assert_array_less( sampsErrs[inliers] - 1.e-14, thresh )
        np.testing.assert_array_less( thresh, sampsErrs[inliers==False] )

        geomErrImg2 = geometricErrorsInImg2( imgPts1_hn, imgPts2_hn, E )
        geomErrImg1 = geometricErrorsInImg2( imgPts2_hn, imgPts1_hn, E.T )

        # OpenCV's findEssentialMat uses the Sampson distance to distinguish inliers from outliers, see Multiple view geometry, chapter 11.4.3, (11.9)
        for pt1,pt2,inlier in zip(imgPts1_hn,imgPts2_hn,inliers):
            EPt1 = E.dot(pt1)
            EtPt2 = E.T.dot(pt2)
            squaredSampsonError = pt2.dot(EPt1)**2 / ( EPt1 [0]**2 + EPt1 [1]**2 +
                                                       EtPt2[0]**2 + EtPt2[1]**2  )
            try:
                self.assertEqual( squaredSampsonError <= thresh**2, inlier )
            except:
                raise

        for pt1,pt2 in zip(imgPts1_hn,imgPts2_hn):
            EPt1 = E.dot(pt1)
            EPt1n = EPt1 / linalg.norm( EPt1[:2] ) # unit-normalize the normal vector of the epipolar line
            # this is the geometric, perpendicular distance of pt2 to the epipolar line of pt1
            geometricError1 = np.abs( pt2.dot(EPt1n) )

            #alternative computation
            #test1 = np.abs( pt2.dot(E).dot(pt1) / linalg.norm( E.dot(pt1)[:2] ) )

            EtPt2 = E.T.dot(pt2)
            EtPt2n = EtPt2 / linalg.norm( EtPt2[:2] )
            # this is the geometric, perpendicular distance of pt1 to the epipolar line of pt2
            geometricError2 = np.abs( pt1.dot(EtPt2n) )
            #test2 = np.abs( pt1.dot(E.T).dot(pt2) / linalg.norm( E.T.dot(pt2)[:2] ) )

            # see Multiple view geometry, chapter 11.4.3, (11.10)
            sumOfSquaredGeometricErrors = pt2.dot(E).dot(pt1)**2 * (  1 / ( EPt1 [0]**2 + EPt1 [1]**2 ) +
                                                                      1 / ( EtPt2[0]**2 + EtPt2[1]**2  ) )
            np.testing.assert_almost_equal( sumOfSquaredGeometricErrors, geometricError1**2 + geometricError2**2 )

        from oriental import adjust
        from oriental.adjust.parameters import ADP
        from oriental.adjust.cost import PhotoTorlegard
        from oriental.adjust.local_parameterization import UnitSphere
        nGood,Rrecon2,tRecon2,mask = cv2.recoverPose( E, imgPts1_hn[inliers,:2],
                                                         imgPts2_hn[inliers,:2] )
        self.assertGreaterEqual( nGood, 5 )

        inliers[inliers] = mask.squeeze() != 0
        print('{} inliers detected as outliers'.format( (inliers==False)[bInliers].sum() ) )
        print('{} outliers detected as inliers'.format( inliers[bOutliers].sum() ) )
        omfika1_ = np.zeros(3)
        prc1_ = np.zeros(3)
        R2_ = Rx200.dot(Rrecon2).dot(Rx200).T
        omfika2_ = ori.omfika( R2_ )
        prc2_ = - R2_.dot( Rx200.dot(tRecon2.squeeze()) )
        prc2_ /= linalg.norm(prc2_)

        np.testing.assert_array_almost_equal( ori.omfika(Rx200.dot(Rocv).dot(Rx200).T), omfika2_ )
        np.testing.assert_array_almost_equal( Rx200.dot(tTrue), prc2_ )

        # let's do an adjustment
        adp = ADP(1000.)
        class CamParams:
            def __init__(self, prc, omfika, ior, adp ):
                self.t = prc
                self.R = ori.omfika(omfika)
                self.ior = ior
                self.adp = adp
        camParams = [ CamParams( prc1_, omfika1_, ior, adp ),
                      CamParams( prc2_, omfika2_, ior, adp ) ]
        objPts_ = np.ascontiguousarray( ori.triangulatePoints( imgPts1[inliers], imgPts2[inliers], *camParams ) )
        problem = adjust.Problem()
        loss = adjust.loss.Wrapper( adjust.loss.Huber(maxResidualNorm/3) )
        resIds1 = []
        resIds2 = []
        for imgPts,prc,omfika,resIds in ( ( imgPts1[inliers], prc1_, omfika1_, resIds1 ),
                                          ( imgPts2[inliers], prc2_, omfika2_, resIds2 ) ):
            for imgPt,objPt in zip( imgPts, objPts_ ):
                resIds.append(
                    problem.AddResidualBlock( PhotoTorlegard(*imgPt),
                                              loss,
                                              prc,
                                              omfika,
                                              ior,
                                              adp,
                                              objPt ) )
        for par in (ior,adp,omfika1_,prc1_):
            problem.SetParameterBlockConstant(par)
        problem.SetParameterization( prc2_, UnitSphere() )
        solveOpts = adjust.Solver.Options()
        solveOpts.linear_solver_ordering = adjust.ParameterBlockOrdering()
        for par in ( prc1_,omfika1_,prc2_,omfika2_,ior,adp ):
            solveOpts.linear_solver_ordering.AddElementToGroup( par, 1 )
        for par in objPts_:
            solveOpts.linear_solver_ordering.AddElementToGroup( par, 0 )
        summary = adjust.Solver.Summary()
        adjust.Solve(solveOpts, problem, summary)
        self.assertTrue( adjust.isSuccess( summary.termination_type ) )
        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.apply_loss_function = False
        residuals, = problem.Evaluate()
        resNormsSqr = residuals[0::2]**2 + residuals[1::2]**2
        for resId1,resId2,resNormSqr in zip( resIds1,resIds2,resNormsSqr ):
            if resNormSqr <= maxResidualNorm**2:
                continue
            for resId in (resId1,resId2):
                problem.GetCostFunctionForResidualBlock( resId ).deAct()
        loss.Reset( adjust.loss.Trivial() )
        adjust.Solve(solveOpts, problem, summary)
        self.assertTrue( adjust.isSuccess( summary.termination_type ) )

        np.testing.assert_array_almost_equal( ori.omfika(Rx200.dot(Rocv).dot(Rx200).T), omfika2_ )
        np.testing.assert_array_almost_equal( Rx200.dot(tTrue), prc2_ )

        redundancy = summary.num_residuals_reduced - summary.num_effective_parameters_reduced
        self.assertGreater( redundancy, 0 )
        sigma0 = ( summary.final_cost * 2 / redundancy ) **.5
        evalOpts.set_parameter_blocks([prc2_,omfika2_]+[el for el in objPts_])
        jacobian, = problem.Evaluate(evalOpts,residuals=False,jacobian=True)
        qxx = adjust.diagQxx(jacobian)[:5]**.5 * sigma0

        dummy=0

    @unittest.skipUnless( config.isDvlp, "Needed data only available on dvlp machines")
    def test_findEssentialMatWithRealData(self):
        import h5py
        from oriental import adjust
        from oriental.adjust import parameters

        ior = np.array([ 2850.012, -1930.463, 4896.952 ])
        adp = parameters.ADP( normalizationRadius=2400. )
        adp[adjust.PhotoDistortion.optPolynomRadial3] = -71.9
        adp[adjust.PhotoDistortion.optPolynomRadial5] = 11.859	
        adp[adjust.PhotoDistortion.optPolynomRadial7] = 2.828

        K = ori.cameraMatrix(ior)

        # while this pair has passed the E-matrix test, I cannot reproduce it here. Very probable reason: the E-matrix is very weakly defined, because both images were taken from the same tripod position!
        #imgPairInFiltered = '111B2704.JPG', '111B2705.JPG'
        imgPairInFiltered = '111B2704.JPG', '111B2723.JPG'
        imgPairNotInFiltered = '111B3142.JPG', '111B3143.JPG'
        with h5py.File(r'E:\Livia\ISPRS_Prague\Sequence1_188Images_PanoFrame\relOriNoMaskNoThinout\features.h5', 'r' ) as features:
            matches = features['matches']
            matchesFiltered = features['matchesFiltered']
            keypts = features['keypts']

            matchesInFiltered = np.array(matches['?'.join(imgPairInFiltered)])
            keyptsInFiltered1 = np.array(keypts[imgPairInFiltered[0]])[:,:2]
            keyptsInFiltered2 = np.array(keypts[imgPairInFiltered[1]])[:,:2]
            assert '?'.join(imgPairInFiltered) in matchesFiltered

            keyPtsUndist = [
                ori.distortion(
                    keypts,
                    ior,
                    ori.adpParam2Struct(adp),
                    ori.DistortionCorrection.undist )
                for keypts in ( keyptsInFiltered1,
                                keyptsInFiltered2 ) ]

            edge2matches = ori.filterMatchesByEMatrix(
                edge2matches={ (0,1) : matchesInFiltered },
                allKeypoints=keyPtsUndist,
                iors=[ior]*2,
                maxEpipolarDist=10.,
                nMinMatches=10,
                inlierRatio=0.25,
                confidenceLevel=0.999 )
            
            self.assertTrue( len(edge2matches)==1 )

            # TODO do accordingly with matchesNotInFiltered
            matchesNotInFiltered = np.array(matches['?'.join(imgPairNotInFiltered)])
            keyptsNotInFiltered1 = np.array(keypts[imgPairNotInFiltered[0]])[:,:2]
            keyptsNotInFiltered2 = np.array(keypts[imgPairNotInFiltered[1]])[:,:2]
            assert '?'.join(imgPairNotInFiltered) not in matchesFiltered
            keyPtsUndist = [
                ori.distortion(
                    keypts,
                    ior,
                    ori.adpParam2Struct(adp),
                    ori.DistortionCorrection.undist )
                for keypts in ( keyptsNotInFiltered1,
                                keyptsNotInFiltered2 ) ]

            edge2matches = ori.filterMatchesByEMatrix(
                edge2matches={ (0,1) : matchesNotInFiltered },
                allKeypoints=keyPtsUndist,
                iors=[ior]*2,
                maxEpipolarDist=10.,
                nMinMatches=10,
                inlierRatio=0.25,
                confidenceLevel=0.999 )
            #self.assertTrue( len(edge2matches)==0 )

            # The E-matrix is practically undefined for this image pair. Thus, edge2matches may or may not return a result
            keyPtsUndistMatched = [
                keyPtsUndist[0][matchesNotInFiltered[:,0],:],
                keyPtsUndist[1][matchesNotInFiltered[:,1],:] ]

            #keyPtsUndistNormCv = [
            #    ( ( np.c_[ keypts, np.zeros(len(keypts)) ] - ior ) / ior[2] )[:,:2] * (1,-1)
            #    for keypts in keyPtsUndist ]
            #H, mask = cv2.findHomography( keyPtsUndistNormCv[0], keyPtsUndistNormCv[1], method=cv2.FM_RANSAC, ransacReprojThreshold = 10. / ior[2] )

            # NOTE cv2.findHomography clips maxIters to <= 2000
            # ori.maxNumItersRANSAC( 4, len(keyPtsUndistMatched[0]), 0.25, 0.999 ) == 1765
            keyPtsUndistMatchedCv = [ keypts * (1,-1) for keypts in keyPtsUndistMatched ]
            H, mask = cv2.findHomography( srcPoints=keyPtsUndistMatchedCv[0], dstPoints=keyPtsUndistMatchedCv[1], method=cv2.FM_RANSAC, ransacReprojThreshold=10., maxIters=7070, confidence=0.999 )

            keyPtsUndistMatchedCvMasked = [ keypts[np.flatnonzero(mask),:] for keypts in keyPtsUndistMatchedCv ]

            keyPtsUndistMatchedCvMaskedNorm = [ ( keypts - ior[:2]*(1,-1) ) / ior[2] for keypts in keyPtsUndistMatchedCvMasked ]

            nSolutions,rotations,translations,normals = cv2.decomposeHomographyMat( H, K )
            for rotation,translation,normal in zip(rotations,translations,normals):
                H_ = rotation + np.outer( translation, normal ) 
                P1 = np.eye( 3, 4 )
                P2 = np.c_[ rotation, translation / linalg.norm(translation) ]
                objPts = cv2.triangulatePoints( P1, P2, keyPtsUndistMatchedCvMaskedNorm[0].T, keyPtsUndistMatchedCvMaskedNorm[1].T )
                objPts = objPts / objPts[3,:]
                #ok1 = np.logical_and( objPts[2,:] > 0, objPts[2,:] < 50. )
                ok1 = objPts[2,:] > 0
                objPts = P2.dot( objPts )
                #ok2 = np.logical_and( objPts[2,:] > 0, objPts[2,:] < 50. )
                ok2 = objPts[2,:] > 0
                nGood = np.logical_and( ok1, ok2 ).sum()
                dummy=1

            keypts0in1 = H.dot( np.c_[ keyPtsUndistMatchedCvMasked[0], np.ones(len(keyPtsUndistMatchedCvMasked[0])) ].T ).T
            keypts0in1 = ( keypts0in1[:,:2].T / keypts0in1[:,2] ).T
            # This may actually fail, because cv2.findHomography first determines the outliers, and then optimizes H based on all inliers.
            self.assertTrue( ( np.sum( (keyPtsUndistMatchedCvMasked[1]- keypts0in1)**2, axis=1)**.5 ).max() < 10. )


            keyPtsUndistMatchedReduced = [ ( keypts - ior[:2] ) / ior[2] for keypts in keyPtsUndistMatched ]
            keyPtsUndistMatchedReducedCv = [ keypts * (1,-1) for keypts in keyPtsUndistMatchedReduced ]
            H2, mask2 = cv2.findHomography( srcPoints=keyPtsUndistMatchedReducedCv[0], dstPoints=keyPtsUndistMatchedReducedCv[1], method=cv2.FM_RANSAC, ransacReprojThreshold=10./ior[2], maxIters=7070, confidence=0.999 )
            # H ~= K.dot(H2).dot(Kinv)
            # precision should be about the same, because cv2.decomposeHomographyMat centers and scales the passed image coordinates.

            pts1 = np.array([[ 0, 0 ],
                             [ 0, 0 ],
                             [ 0, 0 ],
                             [ 0, 0 ]], np.float32 )
            pts2 = np.array([[ 0, 0 ],
                             [ 1, 1 ],
                             [ 2, 2 ],
                             [ 3, 3 ]], np.float32 )
            Hfailed, maskFailed = cv2.findHomography( srcPoints=pts1, dstPoints=pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=1, maxIters=7070, confidence=0.999 )
            self.assertIsNone( Hfailed )
            dummy=1

    @unittest.skip("cannot run this as a distro-test, because it depends on a file that must have been generated externally")
    def test_ori(self):
        fnDir = r"D:\swdvlp64_2015\oriental\tests\data"
        fnImgs = glob.glob( '{}/*.jpg'.format(fnDir) ) 
        keypoints = []
            
        sfm = SfMManager( imageFns=fnImgs,
                          iorGrouping=IORgrouping.sequence,
                          outDir='.',
                          minIntersectAngleGon=5. )
        
        import h5py
        with h5py.File( r'D:\swdvlp64_2012\oriental\tests\relOri\features.h5', 'r' ) as features:
            group = features['keypts']
            for img in sfm.imgs:
                img.keypoints = np.array( group[ os.path.basename( img.fullPath ) ] )
            sfm.edge2matches = dict()
            basename2idx = { os.path.basename( img.fullPath ) : idx for idx,img in enumerate(sfm.imgs) }
            for key,value in features['matches'].items():
                sfm.edge2matches[tuple( basename2idx[el] for el in key.split('?') )] = np.array(value)          
        
        edge2matches = ori.filterMatchesByEMatrix( edge2matches=sfm.edge2matches,
                                                   allKeypoints=[ img.keypoints for img in sfm.imgs ],
                                                   iors=[ img.ior for img in sfm.imgs ],
                                                   maxEpipolarDist=10.,
                                                   nMinMatches=10,
                                                   inlierRatio=0.25 )
        
        print("filterMatchesByEMatrix finished")
        
        ior = np.array([ 10., -10., 10 ])
        adp = np.array([[ -1, 0 ], [ 0, 1000. ], [ 3, 10. ] ])
        imgPts = np.array( [ [10.,11.],
                             [12.,13.],
                             [14.,15.],
                             [16.,17.] ] )
                                
        firstX = imgPts[0,0]
        
        ior = np.array([[ 10., -10., 10 ]])
        undist = ori.distortion( imgPts, ior, adp, ori.DistortionCorrection.undist )
        
        assert firstX == imgPts[0,0]
        assert firstX != undist[0,0]
        
        ori.distortion_inplace( imgPts, ior, adp, ori.DistortionCorrection.undist )
        
        assert firstX != imgPts[0,0]
        
        # pass a slice -> exception
        #ori.distortion_inplace( imgPts[1::2,:], ior, adp, ori.DistortionCorrection.undist )
        
        
        # check filterMatchesByEMatrix
        edge2matches = dict()
        edge2matches[(0,1)] = np.array( [[ 0, 0],
                                            [ 1, 1 ],
                                            [ 2, 2 ],
                                            [ 3, 3 ],
                                            [ 4, 4 ] ] )
        
        leftObs = np.array([[  10.,  10., 2., 3., 4. ],
                               [ -10.,  10., 3., .4, 5. ],
                               [ -10., -10., 3., .4, 5. ],
                               [  10., -10., 3., .4, 5. ],
                               [   0.,   0., 3., .4, 5. ]], dtype=np.float32)
        rightObs = leftObs + 10.
        rightObs[0,:] -= 5.
        rightObs[1,0] -= 5.
        rightObs[1,1] += 5.
        rightObs[-1,0] -= 4.
        allKeypoints = [ leftObs,
                         rightObs ]
        
        Ks = [ np.eye(3), np.eye(3) ]
        
        res = ori.filterMatchesByEMatrix( edge2matches, allKeypoints, Ks, maxEpipolarDist = 10., inlierRatio=0.25 )
        
    @unittest.skipUnless( config.isDvlp, "Needed data only available on dvlp machines")
    def test_trafo2AprioriEOR(self):
        from oriental.utils import stats, crs
        from oriental import adjust

        import numpy as np
        import pickle
        from osgeo import osr
        osr.UseExceptions()

        with open(r'E:\arap\data\2014-12-04_Carnuntum_mit_Hardwareloesung\globalPos.pickle','rb') as fin:
            PRCsWGS84,PRCsLocal,shortNamesWGps,omFiKasWGS84Tangent,omFiKasLocal,omFiKaShortNames = pickle.load( fin )

        PRCsWGS84 = np.r_[ PRCsWGS84[:-9], PRCsWGS84[-8:] ]
        PRCsLocal = np.r_[PRCsLocal[:-9], PRCsLocal[-8:] ]
        #omFiKasWGS84Tangent,omFiKasLocal,
        srcCs = osr.SpatialReference()
        srcCs.SetWellKnownGeogCS( 'WGS84' ) # srcCs.ImportFromEPSG( 4326 )
        tgtCs = osr.SpatialReference()
        meanLonLat = stats.geometricMedian( np.array(PRCsWGS84)[:,:2] )
        utmZone = crs.utmZone( meanLonLat[0] )
        tgtCs.SetWellKnownGeogCS( "WGS84" )
        tgtCs.SetUTM( utmZone, int(meanLonLat[1] >= 0.) )
        csTrafo = osr.CoordinateTransformation( srcCs, tgtCs )
        PRCsGlobal = [ np.array( csTrafo.TransformPoint( *prcWgs84 ) ) for prcWgs84 in PRCsWGS84 ]
        PRCsLocal = np.array(PRCsLocal)
        PRCsGlobal = np.array(PRCsGlobal)
        # y=s*R.dot(x-x0)
        scale, Rt, P0, res = ori.similarityTrafo( x=PRCsGlobal, y=PRCsLocal )
        omfika = ori.omfika( Rt.T )
        scale = np.array([scale])

        # TODO: images may be oriented completely wrongly, so make this robust! Either by using a robust loss, or by using only a subset of the oriented images (with reliable pts)
        trafoProbl = adjust.Problem()
        trafoLoss = adjust.loss.Trivial()
        for PRCLocal,PRCGlobal in zip(PRCsLocal,PRCsGlobal):
            # P = R.dot( p / s ) + P0
            cost = adjust.cost.SimTrafo3d( p=PRCLocal, P=PRCGlobal )
            trafoProbl.AddResidualBlock( cost,
                                            trafoLoss,
                                            P0,
                                            omfika,
                                            scale )

        # R_tangential = R_sim * R_model
        # -> aus IMU-Winkel R_tangential aufbauen
        #for omFiKaWGS84Tangent, omFiKaLocal in zip(omFiKasWGS84Tangent, omFiKasLocal):
        #    cost = adjust.cost.ObservedOmFiKa( omFiKaWGS84Tangent, omFiKaLocal, stdDevRot )
        #    trafoProbl.AddResidualBlock( cost,
        #                                 trafoLoss,
        #                                 omfika )

        options = adjust.Solver.Options()   
        summary = adjust.Solver.Summary()
        P0_orig, omfika_orig, scale_orig = P0.copy(), omfika.copy(), scale.copy()
        adjust.Solve(options, trafoProbl, summary)
        dummy=0

    def test_distortion(self):
        ior = np.array([50.,-50.,70.])
        adp = adjust.parameters.ADP(normalizationRadius=30.)
        adp[adjust.PhotoDistortion.optPolynomRadial3] = 2.
        coords = np.array([[1.,-2.,1.],
                           [3.,-5.,1.],
                           [6.,-7.,1.]],np.float32)
        coords = coords[:,:2]
        # pass a view:
        ori.distortion_inplace( coords, ior, ori.adpParam2Struct( adp ), ori.DistortionCorrection.dist )

        coords = coords.copy()
        # pass an array that owns its data
        ori.distortion_inplace( coords, ior, ori.adpParam2Struct( adp ), ori.DistortionCorrection.dist )

        coords = np.array([[1.,3.,6.],
                           [-2.,-5.,-7.]]).T
        # pass a transposed array
        ori.distortion_inplace( coords, ior, ori.adpParam2Struct( adp ), ori.DistortionCorrection.dist )

        # pass a transposed array with row stride
        ori.distortion_inplace( coords[::2,:], ior, ori.adpParam2Struct( adp ), ori.DistortionCorrection.dist )
        dummy=0

    def test_transform(self):
        A = np.array([[0.95, 0.01, 0.02],
                      [-0.02, 1.01, -0.03],
                      [0.04, -0.82, 0.99]])
        t = np.array([10., 4.02, 3.22])
        trafo = ori.transform.AffineTransform3D( A, t )
        pts_src = np.random.rand(100,3)
        pts_tgt = trafo.forward( pts_src )
        trafo_check = ori.transform.AffineTransform3D.computeAffinity( source=pts_src, target=pts_tgt )
        np.testing.assert_allclose( trafo.A, trafo_check.A )
        np.testing.assert_allclose( trafo.t, trafo_check.t )

    def test_adjustPlane(self):
        pts = np.array([[ 1, 2, 3.001 ],
                        [ 2, 2, 3],
                        [ 1, 5, 3],
                        [ 1, 7, 3],
                        [ 8, 7, 3],
                        [ 5, 7, 3]], float)
        ori.adjustPlane( pts );

    def test_mayOverlap(self):
        def mayOverlap(R, prcShift):
            x0 = np.array([+1., -1., +1.])
            foV = np.array([
                [0., 0., 0.],
                x0 * [-1., -1., -1.],
                x0 * [-1., +1., -1.],
                x0 * [+1., +1., -1.],
                x0 * [+1., -1., -1.] ]);
    
            foV2 = (foV @ R.T) + prcShift;

            foVs = np.array([
                foV.ravel(),
                foV2.ravel() ])

            return ori.mayOverlap(foVs);

        def checkMayOverlap(expectedResult, R=np.eye(3), prcShift=np.zeros(3), msg=''):
            res = mayOverlap(R, prcShift)
            np.testing.assert_array_equal(res, [[0, 1]] if expectedResult else np.empty((0,2)), msg)

        checkMayOverlap(True , prcShift=np.array([.2, 0., 0.])                  , msg='cams look in identical direction, with prcs slightly shifted')
        checkMayOverlap(True , prcShift=np.array([2.1, 0., 0.])                 , msg='cams look in identical direction, with prcs shifted. Test if we really intersect half-spaces, and not only (finite) pyramids' )
        checkMayOverlap(True , R=ori.ry(100.)                                   , msg='cams look in right angle, with identical prcs. If this test fails, then remove it, because FoVs only overlap in (numerically) identical PRCs.' )
        checkMayOverlap(False, R=ori.ry(100.), prcShift=np.array([-1., 0., 0.]) , msg='cams look in right angle, with non-identical prcs shifted such that fields of view do not overlap' )
        checkMayOverlap(True , R=ori.ry(200.), prcShift=np.array([ 0., 0., -1.]), msg='cams look in opposite directions, onto each other' )
        checkMayOverlap(False, R=ori.ry(200.), prcShift=np.array([ 0., 0., +1.]), msg='cams look in opposite directions, away from each other' )
        dummy=1

if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestOri.test_mayOverlap',
                       exit=False )