# -*- coding: cp1252 -*-
from oriental import graph
from . import cameraMatrix, maxNumItersRANSAC 
import numpy as np
import cv2

def filterMatchesByEMatrix( edge, matches, keypoints, iors, ransacEMatMaxDist, inlierRatio, confidenceLevel ):
    
    img1 = graph.ImageConnectivity.Image(edge[0])
    img2 = graph.ImageConnectivity.Image(edge[1])
    
    pt1 = keypoints[0][matches[:,0],:2]   
    pt2 = keypoints[1][matches[:,1],:2]
    
    K1 = cameraMatrix( iors[0] )
    K2 = cameraMatrix( iors[1] )

    K1Inv = np.linalg.inv( K1 )
    K2Inv = np.linalg.inv( K2 )

    pth1 = np.hstack((pt1,np.ones((pt1.shape[0],1)))).T
    pth2 = np.hstack((pt2,np.ones((pt2.shape[0],1)))).T
    pt_normalized1 = K1Inv.dot( pth1 ) 
    pt_normalized2 = K2Inv.dot( pth2 )
    pt_normalized1 = pt_normalized1[:2,:] / pt_normalized1[2,:]
    pt_normalized2 = pt_normalized2[:2,:] / pt_normalized2[2,:]

    # Note: this function is very performance critical, because it is executed nPhos**2 times!
    # @confidenceLevel=99.9%, nModelPoints=5
    # inlierRatio    maxNumIters
    # 10%         -> 690773
    # 20%         ->  21584
    # 25%         ->   7071
    # 37%         ->    993
    # OpenCV-default: maxNumIters=1000
    maxNumIters = maxNumItersRANSAC( nModelPoints=5, inlierRatio=inlierRatio, confidence=confidenceLevel, nDataPoints=matches.shape[0] )


    # directly use image calibrations, essential matrix
    # cv2.findEssentialMat expects normalized image coordinates, and accepts only 1 reprojection error threshold.
    # If the focal lengths of the 2 images are different, then it's hard to find a meaningful threshold!
    threshold = .5*( ransacEMatMaxDist/K1[0,0] + ransacEMatMaxDist/K2[0,0] )
    essentialMatrix,mask = cv2.findEssentialMat( points1=pt_normalized1.T,
                                                 points2=pt_normalized2.T,
                                                 method=cv2.FM_RANSAC,
                                                 threshold=threshold, # default: 1.
                                                 prob=confidenceLevel, # default: 0.999
                                                 maxNumIters=maxNumIters,
                                                 focal=1.,
                                                 pp=(0.,0.)
                                               )
    # mask ... Output array of N elements, every element of which is set to 0 for outliers and to 1 for the other points.
    assert mask.min() >= 0
    assert mask.max() <= 1
    assert mask.shape == (pt1.shape[0],1)
    # -> convert to 1-dim boolean array
    mask = mask.squeeze() > 0
    
    if float(mask.sum()) / matches.shape[0] < inlierRatio:
        # The actual inlier ratio is lower than what we assumed in the estimation of maxNumIters. So let's not trust the result.
        return None

    # Achtung: cv2.recoverPose verwendet einen hartcodierten Schwellwert für die Objektpunktdistanz==50.
    # Punkte, die weiter weg liegen, werden als instabil betrachtet (liegen nahe 'unendlich' weit weg - daher ist keine Entscheidung möglich, ob sie vor oder hinter der Kamera liegen),
    # und nicht für die Entscheidung verwendet, ob die Kombination von R und t eine gute Wahl ist
    nGood,R,t,mask2 = cv2.recoverPose( essentialMatrix, pt_normalized1[:,mask].T, pt_normalized2[:,mask].T )
    
    # mask2 ... Output array of N elements, every element of which is set to 0 for outliers and to 255 (!) for points who have passed the cheirality check.
    assert np.setdiff1d( mask2, [ 0, 255 ]).size == 0
    mask2 = mask2.squeeze() > 0
    assert nGood == mask2.sum(), "nGood == mask2.sum() failed. nGood={} mask2={}".format( nGood, np.array_str(mask2) )
    assert mask2.shape[0] == mask.sum()
    
    #if nGood < 15:
    #    return None
    
    return ( edge, matches[mask,:], essentialMatrix, R, t.squeeze() )

    