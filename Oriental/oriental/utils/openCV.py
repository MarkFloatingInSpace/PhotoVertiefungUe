# -*- coding: cp1252 -*-
import cv2
import numpy as np

# for some reason, OpenCV does not export cv2.cv any more, and there does not seem to be available a replacement for cv2.cv.RGB!
class Cv(object):
    @staticmethod
    def RGB( r, g, b ):
        return ( b, g, r, 0. )
cv2.cv = Cv()

def compDrawEpipolar( myPoints, points, which, fundamentalMatrix, img, rgb=cv2.cv.RGB( 0,0,255 ) ):
    img_epi = img#.copy()
    if 0:
        # ComputeCorrespondEpilines is unavailable in cv2 :(
        ptsMat = cv2.cv.fromarray( points )
        #bestPtsLuftMat = cv2.cv.Reshape( bestPtsLuftMat, 2 )
        #bestPtsLuftMat2 = cv2.cv.CreateMat(  cv2.cv.GetDims(bestPtsLuftMat)[0], 1, cv2.cv.CV_32FC2 )
        #cv2.cv.Convert( bestPtsLuftMat, bestPtsLuftMat2 )
        #bestPtsLuftMat = bestPtsLuftMat2
        
        epiPolarLines = cv2.cv.CreateMat( cv2.cv.GetDims(ptsMat)[0], 1, cv2.cv.CV_32FC3 )
        #epiPolarLines = cv2.cv.CreateMat( cv2.cv.GetDims(ptsMat)[0], 3, cv2.cv.CV_32FC1 )
        cv2.cv.ComputeCorrespondEpilines( ptsMat,
                                          which,
                                          cv2.cv.fromarray(fundamentalMatrix),
                                          epiPolarLines )
        for idx in range( cv2.cv.GetDims(epiPolarLines)[0] ):
            # a*x + b*y + c = 0 
            a,b,c = cv2.cv.Get1D(epiPolarLines,idx)[:-1]
            x = [0, img_epi.shape[1] ]
            y = [0, img_epi.shape[0] ]  
            for idxXY in [0,1]:
                if abs(b) > abs(a):
                    y[idxXY] = ( a * x[idxXY] + c ) / -b
                else:
                    x[idxXY] = ( b * y[idxXY] + c ) / -a
            cv2.line( img_epi,
                      ( int(x[0]), int(y[0]) ),
                      ( int(x[1]), int(y[1]) ),
                      color = rgb,#( 1, 1, 1, 0 )
                      thickness=0,
                      lineType=cv2.LINE_AA
                    )
                                                  
    else:
        # Veränderung in D:\swdvlp64\external\OpenCV\modules\calib3d\include\opencv2\calib3d\calib3d.hpp:642 bewirkt, dass cv2.computeCorrespondEpilines exportiert wird:
        # CV_EXPORTS -> CV_EXPORTS_W
        # allerdings erwartet die Funktion einen array vom Typ float statt double! 
        epiPolarLines = cv2.computeCorrespondEpilines( points.astype(np.float32), which, fundamentalMatrix )
        if epiPolarLines is None: # cv2.computeCorrespondEpilines returns None if an empty array is passed
            return img_epi
        epiPolarLines = epiPolarLines.squeeze()
        for idx in range( epiPolarLines.shape[0] ):
            # a*x + b*y + c = 0 
            # Line coefficients are normalized so that a_i^2+b_i^2=1
            a,b,c = epiPolarLines[idx,:]
            if 1: # wahrscheinlich schneller
                x = [0, img_epi.shape[1] ]
                y = [0, img_epi.shape[0] ]  
                for idxXY in [0,1]:
                    if abs(b) > abs(a):
                        y[idxXY] = ( a * x[idxXY] + c ) / -b
                    else:
                        x[idxXY] = ( b * y[idxXY] + c ) / -a
                nShiftBits = 3 # cv2.line expects integer coordinates! the shift-parameter allows for bit-shifted integers. Let's use 2**3 == 8, meaning that we get a precision of 1/8 pixels at the end points
                shiftFac = 2**nShiftBits
                cv2.line( img_epi,
                          ( int(x[0]*shiftFac), int(y[0]*shiftFac) ),
                          ( int(x[1]*shiftFac), int(y[1]*shiftFac) ),
                          color = rgb,#( 1, 1, 1, 0 )
                          thickness=0,
                          lineType=cv2.LINE_AA,
                          shift=nShiftBits
                        )
                if 1:
                    # plot base lines
                    d = a*myPoints[idx,0] + b*myPoints[idx,1] + c
                    x = [ myPoints[idx,0], myPoints[idx,0]-d*a ]
                    y = [ myPoints[idx,1], myPoints[idx,1]-d*b ]
                    cv2.line( img_epi,
                              ( int(x[0]*shiftFac), int(y[0]*shiftFac) ),
                              ( int(x[1]*shiftFac), int(y[1]*shiftFac) ),
                              color = cv2.cv.RGB( 255,255,0 ),
                              thickness=0,
                              lineType=cv2.LINE_AA,
                              shift=nShiftBits
                            )

            else:
                diag = np.linalg.norm(img_epi.shape[:2])
                n0 = np.array([a,b]);
                #n0 /= np.linalg.norm(n0)    
                shift = -c * n0
                pts = []
                for sign in [1,-1]:
                    pt = shift + sign * diag * np.array([-b,a])
                    pts.append(pt)
                   
                nShiftBits = 3 # cv2.line expects integer coordinates! the shift-parameter allows for bit-shifted integers. Let's use 2**3 == 8, meaning that we get a precision of 1/8 pixels at the end points
                cv2.line( img_epi,
                          ( int(pts[0][0]*shiftFac), int(pts[0][1]*shiftFac) ),
                          ( int(pts[1][0]*shiftFac), int(pts[1][1]*shiftFac) ),
                          color = rgb,
                          thickness=0,
                          lineType=cv2.LINE_AA,
                          shift=nShiftBits
                        )
    return img_epi
