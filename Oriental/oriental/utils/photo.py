# -*- coding: cp1252 -*-
from oriental import adjust, ori, utils
import oriental.adjust.parameters
import oriental.utils.gdal
import oriental.ori.transform

from pathlib import Path

from contracts import contract
import cv2
import numpy as np
from scipy import linalg

# I've tried to parallelize the undistortion of all photos using multiprocessing in Python,
#   boost::interprocess::named_lock in utils::gdal to synchronize file system access, and
#   os.environ['OMP_NUM_THREADS']='1' to prevent ori and cv2 from being parallelized themselves.
# However, that even slowed down the whole process! Seemingly, everything except file access is well parallelized anyway.
# The most expensive operations are probably: decompression/compression in GDAL (which was synchronized by boost::interprocess, so no gain), and ori.undistort
class Undistorter:
    """Simple undistortion of perspective photos, with caching of last used parameters
       Does not change the IOR.
       Simple interpolation in the distorted photo, with fixed neighborhood."""
    def __init__(self):
        self.lastUndistortParams = None

    @contract
    def __call__( self, distFn : Path,
                        ior : 'array[3]',
                        adp : adjust.parameters.ADP,
                        pix2cam : 'None|ITransform2D' = None,
                        mask_px : 'None|array[Nx2](float)' = None ):
        # Create undistorted aerials. Even better would be to pass transformed ADP to SURE, but it seems impossible to transform our model to theirs.
        # Check: IWITNESS (5 parameters – a variation of the Brown model)
        #        INPHOCOEFF (12 parameters).
        pix2cam = pix2cam if not isinstance( pix2cam, ori.transform.IdentityTransform2D ) else None
        mask_px = mask_px if mask_px is None or len(mask_px) > 0 else None

        distFn = distFn.resolve()
        pho,phoInfo = utils.gdal.imread( str(distFn), depth=utils.gdal.Depth.unchanged, info=True )
        self.undistortEntry( pho.shape[0], pho.shape[1], ior, adp, pix2cam, mask_px )

        # Use BORDER_REPLICATE. If pho_rect is going to be saved with lossy compression (JPEG),
        # then this yields less artifacts along mask borders than e.g. BORDER_CONSTANT with e.g. black as border color.
        pho_rect = cv2.remap( pho, self.mapx, self.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE )
        return pho_rect, self.mask, phoInfo, self.ior_rect

    def undistortEntry( self, nRows, nCols, ior, adp, pix2cam, mask_px ):
        if self.lastUndistortParams is not None:
            if nRows == self.lastUndistortParams[0] and \
               nCols == self.lastUndistortParams[1] and \
               np.all( ior == self.lastUndistortParams[2] ) and \
               np.all( adp == self.lastUndistortParams[3] ) and \
               adp.normalizationRadius == self.lastUndistortParams[3].normalizationRadius and \
               adp.referencePoint == self.lastUndistortParams[3].referencePoint and \
               pix2cam is None and self.lastUndistortParams[4] is None and \
               mask_px is None and self.lastUndistortParams[5] is None:
                return
        self.lastUndistortParams = nRows, nCols, ior, adp, pix2cam, mask_px

        gridV = np.mgrid[ :-nRows:-1, :nCols ]
        gridV = np.c_[ gridV[1].flat, gridV[0].flat ].astype(np.float32)
        if pix2cam is None:
            self.ior_rect = ior
        else:
            # Map the center of the rect image to the origin of the cam CS. The PP will be offset accordingly.
            ctr = np.array([ (nCols-1)/2, -(nRows-1)/2 ])
            gridV -= ctr
            gridV *= pix2cam.meanScaleForward()
            self.ior_rect = np.r_[ ctr, 0 ] + ior / pix2cam.meanScaleForward()
        ori.distortion_inplace( gridV, ior, ori.adpParam2Struct( adp ), ori.DistortionCorrection.dist )
        if pix2cam is not None:
            gridV = pix2cam.inverse(gridV)
        gridV[:,1] *= -1.
        self.mapx = gridV[:,0].reshape( (nRows,nCols) )
        self.mapy = gridV[:,1].reshape( (nRows,nCols) )
        self.mapx, self.mapy = cv2.convertMaps( self.mapx, self.mapy, cv2.CV_16SC2 )

        if mask_px is None:
            border = np.r_[
                np.c_[ np.zeros(nRows-1)         , 0 : -nRows+1 : -1           ],
                np.c_[         : nCols-1         , np.full(nCols-1, -nRows+1.) ],
                np.c_[ np.full(nRows-1, nCols-1.), -nRows+1 : 0                ],
                np.c_[ nCols-1 : 0 : -1          , np.zeros(nCols-1)           ]
            ]
        else:
            # if this is a scanned aerial, then use it's mask, as created/used for feature extraction.
            border = []
            segStart = mask_px[0]
            for iEnd in range(1,len(mask_px)+1):
                segEnd = mask_px[iEnd % len(mask_px)]
                direc = segEnd - segStart
                direcLen = linalg.norm(direc)
                if not direcLen:
                    continue # duplicate point?
                direc /= direcLen
                for station in range( int(direcLen) ):
                    border.append( segStart + direc*station )
                    # end point will be appended as start point of the next segment
                segStart = segEnd
            border = np.array(border)
        if pix2cam is not None:
            border = pix2cam.forward(border)
        ori.distortion_inplace( border, ior, ori.adpParam2Struct( adp ), ori.DistortionCorrection.undist )
        if pix2cam is not None:
            border /= pix2cam.meanScaleForward()
            border += ctr
        border[:,1] *= -1
        # due to casting to int, there may result consecutive duplicates. cv2.fillPoly does not seem to have a problem with that.
        border = np.rint( border ).astype(np.int)
        self.mask = np.zeros( self.mapx.shape[:2], dtype=np.uint8 )
        # mask is not self-intersecting, but it may be non-convex (e.g. due to fiducial areas of RMK-A images). Hence, we must not use cv2.fillConvexPoly.
        self.mask = cv2.fillPoly(self.mask, border[np.newaxis], (255, 255, 255))
        self.mask = cv2.erode( self.mask, np.ones((3,3)) ).astype(np.bool)