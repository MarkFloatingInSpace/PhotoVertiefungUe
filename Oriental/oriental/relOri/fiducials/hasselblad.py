# -*- coding: cp1252 -*-

from .interface import IFiducialDetector

from oriental import log, utils
from oriental.adjust.parameters import ADP
from oriental.ori.transform import AffineTransform2D
import oriental.utils.gdal

import math
from os import path

from contracts import contract
import numpy as np
from scipy import linalg
import cv2

logger = log.Logger(__name__)

class Hasselblad( IFiducialDetector ):
    
    def __call__( self,
                  imgFn,
                  camSerial, # unused!
                  filmFormatFocal,
                  plotDir = '',
                  debugPlot = False ):
        """
        Hasselblad cameras have no fiducial marks, thus we need to rely on the image border. That border's shape is quite complex in the vicinity of the image corners.
        Also, for many photos, not all corners have been scanned, and not all edges are fully on the scan (occasionally even less than 50%!).

        Some scans of Hasselblad photos show notably bent borders, e.g. 02020603_051.jpg (right edge), 02020603_052.jpg (right), 02020501_026.jpg (right), some even wavy borders, e.g. 02000616_028.tif,
        while some scans have cleanly straight borders. According to md, these bent/wavy borders are probably a result of the middle-format scanner that was used, and the way the film was laid onto it (i.e. not a result of the camera/cassette).
        Thus, this bending cannot be expected to be the same for all photos of the same film.
        Apart from that, it seems that the width and height of the exposed area are not identical across films - which would call for a transformation with 2 different scales.
        The presence of different scales in x- and y-directions seem to be supported by a calibration protocol for Hasselblad 205FCC (#15ER10223) for f=80mm by IPF from 14.3.2001,
        which states that the absolute values of the x-coordinates of the image corners are, on average, 27.5856mm,
                     while       ---""---            y-coordinates                                       27.7398mm,
        where:
           - the image corners are virtual points: they are the intersections of the image borders (as usually done with e.g. 35mm-film)
           - the x-axis is perpendicular to the side of the image that has the only asymmetric features, which are: 2 triangular areas that extend outside the otherwise square exposed area.
    
        Even though we do not try to model the bent/wavy image borders, we need a way to cope with outliers in the extracted outline of the exposed area.
        To detect outliers, fit a line to short segments along each border using a robust weight function.
        Furtheron, treat those adjusted lines as if they were all collinear.

        There are camera calibrations for Hasselblad cameras and specific lenses and film cassettes available from IPF.
        For these calibrations, 4 virtual image corners are used as fiducial marks, which are the intersections of the borders of the exposed area (thus, assuming straight edges).
        We cannot use them here, because LBA generally gives no information about which lens and which cassette was used.
        """
        if plotDir or debugPlot:
            import oriental.utils.pyplot as plt
        if filmFormatFocal is None:
            raise Exception('For Hasselblad images, passing filmFormatFocal is mandatory')

        if not all(( el==60 for el in filmFormatFocal[:2] )):
            raise Exception('You have found a Hasselblad photo whose film format is not 60x60mm²!?')

        img = utils.gdal.imread( imgFn, bands = utils.gdal.Bands.grey )
    
        # For properly scanned images, a constant threshold works.
        # However, there are scans where light seems to have found its way under the film while being scanned, in the background area.
        # See e.g. 01990601_022.jpg
        # Let's assume that at least 50% of all pixels along the outermost rows/columns of the scan are background. And use robust statistics to estimate a proper threshold.
        # Scans may have bright background along only 1 side. A global threshold thus results in either
        # - the edge next to the bright background not being detected, or
        # - the other edges being zigzag.
        # Thus, use a separate threshold for each edge. Apply each threshold to the triangle formed by the edge and the image center.
        binary = np.ones_like( img )
        selection = np.empty_like( img )
        corners = np.array([ [-1,-1],
                             [-1,selection.shape[0]],
                             [selection.shape[1],selection.shape[0]],
                             [selection.shape[1],-1] ])
        for iBorder,border in enumerate( ( np.s_[:,0],
                                           np.s_[-1,:],
                                           np.s_[:,-1],
                                           np.s_[0,:]  ) ):
            if 0:
                borderMedian = np.median( img[border] )
                sigmaMAD = 1.4826 * np.median( abs( img[border] - borderMedian ) )
                # 15==lower bound. Necessary, because it seems that properly scanned images have very low gray values and practically no gray value variance along the borders, but higher values (still below 15) closer to the exposed area.
                # 50==upper bound. e.g. necessary for 01900503_061.jpg, where less than 50% of the upper raster edge are background.
                threshold = max( 15, min( 50, borderMedian + 2*sigmaMAD ) )
            else:
                surelyBackground = np.percentile( img[border], 30 )
                threshold = max( 15, 2*surelyBackground )
            selection[:] = 0
            tri = np.array([ corners[iBorder],
                           [selection.shape[1]//2,selection.shape[0]//2],
                           corners[(iBorder+1)%4] ])
            cv2.fillConvexPoly( selection, tri, 1 )
            binary[selection.view(bool)] = img[selection.view(bool)] > threshold
        del selection

        if 0: # grabCut; works very well, but uses lots of memory and time
            probableBackgroundMargin = max( 3, round(np.mean(img.shape)*0.01) )
            surelyForegroundMargin = max( 3, round(np.mean(img.shape)*0.05/2) )
            imgc = np.dstack( (img,img,img) )
            mask = np.ones_like( img ) * cv2.GC_PR_FGD
            mask[ surelyForegroundMargin:-surelyForegroundMargin+1,
                  surelyForegroundMargin:-surelyForegroundMargin+1 ] = cv2.GC_FGD

            mask[:,:probableBackgroundMargin] = cv2.GC_PR_BGD
            mask[:,-probableBackgroundMargin:] = cv2.GC_PR_BGD
            mask[:probableBackgroundMargin,:] = cv2.GC_PR_BGD
            mask[-probableBackgroundMargin:,:] = cv2.GC_PR_BGD
            selection = np.empty_like( img )
            for border in ( np.s_[:,0],
                            np.s_[-1,:],
                            np.s_[:,-1],
                            np.s_[0,:]  ):
                surelyBackground = np.percentile( img[border], 30 )
                selection[:] = 0
                selection[border] = img[border]<=surelyBackground
                mask[selection.view(bool)] = cv2.GC_BGD
            res = cv2.grabCut( imgc, mask, rect=None, bgdModel=np.empty((0,)), fgdModel=np.empty((0,)), iterCount=1, mode=cv2.GC_INIT_WITH_MASK )
            mask[mask==cv2.GC_PR_BGD]=cv2.GC_BGD
            binary = mask
        # Note that other methods have been tested, without much success:
        # - Instead of using a hard-coded threshold,
        #   we may estimate a good threshold (separately for each area near a border) by combination of cv2.THRESH_BINARY with cv2.THRESH_OTSU, which tries to minimize the in-class grayvalue spread of 2 classes (fore-/background)
        #   However, that results in zig-zag borders of the thresholded image.
        #   thesh,binary = cv2.threshold( img[:,:200], 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU )
        # - cv2.adaptiveThreshold might be powerful if it supported usage of the median value of the neighbour pixels. However, it only supports the mean.
        #   blockSize=9
        #   at1 = cv2.adaptiveThreshold( img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=blockSize, C=2 )
        #   at2 = cv2.adaptiveThreshold( img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=blockSize, C=-2 )
        #   plt.imshow(at1+at2)
        # - cv.Canny seems completely inappropriate
        if debugPlot:
            plt.figure(1); plt.clf()
            plt.imshow( binary, interpolation='nearest', vmin=0, vmax=1 )
            plt.autoscale(False)

        _,contours,_ = cv2.findContours( binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
        contours = [ contour.squeeze() for contour in contours if len(contour) > np.mean(binary.shape)/2 ]
        if debugPlot:
            for contour in contours:
                plt.plot( contour[...,0], contour[...,1], '-g' )
        if plotDir or debugPlot:
            plt.figure( 3, figsize=(8.0, 8.0) ); plt.clf()
            plt.imshow( img, cmap='gray', interpolation='nearest' )
            plt.xticks([]); plt.yticks([])
            plt.autoscale(False)
            for contour in contours:
                plt.plot( contour[...,0], contour[...,1], '-g' )

        binary[:] = 0
        for contour in contours:
            binary[ contour[...,1], contour[...,0] ] = 1
        # What do the docs of cv2.findContours mean with 'the contours touching the image border will be clipped'? They are inset by 1px, i.e. in the 2nd row, 2nd column, etc.! Thus:
        for slice in ( np.s_[:,:2], np.s_[:,-2:], np.s_[:2,:], np.s_[-2:,:] ):
            binary[slice] = 0

        #border = binary - cv2.erode( binary, np.ones( (3,3), dtype=np.int ) )
        # edgelet positions:
        #   15  14  13  12
        # 0                 11
        # 1                 10
        # 2                 9
        # 3                 8
        #    4   5   6   7

        innerEdgeLets_mm = 27. # orthogonal distance of inner edgelets from the PP.
        # outer edgelets distance is innerEdgeLets_mm*sc_fid

        scFid = 1.
        scY = 1.0035
        al = 0.
        sc = 2*innerEdgeLets_mm / np.mean( img.shape )
        x0 = -innerEdgeLets_mm
        y0 = +innerEdgeLets_mm


        # estimate the fiducial scale scFid (outer edgelets (close to corners) vs. inner edgelets)
        # This hasn't shown to help. The difference in scales in the x- and y-direction is larger.
        # Also, the bending of image borders varies from scan to scan - some are bent, some are wavy, many are straight. Better avoid this additional unknown.
        estimScFid = False

        # estimate a scale for the y-coordinates of the assumed fiducial positions in the cam-CS.
        estimScY = True

        fiducials_px_old = None

        def fiducials_mm():
            # In the middle of 1 of the 4 image borders, we expect the asymmetric features (triangles). 
            # Don't try to fit edgelets in those areas. Thus, the edgelets are not equally distributed along the borders.
            ret = np.array([
                [ -innerEdgeLets_mm*scFid,  2/3*innerEdgeLets_mm   ],
                [ -innerEdgeLets_mm      ,  1/3*innerEdgeLets_mm   ],
                [ -innerEdgeLets_mm      , -1/3*innerEdgeLets_mm   ],
                [ -innerEdgeLets_mm*scFid, -2/3*innerEdgeLets_mm   ],
                [ -2/3*innerEdgeLets_mm  , -innerEdgeLets_mm*scFid ],
                [ -1/3*innerEdgeLets_mm  , -innerEdgeLets_mm       ],
                [  1/3*innerEdgeLets_mm  , -innerEdgeLets_mm       ],
                [  2/3*innerEdgeLets_mm  , -innerEdgeLets_mm*scFid ],
                [  innerEdgeLets_mm*scFid, -2/3*innerEdgeLets_mm   ],
                [  innerEdgeLets_mm      , -1/3*innerEdgeLets_mm   ],
                [  innerEdgeLets_mm      ,  1/3*innerEdgeLets_mm   ],
                [  innerEdgeLets_mm*scFid,  2/3*innerEdgeLets_mm   ],
                [  2/3*innerEdgeLets_mm  ,  innerEdgeLets_mm*scFid ],
                [  1/3*innerEdgeLets_mm  ,  innerEdgeLets_mm       ],
                [ -1/3*innerEdgeLets_mm  ,  innerEdgeLets_mm       ],
                [ -2/3*innerEdgeLets_mm  ,  innerEdgeLets_mm*scFid ] ])
            ret[:,1] *= scY
            return ret

        for iSelection in range(50):
            # re-select contours according to inverse trafo from cam-CS; loop until the selection areas don't change any more.

            sAl, cAl = math.sin(al), math.cos(al)    
            R = np.array([ [ cAl, -sAl ],
                           [ sAl,  cAl ] ]) * sc
            pix2cam = AffineTransform2D( R, np.array([x0,y0]) )
            obsCoord = np.ones( 16, dtype=int )
            obsCoord[:4] = 0
            obsCoord[2*4:3*4] = 0
            selCoord = ( obsCoord + 1 ) % 2

            fiducials_px = pix2cam.inverse( fiducials_mm() )
            if fiducials_px_old is not None and \
               abs( fiducials_px_old[np.arange(16),selCoord].round() - fiducials_px[np.arange(16),selCoord].round() ).max() < 1:
                break
            fiducials_px_old = fiducials_px

            edgelets_px = [None]*16
            for iEdgelet in range(16):
                iSide = iEdgelet // 4
                iPos = iEdgelet % 4
                iCoo = iSide % 2
                oCoo = (iSide+1) % 2
                y = round( fiducials_px[iEdgelet,oCoo] )
                if oCoo==1:
                    y *= -1.
                borderSpacing_px = img.shape[iCoo] / 10
                # make pairs of edgelets in the same half on the same side of an image touch each other, i.e. no gap in between
                edgeletHalfLen_px = round( innerEdgeLets_mm / 6 / sc )
                slice = np.s_[ max( y - edgeletHalfLen_px, 0 ) : min( y + edgeletHalfLen_px+1, binary.shape[iCoo] ) ]
                if iSide==0:
                    sel = binary[ slice, : borderSpacing_px ]
                elif iSide==1:
                    sel = binary[ img.shape[0]-borderSpacing_px :, slice ]
                elif iSide==2:
                    sel = binary[ slice, img.shape[1]-borderSpacing_px : ]
                else:
                    sel = binary[ : borderSpacing_px, slice ]
                if debugPlot:
                    plt.figure(2); plt.clf()
                    plt.imshow( sel, interpolation='nearest' )
                    plt.autoscale(False)

                # np.argmax: In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
                if iSide in (0,3):
                    argm = np.argmax( sel, axis=oCoo )
                elif iSide==1:
                    argm = binary.shape[0] -1 -np.argmax( np.flipud(sel), axis=0 )
                else:
                    argm = binary.shape[1] -1 -np.argmax( np.fliplr(sel), axis=1 )
                arange = np.arange( slice.start, slice.stop )
                if iCoo==0:
                    pts = np.column_stack(( argm, arange ))
                else:
                    pts = np.column_stack(( arange, argm ))
                if iSide in (0,3):
                    pts = pts[ pts[:,iCoo]>1 ] # exclude rows (or cols) where the contour is close to the image border; this also excludes rows (cols) without a contour in any of the columns (rows)
                else:
                    pts = pts[ pts[:,iCoo]<(img.shape[oCoo]-2) ]
                lowerQuartile = slice.start + ( slice.stop - slice.start ) // 4
                upperQuartile = slice.stop  - ( slice.stop - slice.start ) // 4
                nLowestPerQuartile = round( ( slice.stop - slice.start ) // 4 * 0.6 )
                if np.count_nonzero( pts[:,oCoo]<lowerQuartile ) < nLowestPerQuartile or \
                   np.count_nonzero( pts[:,oCoo]>upperQuartile ) < nLowestPerQuartile:
                    continue # not enough contour pixels in the lowest or highest decile
                line = cv2.fitLine( pts, distType=cv2.DIST_HUBER, param=1, reps=0.01, aeps=math.atan2( 0.01, sel.shape[iCoo] ) )
                dists = abs( ( pts - line[2:].squeeze() ).dot( np.array([line[1],-line[0]]) ).squeeze() )
                distThresh = 4
                if debugPlot:
                    plt.plot( pts[ dists<=distThresh, 0 ] - slice.start*iCoo - (img.shape[1]-borderSpacing_px)*(iSide==2), pts[ dists<=distThresh, 1 ] - slice.start*oCoo - (img.shape[0]-borderSpacing_px)*(iSide==1), '.g' )
                    plt.plot( pts[ dists> distThresh, 0 ] - slice.start*iCoo - (img.shape[1]-borderSpacing_px)*(iSide==2), pts[ dists> distThresh, 1 ] - slice.start*oCoo - (img.shape[0]-borderSpacing_px)*(iSide==1), '.r' )
    
                pts = pts[ dists<=distThresh, : ]
                if np.count_nonzero( pts[:,oCoo]<lowerQuartile ) < nLowestPerQuartile or \
                   np.count_nonzero( pts[:,oCoo]>upperQuartile ) < nLowestPerQuartile:
                    continue # not enough contour pixels in the lowest or highest decile
                line = cv2.fitLine( pts, distType=cv2.DIST_L2, param=0., reps=0.01, aeps=math.atan2( 0.01, sel.shape[iCoo] ) ).squeeze()
                if iCoo==0: #  atan2 -> [-pi,pi]
                    ang = math.atan2( line[1], line[0] ) % math.pi
                    if abs(ang - math.pi/2) > 10.*math.pi/180.:
                        continue # the direction of the edgelet deviates too much from the vertical
                else:
                    ang = math.atan2( line[1], line[0] ) if line[0]>0 else math.atan2( -line[1], -line[0] )
                    if abs(ang) > 10.*math.pi/180.:
                        continue # the direction of the edgelet deviates too much from the vertical
                t = ( y - line[oCoo+2] ) / line[oCoo]
                x = line[iCoo+2] + t * line[iCoo]
                edgelets_px[iEdgelet] = (x.item(), y) if iCoo==0 else (y,x.item())
    
            nEdgelets = sum( 1 for el in edgelets_px if el is not None )
            edgelets_px = [ (el[0],-el[1]) if el is not None else None for el in edgelets_px ] # ocv->Orient

            squared2norm_old = np.inf
            for iIter in range(50):
                A = np.empty( (nEdgelets,4+int(estimScY)+int(estimScFid)) )
                b = np.zeros( nEdgelets )
                sAl, cAl = math.sin(al), math.cos(al)    
                R = np.array( [[ cAl, -sAl ],
                               [ sAl,  cAl ]] ) * sc
                pix2cam = AffineTransform2D( R, np.array([x0,y0]) )
                iRow = 0
                fiducials_mm_loc = fiducials_mm()
                for iEdgelet,edgelet in enumerate(edgelets_px):
                    if edgelet is None:
                        continue
                    iSide = iEdgelet // 4
                    iPos = iEdgelet % 4
                    iCoo = iSide % 2
                    x,y = edgelet
                    if iCoo==0: # observation of x-coordinate in cam-CS
                        #            dx/d_scale     dx/d_alpha            dx/dx0 dx/dy0
                        A[iRow,:4] = cAl*x - sAl*y, -sc*sAl*x - sc*cAl*y, 1.,    0.
                    else:
                        #            dy/d_scale      dy/d_alpha           dy/dx0 dy/dy0
                        A[iRow,:4] = sAl*x + cAl*y,  sc*cAl*x - sc*sAl*y, 0.,    1.
                    obs = fiducials_mm_loc[iEdgelet,iCoo]
                    iP=4
                    if estimScY:
                        if iCoo==1:
                            A[iRow,iP] = -obs*scFid
                        else:
                            A[iRow,iP] = 0
                        iP+=1

                    if estimScFid:
                        if iPos in (0,3):
                            #           dx/d_scFid
                            A[iRow,iP] = -obs*scY
                        else:
                            A[iRow,iP] = 0.

                    calc = pix2cam.forward( np.array([x,y]) )[iCoo]
                    if iPos in (0,3):
                        #calc -= scFid*obs
                        calc -= obs
                    else:
                        calc -= obs
                    b[iRow] = 0. - calc
                    iRow += 1

                supplements, squared2norm, rank = linalg.lstsq( A, b )[:3]
                if rank < A.shape[1]:
                    raise Exception('Rank deficit! Probably, because no edgelet on one of the image borders could be detected.')
                sc += supplements[0]
                al += supplements[1]
                x0 += supplements[2]
                y0 += supplements[3]
                iP = 4
                if estimScY:
                    scY += supplements[iP]; iP+=1
                if estimScFid:
                    scFid += supplements[iP]; iP+=1

                if abs(squared2norm_old - squared2norm) < 1.e-6**2:
                    break

                squared2norm_old = squared2norm
            else:
                raise Exception( 'Adjustment of similarity transform has not converged' )

        else:
            raise Exception( 'Selection areas have not stabilized' )

        if debugPlot:
            # inverse transform corner points into raster image, as a visual check
            corners_mm = np.array([ [ -1,  1 ],
                                    [  1,  1 ],
                                    [  1, -1 ],
                                    [ -1, -1 ] ], dtype=float ) * 27.
            corners_px = pix2cam.inverse( corners_mm )
            plt.figure(1)
            plt.plot( corners_px[:,0], -corners_px[:,1], '+y' )
            for edgelet_px,fiducial_mm in utils.zip_equal(edgelets_px,fiducials_mm()):
                if edgelet_px is None:
                    continue
                plt.plot( edgelet_px[0], -edgelet_px[1], '+y' )
                fiducial_px = pix2cam.inverse( fiducial_mm )
                plt.plot( fiducial_px[0], -fiducial_px[1], 'xy' )

        if abs( al ) > 5./200.*np.pi:
            logger.warning( 'Misalignment of film in scanner seems very large: {}gon', al/np.pi*200 )

        # So far, pix2cam is appropriate only if the asymmetric features (2 triangles) are on the left side of the image.
        # Get the orientation within the image plane (landscape/portrait).
        meanGrayVals = np.empty( 4 )
        def get_tri_px( iRotation, fac ):
            al = iRotation * np.pi / 2
            sAl, cAl = math.sin(al), math.cos(al)
            R = np.array([ [ cAl, -sAl ],
                           [ sAl,  cAl ] ])
            mask = np.zeros_like( img, dtype=np.int )
            fac2 = scY if iRotation % 2 == 1 else 1.
            tri_mm = np.array([ [ -27.13*fac2, fac*2.45      ],
                                [ -27.02*fac2, fac*2.45+0.15 ],
                                [ -27.02*fac2, fac*2.45-0.15 ] ]).dot( R.T )
            return np.round( pix2cam.inverse( tri_mm ) * (1,-1) ).astype(np.int)

        for iRotation in range(4):
            mask = np.zeros_like( img, dtype=np.int )
            for fac in (-1.,1.):
                tri_px = get_tri_px( iRotation, fac )
                #if plot:
                #    plt.plot( tri_px[[0,1,2,0],0], tri_px[[0,1,2,0],1], '-r' )
            
                _ = cv2.fillConvexPoly( mask, tri_px, (1,1,1), lineType=cv2.LINE_8 )
                # Cope with both triangles being completely outside the scanned area
            meanGrayVals[iRotation] = img[mask>0].mean() if mask.any() else 0
    
        iRotation = np.argmax(meanGrayVals)
        if plotDir or debugPlot:
            plt.figure(3)
            for fac in (-1.,1.):
                tri_px = get_tri_px( iRotation, fac ).mean(axis=0)
                plt.plot( tri_px[0], tri_px[1], markeredgecolor='yellow', linestyle='', marker='o', markerfacecolor='None', markersize=12 )

        meanGrayVals.sort()
        if meanGrayVals[-1] < 20 or \
           meanGrayVals[-2] > 0 and meanGrayVals[-1] / meanGrayVals[-2] < 1.5:
            logger.warning( 'Rotation detection seems unreliable' )

        edgelets_px = edgelets_px[iRotation*4:] + edgelets_px[:iRotation*4]
        al = iRotation * np.pi / 2
        sAl, cAl = math.sin(al), math.cos(al)
        R = np.array([ [ cAl, -sAl ],
                       [ sAl,  cAl ] ])
        if iRotation % 2:
            scY = 1./scY
            R *= scY

        # Note that the rotation point of pix2cam is not in the origin of the cam-CS
        pix2cam = AffineTransform2D( R.T.dot( pix2cam.A ), R.T.dot( pix2cam.t ) )
        fiducials_px = pix2cam.inverse( fiducials_mm() )

        allResiduals_microns = [None]*16
        for iEdgelet,(edgelet_px,fiducial_mm) in enumerate( utils.zip_equal(edgelets_px,fiducials_mm()) ):
            if edgelet_px is None:
                continue
            iSide = iEdgelet // 4
            iCoo = iSide % 2
            allResiduals_microns[iEdgelet] = ( pix2cam.forward( np.array(edgelet_px) ) - fiducial_mm )[iCoo] * 1000.

        residuals_microns = np.fromiter( (el for el in allResiduals_microns if el is not None), dtype=float )
        rmse_microns = ( (residuals_microns**2).sum() / nEdgelets )**.5

        fiducialPos = np.array( [
            [ -1, 15, 14, 13, 12, -1 ],
            [  0, -1, -1, -1, -1, 11 ],
            [  1, -1, -1, -1, -1, 10 ],
            [  2, -1, -1, -1, -1,  9 ],
            [  3, -1, -1, -1, -1,  8 ],
            [ -1,  4,  5,  6,  7, -1 ] ])

        logger.verbose(
            'Transformation (pix->cam) residuals [µm] for Hasselblad photo {}:\v'.format( path.splitext( path.basename(imgFn) )[0] ) +
            'Local:{}\n'.format('\t'*5) +
               '\n'.join( (
                   '\t'.join( (
                       '{:+5.1f}'.format( allResiduals_microns[el] ) if el >= 0 and allResiduals_microns[el] is not None else ' '*5
                   for el in row ) ) 
               for row in fiducialPos ) ) + '\v' +
            'Global:\t\n'
            'RMSE [µm]\t{:.1f}\n'.format( rmse_microns ) +
            '{}'.format( 'Scale Y\t{:.4f}\n'.format(scY) if estimScY else '' ) +
            '{}'.format( 'Scale outer fiducials\t{}\n'.format(scFid) if estimScFid else '' )
        )

        if scY < 1. or scY > 1.02:
            logger.warning( "Scale of fiducials' y-coordinates is expected to be within [1.,1.02], but is: {}", scY )


        mask_mm = np.array([ [ -1,  1 ],
                             [  1,  1 ],
                             [  1, -1 ],
                             [ -1, -1 ] ] ) * (1.,scY) * 26.7

        mask_px = pix2cam.inverse( mask_mm )
        if plotDir or debugPlot:
            plt.figure(3)
            mask_px_loc = np.vstack((mask_px,mask_px[0,:]))
            plt.plot( mask_px_loc[:,0], -mask_px_loc[:,1], '-r' )
            edgelets_px_ = np.array( [ el for el in edgelets_px if el is not None ] )
            plt.scatter( x=fiducials_px[:,0], y=-fiducials_px[:,1], marker="x", color='r' )
            plt.scatter( x=edgelets_px_[:,0], y=-edgelets_px_[:,1], marker="+", color='g' )
            for iPt,(pt,edgelet_px) in enumerate(utils.zip_equal(fiducials_px,edgelets_px)):
                if edgelet_px is None:
                    continue
                plt.text( pt[0], -pt[1], str(iPt), color='r' )
            plt.title( 'RMSE {:.1f}µm scY {:.4f}'.format(rmse_microns,scY) )
        if plotDir:
            plt.savefig( path.join( plotDir, path.splitext( path.basename(imgFn) )[0] + '.jpg' ), bbox_inches='tight', dpi=150 )

        ior = np.array([ 0., 0., filmFormatFocal[2] ])
        adp = ADP( normalizationRadius = 25. )
        # Also return the information if ior,adp are calibrated, or just rough estimates.
        return ior, adp, pix2cam, mask_px, False, rmse_microns

hasselblad = Hasselblad()

if __name__ == '__main__':

    # 02020607: Kamera: H205FCC, Format 60x60mm, f=80mm
    # http://www.hasselbladhistorical.eu/HW/HWVSys.aspx
    #hasselblad(r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\02020607_023.jpg', camSerial=None, filmFormatFocal=(60,60,80), debugPlot=True)

    # Kamera: H553ELX, Format 60x60mm, f=0 !? 16bit
    hasselblad(r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\02000616_025.tif', camSerial=None, filmFormatFocal=(60,60,None), debugPlot=True)

    # Kamera: Hasselbl f=50.
    # constant threshold of 15 is too low! on the background around the exposed area, gray values up to 30 are present.
    #hasselblad(r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\01990601_022.jpg', plot=2)

    difficultPhos = [
        #'01900503_051.jpg',
        #'01900503_052.jpg',
        #'01900503_057.jpg',
        #'01900503_058.jpg', # background in middle of left raster edge is wrongly detected. Outlier edgelet not excluded from adjustment
        #'01900503_061.jpg', # less than 50% of the upper raster edge are background!
        #'01900503_065.jpg',
        #'02000501_131.jpg', # something serverly went wrong here either during exposure or during scanning: see the upper/left border.
        #'02000501_132.jpg,' # same problem here, see upper part of left border
        #'02000501_131.jpg' # same here
        #'02000616_019.tif' # file seems corrupt: the lower 2 thirds are completely black
        '02000607_035.jpg'
    ]

    for fn in difficultPhos:
        try:
            hasselblad( path.join( r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy', fn ), plotDir='.', debugPlot=True )
        except Exception as ex:
            print( 'Exception raised while processing {}: {}'.format( fn, ex ) )