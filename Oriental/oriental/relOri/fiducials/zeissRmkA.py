# -*- coding: cp1252 -*-

from oriental.relOri.fiducials.interface import IFiducialDetector

from oriental import log, ori, utils
from oriental.adjust.parameters import ADP
from oriental.ori.transform import AffineTransform2D
import oriental.utils.gdal
import oriental.utils.lsm
import oriental.utils.stats
import oriental.utils.pyplot as plt
from oriental.utils import iround

import re
from os import path

from contracts import contract
import numpy as np
from scipy import linalg
import matplotlib as mpl
import matplotlib.patches
import cv2

logger = log.Logger(__name__)

class Window:
    __slots__ = 'xmin ymin xmax ymax'.split()

    @contract
    def __init__(self, xmin : float, ymin : float, xmax : float, ymax : float ):
        for name in self.__slots__:
            setattr(self,name, float(locals()[name]))

    def min(self):
        return self.xmin, self.ymin

    def max(self):
        return self.xmax, self.ymax

    def width(self):
        return self.xmax - self.xmin

    def height(self):
        return self.ymax - self.ymin

    def extents(self):
        return self.width(), self.height()

class ZeissRmkA( IFiducialDetector ):
    # distinguish RMK A cameras that expose onto the photograph the camera serial number and the focal length in addition to the image serial number.
    mask_withCamSerialFocal_mm = np.array([
        [-114.,  114.],
        [  -6.,  114.],
        [  -6.,  109.],
        [   6.,  109.],
        [   6.,  114.],
        [ 107.,  114.],
        [ 107.,   92.],
        [ 114.,   92.],
        [ 114.,    6.],
        [ 109.,    6.],
        [ 109.,   -6.],
        [ 114.,   -6.],
        [ 114.,  -92.],
        [ 107.,  -92.],
        [ 107., -114.],
        [   6., -114.],
        [   6., -109.],
        [  -6., -109.],
        [  -6., -114.],
        [-107., -114.],
        [-107.,  -93.],
        [-114.,  -93.],
        [-114.,   -6.],
        [-109.,   -6.],
        [-109.,    6.],
        [-114.,    6.]], dtype=np.float )

    # same as self.mask_withCamSerialFocal_mm, but without the masked rectangular areas for the exposed camera serial number and the calibrated focal length
    mask_noCamSerialFocal_mm = np.array([
        [-114.,  114.],
        [  -6.,  114.],
        [  -6.,  109.],
        [   6.,  109.],
        [   6.,  114.],
        [ 114.,  114.],
        [ 114.,    6.],
        [ 109.,    6.],
        [ 109.,   -6.],
        [ 114.,   -6.],
        [ 114., -114.],
        [   6., -114.],
        [   6., -109.],
        [  -6., -109.],
        [  -6., -114.],
        [-107., -114.],
        [-107.,  -93.],
        [-114.,  -93.],
        [-114.,   -6.],
        [-109.,   -6.],
        [-109.,    6.],
        [-114.,    6.]], dtype=np.float )

    # Positions of fiducial marks, their indices (principal point of symmetry at origin), the image serial number's area (SR) and the camera CS: x rightwards, y upwards:
    #     ^ Y
    #     |
    #     |
    #
    #     2   
    # 
    # 1  PPA  3   --> X
    #
    # SR  0   
    #
    # The data panel, if scanned, is to the left of the image serial number.

    calibrations_mm = {
        # source: 20204.cam (camera calibration file of Leica Photogrammetry Suite)
        #'20204' : ( np.array([ -0.032, 0.008, 208.158 ]), # PPA(x0,y0),f
        #            np.array([
        #              #   X[mm]    Y[mm]
        #              [   -0.032, -113.001 ],
        #              [ -113.034,    0.001 ],
        #              [   -0.026,  113.01  ],
        #              [  112.974,    0.005 ]
        #           ]),
        #           True ),

        # RMK A 21/23; 12.04.2000 Deutscher Kalibrierdienst
        '20204' : ( np.array([ -0.029, 0.001, 208.160 ]), # PPA(x0,y0),f
                    np.array([
                      #   X[mm]    Y[mm]
                      [   -0.033, -113.009 ],
                      [ -113.042,   -0.008 ],
                      [   -0.034,  112.993 ],
                      [  112.964,   -0.009 ]
                    ]),
                    True
                  ),

        # RMK A 15/23; 30.01.1980 Carl Zeiss Oberkochen
        '21163' : ( np.array([ 0.008, -0.007, 152.425 ]), # PPA(x0,y0),f
                    np.array([
                      #   X[mm]    Y[mm]
                      [   -0.009, -113.025 ],
                      [ -113.012,   -0.031 ],
                      [   -0.002,  112.980 ],
                      [  112.997,   -0.025 ]
                   ]),
                   False
                  ),

        # RMK A 15/23; 05.05.1976 IPF?
        '21216' : ( np.array([ 0.007, -0.003, 152.552 ]), # PPA(x0,y0),f
                    np.array([
                         #   X[mm]    Y[mm]; Fiducial marks have been re-ordered to match our order
                         [    0.010, -113.012 ],
                         [ -112.988,   -0.015 ],
                         [    0.016,  112.978 ],
                         [  113.014,   -0.010 ]
                     ]),
                     None # Never seen a photo of this camera, neither seen a sketch of it in a calibration report. So we really don't know
                  ),

        # RMK A 30/23; 08.09.1999 Deutscher Kalibrierdienst
        '137624' : ( np.array([ 0.001, 0.042, 304.832 ]), # PPA(x0,y0),f
                     np.array([
                         #   X[mm]    Y[mm]; Fiducial marks have been re-ordered to match our order
                         [   -0.005, -112.963 ],
                         [ -113.002,    0.045 ],
                         [   -0.007,  113.034 ],
                         [  112.997,    0.042 ]
                     ]),
                     False # I have never seen a photo from this camera. According to the sketch in the calibration protocol, it does not have data fields for camera serial and focal length
                   )
    }

    @staticmethod
    @contract
    def getImageWindow( img : 'array[RxC](uint8)', plot: int ):
        marginScale = max(img.shape) / 10390 # based on D:\AdV_Benchmark_Historische_Orthophotos\Wuppertal_1969\Tif_16bit\370_3_38.tif
        img = img.copy()
        pyrLvl = 0
        while min(img.shape[:2]) > 500:
            # Note: OpenCV docs say:
            # size of the output image is computed as Size((src.cols+1)/2, (src.rows+1)/2)
            # it downsamples the image by rejecting even rows and columns.
            # This may be interpreted as: columns and rows with even indices are discarded.
            # However, it seems that those with odd indices are discarded!
            # Also, the computation of the output image resolution makes sense only if the latter is true.
            # Hence, we must not apply pyrLvl as shift when transforming image coordinates from a higher pyramid level to the orig image.
            img = cv2.pyrDown(img)
            pyrLvl += 1
        pyrScale = 1. / 2 ** pyrLvl

        marginScale *= pyrScale
        fgdMargin = round( 1300 * marginScale)  # foreground margin: margin along the image borders that is not surely foreground
        bgdMargin = round(fgdMargin * .75)  # background margin
        hext = round(400 * marginScale)  # extents of background along the image border at the fiducials

        mask = np.full(img.shape, cv2.GC_FGD, np.uint8)  # surely foreground

        mask[0:fgdMargin] = cv2.GC_PR_FGD  # probably foreground
        mask[-fgdMargin:] = cv2.GC_PR_FGD
        mask[:, 0:fgdMargin] = cv2.GC_PR_FGD
        mask[:, -fgdMargin:] = cv2.GC_PR_FGD

        hh, hw = [round(el / 2) for el in mask.shape]

        slices = ( np.s_[hh - hext:hh + hext, :bgdMargin],
                   np.s_[hh - hext:hh + hext, -bgdMargin:],
                   np.s_[-bgdMargin:, hw - hext:hw + hext],
                   np.s_[:bgdMargin, hw - hext:hw + hext] )
        #for slic in slices:
        #    mask[slic] = cv2.GC_PR_BGD  # probably background

        # Even with grabCut, we need this threshold.
        # Otherwise, instead of the right image border of 0219870411_007.sid,
        # the cut will go along roads (within the image area).
        thresh = 60
        for slic in slices:
            subMask = mask[slic]
            mask2 = img[slic]<thresh
            subMask[mask2] = cv2.GC_PR_BGD
            subMask[np.logical_not(mask2)] = cv2.GC_PR_FGD

        if plot>1:
            plt.figure(2, clear=True)
            plt.imshow(mask, interpolation='nearest')

        bgdModel = np.empty(0)
        fgdModel = np.empty(0)

        # cv2.grabCut supports 8-bit 3-channel only
        img = np.dstack((img, img, img))

        for it in range(100):
            oldMask = mask.copy()
            mode = cv2.GC_INIT_WITH_MASK if it == 0 else cv2.GC_EVAL
            mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect=None,
                                                   bgdModel=bgdModel, fgdModel=fgdModel, iterCount=1, mode=mode)
            if plot>1:
                plt.figure(2)
                plt.imshow(mask, interpolation='nearest')
            # Don't stop early, even if changes are only small. Updates don't cost much, and small changes sometimes matter.
            if (mask == oldMask).all():
                break
        assert cv2.GC_BGD == 0
        mask[mask == cv2.GC_PR_BGD] = cv2.GC_BGD
        mask[mask != cv2.GC_BGD] = cv2.GC_FGD
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        ctrLabel = labels[round(labels.shape[0] / 2), round(labels.shape[1] / 2)]
        assert ctrLabel != 0, 'Center point detected as background'

        if plot:
            plt.figure(1)
            limits = plt.axis()

            contours, hierarchy = cv2.findContours((labels == ctrLabel).astype(np.uint8),
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            assert len(contours) == 1
            contour = contours[0].squeeze() / pyrScale
            contour = np.r_[contour, contour[:1, :]]
            plt.plot(contour[:, 0], contour[:, 1], '-c', clip_on=False)

        x, y, w, h = stats[ctrLabel, [cv2.CC_STAT_LEFT,
                                      cv2.CC_STAT_TOP,
                                      cv2.CC_STAT_WIDTH,
                                      cv2.CC_STAT_HEIGHT]]

        # note: we need to subtract 1 from width and height to get the right/lower coordinates, before applying pyrScale.
        window = Window( xmin=x/pyrScale,
                         ymin=y/pyrScale,
                         xmax=(x + w - 1)/pyrScale,
                         ymax=(y + h - 1)/pyrScale)

        if plot:
            # plt.plot( window[[0,0,1,1,0],0],
            #          window[[0,1,1,0,0],1], '-r', clip_on=False )

            # Note: we might extract the rotated minimum rectangle from the contour using minAreaRect
            # However, the outermost points of the image area of RMK A photos use to be close to the fiducials,
            # which are close to the centers of the image edges. Hence, the rotation would be very weakly defined,
            # so let's just use the axis-aligned minimum rectangle.
            # To derive a precise, rotated rectangle, we would need to estimate a first one,
            # drop the contour points close to the midpoints of its edges, and then estimate the final, rotated rectangle.
            # rotRect = cv2.boxPoints( cv2.minAreaRect( contour.astype(np.float32)) )
            # rotRect = np.r_[ rotRect, rotRect[0:1,:] ]
            # plt.plot( rotRect[:,0], rotRect[:,1], '-y', clip_on=False )

            plt.axis(limits)

        return window

    @staticmethod
    def getFiducialsMatch( img, window : Window, fiducialArea_mm, fiducialRadius_mm, fiducialOffsetFromBorder_mm, debugPlot ):
        """Doesn't work too well, because fiducialRadius_mm must be quite accurate,
         while the radius seems to vary from camera to camera.
         A work-around was to try with different known radii."""
        x,y = window.min()
        w,h = window.extents()
        mm2px = (w+h)/2/230
        fiducialArea_px = [ el*mm2px for el in fiducialArea_mm ]
        fiducialRadius_px = fiducialRadius_mm * mm2px
        fiducialOffsetFromBorder_px = fiducialOffsetFromBorder_mm * mm2px

        templateSize = iround(fiducialRadius_px*3)
        if templateSize % 2 == 0: # make odd number of cols/rows
            templateSize += 1
        tmplHalfSize = (templateSize-1)//2
        template = np.zeros( [templateSize]*2, dtype=np.uint8 )

        # By the way, that's a good argument to use a centroid operator after coarse location with the template, as that operator does not depend on the radius.

        # cv's drawing functions expect integer coordinates! the shift-parameter allows for bit-shifted integers. Let's use 2**8 == 256, meaning that we get a precision of 1/256 pixels precision
        # only center and radius are shifted, not thickness!
        nShiftBits = 8
        shiftFac = 2**nShiftBits
        cv2.circle( template,
                    center=( int(tmplHalfSize*shiftFac), )*2,
                    radius=iround(fiducialRadius_px*shiftFac),
                    color=(255,255,255,0), # white
                    thickness=-1, #  Thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn
                    lineType=cv2.LINE_AA,
                    shift=nShiftBits )

        if debugPlot:
            plt.figure(3,clear=True)
            plt.imshow( template, interpolation='nearest', cmap='gray', vmin=0, vmax=256 )
            plt.axhline( (template.shape[0]-1)/2 )
            plt.axvline( (template.shape[1]-1)/2 )

        fiducials_px = np.ones( (4,2) )
        for iFiducial in range(4):
            if iFiducial==0:
                row, col = y+h-fiducialOffsetFromBorder_px, x+w/2
            elif iFiducial==1:
                row, col = y+h/2, x+fiducialOffsetFromBorder_px
            elif iFiducial==2:
                row, col = y+fiducialOffsetFromBorder_px, x+w/2
            elif iFiducial==3:
                row, col = y+h/2, x+w-fiducialOffsetFromBorder_px

            fidExtents = fiducialArea_px if iFiducial % 2 == 0 else list(reversed(fiducialArea_px))
            searchRect = ( row-fidExtents[1]/2, row+fidExtents[1]/2+1,
                           col-fidExtents[0]/2, col+fidExtents[0]/2+1 )

            # Note: using slicing syntax, indices beyond the extents of the indexed array do not yield errors,
            # but negative indices would result in empty results. Hence, max(0,el)
            searchRect = tuple( iround(max(0,el)) for el in searchRect )
            searchWin = img[ searchRect[0]:searchRect[1], searchRect[2]:searchRect[3] ]

            if debugPlot:
                plt.figure(4,clear=True)
                plt.imshow( searchWin, cmap='gray', interpolation='nearest' )

            # If image is W x H and template is w x h , then result is (W-w+1) x (H-h+1)
            ccorr =  cv2.matchTemplate( image=searchWin, templ=template, method=cv2.TM_CCOEFF_NORMED )
            iMaximum = ccorr.argmax() # with axis=None, returns the (scalar) flat index of the maximum value
            maxCCoeff = ccorr.flat[iMaximum]

            row, col = np.unravel_index(iMaximum, ccorr.shape)  # transform flat index to row,col

            if debugPlot:
                plt.figure(5,clear=True)
                #plt.contour(ccorr)
                plt.imshow(ccorr, cmap='RdBu', interpolation='nearest', vmin=0.)
                plt.gca().add_artist(mpl.patches.Circle((col, row), fiducialRadius_px, color='k', fill=False))
                # plt.scatter( x=[col], y=[row], marker="x", color='g' )

            # Kraus Bd.2 Kap. 6.1.2.4
            if maxCCoeff < 0.7:
                raise Exception("Maximum cross correlation coefficient is only {:.2}. Detection of fiducial mark #{} failed".format(maxCCoeff,iFiducial))

            # coordinates of max ccoeff w.r.t. upper/left px of searchWin
            row += tmplHalfSize
            col += tmplHalfSize

            if debugPlot:
                plt.figure(4)
                plt.plot( col, row, '.r' )

            # compute centroid for sub-pixel precision
            circleRadiusExtPx = iround(fiducialRadius_px * 1.5)
            for iIter in range(10): # iterate until selection area does not change any more - risk of oscillation?
                ul = ( iround(row-circleRadiusExtPx),
                       iround(col-circleRadiusExtPx) )
                lr = ( ul[0]+2*circleRadiusExtPx+1,
                       ul[1]+2*circleRadiusExtPx+1 )
                # check if searchWin fully contains range. Note: fancy indexing with negative indices is not an error, and fancy indexing beyond the indexed array dimensions is not an error either!
                for dim,coo in ( (dim,coo) for (dim,coords) in enumerate(utils.zip_equal(ul,lr)) for coo in coords ):
                    if coo < 0 or coo > searchWin.shape[dim]:
                        raise Exception( 'Image does not fully contain selection area' )

                searchLoc = searchWin[ ul[0] : lr[0],
                                       ul[1] : lr[1] ]
                total = searchLoc.sum()
                cRow = sum( (iRow*el for iRow,row in enumerate(searchLoc  ) for el in row ) ) / total
                cCol = sum( (iCol*el for iCol,col in enumerate(searchLoc.T) for el in col ) ) / total
                if debugPlot:
                    plt.figure(6,clear=True)
                    plt.imshow(searchLoc, cmap='gray')
                    plt.plot(cCol, cRow, '.r')
                    for fac in ( .5, 1, 1.5 ):
                        plt.gca().add_artist(mpl.patches.Circle((cCol, cRow), fiducialRadius_px*fac, color='r', fill=False))

                row = ul[0] + cRow
                col = ul[1] + cCol
                if round(row-circleRadiusExtPx) == ul[0] and \
                   round(col-circleRadiusExtPx) == ul[1]:
                    break
            else:
                raise Exception('Selection area for centroid operator is oscillating')

            if debugPlot:
                plt.figure(4)
                plt.plot( col, row, '.g' )

            # transform to original image CS
            row += searchRect[0]
            col += searchRect[2]

            fiducials_px[iFiducial,:] = col, -row # -> OrientAL

        return fiducials_px

    @staticmethod
    def getCircle( shape, center, radius):
        # cv's drawing functions expect integer coordinates! the shift-parameter allows for bit-shifted integers. Let's use 2**8 == 256, meaning that we get a precision of 1/256 pixels precision
        # only center and radius are shifted, not thickness!
        nShiftBits = 8
        shiftFac = 2**nShiftBits

        origShape = tuple(shape[:]) # copy
        # for very small radii, we want a smoother edge, so let's draw at twice the resolution and downsample afterwards
        doubled=False
        if radius < 2:
            doubled = True
            shape = [ el*2 for el in shape]
            center = [el*2 for el in center]
            radius *= 2

        img = np.zeros(shape, np.uint8)
        cv2.circle(img,
                   center=tuple(iround(el * shiftFac) for el in center),
                   radius=iround(radius * shiftFac),
                   color=(255, 255, 255, 0),
                   thickness=-1,  # fill
                   lineType=cv2.LINE_AA,
                   shift=nShiftBits)
        if doubled:
            img = cv2.pyrDown(img)

        assert img.shape == origShape
        return img

    @staticmethod
    @contract
    def getFiducialsGrabCut( img : 'array[RxC](uint8)', fiducialArea_px, fiducialRadiusRange_px, fiducialOffsetFromBorder_px, imgName, debugPlot ):
        fidNames = {0: 'bottom', 1: 'left', 2: 'top', 3: 'right'}

        maxR, maxC = [el-1 for el in img.shape]
        # Detect bright discs within the 4 fiducial areas.
        # With combinations to choose?
        # We might collect all and then choose those with similar radii.
        # However, radii may not appear to be the same, if e.g. one fiducial is blurry, while the others are not.
        # Hence, choose the disc for each fiducial area separately.
        fiducials_px = np.ones( (4,2) )
        for iFiducial in range(4):
            if iFiducial==0:
                row, col = maxR-fiducialOffsetFromBorder_px, maxC/2
            elif iFiducial==1:
                row, col = maxR/2, fiducialOffsetFromBorder_px
            elif iFiducial==2:
                row, col = fiducialOffsetFromBorder_px, maxC/2
            elif iFiducial==3:
                row, col = maxR/2, maxC-fiducialOffsetFromBorder_px

            fidExtents = fiducialArea_px if iFiducial % 2 == 0 else list(reversed(fiducialArea_px))
            searchRect = ( row-fidExtents[1]/2, row+fidExtents[1]/2+1,
                           col-fidExtents[0]/2, col+fidExtents[0]/2+1 )

            # Note: using slicing syntax, indices beyond the extents of the indexed array do not yield errors,
            # but negative indices would result in empty results. Hence, max(0,el)
            searchRect = tuple( iround(max(0,el)) for el in searchRect )
            searchWin = img[ searchRect[0]:searchRect[1], searchRect[2]:searchRect[3] ]

            if debugPlot:
                plt.figure(4,clear=True)
                plt.imshow( searchWin, cmap='gray', interpolation='nearest' )

            # cv2.grabCut supports 8-bit 3-channel only
            searchWin = np.dstack((searchWin, searchWin, searchWin))
            mask = np.full(searchWin.shape[:2], cv2.GC_PR_FGD, np.uint8)
            # Simply marking the single outermost columns/rows as probably background (except for the side of the image content)
            # and the rest as probably foreground does not work for D:\\AdV_Benchmark_Historische_Orthophotos\\Wuppertal_1969\\Tif_16bit\\370_1_116.tif:
            # A few rows at the bottom are completely black (0, probably due to a scanner error), and only these would then get detected as background.
            # Same problem with 370_2_100.tif on the right border
            bgdMargin = max(1,iround(min(mask.shape)/4))
            if iFiducial != 0:
                mask[:bgdMargin,:] = cv2.GC_PR_BGD
            if iFiducial != 1:
                mask[:,-bgdMargin:] = cv2.GC_PR_BGD
            if iFiducial != 2:
                mask[-bgdMargin:,:] = cv2.GC_PR_BGD
            if iFiducial != 3:
                mask[:,:bgdMargin] = cv2.GC_PR_BGD
            # Set the border opposite the image content as background, to overcome issues with 370_1_116.tif and 370_2_100.tif
            if iFiducial==0:
                mask[-1,:] = cv2.GC_BGD
            elif iFiducial==1:
                mask[:,0] = cv2.GC_BGD
            elif iFiducial==2:
                mask[0, :] = cv2.GC_BGD
            elif iFiducial==3:
                mask[:,-1] = cv2.GC_BGD
            if debugPlot:
                plt.figure(6, clear=True)
                plt.imshow(mask, interpolation='nearest', cmap='gray')
            bgdModel = np.empty(0)
            fgdModel = np.empty(0)
            for it in range(100):
                oldMask = mask.copy()
                mode = cv2.GC_INIT_WITH_MASK if it == 0 else cv2.GC_EVAL
                mask, bgdModel, fgdModel = cv2.grabCut(searchWin, mask, rect=None,
                                                       bgdModel=bgdModel, fgdModel=fgdModel, iterCount=1, mode=mode)
                if debugPlot:
                    plt.figure(6)
                    plt.imshow(mask, interpolation='nearest')
                # don't stop early: updates don't cost much, and are sometimes important
                if (mask == oldMask).all():
                    break
            mask[mask == cv2.GC_PR_BGD] = cv2.GC_BGD
            mask[mask != cv2.GC_BGD] = cv2.GC_FGD
            if np.count_nonzero(mask) / mask.size > .5: # foreground detected as background
                mask = np.logical_not(mask).astype(np.uint8)*255
            nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            circles = np.zeros( 0,dtype=[(name,float) for name in 'x y radius cc eccentricity'.split()])
            for label in range(1,nLabels): # skip background label
                if debugPlot:
                    plt.figure(6)
                    plt.imshow(labels==label)
                xmin, ymin, width, height = stats[label,
                                                  [cv2.CC_STAT_LEFT,cv2.CC_STAT_TOP,cv2.CC_STAT_WIDTH,cv2.CC_STAT_HEIGHT]]
                if stats[label,cv2.CC_STAT_AREA] < 3: # (labels==label).sum()
                    continue
                if xmin == 0 or ymin == 0 or \
                   xmin+width == labels.shape[1] or ymin+height == labels.shape[0]:
                    continue # contour touches an image border.
                if width < 2 or height < 2:
                    continue # cannot estimate a circle
                # use CHAIN_APPROX_NONE here, since we want to adjust based on all points.
                binaryImg, contours, hierarchy = cv2.findContours((labels == label).astype(np.uint8),
                                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                assert len(contours) == 1, 'Extraction of the external contours of a connected component yielded no or multiple polygons'
                contour = contours[0].squeeze()
                # fit a circle to the contour
                xm, ym = contour.mean(axis=0)
                radius = ( (width-1)/2 + (height-1)/2 ) / 2
                success = False
                for it in range(10):
                    xDiff = contour[:,0] - xm
                    yDiff = contour[:,1] - ym
                    dists = ( xDiff**2 + yDiff**2 )**.5
                    if ( dists < 1.e-3 ).any(): # avoid division by zero
                        break
                    resid = dists - radius
                    common = -1./dists
                    A = np.empty( (contour.shape[0], 3) )
                    A[:,0] = common * xDiff # d/dxm
                    A[:,1] = common * yDiff # d/dym
                    A[:,2] = -1 # d/dradius
                    try:
                        C = linalg.cholesky( A.T @ A, lower=False)
                        delta_x = linalg.cho_solve((C, False), A.T @ -resid )
                    except Exception:
                        break
                    xm += delta_x[0]
                    ym += delta_x[1]
                    radius += delta_x[2]
                    if radius <= 0:
                        break
                    # check if circle is outside whole image area
                    if xm - radius < -.5 or xm + radius > labels.shape[1] or \
                       ym - radius < -.5 or ym + radius > labels.shape[0]:
                        break
                    if np.abs(delta_x).max() < 1.e-3:
                        success = True
                        break
                if not success:
                    continue # no convergence

                if debugPlot:
                    plt.figure(4)
                    plt.gca().add_artist( mpl.patches.Circle((xm, ym), radius, color='r', fill=False))
                    #plt.plot( xm, ym, '.r')

                # We expect the segmented fiducials to be of circular shape
                # i.e. the discs must be surrounded by a strong anough edge that gets detected by grabCut.
                # Hence, if the adjusted circle extends considerably outside the originally detected segment,
                # then the segment is probably not a circle.
                # However, grabCut may detect blurry fiducials too small or too large,
                # so use a threshold that adapts to the segment size.
                # Note that for LSM, we better add a dark background around the disc.
                searchCut = searchWin[ymin : ymin+height, xmin : xmin + width, 0]
                #margin = iround(radius)
                #template = np.zeros( [el+margin*2 for el in searchCut.shape], np.uint8 )
                #tplCenter = tuple(el+margin for el in (xm-xmin, ym-ymin))
                margins = height, width
                slic = np.s_[ymin-margins[0] : ymin+height+margins[0],
                             xmin-margins[1] : xmin+width +margins[1] ]
                if slic[0].start < 0 or slic[0].stop > searchWin.shape[0] or \
                   slic[1].start < 0 or slic[1].stop > searchWin.shape[1]:
                    continue # search window outside of searchWin
                template = __class__.getCircle(
                    shape=[el + marg * 2 for el, marg in zip(searchCut.shape, margins)],
                    center=tuple(el + marg for el, marg in zip( (xm - xmin, ym - ymin), margins[::-1] ) ),
                    radius=radius )
                if debugPlot:
                    plt.figure(7,clear=True)
                    plt.subplot(1,2,1)
                    plt.imshow(template,cmap='gray')
                    plt.subplot(1,2,2)
                    plt.imshow(searchWin[slic[0], slic[1], 0 ],cmap='gray')
                if template[0,:].any() or template[-1,:].any() or \
                   template[:,0].any() or template[:,-1].any():
                    continue # circle outside template

                # LSM
                # window: [ minX, maxY, maxX, minY ] inclusive i.e. this is the outline of pixel borders, not pixel centers. OrientAL img-CS.
                masterWindow = np.array([ searchRect[2] + xmin - margins[1] - .5,
                                         -(searchRect[0] + ymin - margins[0] - .5),
                                          searchRect[2] + xmin + width-1 + margins[1] + .5,
                                         -(searchRect[0] + ymin + height-1 + margins[0] + .5)])

                # Fix the transformation of the synthetic image, since it is small.
                img0 = utils.lsm.Image()
                img0.id = 0
                img0.dataSetPath = utils.gdal.memDataset( template )
                img0.master2slave.tPost = np.array([-(masterWindow[0] + .5),
                                                    -(masterWindow[1] - .5)])
                img1 = utils.lsm.Image()
                img1.id = 1
                img1.dataSetPath = utils.gdal.memDataset( img )

                solveOpts = utils.lsm.SolveOptions()
                solveOpts.preprocess_shifts_master_window_factor = 1 # no initial brute force shifts
                solveOpts.min_num_pix = -1 # full resolution only
                solveOpts.geometricTrafo = utils.lsm.GeometricTrafo.affine
                solveOpts.max_num_iterations = 100
                solveOpts.storeFinalCuts = True
                lsmObj = utils.lsm.Lsm(solveOpts)
                summary = utils.lsm.SolveSummary()
                try:
                    success = lsmObj(masterWindow, [img0, img1], summary)
                except Exception as ex:
                    raise Exception("{}: LSM on {} fiducial mark failed:\n{}".format( imgName, fidNames[iFiducial], str(ex)))
                if not success:
                    continue

                fullResLvl = summary.resolutionLevels[-1]
                if debugPlot:
                    plt.figure(5,clear=True)
                    ax1=plt.subplot(1,3,1)
                    ax1.set_title('synthetic')
                    plt.imshow(fullResLvl.cuts[0],cmap='gray')
                    ax2=plt.subplot(1,3,2)
                    ax2.set_title('observed')
                    plt.imshow(fullResLvl.cuts[1], cmap='gray')
                    diff = fullResLvl.cuts[0] - fullResLvl.cuts[1]
                    maxAbs = np.abs(diff).max()
                    ax3=plt.subplot(1, 3, 3)
                    ax3.set_title('synth - obs')
                    plt.imshow(diff, cmap='RdBu', vmin=-maxAbs, vmax=maxAbs, interpolation='nearest')
                    plt.colorbar(format='%+.2f',orientation='horizontal',ax=[ax1,ax2,ax3])

                if img1.contrast < 0:
                    continue
                det = linalg.det(img1.master2slave.A)
                if det <= 0:
                    continue

                trafo0, trafo1 = [AffineTransform2D(img.master2slave.A, img.master2slave.tPost) for img in [img0,img1]]
                center = np.array([xm-xmin+margins[1],ym-ymin+margins[0]])
                center = trafo1.forward(trafo0.inverse(center * (1,-1))) * (1,-1)
                radius *= det**.5 # since we estimate an affine transform, this is the square root of the ellipse area
                ccorr = cv2.matchTemplate(image=fullResLvl.cuts[0], templ=fullResLvl.cuts[1], method=cv2.TM_CCOEFF_NORMED).item()
                a, b = linalg.svdvals(img1.master2slave.A) # sorted in decreasing order
                eccentricity = ( 1 - b**2 / a**2  )**.5
                if debugPlot:
                    plt.figure(4)
                    #plt.plot( center[0]-searchRect[2], center[1]-searchRect[0], '.c')
                    plt.gca().add_artist(mpl.patches.Circle((center[0]-searchRect[2], center[1]-searchRect[0]), radius, color='c', fill=False))
                    plt.text( center[0]-searchRect[2], center[1]-searchRect[0], str(len(circles)) , color='c' )
                circles = np.resize( circles, len(circles)+1 )
                circles[-1]['x'] = center[0]
                circles[-1]['y'] = center[1]
                circles[-1]['radius'] = radius
                circles[-1]['cc'] = ccorr
                circles[-1]['eccentricity'] = eccentricity

            if not len(circles):
                raise Exception('{}: unable to detect {} fiducial mark'.format(imgName, fidNames[iFiducial]) )
            # Detection of small circles is not too reliable.
            # One of the reasons is that cv2.circle does not do anti-aliasing for small circles!
            # Hence, first search for the largest circle (that is not too large) among the circles with high cc.
            # eccentricity=0.4 -> b = 0.917 * a
            #              0.5 -> b = 0.866 * a
            radiusInRange = np.logical_and(circles['radius'] > fiducialRadiusRange_px[0],
                                           circles['radius'] < fiducialRadiusRange_px[1])
            if not radiusInRange.any():
                radiusInRange[:] = True # If the radii of all circles are out of bounds, then let's consider all circles.
                logger.warning("{}: the radii of all detected circles are out of bounds. Position of {} fiducial mark seems unreliable".format(
                    imgName, fidNames[iFiducial]))
            for maxEccentricity in (.4,.5,.6,np.inf):
                circularAndRadiusInRange = np.logical_and( circles['eccentricity'] < maxEccentricity, radiusInRange )
                if circularAndRadiusInRange.any():
                    iBestCircle = circles[ circularAndRadiusInRange ]['radius'].argmax()
                    circle = circles[circularAndRadiusInRange][iBestCircle]
                    break
            if circle['eccentricity'] > 0.5:
                logger.warning( "{}: the eccentricity {:.2} of the chosen circle seems high. Position of {} fiducial mark seems unreliable".format(
                    imgName, circle['eccentricity'], fidNames[iFiducial] ))

            # Kraus Bd.2 Kap. 6.1.2.4
            if circle['cc'] < 0.7:
                logger.warning("{}: maximum cross correlation coefficient is only {:.2}. Position of {} fiducial mark seems unreliable".format(
                    imgName, circle['cc'], fidNames[iFiducial] ))

            fiducials_px[iFiducial,:] = circle['x'], -circle['y'] # -> OrientAL

        # TODO first, collect all circles for all fiducials.
        # Only afterwards, decide which circles to choose.
        # Criteria: similar radii, and resulting in small RMSE for the similarity transform.
        return fiducials_px

    @staticmethod
    @contract
    def getTextTemplate( text: str, mm2px: float ):
        targetFontHeight = 3.75 * mm2px
        #charSpacing = iround(6.3 * mm2px) # distance between the left sides of subsequent characters
        charSpacing = iround(6. * mm2px) # distance between the left sides of subsequent characters
        thickness = iround(0.6 * mm2px)
        fontScale = 10
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        height_old = np.inf
        for idx in range(10):
            (width,height), baseLine = cv2.getTextSize('0', fontFace=fontFace, fontScale=fontScale, thickness=thickness)
            if height_old == height:
                break
            fontScale *= targetFontHeight / height
            height_old = height
        (width, height), baseLine = cv2.getTextSize(text, fontFace=fontFace, fontScale=fontScale, thickness=thickness)
        width = width + charSpacing * (len(text)-1)
        synth = np.zeros( (height+thickness*2, width), np.uint8)
        for idx, ch in enumerate(text):
            _ = cv2.putText(synth,
                            text=ch,
                            org=(idx * charSpacing, synth.shape[0]-thickness),
                            fontFace=fontFace,
                            fontScale=fontScale,
                            color=(255, 255, 255),
                            thickness=thickness,
                            lineType=cv2.LINE_AA)
        rows = np.any(synth, axis=1)
        cols = np.any(synth, axis=0)
        return synth[np.argmax(rows): synth.shape[0] - np.argmax(rows[::-1]),
                     np.argmax(cols): synth.shape[1] - np.argmax(cols[::-1])]

    @staticmethod
    def getRotation( img, mm2px, imageSerialArea_mm, debugPlot ):
        # Determine rotation by correlation of the image serial's area with synthesized images of every digit [0-9].

        imageSerialArea_px = [iround(el * mm2px) for el in imageSerialArea_mm]


        synths = []
        for digit in range(10):
            synthR = __class__.getTextTemplate('{}'.format(digit), mm2px)
            if digit==0:
                margin = max(1, iround(synthR.shape[0] * 0.05))
                size = [el+margin*2 for el in synthR.shape]
            halfSizeDiff = [ (el1-el2)//2 for el1,el2 in zip(size,synthR.shape)]
            synth = np.zeros(size, np.uint8)
            synth[halfSizeDiff[0]:halfSizeDiff[0]+synthR.shape[0],
                  halfSizeDiff[1]:halfSizeDiff[1]+synthR.shape[1]] = synthR
            synth = np.rot90(synth, 1)
            synths.append(synth)

        rotations = np.zeros(4,[('number', int), ('cc', float)])
        for iRotation in range(4):
            imgR = np.rot90(img, k=-iRotation)
            imgSerial = imgR[-imageSerialArea_px[0]:, : imageSerialArea_px[1]]
            positions = np.zeros(3,[('digit', int), ('cc', float)])
            for iPosition in range(3):
                firstLast = [ iround(imgSerial.shape[0] / 3 * fac ) for fac in (iPosition,iPosition+1) ]
                sel = imgSerial[firstLast[0] : firstLast[1] + 1,:]
                digits = np.zeros(10)
                for digit, synth in enumerate(synths):
                    ccorr = cv2.matchTemplate(image=sel, templ=synth, method=cv2.TM_CCOEFF_NORMED)
                    cc = ccorr.max()
                    digits[digit] = cc
                    if debugPlot > 1:
                        plt.figure(7, clear=True)
                        plt.subplot(1, 3, 1)
                        plt.imshow(sel, cmap='gray', interpolation='nearest')
                        plt.subplot(1, 3, 2)
                        plt.imshow(synth, cmap='gray', interpolation='nearest')
                        row, col = np.unravel_index(ccorr.argmax(), ccorr.shape)
                        overlay = np.zeros((sel.shape[0], sel.shape[1], 3), dtype=np.uint8)
                        overlay[:, :, 0] = sel
                        overlay[row:row + synth.shape[0], col:col + synth.shape[1], 2] = synth
                        plt.subplot(1, 3, 3)
                        plt.imshow(overlay, interpolation='nearest')
                iBest = np.argmax(digits)
                positions[iPosition] = iBest, digits[iBest]
            rotations[iRotation]['number'] = sum( el * 10**idx for idx,el in enumerate( positions['digit'] ) )
            iMedian = np.argsort(positions['cc'])[1]
            rotations[iRotation]['cc'] = positions[iMedian]['cc']

        return np.argmax( rotations['cc'] )

    @staticmethod
    def getRotationWithSerial( img, mm2px, imageSerialArea_mm, debugPlot, text ):
        # Determine rotation by correlation of the image serial's area with a synthesized image.
        # For that purpose, the image searial number is extracted from the file name.

        imageSerialArea_px = [iround(el * mm2px) for el in imageSerialArea_mm]

        synthR = __class__.getTextTemplate(text, mm2px)
        margin = max(1, iround(synthR.shape[0] * 0.05))
        synth = np.zeros([el+margin*2 for el in synthR.shape], np.uint8)
        synth[margin:-margin, margin:-margin] = synthR
        synth = np.rot90(synth, 1)
        maxCCoeffs = np.zeros(4)
        for iRotation in range(4):
            imgR = np.rot90(img, k=-iRotation)
            sel = imgR[-imageSerialArea_px[0]:, : imageSerialArea_px[1]]
            ccorr = cv2.matchTemplate(image=sel, templ=synth, method=cv2.TM_CCOEFF_NORMED)
            maxCCoeffs[iRotation] = ccorr.max()
            if debugPlot:
                plt.figure(7, clear=True)
                plt.subplot(1, 3, 1)
                plt.imshow(sel, cmap='gray', interpolation='nearest')
                plt.subplot(1, 3, 2)
                plt.imshow(synth, cmap='gray', interpolation='nearest')
                row, col = np.unravel_index(ccorr.argmax(), ccorr.shape)
                overlay = np.zeros((sel.shape[0], sel.shape[1], 3), dtype=np.uint8)
                overlay[:, :, 0] = sel
                overlay[row:row + synth.shape[0], col:col + synth.shape[1], 2] = synth
                plt.subplot(1, 3, 3)
                plt.imshow(overlay, interpolation='nearest')

        return np.argmax(maxCCoeffs)

    def __call__( self,
                  imgFn : str,
                  camSerial = None, 
                  filmFormatFocal = None,
                  plotDir = '',
                  debugPlot = False ):
        # Fiducial marks of RMK A are not flashed. Thus, there may be low contrast.
        # Fiducial marks are simple discs. Thus, we cannot reliably recognize them.
        # Scans may not contain the full image content, or have hardly any dark border around it.
        # While e.g. photos from camera #20204 have the camera serial number (top left), the photo serial number (bottom left), and the focal length (bottom right) exposed onto the film,
        # photos from other (older?) RMK A cameras (e.g. #21163) only have the photo serial number exposed.
        # RMK A photos have been scanned in different orientations: the serial number can be in any of the image corners.
        # these seem to be the only features that can be used to determine the image orientation, as there are no asymmetric fiducial marks.
        # -> first, localize the fiducial marks. Subsequently, determine where the serial number has been imaged.

        # The image serial number of D:\\arap\\data\\laserv\\Projekte\\ARAP\\Luftbilder CaseStudy\\0219650801_013.sid
        # is practically unreadable. Hence, we cannot use simple means to detect the location of the image serial number
        # (e.g. histogram analysis: black/white are dominant -> is an image serial number).

        # For camera #20204 (calibrated focal length: 20.8cm, hence: RMK A 21/23; has camera serial and focal length exposed as auxiliary data), a radius of 7px matches well.
        # However, the image 0220020702_044.sid (nominal focal length according to LBA: 30.5cm; RMK A of unknown serial number), shows much smaller fiducial marks with a radius of approx. 2.5px!
        # The deviation of radii is too large to ensure high correlation coefficients. Hence, we need to adapt it.
        # Let's assume for now that the fiducial mark/disc radius depends on the nominal focal length. Nominal focal lengths of RMK A may be, according to 2010 Hobbie - Development of photogramm instr & methods at Carl Zeiss Oberkochen
        # RMK A 8,5/23
        # RMK A 15/23
        # RMK A 21/23
        # RMK A 30/23
        # RMK A 60/23

        # D:\AdV_Benchmark_Historische_Orthophotos\Wuppertal_1969\Tif_16bit:
        # All images in this folder:
        # - thermometer, etc., imaged and scanned at the left side of the images
        # - also, the film manufacturer is scanned at the top,
        # - even the edges of the film itself have been scanned at the top and bottom,
        # - but still, there seems to be dust above and below the film borders?
        # - along the left and right edges, there are bright areas, probably external light during scanning?
        # - background varies in horizontal waves between black and gray (50)
        # - a horizontal band is just below the fiducials, still within the dark areas around them. It is black in the background, and also visible in the image.
        # - all scans have different resolutions.
        # Fiducial mark on the right is very blurry: 370_2_62.tif

        # D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy
        # All images in this folder:
        # - really only the image content has been scanned, so the only background is around the fiducials.
        # Image counter area is detected as foreground: 0219860702_031.jpg

        # Formerly, we assumed here that when the image serial number is exposed in the upper/left image corner (such that the digits are upright), then the x-axis of the cam-sys is towards the right, and its y-axis is up.
        # This would mean that images 01960606_008.sid, etc., have been scanned with the image rotated by 100 gon, and in the raster files, the cam-CS x-axis points upwards, and the y-axis leftwards.
        # However, trying all 4 possible image rotations for all photos 01960606_003.sid - 01960606_008.sid, where the image serial is in the lower/left corner, it turns out that the RMSE is always smallest, if an image rotation of 0 gon is used (i.e. in the raster files, the cam-CS x-axis points right-wards, the y-axis upwards).
        # Looking at 2010 Hobbie - Development of photogramm instr & methods at Carl Zeiss Oberkochen, fig. 7.7 it turns out that for the RMK A, the image serial number is parallel to the optional auxiliary data area (showing the flying height, barometer, etc.).
        # Looking at the Calibration protocol of the RMK TOP 15 #151967, and at e.g. D:\arap\data\Carnuntum_schraeg_u_vertikal\02110507_009.jpg,
        # it turns out that for the RMK TOP, the image serial number is parallel to the auxiliary data panel.
        # Probably, Zeiss did not change that!

        # Let's assume that the camera coordinate system is defined such that the image serial number is upright, in the top left corner. In that case, we use a rotation angle of 0Â°.

        # TODO
        # Searching for the bright disc in the middle of a dark rectangle seems to work for the mdoneus images,
        # which have been scanned with very narrow or even absent image borders.
        # However, it doesn't work for images with large borders e.g. AdV/Wuppertal: the fiducial position is detected outside of the rectangle of the image content!
        # Hence, use grabCut to distinguish fore- and background.

        if camSerial is None:
            calibration = None
        else:
            calibration = self.calibrations_mm.get(camSerial)
    
        isCalibrated = calibration is not None
        if calibration is None:
            if filmFormatFocal is None:
                raise Exception('If camera calibration is unavailable, then the film format and the focal length must be passed')
            logger.warning('{}. Using approximate values for interior orientation.', 'Unknown camera serial number' if camSerial is None else 'Camera calibration unavailable for RMK A with serial number #{}'.format(camSerial) )
            filmWidth_mm,filmHeight_mm,focalLength_mm = filmFormatFocal
            assert (filmWidth_mm is None or filmWidth_mm==230) and (filmHeight_mm is None or filmHeight_mm==230), 'You have found an RMK A whose film format is not 23x23cm²!?'
            calibration = ( np.array([ 0., 0., focalLength_mm ], dtype=float ),
                            np.array([
                                #  X,  Y [mm]
                                [  0, -1 ],
                                [ -1,  0 ],
                                [  0,  1 ],
                                [  1,  0 ]
                            ]) * 113.,
                            None # we don't know, if the image has auxiliary data fields exposed on the image for the camera serial number and the calibrated focal length.
                            ) 

        ior, fiducials_mm, hasCamSerialFocal = calibration

        # note: some RMK A photos have been scanned as RGB-images, even though black-and-white film had been used!
        img = utils.gdal.imread( imgFn, bands = utils.gdal.Bands.grey )

        if plotDir or debugPlot:
            plt.figure(1, figsize=(8.0, 8.0), clear=True)
            plt.axes([0, 0, 1, 1])
            plt.imshow( img, cmap='gray' )
            plt.xticks([]); plt.yticks([])

        window = self.getImageWindow( img, plot= 2 if debugPlot else bool(plotDir) )
        if utils.stats.relativeDifference( window.width(), window.height() ) > 0.1:
            logger.warning('Detected image rectangle deviates considerably from a square')
        fiducialArea_mm = 8.4, 3.7 # width/height of dark are around fiducials. Note that this includes the 2 peaks that reach into it from the image.
        fiducialOffsetFromBorder_mm = (230-113*2)/2
        imageSerialArea_mm = 19, 7
        fiducialRadiusRange_mm = [0.035, 0.17]

        mm2px = ( window.width()/230 + window.height()/230) / 2
        # 0220020702_044.sid: computed radius of left fiducial is only  0.042mm / 2.81px
        fiducialRadiusRange_px = [el * mm2px for el in fiducialRadiusRange_mm]
        fiducials_px = self.getFiducialsGrabCut(img[round(window.ymin): round(window.ymax)+1,
                                                    round(window.xmin): round(window.xmax)+1],
                                                fiducialArea_px = [ el*mm2px for el in fiducialArea_mm ],
                                                fiducialRadiusRange_px=fiducialRadiusRange_px,
                                                fiducialOffsetFromBorder_px=fiducialOffsetFromBorder_mm * mm2px,
                                                imgName=path.basename(imgFn),
                                                debugPlot=debugPlot)
        fiducials_px[:,0] += round(window.xmin)
        fiducials_px[:,1] -= round(window.ymin)
        #for fiducialRadius_mm in (0.15, 0.1, 0.05):
        #    try:
        #            fiducials_px = self.getFiducialsMatch( img, window, fiducialArea_mm, fiducialRadius_mm, fiducialOffsetFromBorder_mm, debugPlot )
        #    except Exception as ex:
        #        detectExc = ex
        #    else:
        #        break
        #else:
        #    raise Exception('Detection of fiducial marks has failed for all known radii for {}:\n{}'.format(imgFn,detectExc) )


        meanSideLenPx  = np.mean( ( ( fiducials_px[:2]-fiducials_px[2:] ) **2 ).sum(axis=1)**.5 )
        meanSideLenCam = np.mean( ( ( fiducials_mm[:2]-fiducials_mm[2:] ) **2 ).sum(axis=1)**.5 )
        # enhanced scale
        mm2px = meanSideLenPx / meanSideLenCam
        #imageSerialArea_px = [ iround(el*mm2px) for el in imageSerialArea_mm ]

        # Determine image rotation, permute fiducials_px accordingly

        # Check all 4 potential image rotations, and select the one with the smallest maximum median in the selection areas.
        # Do it by checking the median gray value in the rectangle of the exposed image serial number, the camera serial number, and the focal length.
        # This means that we rely on 50% or more being plain black background in those areas.
        # Of course, we may run into problems if the actual image content in the 4th corner is also very dark.
        # This might sound promising, but isn't sufficient.

        # getRotationWithSerial is probably more robust than getRotation.
        # Hence, if we know the image serial number, then use getRotationWithSerial.
        args = [ img[round(window.ymin): round(window.ymax)+1,
                     round(window.xmin): round(window.xmax)+1],
                 mm2px,
                 imageSerialArea_mm,
                 debugPlot ]
        text = path.splitext(path.basename(imgFn))[0]
        numbers = re.findall(r'\d+', text)
        if not numbers or \
           len(numbers[-1]) > 3:
            iRotation = self.getRotation( *args )
        else:
            text = '{:03d}'.format(int(numbers[-1])) # make sure it's not less than 3 digits.
            args.append(text)
            iRotation = self.getRotationWithSerial( *args )

        fiducials_px = np.roll( fiducials_px, shift=iRotation, axis=0 )

        pix2cam = AffineTransform2D.computeSimilarity( fiducials_px, fiducials_mm )

        residuals_microns = (pix2cam.forward( fiducials_px ) - fiducials_mm) * 1000.
        rmse_microns = ( (residuals_microns**2).sum() / 4 )**.5 # Since we give the RMSE of the residual norms, we divide by the number of fiducials (4) - and not by the number of coordinates (8)
        resNorms_microns = ( residuals_microns**2 ).sum(axis=1)**.5

        fiducialPos = np.array( [
            [ -1,  2, -1 ],
            [  1, -1,  3 ],
            [ -1,  0, -1 ] ])

        logger.verbose(
            'Transformation (pix->cam) residuals [µm] for RMK A photo {}:\v'.format( path.splitext( path.basename(imgFn) )[0] ) +
            '\t'.join( ('x','y')*3 ) + '\n' +
            '\n'.join( (
                '\t'.join( (
                    '{:+5.1f}\t{:+5.1f}'.format( *residuals_microns[el] ) if el >= 0 else ' '*5 + '\t' + ' '*5
                for el in row ) ) 
            for row in fiducialPos ) ) + '\v' +
            'Residual norms [µm]:\v' +
            '\n'.join( (
                '\t'.join( (
                    '{:5.1f}'.format( resNorms_microns[el] ) if el >= 0 else ' '*5
                for el in row ) ) 
            for row in fiducialPos ) ) + '\v'
            'RMSE [µm]: {:.1f}'.format( rmse_microns ) + '\v'
        )

        # TODO: store the adp according to the calibration protocol. For RMK A, distortion is not negligible!
        adp = ADP( normalizationRadius = 100. )

        # We do not determine if data fields of camera serial and calibrated focal length are present,
        # but simply stay on the safe side: if we don't know it from the calibration, then assume that they are present.
        if hasCamSerialFocal is None:
            hasCamSerialFocal = True

        mask_px = pix2cam.inverse( self.mask_withCamSerialFocal_mm if hasCamSerialFocal else self.mask_noCamSerialFocal_mm )
        if plotDir or debugPlot:
            plt.figure(1)
            mask_px_closed = np.vstack(( mask_px, mask_px[:1,:] ))
            plt.plot( mask_px_closed[:,0], -mask_px_closed[:,1], '-r' )
            fiducialsProj_px = pix2cam.inverse(fiducials_mm)
            #plt.scatter( x=fiducialsProj_px[:,0], y=-fiducialsProj_px[:,1], marker="x", color='g' )
            #plt.scatter( x=fiducials_px[:,0], y=-fiducials_px[:,1], marker="+", color='r' )
            for iFid, fid in enumerate(fiducials_px):
                plt.gca().add_artist(mpl.patches.Circle((fid[0], -fid[1]), radius=fiducialRadiusRange_px[1]*4, color='r', fill=False))
                plt.text(fid[0], -fid[1], str(iFid), color='y')

            smallFidMargin = fiducialRadiusRange_px[1]*4
            for iFid, (fid, reproj) in enumerate(zip(fiducials_px,fiducialsProj_px)):
                sel = np.s_[ iround(-fid[1] - smallFidMargin) : iround(-fid[1] + smallFidMargin)+1,
                             iround( fid[0] - smallFidMargin) : iround( fid[0] + smallFidMargin)+1]
                if not iFid % 2:
                    left = .5
                elif iFid ==1:
                    left = 1/4
                else:
                    left = 3/4
                if iFid % 2:
                    bottom = .5
                elif iFid == 0:
                    bottom = 1/4
                else:
                    bottom = 3/4
                width = .2
                left   -= width/2
                bottom -= width/2
                fidAx = plt.axes([left,bottom,width,width]) # left, bottom, width, height in normalized units
                plt.imshow( img[sel], cmap='gray' )
                lims = plt.axis()
                detect = (  fid[0] - sel[1].start,
                           -fid[1] - sel[0].start )
                reproj = (  reproj[0] - sel[1].start,
                           -reproj[1] - sel[0].start )
                plt.plot(detect[0], detect[1], '.r')
                fidAx.add_artist( mpl.patches.Circle(( detect[0], detect[1] ), radius=fiducialRadiusRange_px[1], color='r', fill=False))
                plt.plot( [detect[0],reproj[0]], [detect[1],reproj[1]], '-c' )
                plt.axis(lims)
                # additional axes in µm, since RMSE are also given in µm.
                fidAx.tick_params(colors='c')
                fidAx.set_xlabel('px', {'color':'c'})
                fidAx.set_ylabel('px', {'color':'c'})
                scale = linalg.det(pix2cam.A)**.5
                axX = fidAx.twinx() # invisible x-axis and an independent y-axis positioned opposite to the original one (i.e. at right)
                axX.set_ylim([el * scale for el in fidAx.get_ylim()])
                axY = fidAx.twiny()
                axY.set_xlim([el * scale for el in fidAx.get_xlim()])
                fidAx.set_frame_on(False)
                for ax in (axX,axY):
                    ax.set_frame_on(False)
                    ax.set_xlabel('mm', {'color':'r'})
                    ax.set_ylabel('mm', {'color':'r'})
                    ax.tick_params(colors='r')

            plt.figtext( .5, .5, 'RMSE: {:.1f}µm'.format(rmse_microns),
                         horizontalalignment='center', verticalalignment='center',
                         size='large',
                         backgroundcolor='w' )
        if plotDir:
            plt.savefig( path.join( plotDir, path.splitext( path.basename(imgFn) )[0] + '.jpg' ),
                         bbox_inches='tight', pad_inches=0, dpi=150 )
        if plotDir or debugPlot:
            plt.close('all')

        # Also return the information if ior,adp are calibrated, or just rough estimates.
        return ior, adp, pix2cam, mask_px, isCalibrated, rmse_microns

zeissRmkA = ZeissRmkA()

if __name__ == '__main__':
    from glob import iglob
    import os


    calls = []
    # fiducialRadius ca. 1mm
    calls.append( dict( imgFn=r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\0219960606_008.sid', camSerial='20204' ) )

    # fiducialRadius ca. 0.6mm
    # LBA says: 'Zeiss RMK' with unknown camera serial. film format: 230x230, focal length=305.0
    calls.append(dict(imgFn=r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\0220020702_044.sid', filmFormatFocal=(230,230,305.) ) )

    # fiducialRadius right very small: 0.035mm
    calls.append(dict(imgFn=r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\0219860404_005.sid'))

    # Hardly visible image serial number.
    calls.append(dict(imgFn=r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\0219650801_013.sid', filmFormatFocal=(230.,230.,152.50) ) )

    # Formerly problems with detection of right image border. Solved with threshold in getImageWindow
    calls.append(dict(imgFn=r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\0219870411_007.sid'))

    imgPatterns = [ path.join( r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy', el ) for el in [
        '0219650801_*.sid',
        '0219750901_*.sid',
        '0219820303_*.sid',
        '0219860404_*.sid',
        '0219860702_*.sid',
        '0219870411_*.sid',
        '0219880503_*.sid',
        '0219900901_*.sid',
        '0219940301_*.sid'
        ]]
    calls += [dict(imgFn=el) for pattern in imgPatterns for el in sorted(iglob(pattern))]

    imgPatterns = [r'P:\Projects\17_AdV_HistOrthoBenchmark\07_Work_Data\Wuppertal_1969\Tif_16bit\*.tif']
    calls += [dict(imgFn=el) for pattern in imgPatterns for el in sorted(iglob(pattern))]

    # instead of the left fiducial, some small dust ist detected -> large RMSE
    calls.append(dict(imgFn=r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\0219900901_044.sid'))

    # instead of the top fiducial, some small dust ist detected -> large RMSE
    calls.append(dict(imgFn=r'P:\Projects\17_AdV_HistOrthoBenchmark\07_Work_Data\Wuppertal_1969\Tif_16bit\370_1_128.tif'))
    # instead of the left fiducial, some small dust ist detected -> large RMSE
    calls.append(dict(imgFn=r'P:\Projects\17_AdV_HistOrthoBenchmark\07_Work_Data\Wuppertal_1969\Tif_16bit\370_2_66.tif'))
    calls.append(dict(imgFn=r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\0219820303_011.sid'))

    #calls=[dict(imgFn=r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\0219860404_008.sid')]
    plotDir = r'D:\swdvlp64_2015\oriental\tests\fiducials\zeissRmkA'
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)

    for call in calls:
        outFn = os.path.join( plotDir,os.path.splitext(os.path.basename(call['imgFn']))[0] + '.jpg')
        if os.path.exists( outFn ):
            continue
        call['plotDir'] = plotDir
        call['debugPlot'] = False
        if not any( el in call for el in ('camSerial', 'filmFormatFocal') ):
            call['filmFormatFocal'] = (230.,230.,152.50)
        try:
            zeissRmkA( **call )
        except Exception as ex:
            print('{} failed: {}'.format(call['imgFn'],ex))