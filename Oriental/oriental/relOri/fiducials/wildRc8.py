# -*- coding: cp1252 -*-
"""Wild RC8
Years of construction: 1956-1972
f = 115/152/210mm
18 x 18cm (Glas- platten) or 23 x 23cm (Film)
http://www.wild-heerbrugg.com/photogrammetry.htm

Concerning fiducials, there seem to have been used at least 2 different types:
Bright disc with dark X on it: D:\Livia\Historical_Images_ValleLunga\alte_IGM_Luftbilder
Bright X (no background), where the junction is left out: D:\AdV_Benchmark_Historische_Orthophotos\Bonn_1962\Tif_8bit
"""
from oriental.relOri.fiducials.interface import IFiducialDetector

from oriental import log, utils
from oriental.adjust.parameters import ADP
from oriental.ori.transform import AffineTransform2D
import oriental.utils.gdal
import oriental.utils.lsm
import oriental.utils.stats
from oriental.utils import iround

import math, os
from os import path

from contracts import contract
import numpy as np
from scipy import linalg
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches
import matplotlib.patheffects as path_effects

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

logger = log.Logger(__name__)

class WildRc8( IFiducialDetector ):
    mask_mm = np.array([
        [-104, +104],
        [-120, +104],
        [-120, -104],
        [-104, -104],
        [-104, -120],
        [+104, -120],
        [+104, -104],
        [+120, -104],
        [+120, +104],
        [+104, +104],
        [+104, +120],
        [-104, +120]], float )

    # TODO we have a calibration for No. 93 on paper: focal length = 210.23, film format = 18 x 18 cm. Used e.g. during Waldstandsflug 1954
    calibrations_mm = {}

    fidNames = {0: 'top/left', 1: 'bottom/left', 2: 'bottom/right', 3: 'top/right'}

    @staticmethod
    @contract
    def getImageWindow( img_ : 'array[RxC](uint8)', plot: int ):
        marginScale = max(img_.shape) / 25312 # based on D:\Livia\Historical_Images_ValleLunga\alte_IGM_Luftbilder\6783.tif
        img = img_.copy()
        pyrScale = 512 / min(img.shape)
        img = cv2.resize( img, dsize=(0,0), fx=pyrScale, fy=pyrScale, interpolation=cv2.INTER_AREA )
        if 0:
            pyrLvl = 0
            while min(img.shape) > 1000:
                # Note: OpenCV docs say:
                # size of the output image is computed as Size((src.cols+1)/2, (src.rows+1)/2)
                # it downsamples the image by rejecting even rows and columns.
                # This may be interpreted as: columns and rows with even indices are discarded.
                # However, it seems that those with odd indices are discarded!
                # Also, the computation of the output image resolution makes sense only if the latter is true.
                # Hence, we must not apply pyrLvl as shift when transforming image coordinates from a higher pyramid level to the orig image.
                img = cv2.pyrDown(img)
                # same as:
                # img = cv2.GaussianBlur( img, (5,5), 0 )[::2,::2]
                # cv2.getGaussianKernel computes sigma from kernel size n as: ((n-1)*0.5 - 1)*0.3 + 0.8
                # For small kernels, cv2.getGaussianKernel returns pre-computed ones.

                # Gaussian blurring and down-sampling to less than 500px like above results in parts of the data panel (barometer and level)
                # getting smeared so much with the image contents that grabCut does not separate them any more for 'D:\\AdV_Benchmark_Historische_Orthophotos\\Bonn_1962\\Tif_8bit\\168_1_2229.tif'
                # To overcome this, we may use a lower image pyramid level, but that would slow down grab cut considerably.
                # Instead, use a bilateral filter instead of Gaussian blurring, which better preserves edges.
                #sigma = ((5-1)*0.5 - 1)*0.3 + 0.8
                #img = cv2.bilateralFilter( img, d=0, sigmaColor=sigma, sigmaSpace=sigma )[::2,::2]
                pyrLvl += 1
            pyrScale = 1. / 2 ** pyrLvl

        marginScale *= pyrScale
        fgdMargin = round( 4000 * marginScale)  # foreground margin: margin along the image borders that is not surely foreground
        fidCornerWidth = round(2500 * marginScale)  # width of image corners that are probably background (fiducial area)

        mask = np.full(img.shape, cv2.GC_FGD, np.uint8)  # surely foreground

        mask[0:fgdMargin] = cv2.GC_PR_FGD  # probably foreground
        mask[-fgdMargin:] = cv2.GC_PR_FGD
        mask[:, 0:fgdMargin] = cv2.GC_PR_FGD
        mask[:, -fgdMargin:] = cv2.GC_PR_FGD

        slices = ( np.s_[:fidCornerWidth, :fidCornerWidth],
                   np.s_[-fidCornerWidth:, :fidCornerWidth],
                   np.s_[-fidCornerWidth:, -fidCornerWidth:],
                   np.s_[:fidCornerWidth, -fidCornerWidth:] )

        # Even with grabCut, let's use a soft threshold that keeps obvious foreground as cv2.GC_PR_FGD.
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
        mask[:] = labels == ctrLabel

        if plot:
            plt.figure(1)
            limits = plt.axis()

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            assert len(contours) == 1
            contour = contours[0].squeeze() / pyrScale
            contour = np.r_[contour, contour[:1, :]]
            plt.plot(contour[:, 0], contour[:, 1], '-c', clip_on=False)

        x, y, w, h = stats[ctrLabel, [cv2.CC_STAT_LEFT,
                                      cv2.CC_STAT_TOP,
                                      cv2.CC_STAT_WIDTH,
                                      cv2.CC_STAT_HEIGHT]]

        # In the estimation of the minimum bounding rectangle,
        # do not consider the peaks to the outside of the image area at the center points of the image edges.
        midEdgePeakHalfWidth = iround(170 * w/289 * pyrScale * 2)
        midEdgePeakHeight    = iround(250 * h/289 * pyrScale * 2)

        # note: we need to subtract 1 from width and height to get the right/lower coordinates, before applying pyrScale.
        hw, hh = [ x + iround((w-1)/2),
                   y + iround((h-1)/2) ]

        mask[hh - midEdgePeakHalfWidth : hh + midEdgePeakHalfWidth + 1,
             : x + midEdgePeakHeight + 1 ] = cv2.GC_BGD
        mask[y + h - 1 - midEdgePeakHeight : ,
             hw - midEdgePeakHalfWidth : hw + midEdgePeakHalfWidth + 1 ] = cv2.GC_BGD
        mask[hh - midEdgePeakHalfWidth : hh + midEdgePeakHalfWidth + 1,
             x + w - 1 - midEdgePeakHeight : ] = cv2.GC_BGD
        mask[ : y + midEdgePeakHeight + 1,
             hw - midEdgePeakHalfWidth : hw + midEdgePeakHalfWidth + 1 ] = cv2.GC_BGD
        if plot > 1:
            plt.figure(2)
            plt.imshow(mask, interpolation='nearest')

        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        ctrLabel = labels[iround(labels.shape[0] / 2), iround(labels.shape[1] / 2)]
        assert ctrLabel != 0, 'Center point detected as background'

        x, y, w, h = stats[ctrLabel, [cv2.CC_STAT_LEFT,
                                      cv2.CC_STAT_TOP,
                                      cv2.CC_STAT_WIDTH,
                                      cv2.CC_STAT_HEIGHT]]

        window = Window( xmin=x/pyrScale,
                         ymin=y/pyrScale,
                         xmax=(x + w - 1)/pyrScale,
                         ymax=(y + h - 1)/pyrScale)

        if plot:
            (xMin,yMin), (xMax,yMax) = window.min(), window.max()
            plt.figure(1)
            plt.plot( [ xMin, xMin, xMax, xMax, xMin ],
                      [ yMin, yMax, yMax, yMin, yMin ], '-y', clip_on=False )

            # Note: we might extract the rotated minimum rectangle from the contour using minAreaRect
            # To derive a precise, rotated rectangle, we would need to estimate a first one,
            # drop the contour points close to the midpoints of its edges, and then estimate the final, rotated rectangle.
            # rotRect = cv2.boxPoints( cv2.minAreaRect( contour.astype(np.float32)) )
            # rotRect = np.r_[ rotRect, rotRect[0:1,:] ]
            # plt.plot( rotRect[:,0], rotRect[:,1], '-y', clip_on=False )

            plt.axis(limits)

        return window

    def __call__( self,
                  imgFn : str,
                  camSerial = None, 
                  filmFormatFocal = None,
                  plotDir = '',
                  debugPlot = False ):
        # Fiducial marks of Wild seem to be flashed.
        # Fiducial marks are located in the image corners. They are bright discs with a dark X in the center.
        # All attributes (thermometer, clock, image counter) are along the same side.
        # There seem to be no other asymmetric features.
        # -> first, localize the fiducial marks. Subsequently, determine where the image attributes have been imaged.

        if camSerial is None:
            calibration = None
        else:
            calibration = self.calibrations_mm.get(camSerial)
    
        isCalibrated = calibration is not None
        if calibration is None:
            if filmFormatFocal is None:
                raise Exception('If camera calibration is unavailable, then the film format and the focal length must be passed')
            logger.warning('{}. Using approximate values for interior orientation.', 'Unknown camera serial number' if camSerial is None else 'Camera calibration unavailable for Wild with serial number #{}'.format(camSerial) )
            filmWidth_mm, filmHeight_mm, focalLength_mm = filmFormatFocal
            assert filmWidth_mm in ( None, 230 ) and filmHeight_mm in ( None, 230 ), 'You have found a Wild camera whose film format is not 23x23cmÂ²!?'
            calibration = ( np.array([ 0., 0., focalLength_mm ], dtype=float ),
                            np.array([
                                #  X,  Y[mm]
                                [ -1,  1 ], # Note: I don't have any calibration protocol for Wild RC8. These are really gross values. Better adjust focal length.
                                [ -1, -1 ],
                                [  1, -1 ],
                                [  1,  1 ]
                            ]) * 113. )

        ior, fiducials_mm = calibration

        img = utils.gdal.imread( imgFn, bands = utils.gdal.Bands.grey, depth=utils.gdal.Depth.u8 )

        if plotDir or debugPlot:
            plt.figure(1, figsize=(8.0, 8.0), clear=True)
            plt.axes([0, 0, 1, 1])
            plt.imshow( img, cmap='gray' )
            plt.xticks([]); plt.yticks([])

        window = self.getImageWindow( img, plot= 2 if debugPlot else bool(plotDir) )
        if utils.stats.relativeDifference( window.width(), window.height() ) > 0.1:
            logger.warning('{}: detected image rectangle deviates considerably from a square', os.path.basename(imgFn) )

        try:
            fiducialCtrOffset_mm = 8.274 # offset of fiducial centre from image window borders
            xHalfWidth_mm = 1.366 # horizontal/vertical offset from center of X to its end points.
            templateHalfWid_mm = 1.6 # half width/height of the background of X
            xThickness_mm = 0.06

            mm2px = (window.width() / 230 + window.height() / 230) / 2

            fiducialCtrOffset_px = fiducialCtrOffset_mm * mm2px
            xHalfWidth_px        = xHalfWidth_mm * mm2px
            templateHalfWid_px   = templateHalfWid_mm * mm2px
            xThickness_px        = xThickness_mm * mm2px
            searchWinHalfWid_px  = templateHalfWid_px*4

            # cv's drawing functions expect integer coordinates! the shift-parameter allows for bit-shifted integers. Let's use 2**8 == 256, meaning that we get a precision of 1/256 pixels precision
            # Only center and radius are shifted, not thickness!
            nShiftBits = 8
            shiftFac = 2**nShiftBits

            template = np.full( [iround(templateHalfWid_px * 2 + 1)] * 2, fill_value=255, dtype=np.uint8 )
            xCenter = np.array( [templateHalfWid_px] * 2, float )
            xs = [iround((xCenter[0] - xHalfWidth_px * sign) * shiftFac) for sign in (1., -1.)]
            ys = [iround((xCenter[1] - xHalfWidth_px * sign) * shiftFac) for sign in (1., -1.)]
            for idx in range(2):
                cv2.line(template,
                         pt1=(xs[0], ys[0]),
                         pt2=(xs[1], ys[1]),
                         color=(0, 0, 0, 0),  # black
                         thickness=iround(xThickness_px),
                         lineType=cv2.LINE_AA,
                         shift=nShiftBits)
                ys.reverse()

            if debugPlot:
                plt.figure(4, clear=True)
                plt.imshow(template, interpolation='nearest', cmap='gray', vmin=0, vmax=256)
                plt.axhline((template.shape[0] - 1) / 2)
                plt.axvline((template.shape[1] - 1) / 2)

            fiducials_px = np.ones( (4,2) )
            leftCols, topRows = [ np.s_[   iround( max( 0, el + fiducialCtrOffset_px - searchWinHalfWid_px ) )
                                         : iround(         el + fiducialCtrOffset_px + searchWinHalfWid_px ) + 1 ] for el in window.min() ]
            rightCols, botRows = [ np.s_[   iround(            el - fiducialCtrOffset_px - searchWinHalfWid_px )
                                          : iround( min( shap, el - fiducialCtrOffset_px + searchWinHalfWid_px) + 1 )] for el, shap in zip( window.max(), img.shape[::-1] ) ]

            minCCs, maxCCs = [], []
            for iRun in range(2):
                if iRun:
                    useMaxCC = np.argmax( np.abs( np.array([ np.median( els ) for els in [minCCs, maxCCs] ]) ) )
                for iFiducial in range(4):  # Start in upper/left corner, proceed counter-clockwise.
                    if iFiducial in (0, 3):
                        rows = topRows
                    else:
                        rows = botRows
                    if iFiducial < 2:
                        cols = leftCols
                    else:
                        cols = rightCols
                    searchWin = img[rows, cols]

                    if debugPlot:
                        plt.figure(8, clear=True)
                        plt.imshow(searchWin, cmap='gray', interpolation='nearest')

                    # If image is W x H and template is w x h , then result is (W-w+1) x (H-h+1)
                    ccorr = cv2.matchTemplate(image=searchWin, templ=template, method=cv2.TM_CCOEFF_NORMED)
                    iMinimum = ccorr.argmin()
                    iMaximum = ccorr.argmax()  # with axis=None, returns the (scalar) flat index of the maximum value
                    minCCoeff = ccorr.flat[iMinimum]
                    maxCCoeff = ccorr.flat[iMaximum]
                    if iRun==0:
                        minCCs.append( minCCoeff )
                        maxCCs.append( maxCCoeff )
                        continue
                    iCC = iMaximum if useMaxCC else iMinimum
                    row, col = np.unravel_index(iCC, ccorr.shape)  # transform flat index to row,col
                    if row==0 or row==ccorr.shape[0]-1 or \
                       col==0 or col==ccorr.shape[1]-1:
                        logger.warning('{}, {} fiducial: Maximum similarity is at the border of the search window. Actual maximum may be outside of it.', os.path.basename(imgFn), __class__.fidNames[iFiducial] )

                    if debugPlot:
                        plt.figure(3, clear=True)
                        plt.scatter(x=[col], y=[row], marker=".", color='r')
                        plt.contour(ccorr)
                        maxAbs = np.abs(ccorr).max()
                        plt.imshow(ccorr, cmap='gray', vmin=-maxAbs, vmax=maxAbs, interpolation='nearest')
                        plt.colorbar(format='%+.2f')

                    # Kraus Bd.2 Kap. 6.1.2.4
                    #if abs(maxCCoeff) < 0.7:
                    #    raise Exception(
                    #        "Maximum cross correlation coefficient is only {:.2}. Detection of {} fiducial mark failed".format(
                    #            maxCCoeff, __class__.fidNames[iFiducial]))

                    # coordinates of max ccoeff w.r.t. upper/left px of searchWin
                    row += (template.shape[0] - 1) / 2
                    col += (template.shape[1] - 1) / 2

                    if debugPlot:
                        plt.figure(8)
                        plt.scatter(x=[col], y=[row], marker='.', color='r')

                    row += rows.start
                    col += cols.start

                    # LSM
                    # window: [ minX, maxY, maxX, minY ] inclusive i.e. this is the outline of pixel borders, not pixel centers. OrientAL img-CS.
                    inset = 2
                    masterWindow = np.array([col - (templateHalfWid_px - inset) - .5,
                                             -(row - (templateHalfWid_px - inset) - .5),
                                             col + (templateHalfWid_px - inset) + .5,
                                             -(row + (templateHalfWid_px - inset) + .5)])

                    # Fix the transformation of the synthetic image, since it is small.
                    img0 = utils.lsm.Image()
                    img0.id = 0
                    img0.dataSetPath = utils.gdal.memDataset(template)
                    img0.master2slave.tPost = np.array([-(masterWindow[0] + .5) + inset,
                                                        -(masterWindow[1] - .5) - inset])
                    img1 = utils.lsm.Image()
                    img1.id = 1
                    img1.dataSetPath = utils.gdal.memDataset(img)

                    solveOpts = utils.lsm.SolveOptions()
                    solveOpts.preprocess_shifts_master_window_factor = 1 # no initial brute force shifts
                    solveOpts.min_num_pix = -1 # full resolution only
                    solveOpts.geometricTrafo = utils.lsm.GeometricTrafo.rigid
                    solveOpts.max_num_iterations = 200
                    solveOpts.storeFinalCuts = debugPlot
                    lsmObj = utils.lsm.Lsm(solveOpts)
                    images = [img0, img1]
                    summary = utils.lsm.SolveSummary()
                    try:
                        success = lsmObj(masterWindow, images, summary)
                    except Exception as ex:
                        raise Exception("LSM on {} fiducial mark failed:\n{}".format(__class__.fidNames[iFiducial], str(ex)))
                    if not success:
                        raise Exception("LSM on {} fiducial mark failed:\n{}".format(__class__.fidNames[iFiducial], summary.resolutionLevels[-1].message))

                    if debugPlot:
                        fullResLvl = summary.resolutionLevels[-1]
                        plt.figure(5, clear=True)
                        ax1 = plt.subplot(1, 3, 1)
                        ax1.set_title('synthetic')
                        plt.imshow(fullResLvl.cuts[0], cmap='gray')
                        ax2 = plt.subplot(1, 3, 2)
                        ax2.set_title('observed')
                        plt.imshow(fullResLvl.cuts[1], cmap='gray')
                        diff = fullResLvl.cuts[0] - fullResLvl.cuts[1]
                        maxAbs = np.abs(diff).max()
                        ax3 = plt.subplot(1, 3, 3)
                        ax3.set_title('synth - obs')
                        plt.imshow(diff, cmap='RdBu', vmin=-maxAbs, vmax=maxAbs, interpolation='nearest')
                        plt.colorbar(format='%+.2f', orientation='horizontal', ax=[ax1, ax2, ax3])

                    for im in images:
                        assert not np.any(im.master2slave.tPre)
                    trafo0, trafo1 = [AffineTransform2D(img.master2slave.A, img.master2slave.tPost) for img in images]
                    center = trafo1.forward(trafo0.inverse(xCenter * (1., -1.)))

                    if debugPlot:
                        plt.figure(8)
                        plt.scatter(x=center[0] - cols.start, y=-center[1] - rows.start, marker='.', color='g')

                    fiducials_px[iFiducial, :] = center

            meanDiag_px  = np.mean( ( ( fiducials_px[:2]-fiducials_px[2:] ) **2 ).sum(axis=1)**.5 )
            meanDiag_mm = np.mean( ( ( fiducials_mm[:2]-fiducials_mm[2:] ) **2 ).sum(axis=1)**.5 )
            # enhanced scale
            mm2px = meanDiag_px / meanDiag_mm

            # Determine rotation in scanner plane
            indicatorsY = [ iround(-fiducials_px[-1,1]+el*mm2px) for el in [46.67, 91.75, 133.93] ]
            indicatorRadius = iround(6.13*mm2px)
            indicatorImg = np.zeros( (indicatorsY[-1] - indicatorsY[0] + 2 * indicatorRadius, 2 * indicatorRadius), np.uint8 )
            for indY in indicatorsY:
                circleCenter = indicatorRadius, indY+indicatorRadius-indicatorsY[0]
                cv2.circle(indicatorImg,
                           center=tuple(round(el * shiftFac) for el in circleCenter),
                           radius=round(indicatorRadius * shiftFac),
                           color=(255, 255, 255, 0),  # white
                           thickness=-1,
                           # Thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn
                           lineType=cv2.LINE_AA,
                           shift=nShiftBits )
            if debugPlot:
                plt.figure(6, clear=True)
                plt.imshow( indicatorImg, cmap='gray', interpolation='nearest' )

            metaDataHalfOffsetAlong_px = 72.52 * mm2px # (vertical) offset of top template edge from mid point of the 2 neighboring fiducial centers
            metaDataOffsetAcross_px = 11.15 * mm2px # (horizontal) offset of the inner border of the meta data discs from the mid point of the 2 neighboring fiducial centers
            maxCCorr = None
            iBestRot = -1
            # 0 means: indicators along the right image border. 1: indicators along top image border
            for iRot in range(4):
                indicatorImgRot = np.rot90( indicatorImg, k=iRot )
                # xStart and yStart need to be adapted to iRot!
                if iRot==0:
                    midPoint = np.mean( fiducials_px[[2,3]], axis=0) * (1,-1)
                    cStart = iround( midPoint[0] + metaDataOffsetAcross_px )
                    rStart = iround( midPoint[1] - metaDataHalfOffsetAlong_px )
                elif iRot==1:
                    midPoint = np.mean(fiducials_px[[3,0]], axis=0) * (1, -1)
                    cStart = iround( midPoint[0] - metaDataHalfOffsetAlong_px )
                    rStart = iround( midPoint[1] - metaDataOffsetAcross_px - indicatorImgRot.shape[0] + 1 )
                elif iRot==2:
                    midPoint = np.mean(fiducials_px[[0, 1]], axis=0) * (1, -1)
                    cStart = iround(midPoint[0] - metaDataOffsetAcross_px - indicatorImgRot.shape[1] + 1 )
                    rStart = iround(midPoint[1] + metaDataHalfOffsetAlong_px - indicatorImgRot.shape[0] + 1 )
                elif iRot==3:
                    midPoint = np.mean(fiducials_px[[1,2]], axis=0) * (1, -1)
                    cStart = iround( midPoint[0] + metaDataHalfOffsetAlong_px - indicatorImgRot.shape[1] + 1 )
                    rStart = iround( midPoint[1] + metaDataOffsetAcross_px )
                if cStart < 0 or rStart < 0:
                    continue # outside
                imgLoc = img[rStart: rStart + indicatorImgRot.shape[0], cStart: cStart + indicatorImgRot.shape[1]]
                if debugPlot:
                    plt.figure(7, clear=True)
                    plt.imshow( imgLoc, cmap='gray', interpolation='nearest' )
                if imgLoc.shape < indicatorImgRot.shape:
                    continue # outside
                assert imgLoc.shape == indicatorImgRot.shape
                ccorr = cv2.matchTemplate(image=imgLoc, templ=indicatorImgRot, method=cv2.TM_CCOEFF_NORMED).item()
                if maxCCorr is None or ccorr > maxCCorr:
                    maxCCorr = ccorr
                    iBestRot = iRot
            if maxCCorr is None:
                raise Exception('Determination of rotation in the scanner plane failed.')

            fiducials_px = np.roll( fiducials_px, shift=-iBestRot, axis=0 )

            pix2cam = AffineTransform2D.computeSimilarity(fiducials_px, fiducials_mm)

            residuals_microns = (pix2cam.forward( fiducials_px ) - fiducials_mm) * 1000.
            rmse_microns = ((residuals_microns ** 2).sum() / 4) ** .5 # Since we give the RMSE of the residual norms, we divide by the number of fiducials (4) - and not by the number of coordinates (8)
            resNorms_microns = ( residuals_microns**2 ).sum(axis=1)**.5

            msg = [ '\t'.join( ('x','y','norm')*2 ) ]
            for idxs in [(0,3),(1,2)]:
                msg.append( '\t'.join( '{:+5.1f}\t{:+5.1f}\t{:+5.1f}'.format( *residuals_microns[idx,:], resNorms_microns[idx] ) for idx in idxs ) )
            logger.verbose( 'Transformation (pix->cam) residuals [µm] for Wild RC8 photo {}:\v'.format( path.splitext( path.basename(imgFn) )[0] ) +
                            '\n'.join( msg ) + '\v' +
                            'RMSE [µm]: {:.1f}'.format(rmse_microns) )

            mask_px = pix2cam.inverse( self.mask_mm )
            if plotDir or debugPlot:
                plt.figure(1)
                mainAx = plt.gca()
                for endpt,col in [((100,0),'r'),
                                  ((0,100),'g')]:
                    arrow = pix2cam.inverse(np.array([[0,0],
                                                      endpt],float)) * (1,-1)
                    diff = arrow[1] - arrow[0]
                    mainAx.arrow( arrow[0,0], arrow[0,1], diff[0], diff[1], head_width=linalg.norm(diff)/30, color=col )
                mask_px_closed = np.vstack(( mask_px, mask_px[0,:] ))
                plt.plot( mask_px_closed[:,0], -mask_px_closed[:,1], '-r' )
                fiducialsProj_px = pix2cam.inverse(fiducials_mm)

                fiducialPrintRadius_px = xHalfWidth_mm * 2**.5 * mm2px
                for iFid, fid in enumerate(fiducials_px):
                    mainAx.add_artist(mpl.patches.Circle((fid[0], -fid[1]), radius=fiducialPrintRadius_px, color='r', fill=False))
                    #plt.text(fid[0], -fid[1], str(iFid), color='y')

                smallFidHalfWid_px = xHalfWidth_px*2

                if 0: # adapt the residual scale to the norm of the residuals. Drawback: different plots have different residual scales.
                    # round to int, for better readability.
                    medResidLen_microns = iround( np.median( np.sum( residuals_microns ** 2, axis=1 ) ) **.5 )
                    medResidLen_px = medResidLen_microns / 1000 * mm2px
                    residScale = .5 * smallFidHalfWid_px / medResidLen_px
                else: # Constant scale, making different plots comparable. Correctly detected fiducial center residual norms seem to be in range [0;100]
                    residScale = smallFidHalfWid_px / ( 150. / 1000. * mm2px )

                mainAx2Data = mainAx.transAxes + mainAx.transData.inverted()
                data2mainAx = mainAx2Data.inverted()
                for iFid, (fid, reproj) in enumerate(zip(fiducials_px,fiducialsProj_px)):
                    sel = np.s_[ iround(-fid[1] - smallFidHalfWid_px) : iround(-fid[1] + smallFidHalfWid_px)+1,
                                 iround( fid[0] - smallFidHalfWid_px) : iround( fid[0] + smallFidHalfWid_px)+1]
                    center = pix2cam.inverse(fiducials_mm[iFid] / 2) * (1, -1)
                    left, bottom = data2mainAx.transform_point( center )
                    width = .2
                    left   -= width/2
                    bottom -= width/2
                    fidAx = plt.axes([left, bottom, width, width]) # left, bottom, width, height in normalized units
                    plt.imshow( img[sel], cmap='gray' )
                    lims = plt.axis()
                    reproj = fid + ( reproj - fid ) * residScale
                    detect = (  fid[0] - sel[1].start,
                               -fid[1] - sel[0].start )
                    reproj = (  reproj[0] - sel[1].start,
                               -reproj[1] - sel[0].start )
                    plt.plot(detect[0], detect[1], '.r')
                    fidAx.add_artist( mpl.patches.Circle(( detect[0], detect[1] ), radius=fiducialPrintRadius_px, color='r', fill=False))
                    plt.plot( [detect[0],reproj[0]], [detect[1],reproj[1]], '-c' )
                    txt = fidAx.text( 10, 10, str(iround(resNorms_microns[iFid])) + 'µm', color='c', horizontalalignment='left', verticalalignment='top' )
                    txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                                          path_effects.Normal()])
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

                txt = plt.figtext( .5, .5, 'RMSE: {:.1f}µm'.format(rmse_microns),
                                   horizontalalignment='center', verticalalignment='center',
                                   size='large', color='w' )
                txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='k'),
                                      path_effects.Normal()])

        except:
            plotFn = path.splitext( path.basename(imgFn) )[0] + '_failed.jpg'
            raise
        else:
            plotFn = path.splitext( path.basename(imgFn) )[0] + '.jpg'
        finally:
            if plotDir:
                plt.figure(1)
                os.makedirs( plotDir, exist_ok=True )
                plt.savefig( path.join( plotDir, plotFn ), bbox_inches='tight', pad_inches=0, dpi='figure' )
            if plotDir or debugPlot:
                plt.close('all')

        # TODO: store the adp according to the calibration protocol. For Wild, distortion may not be negligible!
        adp = ADP( normalizationRadius = 100. )

        logger.verbose(f'{path.basename(imgFn)} fiducials\n' +
                        'idx\tx\ty\n' +
                        '\n'.join(f'{idx}\t' + '\t'.join(f'{coo:.2f}' for coo in coos) for idx, coos in enumerate(fiducials_px)))
        # Also return the information if ior,adp are calibrated, or just rough estimates.
        return ior, adp, pix2cam, mask_px, isCalibrated, rmse_microns

wildRc8 = WildRc8()

if __name__ == '__main__':
    wildRc8(r'P:\Studies\17_AdV_HistOrthoBenchmark\07_Work_Data\Bonn_1962\Tif_8bit\168_1_2213.TIF', filmFormatFocal=(230., 230., 153.34), plotDir='fiducials', debugPlot=True)
    wildRc8(r'P:\Projects\16_SEHAG\07_Work_Data\AerialImages\Kaunertal\1953C\Flug 1\1953_C_2865.tif', filmFormatFocal=(230., 230., 210.11), plotDir='fiducials', debugPlot=True)