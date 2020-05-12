# -*- coding: cp1252 -*-
"""Unknown Soviet camera model used in AdV Potsdam 1953 data set.
At the AdV workshop, some said it's either a Zeiss RMK 20/30, or a replica of that camera produced in the USSR.
In fact, Zeiss RMK 20 had a film format of 30x30cm, with film perforated at the sides - just like the Potsdam 1953 data set.
Also, the dark area along the center of each image border is very similar the the much younger Zeiss RMK A.
f = 200.52mm
film format 30 x 30cm
Scanned at 1200dpi.

This camera may have no real fiducial marks at all. At least, none are visible.
However, a dark area in the center of each image border extends into the image area.
Its shape is similar to the one known from Zeiss RMK A.
As with the RMK A, the shapes of these 4 areas are not completely identical.
"""
from .interface import IFiducialDetector

from oriental import log, utils
from oriental.adjust.parameters import ADP
from oriental.ori.transform import AffineTransform2D
import oriental.utils.gdal
import oriental.utils.lsm
import oriental.utils.stats
from oriental.utils import iround

import os
from os import path

import numpy as np
from scipy import linalg
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches
import matplotlib.patheffects as path_effects


logger = log.Logger(__name__)


class ZeissRmk20(IFiducialDetector):
    mask_mm = np.array([
        [-120, +149],
        [-120, +123],
        [-149, +123],
        [-149, + 14],
        [-144, + 14],
        [-144, - 14],
        [-149, - 14],
        [-149, -120],
        [-120, -120],
        [-120, -149],
        [- 14, -149],
        [- 14, -144],
        [+ 14, -144],
        [+ 14, -149],
        [+149, -149],
        [+149, - 14],
        [+144, - 14],
        [+144, + 14],
        [+149, + 14],
        [+149, +149],
        [+ 14, +149],
        [+ 14, +144],
        [- 14, +144],
        [- 14, +149] ], float)


    calibrations_mm = {}

    fidNames = {0: 'left', 1: 'bottom', 2: 'right', 3: 'top'}

    def __call__(self,
                 imgFn: str,
                 camSerial='314005',
                 filmFormatFocal=(300,300,200.52),
                 plotDir='',
                 debugPlot=False):
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
                raise Exception(
                    'If camera calibration is unavailable, then the film format and the focal length must be passed')
            #logger.warning('{}. Using approximate values for interior orientation.',
            #               'Unknown camera serial number' if camSerial is None else 'Camera calibration unavailable for Wild with serial number #{}'.format(
            #                   camSerial))
            filmWidth_mm, filmHeight_mm, focalLength_mm = filmFormatFocal
            assert filmWidth_mm in (None, 300) and filmHeight_mm in (None, 300), 'You have found a Wild camera whose film format is not 30x30cm²!?'
            calibration = (np.array([0., 0., focalLength_mm], dtype=float),
                           np.array([
                               # X,  Y[mm]
                               [-1,  0],
                               [ 0, -1],
                               [ 1,  0],
                               [ 0,  1]
                           ]) * 146.84 +
                           np.array([[-195.387,  247.057], # mean residuals over all images
                                     [ 137.774,   65.002],
                                     [ 173.813,   -6.916],
                                     [-116.199, -305.142]]) / 1000.
                           )

        ior, fiducials_mm = calibration

        img = utils.gdal.imread(imgFn, bands=utils.gdal.Bands.grey, depth=utils.gdal.Depth.u8)

        if plotDir or debugPlot:
            plt.figure(1, figsize=(8.0, 8.0), clear=True)
            plt.axes([0, 0, 1, 1])
            #plt.imshow(img, cmap='gray')
            plt.imshow(((img / 255.) ** .5 * 255).astype(np.uint8), cmap='gray')
            plt.xticks([])
            plt.yticks([])

        marginScale = max(img.shape) / 14817  # based on D:\AdV_Benchmark_Historische_Orthophotos\BB_Testdaten-1953\Luftbilder\3751.tif

        prFgdMargin = iround( 680 * marginScale)  # probably foreground margin: margin along the image borders that is not surely foreground
        prBgdMargin = iround(prFgdMargin * .75)  # probably background margin
        hext = iround(930 * marginScale)  # extents of background along the image border at the fiducials
        bgdMarginTopBottom = iround(80*marginScale) # surely background
        bgdMarginLeftRight = iround(300 * marginScale)  # surely background

        mask = np.full(img.shape, cv2.GC_FGD, np.uint8)  # surely foreground

        mask[:prFgdMargin] = cv2.GC_PR_FGD  # probably foreground
        mask[-prFgdMargin:] = cv2.GC_PR_FGD
        mask[:, :prFgdMargin] = cv2.GC_PR_FGD
        mask[:, -prFgdMargin:] = cv2.GC_PR_FGD

        hh, hw = [round(el / 2) for el in mask.shape]

        slices = ( np.s_[hh - hext:hh + hext, :prBgdMargin],
                   np.s_[hh - hext:hh + hext, -prBgdMargin:],
                   np.s_[-prBgdMargin:, hw - hext:hw + hext],
                   np.s_[:prBgdMargin, hw - hext:hw + hext] )

        # Even with grabCut, let's use a soft threshold that keeps obvious foreground as cv2.GC_PR_FGD.
        thresh = 60
        for slic in slices:
            subMask = mask[slic]
            mask2 = img[slic] < thresh
            subMask[mask2] = cv2.GC_PR_BGD

        mask[:bgdMarginTopBottom, :] = cv2.GC_BGD
        mask[-bgdMarginTopBottom:, :] = cv2.GC_BGD
        mask[:, :bgdMarginLeftRight] = cv2.GC_BGD
        mask[:, -bgdMarginLeftRight:] = cv2.GC_BGD

        if debugPlot:
            plt.figure(2, clear=True)
            plt.imshow(mask, interpolation='nearest')

        def getTemplate():
            nShiftBits = 8
            shiftFac = 2**nShiftBits
            # Let's model the wavy image content border as a cosine.
            period = 130*marginScale
            amplitude = 20*marginScale
            # Create a template with odd columns and rows, and center the cosine on its central pixel.
            # Hence, the template is symmetric about its vertical and horizontal center lines.
            templRows, templCols = ( iround(np.ceil(el/2))*2+1 for el in [ period*1.5, amplitude*2.5 ] )
            templRowCtr, templColCtr = ( el//2 for el in [templRows, templCols] )
            template = np.zeros( (templRows, templCols), np.uint8 )
            y = np.arange( -1, templRows+1 )
            x = amplitude * np.cos( ( y - templRowCtr ) / period * 2*np.pi ) + templColCtr
            pts = np.vstack(( [templCols,-1],
                              np.array([ x, y ]).T,
                              [templCols,templRows] ))
            cv2.fillPoly(template, [np.rint(pts*shiftFac).astype(np.int32)], color=255, lineType=cv2.LINE_AA, shift=nShiftBits )
            return template, np.column_stack((x-templColCtr, y-templRowCtr))
        template, curve_centered = getTemplate()

        bgdModel = np.empty(0)
        fgdModel = np.empty(0)
        wid = iround(700*marginScale)
        lroffset = iround(250*marginScale)
        hext = iround(400 * marginScale)  # extents of background along the image border at the fiducials
        slices = ( np.s_[hh - hext : hh + hext, lroffset : wid],
                   np.s_[mask.shape[0]-wid :  , hw - hext : hw + hext],
                   np.s_[hh - hext : hh + hext, mask.shape[1]-wid : -lroffset],
                   np.s_[0 : wid              , hw - hext : hw + hext] )
        fiducials_px = np.ones((4, 2))
        curves = [None]*4
        for iFiducial, slic in enumerate(slices):
            img_ = img[slic]
            # cv2.grabCut supports 8-bit 3-channel only
            img_ = np.dstack((img_, img_, img_))
            mask_ = mask[slic]
            if debugPlot:
                plt.figure(3,clear=True)
                plt.imshow(mask_, interpolation='nearest')
                plt.figure(4,clear=True)
                plt.imshow( ( ( img_ / 255. )**.5 * 255 ).astype(np.uint8), cmap='gray')
            for it in range(100):
                oldMask = mask_.copy()
                mode = cv2.GC_INIT_WITH_MASK if it == 0 else cv2.GC_EVAL
                mask_, bgdModel, fgdModel = cv2.grabCut(img_, mask_, rect=None,
                                                       bgdModel=bgdModel, fgdModel=fgdModel, iterCount=1, mode=mode)
                if debugPlot:
                    plt.figure(3)
                    plt.imshow(mask_, interpolation='nearest')
                # Don't stop early, even if changes are only small. Updates don't cost much, and small changes sometimes matter.
                if (mask_ == oldMask).all():
                    break

            assert cv2.GC_BGD == 0
            mask_[mask_ == cv2.GC_PR_BGD] = cv2.GC_BGD
            mask_[mask_ != cv2.GC_BGD] = cv2.GC_FGD

            templ = np.rot90(template, k=iFiducial)
            templRowCtr, templColCtr = (el // 2 for el in templ.shape)

            # If image is W x H and template is w x h , then result is (W-w+1) x (H-h+1)
            ccorr =  cv2.matchTemplate( image=mask_, templ=templ, method=cv2.TM_CCOEFF_NORMED )
            iMaximum = ccorr.argmax() # with axis=None, returns the (scalar) flat index of the maximum value
            maxCCoeff = ccorr.flat[iMaximum]

            row, col = np.unravel_index(iMaximum, ccorr.shape)  # transform flat index to row,col

            if debugPlot:
                plt.figure(5,clear=True)
                #plt.contour(ccorr)
                plt.imshow(ccorr, cmap='RdBu', interpolation='nearest', vmin=0.)
                plt.plot( col, row, 'xk' )

            # Kraus Bd.2 Kap. 6.1.2.4
            if maxCCoeff < 0.7:
                logger.warning("{}: maximum cross correlation coefficient is only {:.2}. Detection of {} fiducial mark failed".format(os.path.basename(imgFn), maxCCoeff,__class__.fidNames[iFiducial]))
                fiducials_px[iFiducial, :] = None
                continue

            # coordinates of max ccoeff w.r.t. upper/left px of mask_
            row += templRowCtr
            col += templColCtr
            if debugPlot:
                plt.figure(3)
                plt.plot(col, row, 'or')

            # LSM
            # window: [ minX, maxY, maxX, minY ] inclusive i.e. this is the outline of pixel borders, not pixel centers. OrientAL img-CS.
            inset = 2
            masterWindow = np.array([col - (templColCtr - inset) - .5,
                                     -(row - (templRowCtr - inset) - .5),
                                     col + (templColCtr - inset) + .5,
                                     -(row + (templRowCtr - inset) + .5)])

            # Fix the transformation of the synthetic image, since it is small.
            img0 = utils.lsm.Image()
            img0.id = 0
            img0.dataSetPath = utils.gdal.memDataset(templ)
            img0.master2slave.tPost = np.array([-(masterWindow[0] + .5) + inset,
                                                -(masterWindow[1] - .5) - inset])
            img1 = utils.lsm.Image()
            img1.id = 1
            mask4Lsm = mask_ * 255
            img1.dataSetPath = utils.gdal.memDataset(mask4Lsm)

            solveOpts = utils.lsm.SolveOptions()
            solveOpts.preprocess_shifts_master_window_factor = 1  # no initial brute force shifts
            solveOpts.min_num_pix = -1  # full resolution only
            solveOpts.geometricTrafo = utils.lsm.GeometricTrafo.similarity
            solveOpts.radiometricTrafo = utils.lsm.RadiometricTrafo.identity
            solveOpts.max_num_iterations = 200
            solveOpts.storeFinalCuts = debugPlot
            lsmObj = utils.lsm.Lsm(solveOpts)
            images = [img0, img1]
            summary = utils.lsm.SolveSummary()
            try:
                success = lsmObj(masterWindow, images, summary)
            except Exception as ex:
                raise Exception(
                    "LSM on {} fiducial mark failed:\n{}".format(__class__.fidNames[iFiducial], str(ex)))
            if not success:
                raise Exception("LSM on {} fiducial mark failed:\n{}".format(__class__.fidNames[iFiducial], summary.resolutionLevels[-1].message))

            if debugPlot:
                fullResLvl = summary.resolutionLevels[-1]
                plt.figure(6, clear=True)
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
            center = trafo1.forward(trafo0.inverse(np.array([templColCtr, -templRowCtr],float)))
            R = np.eye(2)
            for idx in range(iFiducial):
                R = R @ np.array([[0,-1],[1,0]])
            curve_ = R @ ( curve_centered * (1,-1) ).T
            curve_ = trafo1.forward(trafo0.inverse(curve_.T + (templColCtr, -templRowCtr) ))
            if debugPlot:
                plt.figure(3)
                plt.plot(center[0], -center[1], '.g')
                plt.plot(curve_[:,0], -curve_[:,1], '-y')

            center[0] += slic[1].start
            center[1] -= slic[0].start
            curve_[:,0] += slic[1].start
            curve_[:,1] -= slic[0].start
            curves[iFiducial] = curve_

            fiducials_px[iFiducial, :] = center

        ok = np.logical_and( np.isfinite(fiducials_px[:,0]), np.isfinite(fiducials_px[:,1]) )
        if ok.sum() < 2:
            raise Exception('Too few fiducials successfully detected: only {}.'.format(ok.sum()))

        if ok.sum() >= 3:
            pix2cam = AffineTransform2D.computeAffinity(fiducials_px[ok], fiducials_mm[ok])
        else:
            pix2cam = AffineTransform2D.computeSimilarity(fiducials_px[ok], fiducials_mm[ok])
        if ok.all():
            if utils.stats.relativeDifference(linalg.norm(fiducials_px[2]-fiducials_px[0]),
                                              linalg.norm(fiducials_px[3]-fiducials_px[1])) > 0.1:
                logger.warning('{}: detected image rectangle deviates considerably from a square', os.path.basename(imgFn))

            # If not all 4 have been successfully detected, then there are no residuals, anyway
            residuals_microns = (pix2cam.forward( fiducials_px[ok] ) - fiducials_mm[ok]) * 1000.
            rmse_microns = ( (residuals_microns**2).sum() / ok.sum() )**.5 # Since we give the RMSE of the residual norms, we divide by the number of fiducials (4) - and not by the number of coordinates (8)
            resNorms_microns = ( residuals_microns**2 ).sum(axis=1)**.5

            if plotDir and False: # Collect residuals, in order to improve fiducials_mm a priori
                import pickle
                from contextlib import suppress
                pickleFn = os.path.join( plotDir, 'residuals_microns.pickle' )
                residuals_microns_restored = []
                with suppress(FileNotFoundError):
                    with open(pickleFn, 'rb') as fin:
                        residuals_microns_restored = pickle.load(fin)
                residuals_microns_restored.append(residuals_microns)
                with open(pickleFn, 'wb') as fout:
                    pickle.dump(residuals_microns_restored, fout, pickle.HIGHEST_PROTOCOL)


            fiducialPos = np.array( [
                [ -1,  3, -1 ],
                [  0, -1,  2 ],
                [ -1,  1, -1 ] ])

            logger.verbose(
                'Transformation (pix->cam) residuals [µm] for Potsdam photo {}:\v'.format( path.splitext( path.basename(imgFn) )[0] ) +
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
        else:
            resNorms_microns = np.zeros(4)
            rmse_microns = 0.


        mask_px = pix2cam.inverse( self.mask_mm )

        if plotDir or debugPlot:
            mm2px = 1. / pix2cam.meanScaleForward()
            plt.figure(1)

            mainAx = plt.gca()
            for endpt, color in [((135, 0), 'r'),
                               ((0, 135), 'g')]:
                arrow = pix2cam.inverse(np.array([[0, 0],
                                                  endpt], float)) * (1, -1)
                diff = arrow[1] - arrow[0]
                mainAx.arrow(arrow[0, 0], arrow[0, 1], diff[0], diff[1], head_width=linalg.norm(diff) / 30,
                             color=color)
            mask_px_closed = np.vstack((mask_px, mask_px[0, :]))
            plt.plot(mask_px_closed[:, 0], -mask_px_closed[:, 1], '-r')
            fiducialsProj_px = pix2cam.inverse(fiducials_mm)

            fiducialPrintRadius_px = 1 * mm2px
            for iFid, fid in enumerate(fiducials_px):
                if np.isfinite(fid).all():
                    mainAx.add_artist(
                        mpl.patches.Circle((fid[0], -fid[1]), radius=fiducialPrintRadius_px, color='r', fill=False))

            smallFidHalfWid_px = 4 * mm2px

            # Constant scale, making different plots comparable.
            # Correctly detected fiducial center residual norms seem to be in range [0;400]
            # With the improved fiducials_mm, residual norms are now in the range [0;100]
            residScale = smallFidHalfWid_px / (150. / 1000. * mm2px)

            mainAx2Data = mainAx.transAxes + mainAx.transData.inverted()
            data2mainAx = mainAx2Data.inverted()
            for iFid, (fid, reproj,curve) in enumerate(zip(fiducials_px, fiducialsProj_px, curves)):
                if not np.isfinite(fid).all():
                    continue
                sel = np.s_[iround(-fid[1] - smallFidHalfWid_px): iround(-fid[1] + smallFidHalfWid_px) + 1,
                            iround(fid[0] - smallFidHalfWid_px): iround(fid[0] + smallFidHalfWid_px) + 1]
                center = pix2cam.inverse(fiducials_mm[iFid] / 2) * (1, -1)
                left, bottom = data2mainAx.transform_point(center)
                width = .2
                left -= width / 2
                bottom -= width / 2
                fidAx = plt.axes([left, bottom, width, width])  # left, bottom, width, height in normalized units
                plt.imshow(img[sel], cmap='gray')
                lims = plt.axis()
                reproj = fid + (reproj - fid) * residScale
                detect = (fid[0] - sel[1].start,
                          -fid[1] - sel[0].start)
                reproj = (reproj[0] - sel[1].start,
                          -reproj[1] - sel[0].start)
                plt.plot(curve[:,0] - sel[1].start, -curve[:,1] - sel[0].start, '-y')
                plt.plot(detect[0], detect[1], '.r')
                fidAx.add_artist(
                    mpl.patches.Circle((detect[0], detect[1]), radius=fiducialPrintRadius_px, color='r',
                                       fill=False))
                plt.plot([detect[0], reproj[0]], [detect[1], reproj[1]], '-c')
                txt = fidAx.text(10, 10, str(iround(resNorms_microns[iFid])) + 'µm', color='c',
                                 horizontalalignment='left', verticalalignment='top')
                txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                                      path_effects.Normal()])
                plt.axis(lims)
                # additional axes in µm, since RMSE are also given in µm.
                fidAx.tick_params(colors='c')
                fidAx.set_xlabel('px', {'color': 'c'})
                fidAx.set_ylabel('px', {'color': 'c'})
                scale = linalg.det(pix2cam.A) ** .5
                axX = fidAx.twinx()  # invisible x-axis and an independent y-axis positioned opposite to the original one (i.e. at right)
                axX.set_ylim([el * scale for el in fidAx.get_ylim()])
                axY = fidAx.twiny()
                axY.set_xlim([el * scale for el in fidAx.get_xlim()])
                fidAx.set_frame_on(False)
                for ax in (axX, axY):
                    ax.set_frame_on(False)
                    ax.set_xlabel('mm', {'color': 'r'})
                    ax.set_ylabel('mm', {'color': 'r'})
                    ax.tick_params(colors='r')

            txt = plt.figtext(.5, .5, 'RMSE: {:.1f}µm'.format(rmse_microns),
                              horizontalalignment='center', verticalalignment='center',
                              size='large', color='w')
            txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='k'),
                                  path_effects.Normal()])


        plotFn = path.splitext(path.basename(imgFn))[0] + '.jpg'
        if plotDir:
            plt.figure(1)
            os.makedirs(plotDir, exist_ok=True)
            plt.savefig(path.join(plotDir, plotFn), bbox_inches='tight', pad_inches=0, dpi=150)
        if plotDir or debugPlot:
            plt.close('all')

        # TODO: store the adp according to the calibration protocol. For Wild, distortion may not be negligible!
        adp = ADP(normalizationRadius=100.)

        # Also return the information if ior,adp are calibrated, or just rough estimates.
        return ior, adp, pix2cam, mask_px, isCalibrated, rmse_microns


zeissRmk20 = ZeissRmk20()
