# -*- coding: cp1252 -*-

from oriental.relOri.fiducials.interface import IFiducialDetector
from oriental import log, utils
from oriental.adjust.parameters import ADP
from oriental.ori.transform import AffineTransform2D
import oriental.utils.gdal

import re, math
from os import path
from collections import namedtuple

from contracts import contract
import numpy as np
from scipy import linalg
import cv2

logger = log.Logger(__name__)

class ZeissRmkTop( IFiducialDetector ):
    
    # mask, digitized on 02110507_011.jpg and transformed into the camera CS
    mask_mm = np.array([
        [ -107,  114 ],
        [   -5,  114 ],
        [   -5,  108 ],
        [    5,  108 ],
        [    5,  114 ],
        [  108,  114 ],
        [  108,  108 ],
        [  114,  108 ],
        [  114,    5 ],
        [  108,    5 ],
        [  108,   -5 ],
        [  114,   -5 ],
        [  114, -108 ],
        [  108, -108 ],
        [  108, -114 ],
        [    5, -114 ],
        [    5, -108 ],
        [   -5, -108 ],
        [   -5, -114 ],
        [ -107, -114 ],
        [ -107,  -85 ],
        [ -114,  -85 ],
        [ -114,   -5 ],
        [ -108,   -5 ],
        [ -108,    5 ],
        [ -114,    5 ],
        [ -114,   94 ],
        [ -107,   94 ] ], dtype=np.float )

    # Daten lt. Kalibrierschein:
    # calibrated focal length = 152.642mm
    # Positionen (principal point of symmetry ist im Ursprung, X nach rechts, Y nach oben):
    #
    # 7   3   5
    # 
    # 2  PPA  1  
    #
    # 6   4   8
    #
    calibrations_mm = {
        '151967' : ( np.array([ 0.008, 0.004, 152.642 ]), # PPA (principal point of autocollimation)
                     np.array([
                       #   X[mm]    Y[mm]
                       [ 113.008,    0.011], # fiducial1
                       [-112.987,    0.011], # fiducial2
                       [   0.012,  113.010], # fiducial3
                       [   0.011, -112.994], # fiducial4
                       [ 113.017,  113.011], # fiducial5
                       [-112.987, -112.989], # fiducial6
                       [-112.986,  113.006], # fiducial7
                       [ 113.012, -112.989]  # fiducial8  
                   ]) )
    }
    
    @contract
    def __call__( self,
                  imgFn : str,
                  camSerial = None, 
                  filmFormatFocal = None,
                  plotDir = '',
                  debugPlot = False,
                  subSelect : bool = True ):
        """
        subSelect: re-compute correlation on sub-selection of search window (the inner disc, excluding any surrounding features, like a cross)
            This does not seem to have a noticeable effect on the RMSE.
            However, in case the scanned image and hence the fiducial marks are rotated against the pixel CS,
            then this is crucial, because the inner disc is, of course, rotationally symmetric - unlike the outer features of the fiducial marks.

        """
        # Instead of reading the whole image into memory, memory demands are lower if only the small image areas that are actually needed are read (on demand)
        # However, that seems to increase processing times due to multiple read operations.

        if plotDir or debugPlot:
            import oriental.utils.pyplot as plt

        # detect & identify fiducial marks of scanned image from Zeiss RMK Top 15, with known scanning resolution.
        # RMK Top15 has 8 fiducial marks.
        # For 8 fiducial marks, Kraus recommends a bilinear transformation: Kraus A 3.2.1.2: Tab. 3.2-4
        # Due to considerations above, we still use an affine transformation.
        # For automatic localization of fiducial marks, Kraus recommends to assume the rotation within the image plane to be known as either 0°, 90°, 180°, or 270°,
        # and the scan resolution to be known.
        # Kraus recommends the maximum of the cross correlation coefficient for fiducial mark detection,
        # with a lower threshold of 0.7 for a successful detection,
        # followed by an interpolation with a degree 2 polynomial for sub-pixel precision.
    
        # fiducials of this camera are red!
        imgRGB, imgInfo = utils.gdal.imread( imgFn, bands = utils.gdal.Bands.rgb, info = True )
        if imgInfo.bands != utils.gdal.Bands.rgb: # imread has converted a single channel file to an RGB image in memory.
            raise Exception( "RMK Top photos are expected to be scanned with a channel for red" )
        img = imgRGB[:,:,0]

        # For now, let's assume that meta-data is available that tells us the scan resolution and the orientation of the image in the scanner.
        # The UFG-scanner used for RMK-cameras stores scans in MrSID-format.
        # It stores meta information about the scan parameters in the Exif-tag 'ImageDescription'
        # We use ExifTool to extract image meta data. However, it cannot read MrSID!
        # Thus, forget about the meta data.
        #imageDescription = subprocess.check_output(
        #    [ 'exiftool',
        #      '-EXIF:ImageDescription',
        #      '-json',
        #      '-quiet', # don't clutter the screen of our main application with messages from ExifTool like '276 image files read' 
        #      imgFn
        #    ],
        #    universal_newlines=True )
        #lines = json.loads( imageDescription )[0]
        #imageDescription = lines.get('ImageDescription')

        if not imgInfo.description:
            scanResolution_microns = 230*1.e3 / ( min( img.shape ) * 0.98 )
        else:
            info = { line.split(":",1)[0].strip():line.split(":",1)[1].strip() for line in imgInfo.description.splitlines() }
            #assert info['Rotation'] == '270'
            pattern = re.compile( r".*?\(\s*(.*?)\s*microns\s\)" )
            scanResolutionX_microns,substituted = pattern.subn( r"\1", info['XAxisResolution'] )
            assert substituted
            scanResolutionY_microns,substituted = pattern.subn( r"\1", info['YAxisResolution'] )
            assert substituted
            # microns per pixel
            scanResolution_microns = np.mean(( float(scanResolutionX_microns),
                                               float(scanResolutionY_microns) ))
    
        calibration = self.calibrations_mm.get(camSerial)
    
        isCalibrated = calibration is not None
        if calibration is None:
            if filmFormatFocal is None:
                raise Exception('If camera calibration is unavailable, then the film format and the focal length must be passed')
            filmWidth_mm,filmHeight_mm,focalLength_mm = filmFormatFocal
            assert (filmWidth_mm is None or filmWidth_mm==230) and (filmHeight_mm is None or filmHeight_mm==230), 'You have found an RMK TOP whose film format is not 23x23cm²!?'
            calibration = ( np.array([ 0., 0., focalLength_mm ], dtype=float ),
                            np.array([
                                #   X[mm]    Y[mm]
                                [  113.,    0. ],
                                [ -113.,    0. ],
                                [    0.,  113. ],
                                [    0., -113. ],
                                [  113.,  113. ],
                                [ -113., -113. ],
                                [ -113.,  113. ],
                                [  113., -113. ]
                            ])
                          ) 

        ior, fiducials_mm = calibration

        # linker Bildrand: Spalte 1080 re: 15918 Diff=14838 
        # 14838 * 15.5 mu/px / 1000 = 230mm # stimmt überein mit Bildformat! Die tatsächlich belichtete Fläche ist größer als das nominelle Bildformat (23cmx23cm) wegen Nebenabbildungen (Datumsstempel, etc.)
    
        # selbst gemessener Durchmesser der Mittellinie der Ringe der Rahmenmarken: 100px -> 1.55mm
        # 1.55mm ist unrunder Wert, daher annehmen, dass der Durchmesser von Zeiss mit 1.5mm gewählt wurde?
        # 1.5mm * 1000 / 15.5 mu/px = 96.77px
        # selbst gemessene Breite der Ringe der Rahmenmarken (Differenz innerer, äußerer Ring): 20px -> 0.31mm
        # 0.31 ist unrund, daher Annahme, dass Zeiss 0.3mm gewählt hat
        # selbst gemessener Durchmesser des zentralen Kreises der Rahmenmarken: 10px -> 0.155mm
        # 0.155mm ist unrund, daher Annahme, dass Zeiss 0.15mm gewählt hat.
    
        # -> synthetisches Bild erstellen, mit ungerader Anzahl an Spalten & Zeilen
        def ringRadiusPx():
           return 1.55 / 2. * 1000 / scanResolution_microns
        ringWidthPx    = 0.3 * 1000 / scanResolution_microns
        def centralDiscRadiusPx():
           return 0.15 / 2. * 1000 / scanResolution_microns
    
        templateWidth = int( math.ceil(ringRadiusPx()*2 + ringWidthPx*3) )
        if templateWidth % 2 == 0:
            templateWidth += 1
        templateX = np.zeros( (templateWidth,templateWidth), dtype=np.uint8 )
        ctr = (templateWidth-1)//2
        # cv's drawing functions expect integer coordinates! the shift-parameter allows for bit-shifted integers. Let's use 2**8 == 256, meaning that we get a precision of 1/256 pixels precision
        # only center and radius are shifted, not thickness!
        nShiftBits = 8
        shiftFac = 2**nShiftBits
        white = (255,255,255,0)
        cv2.circle( templateX,
                    center=(int(ctr*shiftFac),)*2,
                    radius=int(centralDiscRadiusPx()*shiftFac),
                    color=white,
                    thickness=-1, #  Thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn
                    lineType=cv2.LINE_AA,
                    shift=nShiftBits )
    
        templateSpotBorder = int(2 * centralDiscRadiusPx())
        templateSpot = templateX[ ctr-templateSpotBorder:ctr+templateSpotBorder+1,
                                  ctr-templateSpotBorder:ctr+templateSpotBorder+1 ].copy()
        if False and debugPlot:
            plt.figure(7); plt.clf()
            plt.imshow( templateSpot, interpolation='nearest', cmap='gray' )
            plt.axhline( (templateSpot.shape[0]-1)/2 )
            plt.axvline( (templateSpot.shape[0]-1)/2 )
    
        cv2.circle( templateX,
                    center=(int(ctr*shiftFac),)*2,
                    radius=int(ringRadiusPx()*shiftFac),
                    color=white,
                    thickness=int(ringWidthPx), #  Thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn
                    lineType=cv2.LINE_AA,
                    shift=nShiftBits )
    
        templateP = templateX.copy()
    
        for idx,template in enumerate((templateP,templateX)):
            for iAngle in range(4):
                angle = iAngle*np.pi/2.
                if idx==1:
                    angle += np.pi/4.
                start = ( ctr + 4*centralDiscRadiusPx()*math.cos( angle ),
                          ctr + 4*centralDiscRadiusPx()*math.sin( angle ) )
                end   = ( ctr + ( ringRadiusPx() + ringWidthPx )*math.cos( angle ),
                          ctr + ( ringRadiusPx() + ringWidthPx )*math.sin( angle ) )
                pt1 = tuple( int(pt*shiftFac) for pt in start )  
                pt2 = tuple( int(pt*shiftFac) for pt in end )
                cv2.line( template,
                          pt1=pt1,
                          pt2=pt2,
                          color=white,
                          thickness=int(centralDiscRadiusPx()*2),
                          lineType=cv2.LINE_AA,
                          shift=nShiftBits )
        if False and debugPlot:
            plt.figure(1)                  
            plt.imshow( templateX, interpolation='nearest', cmap='gray' )
            plt.axhline( (templateX.shape[0]-1)/2 )
            plt.axvline( (templateX.shape[0]-1)/2 )
            plt.figure(4)
            plt.imshow( templateP, interpolation='nearest', cmap='gray' )
            plt.axhline( (templateP.shape[0]-1)/2 )
            plt.axvline( (templateP.shape[0]-1)/2 )
    
        # fiducial #7 is approx. at row/col:
        fiducials_rc = np.empty_like( fiducials_mm ) 
        fiducials_rc[7-1,:] = 273, 1233
        fiducials_rc[7-1,:] *= scanResolution_microns / 15.5
        # first fiducial: big search window;
        #searchWinHalfWidth = int(ringRadiusPx()*10)
        searchWinHalfWidth = np.mean(img.shape)//16
        Window = namedtuple( 'Window', ['startRow', 'endRow', 'startCol', 'endCol'] ) # start=inclusive, end=exclusive
    
        detectionOrder = ( 7, 8, 5, 6, 1, 2, 3, 4 )
        for idx,iFiducial in enumerate(detectionOrder):
            if iFiducial <= 4:
                template = templateP
            else:
                template = templateX
            if idx==0:
                assert iFiducial==7 # the only for which we have set fiducials_rc
            elif idx==1:
                fiducials_rc[iFiducial-1,:] = fiducials_rc[detectionOrder[0]-1,:]
                fiducials_rc[iFiducial-1,0] -= ( fiducials_mm[iFiducial-1,1]-fiducials_mm[detectionOrder[0]-1,1] ) * 1000. / scanResolution_microns
                fiducials_rc[iFiducial-1,1] += ( fiducials_mm[iFiducial-1,0]-fiducials_mm[detectionOrder[0]-1,0] ) * 1000. / scanResolution_microns
            elif idx==2:
                # estimate 2d similarity trafo mm->pix based on previous 2 points
                pt0px = fiducials_rc[detectionOrder[0]-1,:]
                pt1px = fiducials_rc[detectionOrder[1]-1,:]
                pt0mm = fiducials_mm[detectionOrder[0]-1,:]
                pt1mm = fiducials_mm[detectionOrder[1]-1,:]
            
                diffpx = pt1px - pt0px
                diffmm = pt1mm - pt0mm 
                scale =   linalg.norm( diffpx ) \
                        / linalg.norm( diffmm )
                angle =   math.atan2( diffpx[1], diffpx[0] ) \
                        - math.atan2( diffmm[1], diffmm[0] )
                R = np.array([ [ math.cos(angle), -math.sin(angle) ],
                               [ math.sin(angle),  math.cos(angle) ] ])
                offset = pt0px - scale * R.dot(pt0mm)
                fiducials_rc[iFiducial-1,:] = offset + scale * R.dot( fiducials_mm[iFiducial-1,:] )
                # TODO: we might narrow the search window based on the difference of last estimated vs. found fiducial mark positions.
                searchWinHalfWidth = int(ringRadiusPx()*2)
            else:
                # TODO: refine transformation?
                fiducials_rc[iFiducial-1,:] = offset + scale * R.dot( fiducials_mm[iFiducial-1,:] )
            
            selection = Window( min( img.shape[0], max( 0, int( fiducials_rc[iFiducial-1,0]-searchWinHalfWidth   ) ) ),
                                min( img.shape[0], max( 0, int( fiducials_rc[iFiducial-1,0]+searchWinHalfWidth+1 ) ) ),
                                min( img.shape[1], max( 0, int( fiducials_rc[iFiducial-1,1]-searchWinHalfWidth   ) ) ),
                                min( img.shape[1], max( 0, int( fiducials_rc[iFiducial-1,1]+searchWinHalfWidth+1 ) ) ) )
            searchWin = img[ selection.startRow:selection.endRow, selection.startCol:selection.endCol ]

            # check equivalence - maximum absolute deviations are 1 (why are there deviations at all?)
            # plt.figure(6); plt.clf(); plt.imshow( searchWin.astype(np.float) - searchWin2.astype(np.float), cmap='gray' ); plt.colorbar();
        
            # If image is W x H and template is w x h , then result is (W-w+1) x (H-h+1)
            ccorr =  cv2.matchTemplate( searchWin, template, cv2.TM_CCOEFF_NORMED )
            iMaximum = ccorr.argmax() # with axis=None, returns the (scalar) flat index of the maximum value 
            maxCCoeff = ccorr.flat[iMaximum]
            # Kraus Bd.2 Kap. 6.1.2.4
            if maxCCoeff < 0.7:
                raise Exception("Maximum cross correlation coefficient is only {:.2}. Fiducial mark detection failed".format(maxCCoeff))
            row,col = np.unravel_index( iMaximum, ccorr.shape ) # transform flat index to row,col
        
            if debugPlot:
                plt.figure(3); plt.clf()
                plt.scatter( x=[col], y=[row], marker=".", color='r' )
                plt.contour( ccorr )
                plt.imshow( ccorr, cmap='gray', interpolation='nearest' )
                plt.colorbar()
        
            if subSelect:
                # Kraus recommends to now correlate only the area around the central spot/disc 
                # However, this does not seem to reduce the final RMSE!
            
                # coordinates of max ccoeff w.r.t. upper/left px of searchWindow 
                row += (template.shape[0]-1)//2
                col += (template.shape[1]-1)//2
            
                if debugPlot:
                    plt.figure(2); plt.clf()
                    plt.imshow( searchWin, cmap='gray', interpolation='nearest' )
                    plt.scatter( x=[col], y=[row], marker=".", color='r' )
    
                spotSearchWinHalfWidth = (templateSpot.shape[0]-1)//2 + 3
                subSelection = Window(
                  startRow = row - spotSearchWinHalfWidth,
                  startCol = col - spotSearchWinHalfWidth,
                  endRow   = row + spotSearchWinHalfWidth+1,
                  endCol   = col + spotSearchWinHalfWidth+1 )
                subSearchWin = searchWin[ subSelection.startRow:subSelection.endRow, subSelection.startCol:subSelection.endCol ]
                assert subSearchWin.shape == (spotSearchWinHalfWidth*2+1,)*2
                if debugPlot:        
                    plt.figure(8); plt.clf()
                    plt.imshow( subSearchWin, cmap='gray', interpolation='nearest' )
                # If image is W x H and template is w x h , then result is (W-w+1) x (H-h+1)
                ccorr =  cv2.matchTemplate( subSearchWin, templateSpot, cv2.TM_CCOEFF_NORMED )
                iMaximum = ccorr.argmax() # with axis=None, returns the (scalar) flat index of the maximum value 
                maxCCoeff = ccorr.flat[iMaximum]
                # Kraus B 6.1.2.4
                if maxCCoeff < 0.7:
                    raise Exception("Maximum cross correlation coefficient is only {:.2}. Fiducial mark detection failed".format(maxCCoeff))
                row,col = np.unravel_index( iMaximum, ccorr.shape ) # transform flat index to row,col
            else:
                subSelection = Window(
                  startRow = 0,
                  startCol = 0,
                  endRow   = 0,
                  endCol   = 0 )        
            
            # sub-pixel precision via lsq-adjustment of 2nd-order polynomial
            # in the 3x3-neighborhood of the pixel with the maximum ccoeff 
            # ccorr[row-1:row+2,col-1:col+2] - ccorr[row,col]
            # Kraus B 6.1.2.2:
            # Note: alternatively, we may apply LSM with a 2-element-offset vector and a single rotation parameter (not affine, but conformal).
            #       in that case, it would be advantageous to consider the whole fiducial mark, and not only its central disc.
            for neighborhood in range(1,5):
                nObs = (neighborhood*2 + 1)**2
                A = np.empty( (nObs,6) )
                r = np.empty( (nObs,) )
                iFlat = 0
                for iRow in range(-neighborhood,neighborhood+1):
                    for iCol in range(-neighborhood,neighborhood+1):
                        if iRow+row < 0 or iRow+row >= ccorr.shape[0] or \
                           iCol+col < 0 or iCol+col >= ccorr.shape[1]:
                            raise Exception("Cannot compute adjustment in such a large neighborhood")
                        r[iFlat] = ccorr[iRow+row,iCol+col]
                        A[iFlat,:] = ( 1, iRow, iCol, iRow*iCol, iRow**2, iCol**2 )
                        iFlat += 1    
                # use scipy
                #a = linalg.inv( A.T.dot(A) ).dot( A.T.dot(r) )
                a = linalg.lstsq( A, r, overwrite_a=True, overwrite_b=True )[0]
                # (6.1-3): compute the maximum of the 2nd-order polynomial
                A2 = np.array([ [ 2*a[4],   a[3] ],
                                [   a[3], 2*a[5] ] ])
                #subPix = linalg.inv( A2 ).dot( -np.array( [a[1], a[2]] ) )
                subPix = linalg.solve( A2, -a[1:3] )
                if np.abs(subPix).max() < neighborhood/2.: # Otherwise, let's consider a larger neighbourhood.
                    break
            else:
                raise Exception("Refinement of the position of the cross correlation coefficient maximum to subpixel resolution failed.")
        
            if subSelect:
                row += (templateSpot.shape[0]-1)/2
                col += (templateSpot.shape[1]-1)/2
            else:
                row += (template.shape[0]-1)/2
                col += (template.shape[1]-1)/2
    
                if debugPlot:
                    plt.figure(2); plt.clf()
                    plt.autoscale(False)
                    plt.imshow( searchWin, cmap='gray', interpolation='nearest' )

            
            if debugPlot: 
                plt.scatter( x=[col], y=[row], marker=".", color='r' )
        
            row += subPix[0]
            col += subPix[1]
        
            if debugPlot:
                plt.scatter( x=[col], y=[row], marker=".", color='g' )
                theta = np.linspace( 0., 2.*np.pi )
                plt.plot( col + centralDiscRadiusPx()*np.cos(theta), row + centralDiscRadiusPx()*np.sin(theta), '-r' )
        
            newVal = np.array([ row+selection.startRow+subSelection.startRow,
                                col+selection.startCol+subSelection.startCol])
            logger.debug( "fiducial #{} maxCCoeff={:.2} Prediction error(r,c)=({:.2f}, {:.2f})[px]", iFiducial, maxCCoeff, *(fiducials_rc[iFiducial-1,:] - newVal) )
            fiducials_rc[iFiducial-1,:] = newVal
        
            if debugPlot:
                # check the template's shape: overlay template und searchWin
                # Note: this plot rounds sub-pixel locations!
                fiducialsPxInt = fiducials_rc[iFiducial-1,:].astype(np.int)
                tplHalfWidth = (template.shape[0]-1)//2
                img2 = img[ fiducialsPxInt[0]-tplHalfWidth:fiducialsPxInt[0]+tplHalfWidth+1,
                            fiducialsPxInt[1]-tplHalfWidth:fiducialsPxInt[1]+tplHalfWidth+1 ]
                plt.figure(5); plt.clf()
                if 0:
                    blend = np.dstack( ( img2,
                                         np.zeros(template.shape, dtype=np.uint8),
                                         (template*0.5).astype(np.uint8)
                                        ) )
                    plt.imshow( blend, interpolation='nearest' )
                else:
                    diff = img2.astype(np.float) - template.astype(np.float) / 255. * img2.astype(np.float)[template>0].mean()
                    maxAbs = np.abs(diff).max()
                    plt.imshow( diff, interpolation='nearest', cmap='PuOr', vmin=-maxAbs, vmax=maxAbs )
                    plt.colorbar()
    

        # refine:
        scanResolution_microns = 1000 * ( cv2.contourArea( fiducials_mm[(7-1,3-1,5-1,1-1,8-1,4-1,6-1,2-1),:].astype(np.float32) ) /
                                          cv2.contourArea( fiducials_rc[(7-1,3-1,5-1,1-1,8-1,4-1,6-1,2-1),:].astype(np.float32) ) ) ** .5

        # Having localized the fiducial marks, now identify them (by the digit exposed next to each of them)
        digitOffset = ringRadiusPx() * 1.82 # horizontal offset of center of digit from center of fiducial
        digitHeight = 0.95*ringRadiusPx()
        digitThickness = centralDiscRadiusPx() * 1.8
        templateWH = round(digitHeight), round(1.5*digitHeight)
        templateWH = tuple(( el+1 if el%2==0 else el for el in templateWH ))
        nShiftBits = 8
        ccorrs = np.empty((4,8))
        for iRotation in range(4):
            fiducials = np.roll( fiducials_rc, shift=iRotation*2, axis=0 )
            for iFiducial,pt_rc in enumerate(fiducials, start=1):
                if iFiducial == 1:
                    digit = [ np.array([[1,-2],[1,2]]) ]
                elif iFiducial == 2:
                    digit = [ np.array([[-1,-2],[1,-2],[1,0],[-1,0],[-1,2],[1,2]]) ]
                elif iFiducial == 3:
                    digit = [ np.array([[-1,-2],[1,-2],[1,2],[-1,2]]), np.array([[-1,0],[1,0]]) ]
                elif iFiducial == 4:
                    digit = [ np.array([[-1,-2],[-1,0],[1,0]]), np.array([[1,-2],[1,2]]) ]
                elif iFiducial == 5:
                    digit = [ np.array([[1,-2],[-1,-2],[-1,0],[1,0],[1,2],[-1,2]]) ]
                elif iFiducial == 6:
                    digit = [ np.array([[1,-2],[-1,-2],[-1,2],[1,2],[1,0],[-1,0]]) ]
                elif iFiducial == 7:
                    digit = [ np.array([[-1,-2],[1,-2],[1,2]]) ]
                else:
                    assert iFiducial == 8
                    digit = [ np.array([[-1,-2],[1,-2],[1,2],[-1,2],[-1,-2]]), np.array([[-1,0],[1,0]]) ]
                template = np.zeros( templateWH[::-1], dtype=np.uint8 )
                templateWH_half = tuple(( round((el-1)/2) for el in templateWH ))
                digitScale = digitHeight / 4
                pts = [ ( (el*digitScale+templateWH_half) * 2**nShiftBits ).round().astype(np.int) for el in digit ]
                cv2.polylines( template, pts, isClosed=False, color=(255,255,255), thickness=round(digitThickness), lineType=cv2.LINE_AA, shift=nShiftBits )
        
                searchWin = img[ int(round(pt_rc[0])-templateWH_half[1])             : int(round(pt_rc[0])+templateWH_half[1]+1),
                                 int(round(pt_rc[1]+digitOffset)-templateWH_half[0]) : int(round(pt_rc[1]+digitOffset)+templateWH_half[0]+1) ]
                if False and debugPlot: # this loop creates many plots. as long as plotting is buggy, that may crash the interpreter
                    plt.figure(11); plt.clf()
                    plt.subplot(1,3,1)
                    plt.imshow( template, interpolation='nearest', cmap='gray' )
                    plt.axhline( templateWH_half[1] )
                    plt.axvline( templateWH_half[0] )
                    plt.subplot(1,3,2)
                    plt.imshow( searchWin, interpolation='nearest', cmap='gray' )
                    plt.axhline( templateWH_half[1] )
                    plt.axvline( templateWH_half[0] )
                    plt.subplot(1,3,3)
                    diff = ( template - template.mean() ) / template.std(ddof=1) - ( searchWin - searchWin.mean() ) / searchWin.std(ddof=1)
                    maxAbs = abs(diff).max()
                    plt.imshow( diff, interpolation='nearest', cmap='RdBu', vmin=-maxAbs, vmax=maxAbs )
                    plt.axhline( templateWH_half[1] )
                    plt.axvline( templateWH_half[0] )
                    plt.colorbar()

                ccorr = cv2.matchTemplate( image=searchWin, templ=template, method=cv2.TM_CCOEFF_NORMED ).item()
                ccorrs[iRotation,iFiducial-1] = ccorr

        iRotation = np.argmax( np.median( ccorrs, axis=1 ) )
        fiducials_rc = np.roll( fiducials_rc, shift=iRotation*2, axis=0 )
        fiducials_px = fiducials_rc[:,::-1].copy()
        fiducials_px[:,1] *= -1. # -> oriental-pix-CS

        # fiducials_mm = px2cam[:,:2].dot( fiducials_px ) + px2cam[:,2]
        #A = np.empty( (8*2, 6 ) )
        #b = np.zeros( (8*2,) )
        #for iFiducial in range(fiducials_px.shape[0]-1):
        #    fiducial_px = fiducials_px[iFiducial+1,:]
        #    px_x = fiducial_px[0]
        #    px_y = fiducial_px[1]
        #    A[iFiducial*2  ,:] = px_x, px_y, 1,    0,    0, 0
        #    A[iFiducial*2+1,:] =    0,    0, 0, px_x, px_y, 1
        #    b[iFiducial*2:iFiducial*2+2] = fiducials_mm[iFiducial+1,:]
        A = np.zeros( (8*2, 6 ) )
        A[0::2,0:2] = fiducials_px; A[0::2,2] = 1
        A[1::2,3:5] = fiducials_px; A[1::2,5] = 1
        b = np.ravel( fiducials_mm )

        pix2cam,squared2norm = linalg.lstsq( A, b )[:2]
        pix2cam = pix2cam.reshape( (2,3) )

        pix2cam = AffineTransform2D( pix2cam[:,:2], pix2cam[:,2] )

        # check trafo:
        # squared2norm = ( (fiducials_mm_check - fiducials_mm)**2 ).sum()  
        residuals_microns = (pix2cam.forward( fiducials_px ) - fiducials_mm) * 1000.

        rmse_microns = ( squared2norm / residuals_microns.size )**.5 * 1000
        # RMSEs for Carnuntum_Vertikalbilder_Bundesheer_2011:
        # 02110507_011.jpg: 6.9µm
        # Why not better?
        # Kraus B 6.1.2.3 mentions as typically achievable values: 1/10px to 1/5px;
        # for a scanning resolution of 15µm, this makes 2µm to 3µm

        resNorms_microns = ( residuals_microns**2 ).sum(axis=1)**.5

        # inverse trafo:
        residuals_px = pix2cam.inverse( fiducials_mm ) - fiducials_px
        resNorms_px = ( residuals_px**2 ).sum(axis=1)**.5
        rmse_px = ( resNorms_px.dot(resNorms_px) / residuals_px.size )**.5

        fiducialPos = np.array( [
            [ 7, 3, 5 ],
            [ 2,-1, 1 ],
            [ 6, 4, 8 ] ])

        if not isCalibrated:
            # don't issue this warning until here, because: LBA doesn't say if a photo was taken with RMK TOP or RMK A. Thus, we first try TOP, then A.
            # If it is not TOP, then we expect this function to return early. In that case, don't clutter the sceen with this warnung.
            logger.warning('{}. Using approximate values for interior orientation.', 'Unknown camera serial number' if camSerial is None else 'Camera calibration unavailable for RMK A with serial number #{}'.format(camSerial) )

        logger.verbose(
            'Transformation (pix->cam) residuals [�m] for RMK TOP photo {}:\v'.format( path.splitext( path.basename(imgFn) )[0] ) +
            '\t'.join( ('x','y')*3 ) + '\n' +
            '\n'.join( (
                '\t'.join( (
                    '{:+5.1f}\t{:+5.1f}'.format( *residuals_microns[el-1] ) if el > 0 else ' '*5 + '\t' + ' '*5
                for el in row ) ) 
            for row in fiducialPos ) ) + '\v' +
            'Residual norms [�m]:\v' +
            '\n'.join( (
                '\t'.join( (
                    '{:5.1f}'.format( resNorms_microns[el-1] ) if el > 0 else ' '*5
                for el in row ) ) 
            for row in fiducialPos ) ) + '\v'
            'RMSE [\N{GREEK SMALL LETTER MU}m]\t{:.1f}\n'.format( rmse_microns ) +
            'Scan resolution [�m]\t{:.1f}'.format(scanResolution_microns)
        )

        mask_px = pix2cam.inverse( self.mask_mm  )

        if debugPlot:
            plt.close('all') # img may be large. Save memory if debugPlot has produced previous plots.
        if plotDir or debugPlot:
            plt.figure(10, figsize=(8.0, 8.0)); plt.clf()
            plt.imshow( imgRGB )
            plt.xticks([]); plt.yticks([])
            plt.autoscale(False)
            mask_px_closed = np.vstack(( mask_px, mask_px[:1,:] ))
            plt.plot( mask_px_closed[:,0], -mask_px_closed[:,1], '-r' )
            fiducialsProj_px = pix2cam.inverse(fiducials_mm)
            plt.scatter( x=fiducialsProj_px[:,0], y=-fiducialsProj_px[:,1], marker="x", color='r' )
            plt.scatter( x=fiducials_px[:,0], y=-fiducials_px[:,1], marker="+", color='g' )
            for iPt,pt in enumerate(fiducials_px):
                plt.text( pt[0], -pt[1], str(iPt), color='y' )
            plt.title( 'RMSE [�m]: {:.1f}'.format(rmse_microns) )
        if plotDir:
            plt.savefig( path.join( plotDir, path.splitext( path.basename(imgFn) )[0] + '.jpg' ), bbox_inches='tight', dpi=150 )


        # TODO: warn the user if rotation/skew of affine trafo have weird values
        # TODO: store the adp according to the calibration protocol. For now, return a dummy.
        adp = ADP( normalizationRadius = linalg.norm( fiducials_mm[-1] - fiducials_mm[-2] ) / 3. )

        len0 = linalg.norm(pix2cam.A[0,:])
        len1 = linalg.norm(pix2cam.A[1,:])
        lenRatio = min(len0,len1) / max(len0,len1)
        if lenRatio < 0.95:
            logger.warning( 'Lengths of pixel CS axes transformed to camera CS deviate notably! Their ratio is: {:.3f}', lenRatio )
        angle_gon = math.acos( ( pix2cam.A[0,:] / len0 ).dot( 
                                 pix2cam.A[1,:] / len1 ) ) * 200. / np.pi
        if abs( angle_gon - 100. ) > 3:
            logger.warning( 'Angle between pixel CS axes transformed to camera CS deviates notably from the right angle: {:.1f}', angle_gon )

        return ior, adp, pix2cam, mask_px, isCalibrated, rmse_microns

        #    # compute bilinear transformation:
        #    # [ Xmm   = [ a0 a1 a2 a3   * [ 1
        #    #   Ymm ]     a4 a5 a6 a7 ]     Xpx
        #    #                               Ypx
        #    #                               Xpx*Ypx ]
        #    A = np.empty( (8*2, 8 ) )
        #    b = np.empty( (8*2,) )
        #    for iFiducial in range(fiducials_px.shape[0]-1):
        #        fiducial_px = fiducials_px[iFiducial+1,:]
        #        px_x = fiducial_px[0]
        #        px_y = fiducial_px[1]
        #        A[iFiducial*2  ,:] = ( 1, px_x, px_y, px_x*px_y, 0, 0, 0, 0 )
        #        A[iFiducial*2+1,:] = ( 0, 0, 0, 0 , 1, px_x, px_y, px_x*px_y )
        #        b[iFiducial*2:iFiducial*2+2] = fiducials_mm[iFiducial+1,:]
        #    x,squared2norm = linalg.lstsq( A, b )[:2]
        #    px2mm = x.reshape( (2,4) )
        #    # check trafo:
        #    fiducials_mm_check = px2mm.dot( np.vstack(( np.ones( (8,) ),
        #                                                fiducials_px[1:,0],
        #                                                fiducials_px[1:,1],
        #                                                fiducials_px[1:,0]*fiducials_px[1:,1] )) ).T
        #    # squared2norm = ( (fiducials_mm_check - fiducials_mm[1:,:])**2 ).sum()  
        #
        #    resNorms = ( ( (fiducials_mm_check - fiducials_mm[1:,:]) * 1000. )**2 ).sum(axis=1)**.5
        #
        #    # TODO: z.B. Orthogonalitätsfehler ausgeben, s. B 7.1.1
        #
        #    # Note: the following does NOT yield the inverse of px2mm!
        #
        #    # compute bilinear transformation:
        #    # [ Xpx   = [ a0 a1 a2 a3   * [ 1
        #    #   Ypx ]     a4 a5 a6 a7 ]     Xmm
        #    #                               Ymm
        #    #                               Xmm*Ymm ]
        #    A = np.empty( (8*2, 8 ) )
        #    b = np.empty( (8*2,) )
        #    for iFiducial in range(fiducials_mm.shape[0]-1):
        #        fiducial_mm = fiducials_mm[iFiducial+1,:]
        #        mm_x = fiducial_mm[0]
        #        mm_y = fiducial_mm[1]
        #        A[iFiducial*2  ,:] = ( 1, mm_x, mm_y, mm_x*mm_y, 0, 0, 0, 0 )
        #        A[iFiducial*2+1,:] = ( 0, 0, 0, 0 , 1, mm_x, mm_y, mm_x*mm_y )
        #        b[iFiducial*2:iFiducial*2+2] = fiducials_px[iFiducial+1,:]
        #    x,squared2norm = linalg.lstsq( A, b )[:2]
        #    mm2px = x.reshape( (2,4) )
        #    # check trafo:
        #    fiducials_px_check = mm2px.dot( np.vstack(( np.ones( (8,) ),
        #                                                fiducials_mm[1:,0],
        #                                                fiducials_mm[1:,1],
        #                                                fiducials_mm[1:,0]*fiducials_mm[1:,1] )) ).T
        #    # squared2norm = ( (fiducials_px_check - fiducials_px[1:,:])**2 ).sum()  
        #    resNorms = ( (fiducials_px_check - fiducials_px[1:,:])**2 ).sum(axis=1)**.5
        #
        #    ior = np.array([ fiducials_mm[0,0], fiducials_mm[0,1], 152.642 ])
        #    adp = ADP( normalizationRadius = linalg.norm( fiducials_mm[8]-fiducials_mm[7] * 1./3. ) )
        #    return ior, adp, px2mm, mm2px

zeissRmkTop = ZeissRmkTop()

if __name__ == '__main__':
    zeissRmkTop( r"E:\arap\data\Carnuntum_schraeg_u_vertikal\0220110507_009.jpg", '151967', debugPlot=True )
