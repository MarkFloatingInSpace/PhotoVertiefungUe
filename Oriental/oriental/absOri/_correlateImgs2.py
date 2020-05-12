# -*- coding: cp1252 -*-

import math
import os
from os.path import basename
from collections import namedtuple
import numpy as np
from scipy import linalg
import sqlite3.dbapi2 as db
from osgeo import gdal
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()
import cv2

from contracts import contract, new_contract

from oriental import config, ori, log
from oriental.utils import ( traitlets,
                             pyplot as plt,
                             pyplot_utils as plt_utils )

from .bruteForce import BruteForceResult, bruteForce
from .lsm import lsm, LSMResult
from .manualTrafo import interpolateBilinear

figsize = (5,4)

@contract
def ensureRGB( dataSet : gdal.Dataset ) -> None:
    if dataSet.RasterCount != 3 or \
       dataSet.GetRasterBand(1).GetColorInterpretation() != gdal.GCI_RedBand or \
       dataSet.GetRasterBand(2).GetColorInterpretation() != gdal.GCI_GreenBand or \
       dataSet.GetRasterBand(3).GetColorInterpretation() != gdal.GCI_BlueBand:
        raise Exception("GDAL dataset is not an RGB image.")

@contract
def rect2obj(  rect_xy : 'array[2](float)',
               H : 'array[3x3](float)',
               dbFn : str,
               imgFn : str,
               normalPtcl : 'array[4](float)',
               manualAbsOri ) -> 'array[3](float)':
    "transform the image point warp_xy in the warped, rectified AP to object space, using the simplest object model: the adjusting plane"

    # transform from the rectified AP to the original AP
    # imgRect.H is the non-inverse homographic transform from the aerial photo to rect, thus:
    # [ x_d, y_d, w_d ].T = H.dot( [ x_s, y_s, 1 ].T )
    # with _s ... source coordinates (aerial)
    #      _d ... destination coordinates (rect)
    cond,Hinv = cv2.invert( H )
    bf_xy = cv2.perspectiveTransform( rect_xy[np.newaxis,np.newaxis,:], Hinv ).flatten()

    # intersect the observation ray with the adjusting object plane.
    origImgFn = os.path.join( os.path.dirname(os.path.dirname(imgFn)), os.path.basename(imgFn) )
    with db.connect( dbFn ) as relOri:
        X0,Y0,Z0,r1,r2,r3,x0,y0,z0 = relOri.execute("""
            SELECT images.X0,
                    images.Y0,
                    images.Z0,
                    images.r1,
                    images.r2,
                    images.r3,
                    cameras.x0,
                    cameras.y0,
                    cameras.z0
            FROM cameras
	            JOIN images
		        ON images.camID==cameras.id
	        WHERE images.path==?
        """, (origImgFn,) ).fetchone()

    ctrCam = np.array([  bf_xy[0] - x0,
                        -bf_xy[1] - y0,
                                  - z0 ] )
    R = ori.omfika( np.array([r1, r2, r3]) )
    r0 = R.dot( ctrCam )
    r0 /= linalg.norm(r0)
    P0 = np.array([X0,Y0,Z0])

    n0 = normalPtcl[:3]
    d = normalPtcl[3]

    # ( P0 + k*r0 ).dot( n0 ) = d
    k = ( d - P0.dot(n0) ) / r0.dot(n0)

    bf_mod = P0 + k * r0

    # model = s*R.dot(object-x0)
    s_obj,R_obj,x0_obj = manualAbsOri
    bf_obj = R_obj.T.dot( bf_mod / s_obj ) + x0_obj

    return bf_obj

class Result(traitlets.HasStrictTraits):
    fn = traitlets.CUnicode()
    searchPos_m    = traitlets.NDArray( dtype=np.float, shape=(2,) )
    flyingHeight_m = traitlets.CFloat()
    azimuth_gon    = traitlets.CFloat()
    bruteForce     = traitlets.Instance(BruteForceResult)
    lsm            = traitlets.Instance(LSMResult)
    error_type     = traitlets.Unicode()

new_contract('Result', lambda x: isinstance(x, Result) )

def successfulResults(results):
    for el in results:
        if el.error_type=='' and \
           el.bruteForce and el.bruteForce.error_type =='' and \
           el.lsm and el.lsm.error_type=='':
            yield el

@contract
def histo2d( figNum : int,
             lsm_error_m : 'array(float)',
             results : list,
             attr : str ):

    def getDottedAttr( obj, attr ):
        attrNames = attr.split('.')
        subObj = getattr( obj, attrNames[0] )
        if len(attrNames)==1:
            return subObj
        return getDottedAttr( subObj, '.'.join( attrNames[1:] ) )

    values = np.fromiter( ( getDottedAttr( el, attr ) for el in successfulResults(results) ), dtype=np.float )
    plt.figure(figNum, figsize=figsize); plt.clf()
    plt.hist2d( x=lsm_error_m, y=values, cmin=1, bins=50 )
    plt.xlabel('lsm.error_m')
    plt.ylabel(attr)
    plt.colorbar()
    return plt_utils.embedPNG()

@contract
def logResults( results : 'list(Result)',
                logger : log.Logger,
                xml : str = '' ) -> None:
    # Auswertung
    nTotal = len(results)
    msg = "Statistics\t\n"
    msg += "#tried combinations\t{}\n".format(nTotal)
    nTmplOutside = sum( (1 for el in results if el.error_type=='tmpl-outside') )
    msg += "#tmpl-outside\t{} of {}\n".format( nTmplOutside,nTotal)
    nTotal -= nTmplOutside
    nBfOutside = sum( (1 for el in results if el.bruteForce.error_type=='bf-outside') )
    msg += "#bf-outside\t{} of {}\n".format( nBfOutside, nTotal )
    nTotal -= nBfOutside
    nLsmSingular = sum( (1 for el in results if el.lsm and el.lsm.error_type=='lsm-singular') )
    msg += "#lsm-singular\t{} of {}\n".format( nLsmSingular, nTotal )
    nLsmOutside = sum( (1 for el in results if el.lsm and el.lsm.error_type=='lsm-outside') )
    msg += "#lsm-outside\t{} of {}\n".format( nLsmOutside, nTotal )
    nTotal -= nLsmSingular + nLsmOutside
    nLsmMaxIter = sum( (1 for el in results if el.lsm and el.lsm.error_type=='lsm-maxiter') )
    msg += "#lsm-maxiter\t{} of {}\n".format( nLsmMaxIter, nTotal )
    nTotal -= nLsmMaxIter
    msg += "#successful\t{} of {}\n".format( nTotal, len(results) )
    logger.info(msg)

    bf_r = np.fromiter( ( el.bruteForce.r for el in successfulResults(results) ), dtype=np.float )
    bf_r2nd = np.fromiter( ( el.bruteForce.r2nd for el in successfulResults(results) ), dtype=np.float )
    lsm_error_m = np.fromiter( ( el.lsm.error_m for el in successfulResults(results) ), dtype=np.float )
    # maximum of absolute correlation coefficients, excluding those on the main diagonal
    lsm_rMaxAbs = np.fromiter( ( np.abs( np.triu( el.lsm.Rxx[:-2,:-2], 1 ) ).max() for el in successfulResults(results) ), dtype=np.float )
    lsm_stdDevX = np.fromiter( ( el.lsm.stdDevs[0] for el in successfulResults(results) ), dtype=np.float )
    lsm_stdDevY = np.fromiter( ( el.lsm.stdDevs[3] for el in successfulResults(results) ), dtype=np.float )

    assert lsm_error_m.shape[0] == nTotal
    #assert next( el.error_type for el in results if el.flyingHeight_m==150. and el.azimuth_gon==125. ) == '' # these approx. values are closest to the true ones.

    if nTotal < 2:
        logger.warning("Less than 2 successful results, not printing histograms")
    else:
        plt.figure(1, figsize=figsize); plt.clf()
        plt.hist( lsm_error_m, bins=50 )
        plt.xlabel('lsm.error_m')
        plt.ylabel('count')
        xml += plt_utils.embedPNG()

        for idx,attr in enumerate( ('bruteForce.r', 'bruteForce.rw') ):
            xml += histo2d( idx+100, lsm_error_m, results, attr )

        plt.figure(2, figsize=figsize); plt.clf()
        plt.hist2d( x=lsm_error_m, y=bf_r2nd/bf_r, cmin=1, bins=50 )
        plt.xlabel('lsm.error_m')
        plt.ylabel('bruteForce.r2nd/bruteForce.r')
        plt.colorbar()
        xml += plt_utils.embedPNG()

        for idx,attr in enumerate( ('lsm.niter', 'lsm.s0', 'lsm.r', 'lsm.rw') ):
            xml += histo2d( idx+100, lsm_error_m, results, attr )

        for idx,attr in enumerate( ('lsm_rMaxAbs','lsm_stdDevX','lsm_stdDevY') ):
            plt.figure(idx+3, figsize=figsize); plt.clf()
            plt.hist2d( x=lsm_error_m, y=locals()[attr], cmin=1, bins=50 )
            plt.xlabel('lsm.error_m')
            plt.ylabel(attr)
            plt.colorbar()
            xml += plt_utils.embedPNG()

        logger.infoRaw( xml )

    goodResults = [ el for el in successfulResults(results) if el.lsm.error_m < 1. ]
    weakResults = [ el for el in successfulResults(results) if el.lsm.error_m >= 1. and el.lsm.error_m < 10. ]
    logger.info( "#Results with lsm.error_m < 1.: {}", len(goodResults) )
    logger.info( "#Results with 1 <= lsm.error_m < 10.: {}", len(weakResults) )

@contract
def correlateImgs2( imgRects   : 'array[N]',
                    normalPtcl : 'array[4](float)',
                    fnOrtho : str,
                    dbFn    : str,
                    plot    : bool = False,
                    weightedLsm : bool = False,
                    manualAbsOri : 'tuple( float, array[3x3](float), array[3](float) ) | None' = None,
                    fnDSM : 'str | None' = None,
                    globalRadiometricAdmustment : bool = False ):
    if plot:
        #plotDir = r"H:\140316_ISPRS_Comm_V_Gardasee\04 paper\figures_work"
        plotDir = r"D:\arap\data\Carnuntum_UAS_Geert\moscow"
        from ..utils.BlockingKernelManager import client
        #client.shell_channel.execute("matplotlib.rcParams['font.size'] = 15")
        #client.shell_channel.execute("matplotlib.rcParams['figure.dpi'] = 100")
        #client.shell_channel.execute("matplotlib.rcParams['savefig.dpi'] = 100")

        # don't convert text to paths!
        # + Makes svg file sizes smaller.
        # - Fonts must be installed on viewing machine.
        client.shell_channel.execute("matplotlib.rcParams['svg.fonttype'] = 'none'")

    logger = log.Logger("absOri")

    # search for homologous points in ortho and 3240461
    # use 3240461, because it is quite vertical, and rather small scale
    # 3240461 is not (yet) part of LBA
    # -> assume necessary data that would be part of LBA

    # Seitenlänge des quadratischen Suchfensters im Objektraum definieren:
    searchWinSideLen_m = 20.

    planarOffsetHalfRange_m = 60.
    planarOffsetStepSize_m = searchWinSideLen_m/2.
    flyingHeightHalfRange_m = 0.#20.
    flyingHeightStepSize_m  = 2.5
    azimuthStepSize_gon     = 5.

    logger.info(   'Parameters:\t\n'
                   'searchWinSideLen_m\t{}\n'     .format(searchWinSideLen_m)
                 + 'planarOffsetHalfRange_m\t{}\n'.format(planarOffsetHalfRange_m)
                 + 'planarOffsetStepSize_m\t{}\n' .format(planarOffsetStepSize_m)
                 + 'flyingHeightHalfRange_m\t{}\n'.format(flyingHeightHalfRange_m)
                 + 'flyingHeightStepSize_m\t{}\n' .format(flyingHeightStepSize_m)
                 + 'azimuthStepSize_gon\t{}\n'    .format(azimuthStepSize_gon)
                 + 'weightedLsm\t{}\n'            .format(weightedLsm)
                 + 'globalRadiometricAdmustment\t{}\n'.format(globalRadiometricAdmustment)  )

    # UAV                                       # Rechts, Hoch
    UavParams = namedtuple( 'UavParams', [ 'fn', 'Y', 'X', 'flyingHeight' ] )
    
    uavParamsList = [ 
                      #UavParams( '3240461_DxO_no_dist_corr', 38632.11, 330299.67, 75. ) # Zentrum des Amphitheaters

                      UavParams( '3240461_DxO_no_dist_corr', 38607.23, 330309.80, 149.29 )
                    ]

    dsOrtho = gdal.Open( fnOrtho, gdal.GA_ReadOnly )
    ensureRGB( dsOrtho )
    ortho2wrld = dsOrtho.GetGeoTransform()
    okay,wrld2ortho = gdal.InvGeoTransform( ortho2wrld ); assert okay==1, "inversion of transform failed"
    # ReadAsArray() returns an array with shape (depth,nRows,nCols)
    ortho = np.rollaxis( dsOrtho.ReadAsArray(), 0, 3 )

    orthoGray = cv2.cvtColor( ortho, cv2.COLOR_RGB2GRAY )
    if globalRadiometricAdmustment:
        # TODO: restrict the image are used for normalization to the area covered in the search
        orthoGray = ( ( orthoGray - orthoGray.mean() ) / orthoGray.std(ddof=1) ).astype( np.float32 ) # cv2.matchTemplate accepts that

    if fnDSM is not None:
        # interpolate terrain height at template centre
        dsDsm = gdal.Open( fnDSM, gdal.GA_ReadOnly )
        assert dsDsm.RasterCount == 1
        assert dsDsm.GetRasterBand(1).GetColorInterpretation() == gdal.GCI_Undefined
        dsm2wrld = dsDsm.GetGeoTransform()
        okay,wrld2dsm = gdal.InvGeoTransform( dsm2wrld ); assert okay==1

    overallResults = []
    iTry = 0
    for iUavParams,uavParams in enumerate(uavParamsList):
        print( "-------------\nUavParams #{}".format(iUavParams) )

        with db.connect( dbFn ) as relOri:
            X0,Y0,Z0,r1,r2,r3,parameterization,x0,y0,z0 = relOri.execute("""
                SELECT images.X0,
                       images.Y0,
                       images.Z0,
                       images.r1,
                       images.r2,
                       images.r3,
                       images.parameterization,
                       cameras.x0,
                       cameras.y0,
                       cameras.z0
                FROM images
                    JOIN cameras
                        ON images.camID==cameras.ID
                WHERE images.path GLOB "*{}.???"
            """.format( basename(uavParams.fn) ) ).fetchone()
            PRC_mod = np.array([X0,Y0,Z0])
            assert parameterization=='omfika'
            omfika_mod = np.array([r1,r2,r3])
            ior_mod = np.array([x0,y0,z0])

        iRect = [ idx for idx in range(len(imgRects)) if os.path.splitext( os.path.basename( imgRects[idx].path ) )[0] == uavParams.fn ][0]
        imgRect = imgRects[iRect]
        dsRect = gdal.Open( imgRect.path, gdal.GA_ReadOnly )
        ensureRGB( dsRect )
        rect = np.rollaxis( dsRect.ReadAsArray(), 0, 3 )
        rect = cv2.cvtColor( rect, cv2.COLOR_RGB2GRAY ) # Bildränder?
        if globalRadiometricAdmustment:
            # exclude image borders without information that were created during rectification i.e. consider imgRect.rc
            corners = imgRect.rc[:,::-1] # trafo row/cols -> x,y
            mask = np.zeros( rect.shape, dtype=np.uint8 )
            mask = cv2.fillConvexPoly( mask, corners.astype(np.int), (1,1,1) ) 
            rectMasked = np.ma.masked_array( rect, mask!=1 )   
            rect = ( ( rect - rectMasked.mean() ) / rectMasked.std(ddof=1) ).astype( np.float32 ) # cv2.matchTemplate accepts that
        
        #  38607.23   330309.80
        #y=38607.23 x=330249.8
        #  38627.23   330299.8 # Zentrum des Amphitheaters
        #tmplCtrsWrlX = np.arange( uavParams.Y-planarOffsetHalfRange_m,
        #                          uavParams.Y+planarOffsetHalfRange_m + .01,
        #                          planarOffsetStepSize_m )
        #tmplCtrsWrlY = np.arange( uavParams.X-planarOffsetHalfRange_m,
        #                          uavParams.X+planarOffsetHalfRange_m + .01,
        #                          planarOffsetStepSize_m )
        tmplCtrsWrlX = np.array([  38607.23,  38607.23,  38627.23 ])
        tmplCtrsWrlY = np.array([ 330309.80, 330249.80, 330299.80 ])
        for iTmplCtrWrlX,(tmplCtrWrlX,tmplCtrWrlY) in enumerate(zip(tmplCtrsWrlX,tmplCtrsWrlY)):
            for dummy in (None,):
                iTmplCtrWrlY = iTmplCtrWrlX
        #for iTmplCtrWrlX,tmplCtrWrlX in enumerate(tmplCtrsWrlX):
        #    for iTmplCtrWrlY,tmplCtrWrlY in enumerate(tmplCtrsWrlY):
                logger.info( "Pos {} of {}: Rechts/Hoch:{}/{}",
                             iTmplCtrWrlX * len(tmplCtrsWrlX) + iTmplCtrWrlY + 1,
                             len(tmplCtrsWrlX)*len(tmplCtrsWrlY),
                             tmplCtrWrlX,
                             tmplCtrWrlY )
                results = []
                # extract template
                tmpl_ctr_wrl = np.array( ( tmplCtrWrlX, tmplCtrWrlY ) )

                # beziehe die Bildkoordinaten auf den Mittelpunkt des linken/oberen Pixels -> - 0.5
                # runde auf ganze Pixel
                # Koordinatenreihenfolge: row/col statt col/row -> reversed
                tmpl_ctr_px_rc = np.fromiter( ( round( el - .5 ) for el in reversed( gdal.ApplyGeoTransform( wrld2ortho, *tmpl_ctr_wrl ) ) ),
                                              dtype=np.int )

                # Annahme: Orthophoto is axis-aligned zu Welt-KS, mit quadratischer Pixelgröße
                assert wrld2ortho[2] == wrld2ortho[4] == 0.
                assert wrld2ortho[1] == -wrld2ortho[5]
                tmpl_halfSearchWinSideLen_px = searchWinSideLen_m/2. * wrld2ortho[1]
    
                tmpl_lu_px_rc = tmpl_ctr_px_rc - tmpl_halfSearchWinSideLen_px
                tmpl_rl_px_rc = tmpl_ctr_px_rc + tmpl_halfSearchWinSideLen_px # inclusive!

                # check if ortho wholly contains tmpl_lu_px_rc, tmpl_rl_px_rc
                if np.any( tmpl_lu_px_rc < [ -.5, -.5 ] ) or \
                   np.any( tmpl_rl_px_rc > np.array(ortho.shape[:2]) -.5 ):
                    result.error_type = 'tmpl-outside'
                    continue
    
                tmpl = orthoGray[ tmpl_lu_px_rc[0]:tmpl_rl_px_rc[0]+1,
                                  tmpl_lu_px_rc[1]:tmpl_rl_px_rc[1]+1 ]
                #tmpl = cv2.cvtColor( tmpl, cv2.COLOR_RGB2GRAY )

                if plot:
                    plt.figure(2, tight_layout=True); plt.clf()
                    plt.imshow( tmpl, interpolation='nearest', cmap='gray', vmin=0, vmax=255 )
                    plt.title('OPM')
                    #plt.savefig( os.path.join( plotDir, str(iUavParams) + '_' + uavParams.fn + "_ortho_detail.pdf" ), bbox_inches='tight', transparent=True )
    
                if manualAbsOri is not None and \
                   fnDSM is not None:
                    # interpolate terrain height at template centre
                    tmpl_ctr_dsm = np.array( gdal.ApplyGeoTransform( wrld2dsm, *tmpl_ctr_wrl ) )
                    tmpl_ctr_dsm_px = tmpl_ctr_dsm - 0.5
                    tmpl_ctr_dsm_px_org = np.floor( tmpl_ctr_dsm_px )
                    dsm = dsDsm.ReadAsArray( xoff=int(tmpl_ctr_dsm_px_org[0]),
                                             yoff=int(tmpl_ctr_dsm_px_org[1]),
                                             xsize=2,
                                             ysize=2 )
                    z = interpolateBilinear( dsm, tmpl_ctr_dsm_px - tmpl_ctr_dsm_px_org )
                    tmpl_ctr_wrl_3d = np.append( tmpl_ctr_wrl, [z] )

                    # model = s*R.dot(object-x0)
                    s_obj,R_obj,x0_obj = manualAbsOri
                    # transform template center from object (ortho) to model
                    tmplCtrModel = s_obj * R_obj.dot( tmpl_ctr_wrl_3d - x0_obj )
                    # transform from model to oblique AP
                    # TODO: radial distortion!
                    tmplCtrAP = ori.projection( tmplCtrModel, PRC_mod, omfika_mod, ior_mod )
                    # ORIENT -> matplotlib
                    tmplCtrAP[1] *= -1.
                    # transform from oblique AP to rectified AP
                    # imgRect.H is the non-inverse homographic transform from the aerial photo to rect, thus:
                    # [ x_d, y_d, w_d ].T = H.dot( [ x_s, y_s, 1 ].T )
                    # with _s ... source coordinates (aerial)
                    #      _d ... destination coordinates (rect)
                    tmplCtrAP_rect = imgRect.H.dot( np.append( tmplCtrAP, [1] ) )
                    tmplCtrAP_rect = tmplCtrAP_rect[:2] / tmplCtrAP_rect[2]

                    # This is an approximation: PRC-z-object-coordinate minus terrain height at template centre
                    trueFlyingHeight_m = ( R_obj.T.dot( PRC_mod / s_obj ) + x0_obj )[2] - z
                    logger.info("True flying height above ground [m]: {:.3f}".format( trueFlyingHeight_m ) )

                    # This is another approximation: angle between cam-sys-x-axis and obj-sys-x-axis
                    x_mod = ori.omfika( omfika_mod ).dot( np.array([1.,0.,0.]) )
                    x_obj = R_obj.T.dot( x_mod )
                    trueAzimuth_gon = -math.atan2( x_obj[1], x_obj[0] ) / math.pi * 200.
                    logger.info( "True azimuth [gon]: {:.3f}".format( trueAzimuth_gon ) )
                else:
                    tmplCtrAP_rect = None

                # Ungefähren Maßstab abschätzen aus IOR und Flughöhe
                # Über den Normalvektor der Punktwolke wissen wir zwar ca. die Rotation des phos relativ zum Boden,
                # nicht aber die Position des Suchfensters im Bild.
                # Im (schrägen) Luftbild variiert der Bildmaßstab je nach Position.
                # Nicht aber im entzerrten Bild!
                # -> einfach Suchfenster-Seitenlänge skalieren mit Verhältnis Flughöhe zu Brennweite des entzerrten Bilds!
                rect_focal = imgRect.K[0,0]

                flyingHeights = np.arange( uavParams.flyingHeight - flyingHeightHalfRange_m,
                                           uavParams.flyingHeight + flyingHeightHalfRange_m+0.01,
                                           flyingHeightStepSize_m )
                for iFlyingHeight,flyingHeight in enumerate( flyingHeights ):
                    print("=======")
                    print("Flying height #{}: {} [m]".format( iFlyingHeight, flyingHeight ) )
                    rect_halfSearchWinSideLen_px = int( round( searchWinSideLen_m / 2. / flyingHeight * rect_focal ) )  
    
                    # for varying azimuths, we need to warp/rotate the rectified image, such that the search window will be axis-aligned
                    # Achtung: es wird um die li/obere Bildecke gedreht!
                    #azimuths = np.arange( 0., 400., azimuthStepSize_gon )
                    azimuths = [trueAzimuth_gon]
                    for iAzimuth,azimuth in enumerate( azimuths ):
                        iTry += 1
                        print("------- Azimuth #{}: {} [gon]; try {} of {}".format(
                            iAzimuth, azimuth, iTry,
                            len(uavParamsList)*len(tmplCtrsWrlX)*len(tmplCtrsWrlY)*len(flyingHeights)*len(azimuths) ) )

                        result = Result()
                        results.append( result )
                        result.fn = uavParams.fn
                        result.searchPos_m = tmpl_ctr_wrl
                        result.flyingHeight_m = flyingHeight
                        result.azimuth_gon = azimuth

                        result.bruteForce,rect_warp,warp_rc = bruteForce(
                            rect = rect,
                            tmpl = tmpl,
                            imgRect = imgRect,
                            azimuth_gon = azimuth,
                            tmpl_halfSearchWinSideLen_px = int(tmpl_halfSearchWinSideLen_px),
                            rect_halfSearchWinSideLen_px = rect_halfSearchWinSideLen_px,
                            tmplCtrAP_rect=tmplCtrAP_rect,
                            method = cv2.TM_CCOEFF_NORMED if not globalRadiometricAdmustment else cv2.TM_SQDIFF,
                            plot = str(iTmplCtrWrlX) )

                        if len( result.bruteForce.error_type ):
                            continue

                        if manualAbsOri is not None:
                            bf_obj = rect2obj(
                                rect_xy = result.bruteForce.rect_xy,
                                H = imgRect.H,
                                dbFn = dbFn,
                                imgFn = imgRect.path,
                                normalPtcl = normalPtcl,
                                manualAbsOri = manualAbsOri )
                            result.bruteForce.error_m = linalg.norm( bf_obj[:2] - tmpl_ctr_wrl )
                            print("error 2D brute force [m]: {:.3f}".format(result.bruteForce.error_m))


                        # LSM
                        estimateContrast = True
                        result.lsm = lsm(
                            template=tmpl,
                            picture=rect_warp,
                            picture_shift_rc = warp_rc,
                            estimateContrast=estimateContrast,
                            plot=plot,
                            weight=weightedLsm )
                        if len(result.lsm.error_type):
                            continue
                        inverseTrafoLsm = result.lsm.M
                        #if plot:
                        #    plt.savefig( os.path.join( plotDir, 'lsm', str(iUavParams) + '_' + uavParams.fn + "_lsm.pdf" ), bbox_inches='tight' )

                        # transform the center of the detected window to the warped UP
                        # Note: inverseTrafoLsm rotates about the origin of the destination coordinate system, which is shifted by -knlHSz i.e. -2
                        ctrWarp_xy = inverseTrafoLsm[:,:2].dot( np.ones(2)*(tmpl_halfSearchWinSideLen_px+2) ) + inverseTrafoLsm[:,2]
                        if plot:
                            plt.figure(4); plt.clf()
                            plt.imshow( rect_warp, interpolation='nearest', cmap='gray', vmin=0, vmax=255 )
                            plt.scatter( *ctrWarp_xy, s=150, marker='o', edgecolors='m', facecolors='none' )

                        # transform from the warped AP to the rectified AP
                        # M is the non-inverse affine transform from rect to rect_warp
                        Minv = cv2.invertAffineTransform( result.bruteForce.M )
                        result.lsm.rect_xy = Minv[:,:2].dot( ctrWarp_xy ) + Minv[:,2]

                        if manualAbsOri is not None and \
                           fnDSM is not None:
                            # transform ctrWarp_xy from rect_warp to object space using a surface model in model space,
                            # and compute the error in object space.
                            bf_obj = rect2obj(
                                rect_xy = result.lsm.rect_xy,
                                H = imgRect.H,
                                dbFn = dbFn,
                                imgFn = imgRect.path,
                                normalPtcl = normalPtcl,
                                manualAbsOri = manualAbsOri )
                            result.lsm.error_m = linalg.norm( bf_obj[:2] - tmpl_ctr_wrl )
                            result.lsm.pos_obj = bf_obj
                            print("error 2D LSM [m]: {:.3f}".format(result.lsm.error_m))

                        continue
                
                plt.figure(2, figsize=figsize ); plt.clf()
                plt.imshow( tmpl, interpolation='nearest', cmap='gray', vmin=0, vmax=255 )
                plt.title('OPM')
                xml = plt_utils.embedPNG()

                logResults( results, logger, xml )
                overallResults += results

    return overallResults