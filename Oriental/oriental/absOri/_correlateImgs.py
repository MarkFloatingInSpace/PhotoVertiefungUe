# -*- coding: cp1252 -*-

# Probleme bei Korrelation des entzerrten Luftbilds & Ortho:
# Nicht einmal, wenn ich das Azimuth und den Maßstab des Ausschnitts des entzerrten Luftbilds manuell recht genau vorgebe,
# liegt das Maximum des Korrelationskoeffizienten an der gewünschten Stelle, sondern ganz woanders.
# Vermutlich liegt das an den starken Schatten, die sowohl im Ortho, als auch im Luftbild vorhanden sind.
# Abhilfe könnte die Verwendung eines alternativen Farbraums schaffen, der v.a. den Farbton und die Helligkeit, nicht aber die Sättigung beachtet
#   - s. dazu z.B. Artikel von Blauensteiner (PRIP): http://www.researchgate.net/publication/255615120_On_Colour_Spaces_for_Change_Detection_and_Shadow_Suppression
# unabhängig vom Problem mit den Schatten: im Carnuntum-Datensatz sind v.a. die Mauern des Amphitheaters stabile Features.
# Diese Features sind aber nur wenige cm breit. Deshalb ist der Peak der sich ergibt, wenn ich das Ortho mit sich selbst korreliere, auch nur wenige Pixel breit.
# Das bedeutet wiederum, dass die Korrelation praktisch mit voller Auflösung und sehr guter Näherung des Maßstabs und des Azimuths durchgeführt werden muss,
# weil sich ansonsten gar kein Maximum des Korrelationskoeffizienten an der korrekten Stelle herausbildet.

# Alternativen:
# (1) Kanten extrahieren -> Binärbilder korrelieren. Kantenextraktor liefert leider auch Schattengrenzen und Kanten in Vegetation
# (2) MSER: liefert v.a. auch Schatten. Allerdings müssten für eine genauere Untersuchung wohl die Parameter angepasst werden: 
##mser = cv2.MSER()
##img = rect_warp.copy()
##regions = mser.detect(img, None)
##cv2.polylines( img=img, pts=regions, isClosed=1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA )
##plt.figure(11)
##plt.imshow(img, cmap='gray')

# Ideen/Erweiterungen:
# bevorzugt Luftbilder korrelieren, deren optische Achsen einen kleinen Winkel mit der Normalen der Punktwolke einschließen, damit Bodenunebenheiten eine kleinere Rolle spielen.

# Rauheit des DSM berechnen, und rauhe Bereiche ìm Orthophoto maskieren (Vegetation), d.h. von Korrelation ausschließen?
# -> Das geht bei der Auflösung des DSM von 1m schlecht. Evtl. funzt aber: nDSM = DSM minus DTM, mit Schwellwert, z.B. 2m
# damit Bereiche keinen Einfluss auf CCOEFF haben: Mittelwert der nicht-Vegetationspunkte im Suchfenster rechnen, und Vegetationspunkte auf diesen Mittelwert setzen. 

# mit pho 52 und Bereich um Amphitheater anfangen, denn darin ist nichts enthalten, was nicht auch im Orthophoto wäre (Parkplatz, Zufahrt)
# Durchmesser der Arena: ca. 50m
# Breite der Trompete der Einmündung in die Bundesstraße: ca. 55m
# Breite der Bundesstraße: ca. 11.5m

#film = '02110503'
#bildnr = 52
    
#with db.connect( config.dbLBA ) as conn:
#    rows = conn.execute("""
#        SELECT hoch
#        FROM bild
#        WHERE bild=?
#    """, ( "{}.{}".format(film,bildnr) ,) ).fetchall()
#    flyingHeight = float( rows[0][0] )  
        
#flyingHeight = 375.        
#flyingHeight = 500.
    
# zentraler Punkt des Amphitheaters, mit QGIS abgegriffen:
#Y =  38632.11  # [m]
#X = 330299.67  # [m]
#Z =    184.051 # [m]
    
# Punkt im Süden der Arena, mit Mauern rundherum
#Y =  38621.5862466
#X = 330254.839613
#Z =    185.693
    
# Innenseite der westlichen Mauerecke beim südlichen Ausgang der Arena:
#Y =  38621.2864472
#X = 330267.155936
    
# Mauergruppe südwestlich der westlichen Mauerecke beim südlichen Ausgang der Arena:
#Y =  38610.6080974
#X = 330258.13554
    
# Mauergruppe östlich des südlichen Ausgangs der Arena
#Y =  38656.4687995
#X = 330262.687889
    
#Y =  38655.3728636
#X = 330264.064834
    
#Y =  38663.8312406
#X = 330252.459154
    
#Y =  38603.4142618
#X = 330310.96527

#ctrSoll_rc = np.array( [1859, 3022] ) # row/col des Zentrums des Amphitheaters, im entzerrten Luftbild, manuell bestimmt.
#ctrSoll_rc = np.array( [2363, 3194] ) # row/col des Punktes im Süden des Amphitheaters, im entzerrten Luftbild, manuell bestimmt.
#ctrSoll_rc = np.array([ 2262, 3121 ]) # row/col der inneren Mauerecke auf der Westseite des südl. Ausgangs der Arena, manuell bestimmt.
#ctrSoll_rc = np.array([ 2402, 3066 ]) # row/col der Mauergruppe südwestlich der inneren Mauerecke auf der Westseite des südl. Ausgangs der Arena, manuell bestimmt.
        
#ctrSoll_rc = np.array([ 2223, 2529 ]) # row/col der Mauergruppe östlich des südlichen Ausgangs der Arena im entzerrten UAV-Bild 3240531_DxO_no_dist_corr
#ctrSoll_rc = np.array([ 2304, 2599 ])  
#ctrSoll_rc = np.array([ 1662, 2058 ]) # 3240531_DxO_no_dist_corr
#ctrSoll_rc = np.array([ 843, 1685 ]) # 3240658_DxO_no_dist_corr
#ctrSoll_rc = np.array([ 1866, 2853 ]) # 3240662_DxO_no_dist_corr

import os
from collections import namedtuple
import numpy as np
import numpy.ma as ma
from scipy import linalg
import sqlite3.dbapi2 as db
from osgeo import gdal
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()
import cv2

from contracts import contract

from oriental import config, ori

from .lsm import lsm, ccoeffNormed, ccoeffWeighted, plotWithDiffAndProduct


def correlateImgs( imgRects, normalPtcl, fnOrtho, dbFn, plot=False ):
    if plot:
        plotDir = r"H:\140316_ISPRS_Comm_V_Gardasee\04 paper\figures_work"
        import oriental.utils.pyplot as plt
        from ..utils.BlockingKernelManager import client
        client.shell_channel.execute("matplotlib.rcParams['font.size'] = 15")



    # UAV
    UavParams = namedtuple('UavParams', [ 'fn', 'Y', 'X', 'flyingHeight', 'ctrSoll_rc', 'azimuth_gon' ] )
    
    uavParamsList = [ #UavParams( '3240662_DxO_no_dist_corr', 38603.4142618, 330310.96527, 75., np.array([ 1866, 2853 ]), 237. ),
                      UavParams( '3240662_DxO_no_dist_corr', 38603.4142618, 330310.96527, 75., np.array([ 1831, 2784 ]), 237. ), # good
                      UavParams( '3240525_DxO_no_dist_corr', 38655.3728636, 330264.06483, 67., np.array([ 2365, 2328 ]), 210. ), # good
                      UavParams( '3240543_DxO_no_dist_corr', 38610.6080974, 330258.13554, 70., np.array([ 1615, 3561 ]), 200. ), # good
                      UavParams( '3240662_DxO_no_dist_corr', 38632.11     , 330299.67   , 75., np.array([ 2233, 1083 ]), 237. ), # false location of cv.matchTemplate(.).max()
                      UavParams( '3240512_DxO_no_dist_corr', 38658.2507165, 330292.803333, 65., np.array([ 1719, 2627 ]), 210. ), # good
                      UavParams( '3240512_DxO_no_dist_corr', 38658.2507165, 330292.803333, 75., np.array([ 1719, 2627 ]), 210. ), # flying height wrong by 10m. LSM converges on iter #123 to right pos, but scaleRatio=1.4
                      UavParams( '3240512_DxO_no_dist_corr', 38658.2507165, 330292.803333, 75., np.array([ 1719, 2627 ]), 220. )  # azimuth wrong by 10g.
                    ]

    dsOrtho = gdal.Open( fnOrtho, gdal.GA_ReadOnly )
    ortho2wrld = dsOrtho.GetGeoTransform()
    #luCorner_luPixel_wrld = gdal.ApplyGeoTransform( ortho2wrld, 0, 0 )
    #rlCorner_rlPixel_wrld = gdal.ApplyGeoTransform( ortho2wrld, dsOrtho.RasterXSize, dsOrtho.RasterYSize )
    okay,wrld2ortho = gdal.InvGeoTransform( ortho2wrld )
    assert okay==1, "inversion of transform failed"

    # ReadAsArray() returns an array with shape (depth,nRows,nCols)
    ortho = np.rollaxis( dsOrtho.ReadAsArray(), 0, 3 )
    if False:
        plt.figure(1, tight_layout=True); plt.clf()
        # imshow sets the axes to make x increase to the right (as default), and y increase downwards
        plt.imshow( ortho, interpolation='nearest' )
        plt.title('whole ortho')
        # matplotlib's raster coo.sys. has it's origin in the center of the top/left pixel!
        #plt.scatter( 0, 0, marker="o", color='r' )
    
        cOrtho,rOrtho = gdal.ApplyGeoTransform( wrld2ortho, uavParams.Y, uavParams.X )
        # PixelIsArea! -> beziehe die Bildkoordinaten auf den Mittelpunkt des linken/oberen Pixels, statt auf dessen linke/obere Ecke!
        cOrtho -= .5
        rOrtho -= .5
            
        plt.scatter( x=cOrtho, y=rOrtho, marker="o", color='r' )
    
    for iUavParams in range(len(uavParamsList)):
        print( "-------------\nUavParams #{}".format(iUavParams) )
        uavParams = uavParamsList[iUavParams]


        # Seitenlänge des quadratischen Suchfensters im Objektraum definieren:
        searchWinSideLen_m = 20.

        # extract template
        # Annahme: Orthophoto is axis-aligned zu Welt-KS
        #tmpl_lu_wrl = ( Y - searchWinSideLen_m/2., X + searchWinSideLen_m/2. )
        #tmpl_rl_wrl = ( Y + searchWinSideLen_m/2., X - searchWinSideLen_m/2. )
        tmpl_ctr_wrl = ( uavParams.Y, uavParams.X )
        tmpl_lm_wrl = ( uavParams.Y - searchWinSideLen_m/2., uavParams.X ) # point on the left edge, same column as tmpl_ctr_wrl, 'left-middle'

        # beziehe die Bildkoordinaten auf den Mittelpunkt des linken/oberen Pixels -> - 0.5
        # runde auf ganze Pixel
        # Koordinatenreihenfolge: row/col statt col/row -> reversed
        tmpl_ctr_px_rc = np.array( [ round( el - .5 ) for el in reversed( gdal.ApplyGeoTransform( wrld2ortho, *tmpl_ctr_wrl ) ) ], dtype=np.int )
        tmpl_lm_px_rc  = np.array( [ round( el - .5 ) for el in reversed( gdal.ApplyGeoTransform( wrld2ortho, *tmpl_lm_wrl  ) ) ], dtype=np.int )
        assert tmpl_ctr_px_rc[0] == tmpl_lm_px_rc[0]
        tmpl_halfSearchWinSideLen_px = tmpl_ctr_px_rc[1] - tmpl_lm_px_rc[1]
    
        # the original tmpl_ctr_wrl, moved to the nearest pixel center
        #tmpl_ctr_wrl = gdal.ApplyGeoTransform( ortho2wrld, *( tmpl_ctr_px_rc[::-1] + .5 ) )
        
        tmpl_lu_px_rc = tmpl_ctr_px_rc - tmpl_halfSearchWinSideLen_px
        tmpl_rl_px_rc = tmpl_ctr_px_rc + tmpl_halfSearchWinSideLen_px # inclusive!

        # check if ortho wholly contains tmpl_lu_px_rc, tmpl_rl_px_rc
        if np.any( tmpl_lu_px_rc < [ -.5, -.5 ] ) or \
           np.any( tmpl_rl_px_rc > np.array(ortho.shape[:2]) -.5 ):
            raise Exception("Not implemented: template extends outside orthophoto")
    
        tmpl = ortho[ tmpl_lu_px_rc[0]:tmpl_rl_px_rc[0]+1,
                      tmpl_lu_px_rc[1]:tmpl_rl_px_rc[1]+1, :]

        grayScale = True # correlate RGB or grayscale images?
        if grayScale:
            tmpl=cv2.cvtColor( tmpl, cv2.COLOR_RGB2GRAY )

        if plot:
            plt.figure(2, tight_layout=True); plt.clf()
            plt.imshow( tmpl, interpolation='nearest', cmap='gray' if grayScale else None )
            plt.title('OPM')
            plt.savefig( os.path.join( plotDir, str(iUavParams) + '_' + uavParams.fn + "_ortho_detail.pdf" ), bbox_inches='tight', transparent=True )
    
        iRect = [ idx for idx in range(len(imgRects)) if os.path.splitext( os.path.basename( imgRects[idx].path ) )[0] == uavParams.fn ][0]
        imgRect = imgRects[iRect]

        # Ungefähren Maßstab abschätzen aus IOR und Flughöhe
        # Über den Normalvektor der Punktwolke wissen wir zwar ca. die Rotation des phos relativ zum Boden,
        # nicht aber die Position des Suchfensters im Bild.
        # Im (schrägen) Luftbild variiert der Bildmaßstab je nach Position.
        # Nicht aber im entzerrten Bild!
        # -> einfach Suchfenster-Seitenlänge skalieren mit Verhältnis Flughöhe zu Brennweite des entzerrten Bilds!
        rect_focal = imgRect.K[0,0]
        #flyingHeight = 70.

        rect_halfSearchWinSideLen_px = int( round( searchWinSideLen_m / 2. / uavParams.flyingHeight * rect_focal ) )  
    
        dsRect = gdal.Open( imgRect.path, gdal.GA_ReadOnly )
        rect = np.rollaxis( dsRect.ReadAsArray(), 0, 3 )
    
        if grayScale:
            rect=cv2.cvtColor( rect, cv2.COLOR_RGB2GRAY ) # Bildränder?
    
        if plot:
            # Achtung: das sind Koord. im entzerrten Bild!
            rect_lu_px_rc = uavParams.ctrSoll_rc - rect_halfSearchWinSideLen_px
            rect_rl_px_rc = uavParams.ctrSoll_rc + rect_halfSearchWinSideLen_px
        
            plt.figure(3, tight_layout=True); plt.clf()
            plt.imshow( rect[ rect_lu_px_rc[0] : rect_rl_px_rc[0]+1,
                              rect_lu_px_rc[1] : rect_rl_px_rc[1]+1 ],
                        interpolation='nearest', cmap='gray' if grayScale else None )
            plt.title('rect, orig')

        # for varying azimuths, we need to warp/rotate the rectified image, such that the search window will be axis-aligned
        # Achtung: es wird um die li/obere Bildecke gedreht!
        azimuth = uavParams.azimuth_gon / 200. * np.pi 
        M = np.array([ [ np.cos(azimuth), -np.sin(azimuth), 0. ],
                       [ np.sin(azimuth),  np.cos(azimuth), 0. ]] )
        # consider searchWinSideLen_px:
        # Scale the image, such that searchWinSideLen_px becomes the same size as in the template
        scale = float(tmpl_halfSearchWinSideLen_px) / rect_halfSearchWinSideLen_px
        M *= scale

        # relevant bug! http://code.opencv.org/issues/3212#note-1
        # it seems that cv2.warpAffine really only considers the minimal neighborhood in the source image,
        # as needed by the chosen interpolation method.
        # e.g. for INTER_LINEAR (bilinear interpolation),
        # it seems to use only the 4 nearest pixels in the source image.
        # For INTER_AREA, one might expect it to compute an area-weighted mean of all source pixels that are mapped onto the area of the target pixels.
        # However, it seems to use only the minimal number of neighboring pixels in the source image.
        # Unlike cv2.resize!
        # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) ? dst
        if 0:
            dsize2 = ( round( scale*rect.shape[1] ), round( scale*rect.shape[0] ) )
            rectResamp = cv2.resize(
                rect,
                #dsize=(0,0),
                #fx=scale,
                #fy=scale, 
                dsize=dsize2,
                interpolation=cv2.INTER_AREA )
            plt.figure(13, tight_layout=True); plt.clf()
            plt.imshow( rectResamp, interpolation='nearest', cmap='gray' if grayScale else None )
            plt.title('whole rect, resize.')

            M2 = np.array([ [ scale,     0., 0. ],
                            [     0., scale, 0. ] ])
            rect_warp2 = cv2.warpAffine(
                src=rect,
                M=M2,
                dsize=dsize2,
                flags=cv2.INTER_AREA ,# cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.,0.,0.,0.) )

            plt.figure(23, tight_layout=True); plt.clf()
            plt.imshow( rect_warp2, interpolation='nearest', cmap='gray' if grayScale else None )
            plt.title('whole rect, warpAffine.')

        # What's a good std.dev. for a Gaussian kernel that does anti-aliasing for the given scale, before down-sampling during cv2.warpAffine?
        # With pyrDown, OpenCV down-scales images by a factor of 2
        # and uses a Kernel of size 5x5, with a std.dev. (close to) 1.1,
        # which corresponds to the default value of sigma in getGaussianKernel, if only the kernel size is given:
        # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 -> 1.1 for ksize==5
        #sigma=1.1*0.5/scale
        # skimage.pyramid_reduce (also for scale=0.5) proposes:
        sigma = 2. / scale / 6.
        rect = cv2.GaussianBlur( rect, ksize=(0,0), sigmaX=sigma )

        if 0:
            rect_warp3 = cv2.warpAffine(
                src=rect_warp3,
                M=M2,
                dsize=dsize2,
                flags=cv2.INTER_AREA ,# cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.,0.,0.,0.) )

            plt.figure(33); plt.clf()
            plt.imshow( rect_warp3, interpolation='nearest', cmap='gray' if grayScale else None )
            plt.title('whole rect, gaussian warpAffine.')

        # transformiere die Ecken des Originalbildes im entzerrten Bild.
        # M: rect_warp_xy = M * [ rect_xy 1 ]
        corners_trafo = np.hstack(( imgRect.rc[:,::-1], np.ones((4,1)) )).dot(M.T)
        min_xy = np.min( corners_trafo, axis=0 )
        max_xy = np.max( corners_trafo, axis=0 )
        M[:,2] = -min_xy
        corners_trafo = np.hstack(( imgRect.rc[:,::-1], np.ones((4,1)) )).dot(M.T)
        ctrSoll_warp_rc = np.hstack(( uavParams.ctrSoll_rc[::-1], np.ones(1) )).dot(M.T)[::-1]

        # OpenCV wants image sizes as tuple(nCols,nRows) -> rect.shape[1::-1]
        #dsize = ( np.array(rect.shape[1::-1],dtype=np.float) * scale ).round().astype(np.int)
        dsize = np.ceil( max_xy - min_xy ).astype(np.int)
        rect_warp = cv2.warpAffine(
            src=rect,
            M=M,
            dsize=tuple(dsize.tolist()),
            flags=cv2.INTER_AREA ,# cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.,0.,0.,0.) )
    
        if plot:
            plt.figure(4, tight_layout=True); plt.clf()
            plt.imshow( rect_warp, interpolation='nearest', cmap='gray' if grayScale else None )
            plt.title('whole rect, transf.')
        
            rect_warp_halfSearchWinSideLen_px = int(round( float(rect_halfSearchWinSideLen_px) * scale ))
        
            rect_warp_lu_px_rc = ctrSoll_warp_rc - rect_warp_halfSearchWinSideLen_px
            rect_warp_rl_px_rc = ctrSoll_warp_rc + rect_warp_halfSearchWinSideLen_px

            plt.figure(5, tight_layout=True); plt.clf()
            plt.imshow( rect_warp[ rect_warp_lu_px_rc[0] : rect_warp_rl_px_rc[0]+1,
                                   rect_warp_lu_px_rc[1] : rect_warp_rl_px_rc[1]+1 ],
                        interpolation='nearest', cmap='gray' if grayScale else None )
            plt.title('Rectified UP')
            plt.savefig( os.path.join( plotDir, str(iUavParams) + '_' + uavParams.fn + "_detail.pdf" ), bbox_inches='tight', transparent=True )

        if 0:
            sigma = 1.5 # px
            rect_warp = cv2.GaussianBlur( rect_warp, ksize=(0,0), sigmaX=sigma )  
            tmpl      = cv2.GaussianBlur( tmpl     , ksize=(0,0), sigmaX=sigma )
                               
        # cv2.matchTemplate(image, templ, method[, result]) ? result
        # image – Image where the search is running. It must be 8-bit or 32-bit floating-point.
        # templ – Searched template. It must be not greater than the source image and have the same data type.
        # If image is W x H and templ is w x h , then result is (W-w+1) x (H-h+1)
        # In case of a color image, template summation in the numerator and each sum in the denominator is done over all of the channels and separate mean values are used for each channel.
        #   That is, the function can take a color template and a color image. The result will still be a single-channel image, which is easier to analyze.
        # cv2.matchTemplate is not parallelized!
        # Use scipy instead?
        # -> move to C++; parallelize considering memory consumption!     

        ccoeff = cv2.matchTemplate(
            image=rect_warp, # Bildränder?
            templ=tmpl, 
            method=cv2.TM_CCOEFF_NORMED
        )
        # Evtl. am einfachsten mit Math. Morphologie am Rasterbild:
        mask = np.zeros( rect_warp.shape, dtype=np.uint8 )
        mask = cv2.fillConvexPoly( mask, corners_trafo.astype(np.int), (1,1,1) ) 
        mask = cv2.erode( src=mask,
                          kernel=np.ones( tmpl.shape ),
                          borderType=cv2.BORDER_CONSTANT,
                          borderValue=(0,0,0) )
        mask = mask[ tmpl_halfSearchWinSideLen_px:-tmpl_halfSearchWinSideLen_px,
                     tmpl_halfSearchWinSideLen_px:-tmpl_halfSearchWinSideLen_px ]
    
        ccoeff = ma.masked_array( ccoeff, mask==0 )   
     
        rMax,cMax = np.unravel_index( ccoeff.argmax(), ccoeff.shape )
        print( "Max. ccoeff: {:.2f}".format( ccoeff.max() ) )

        if plot:
            maxInRect = np.array([rMax,cMax]) + tmpl_halfSearchWinSideLen_px
            rect_warp_cut = rect_warp[ maxInRect[0]-tmpl_halfSearchWinSideLen_px : maxInRect[0]+tmpl_halfSearchWinSideLen_px+1,
                                       maxInRect[1]-tmpl_halfSearchWinSideLen_px : maxInRect[1]+tmpl_halfSearchWinSideLen_px+1 ]
            rTest = ccoeffNormed( tmpl, rect_warp_cut )
            plt.figure( 10, tight_layout=True ); plt.clf()
            plotWithDiffAndProduct( tmpl, rect_warp_cut )
            # Adjust brightness and contrast.

            print( "ccoeff weighted @detected: {:.2f}".format( ccoeffWeighted( tmpl, rect_warp_cut ) ) )
            rect_warp_cut = rect_warp[ ctrSoll_warp_rc[0]-tmpl_halfSearchWinSideLen_px : ctrSoll_warp_rc[0]+tmpl_halfSearchWinSideLen_px+1,
                                       ctrSoll_warp_rc[1]-tmpl_halfSearchWinSideLen_px : ctrSoll_warp_rc[1]+tmpl_halfSearchWinSideLen_px+1 ]
            print( "ccoeff weighted @ground truth: {:.2f}".format( ccoeffWeighted( tmpl, rect_warp_cut ) ) )

        # TODO: beschneiden auf gültigen Bereich: exklusive Bildränder, die durch Entzerrung entstanden sind.
        if plot:
            plt.figure(6, tight_layout=True); plt.clf()
        
            maxAbs = ccoeff.max()#np.abs( ccoeff ).max()
            plt.imshow( ccoeff, interpolation='nearest', cmap='RdBu', vmin=-maxAbs, vmax=maxAbs )
            plt.autoscale(False)
            plt.colorbar(format='%+.2f')
            #plt.scatter( x=ctrSoll_warp_rc[1]-tmpl_halfSearchWinSideLen_px,
            #             y=ctrSoll_warp_rc[0]-tmpl_halfSearchWinSideLen_px,
            #             marker='+',
            #             color='k' )
            if 0:
                xy = np.vstack(( corners_trafo,
                                 corners_trafo[0,:] )) - tmpl_halfSearchWinSideLen_px
                # xy are now the corners of the rectified image, in the coo.sys. of ccoeff.
                plt.plot( xy[:,0],
                          xy[:,1],
                          '-k' )
            # However, that region contains along its borders areas that are affected by 
            # the undefined image content outside of the original (unrectified) image.
            # We are thus interested in the location of those corner points,
            # offset towards their interior by tmpl_halfSearchWinSideLen_px
            # Howto compute those offset points? OpenCV does not provide that. OGR -> GEOS does.
            # Anyway, it seems easier to offset the corner points of the unrectified image
            # in the coo.sys of the unrectified image, and then transform those points:
            # For that, we'd either need the extent of the unrectified aerial image,
            # or we transform the corners in the rectified image back into the coo.sys of the orig image.
            # TODO ...

         
            #plt.scatter( x=cMax, y=rMax, s=100, marker='x', color='g' )
            plt.scatter( x=cMax, y=rMax, s=150, marker='o', edgecolors='k', facecolors='none' )
         
            if 1: # better not in paper? 
                soll_xy = ctrSoll_warp_rc[::-1] - tmpl_halfSearchWinSideLen_px
                error_m = linalg.norm( np.array( [cMax, rMax], dtype=np.float ) - soll_xy ) * ortho2wrld[1]
                print("error brute force [m]: {}".format(error_m))
                plt.scatter( *soll_xy, s=150, marker='x', color='m' )

            # Blauensteiner gewichtet Hue mit Saturation, denn:
            # je geringer die Saturation, desto schlechter definiert ist der Hue!
            # -> Korrelation des Farbtons, gewichtet mit der Saturierung?
            # Nein, denn dadurch würden z.B. die weißen Mauern des Amphitheaters gar nicht zum Ergebnis beitragen!
            # lt. Norbert ist es recht aussichtslos, ohne physikalische Modelle und ohne radiometrisch kalibrierte Kameras/Photos, Schattenbereiche zu bestimmen. 
        
            rowSum = mask.sum(axis=0)
            plt.xlim( rowSum.nonzero()[0][0]-0.5, rowSum.nonzero()[0][-1]+0.5 )
            colSum = mask.sum(axis=1)
            plt.ylim( colSum.nonzero()[0][-1]+0.5, colSum.nonzero()[0][0]-0.5 )
            plt.title('Correlation Coefficient')
            plt.savefig( os.path.join( plotDir, str(iUavParams) + '_' + uavParams.fn + "_corrcoeff.pdf" ), bbox_inches='tight' )

        # LSM
        estimateContrast = True
        inverseTrafoLsm = lsm(
            template=tmpl,
            picture=rect_warp,
            picture_shift_rc = np.array([ rMax, cMax ])+tmpl_halfSearchWinSideLen_px,
            estimateContrast=estimateContrast,
            plot=plot )
        if plot:
            plt.savefig( os.path.join( plotDir, 'lsm', str(iUavParams) + '_' + uavParams.fn + "_lsm.pdf" ), bbox_inches='tight' )

        # transform the center of the detected window to the warped UP
        # Note: inverseTrafoLsm rotates about the origin of the destination coordinate system, which is shifted by -knlHSz i.e. -2
        ctrWarp_xy = inverseTrafoLsm[:,:2].dot( np.ones(2)*(tmpl_halfSearchWinSideLen_px+2) ) + inverseTrafoLsm[:,2]
        plt.figure(4)
        plt.scatter( *ctrWarp_xy, s=150, marker='+', color='g' )

        # transform that point from the warped UP to the rectified UP
        # M is the non-inverse affine transform from rect to rect_warp
        Minv = cv2.invertAffineTransform( M )
        ctrRect_xy = Minv[:,:2].dot( ctrWarp_xy ) + Minv[:,2]
        plt.figure(13, tight_layout=True); plt.clf()
        plt.imshow( rect, cmap='gray', vmin=0, vmax=255, interpolation='nearest' )
        plt.title('whole rect, orig')
        plt.scatter( *ctrRect_xy, s=150, marker='+', color='g' )

        # transform that point from the rectified UP to the original UP
        # imgRect.H is the non-inverse homographic transform from the aerial photo to rect, thus:
        # [ x_d, y_d, w_d ].T = H.dot( [ x_s, y_s, 1 ].T )
        # with _s ... source coordinates (aerial)
        #      _d ... destination coordinates (rect)
        cond,Hinv = cv2.invert( imgRect.H )
        ctrAerial_xy = cv2.perspectiveTransform( ctrRect_xy[np.newaxis,np.newaxis,:], Hinv ).flatten()

        # intersect the observation ray with the adjusting object plane.
        origImgFn = os.path.join( os.path.dirname(os.path.dirname(imgRect.path)), os.path.basename(imgRect.path) )
        with db.connect( dbFn ) as relOri:
            x0,y0,z0 = relOri.execute("""
                SELECT cameras.x0,
                       cameras.y0,
                       cameras.z0
                FROM cameras
	                JOIN images
		                ON images.camID==cameras.id
	            WHERE images.path==?
            """, (origImgFn,) ).fetchone()

            r1,r2,r3,X0,Y0,Z0 = relOri.execute("""
                SELECT r1, r2, r3,
                       X0, Y0, Z0
                FROM images
                WHERE images.path==?
            """, (origImgFn,) ).fetchone()

        ctrCam = np.array([  ctrAerial_xy[0] - x0,
                            -ctrAerial_xy[1] - y0,
                                             - z0 ] )
        R = ori.omfika( np.array([r1, r2, r3]) )
        P0 = np.array([X0,Y0,Z0])
        r0 = R.dot( ctrCam )
        r0 /= np.linalg.norm(r0)

        n0 = normalPtcl[:3]
        d = normalPtcl[3]

        # ( P0 + k*r0 ).dot( n0 ) = d
        k = ( d - P0.dot(n0) ) / r0.dot(n0)

        ctrObj = P0 + k * r0

        # If the LSM result seems trustworthy, then select the best other image that also images our object point.
        # Compute the normal vector in our object point (or simply use the object plane).
        # Compute the homography from our UP to the other UP for the area around the object point.
        # Simplify the homography to an affine transform and pass it to LSM for the other point.
        # -> Re-forward intersect the object point based on the unchanged image point in our UP and the LSM-adjusted image point in the other UP.
        if iUavParams==0:
            otherPhoFn = os.path.join( os.path.dirname(os.path.dirname(imgRect.path)), '3240582_DxO_no_dist_corr.tif' )
            with db.connect( dbFn ) as relOri:
                x0,y0,z0 = relOri.execute("""
                    SELECT cameras.x0,
                           cameras.y0,
                           cameras.z0
                    FROM cameras
	                    JOIN images
		                    ON images.camID==cameras.id
	                WHERE images.path==?
                """, (otherPhoFn,) ).fetchone()

                r1,r2,r3,X0,Y0,Z0 = relOri.execute("""
                    SELECT r1, r2, r3,
                           X0, Y0, Z0
                    FROM images
                    WHERE images.path==?
                """, (otherPhoFn,) ).fetchone()

                ctrInOther = ori.projection( ctrObj,
                                             np.array([X0,Y0,Z0]),
                                             np.array([r1,r2,r3]),
                                             np.array([x0,y0,z0]) )

            dsOther = gdal.Open( otherPhoFn, gdal.GA_ReadOnly )
            other = np.rollaxis( dsOther.ReadAsArray(), 0, 3 )
            other=cv2.cvtColor( other, cv2.COLOR_RGB2GRAY )
            if plot:
                plt.figure( 12, tight_layout=True ); plt.clf()
                plt.imshow( other, cmap='gray', vmin=0, vmax=255, interpolation='nearest' )
                plt.scatter( ctrInOther[0,0], -ctrInOther[0,1] )
                plt.title('other')

        continue

        # transform the center of the final LSM search window
        # to the rect. UP - CS
        lsmInCcoeff = inverseTrafoLsm[:,:2].dot( np.ones(2)*tmpl_halfSearchWinSideLen_px ) + inverseTrafoLsm[:,2]
        # If image is W x H and templ is w x h , then result is (W-w+1) x (H-h+1)
        lsmInCcoeff -= tmpl_halfSearchWinSideLen_px
        # plot the resulting central position of the final LSM search window into the figure of correlation coefficients.
        plt.figure(6)
        plt.scatter( *lsmInCcoeff, s=150, marker='+', color='g' )
        error_m = linalg.norm( lsmInCcoeff - soll_xy ) * ortho2wrld[1]
        print("error LSM [m]: {}".format(error_m))

        # M is the (non-inverse) affine transformation from
        # rect (the rectified UP) to rect_warp (the picture passed to LSM and cv2.matchTemplate,
        # which is rotated, scaled and offset according to our 'brute-force-search' approx. parameters)

        # inverseTrafoLsm is the inverse affine transformation from
        # the picture passed to LSM to the final search window of LSM
        M2 = cv2.invertAffineTransform( inverseTrafoLsm )

        R1 = M [:,:2]
        R2 = M2[:,:2]
        t1 = M [:,2]
        t2 = M2[:,2]
        combinedTrafo = np.column_stack((
            R2.dot(R1),
            R2.dot(t1)+t2 ))

        # halve the size of the template,
        # resample the rectified UP again,
        # and run LSM again
        template_halfSearchWinSideLen_px = int( tmpl_halfSearchWinSideLen_px / 2 )
        template_lu_px_rc = tmpl_ctr_px_rc - template_halfSearchWinSideLen_px
        template_rl_px_rc = tmpl_ctr_px_rc + template_halfSearchWinSideLen_px # inclusive!
        template = ortho[ template_lu_px_rc[0]:template_rl_px_rc[0]+1,
                          template_lu_px_rc[1]:template_rl_px_rc[1]+1, 0]

        cut = cv2.warpAffine(
            src=rect,
            M=combinedTrafo,
            dsize=tmpl.shape,
            flags=cv2.INTER_AREA #| cv2.WARP_INVERSE_MAP
        )
        if 0:
            plt.figure(300, tight_layout=True); plt.clf()
            plt.imshow( cut, interpolation='nearest', cmap='gray' )
            plt.autoscale(False)
            plt.scatter( x=(cut.shape[1]-1)/2, y=(cut.shape[0]-1)/2, s=150, marker='o', edgecolors='k', facecolors='none' )
            plt.title('picture 2')

        inverseTrafoLsm2 = lsm(
            template=template,
            picture=cut,
            picture_shift_rc = np.ones(2)*tmpl_halfSearchWinSideLen_px,
            estimateContrast=estimateContrast,
            plot=plot )

        # transform the center of the final LSM search window
        # to the rect. UP - CS
        lsmInCcoeff = inverseTrafoLsm2[:,:2].dot( np.ones(2)*template_halfSearchWinSideLen_px ) + inverseTrafoLsm2[:,2]
        lsmInCcoeff = inverseTrafoLsm[:,:2].dot( lsmInCcoeff ) + inverseTrafoLsm[:,2]
        # If image is W x H and templ is w x h , then result is (W-w+1) x (H-h+1)
        lsmInCcoeff -= tmpl_halfSearchWinSideLen_px

        if plot:
            # plot the resulting central position of the final LSM search window into the figure of correlation coefficients.
            # note: for 3240662_DxO_no_dist_corr, the second call to LSM results in x0 being moved by 6px. However, the central point stays almost where it was!
            plt.figure(6)
            plt.scatter( *lsmInCcoeff, s=150, marker='x', color='g' )
        dummy=1