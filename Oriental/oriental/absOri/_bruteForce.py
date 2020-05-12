# -*- coding: cp1252 -*-

from contracts import contract, new_contract
import numpy as np
import numpy.ma as ma
import cv2

from oriental.utils import traitlets

from .lsm import ccoeffNormed, ccoeffWeighted, plotWithDiffAndProduct

class BruteForceResult(traitlets.HasStrictTraits):
    rect_xy    = traitlets.NDArray( dtype=np.float, shape=(2,) )
    M          = traitlets.NDArray( dtype=np.float, shape=(2,3) )
    r          = traitlets.CFloat()
    r2nd       = traitlets.CFloat()
    rw         = traitlets.CFloat()
    error_m    = traitlets.CFloat()
    error_type = traitlets.Unicode()

new_contract('BruteForceResult', BruteForceResult )

@contract
def bruteForce( rect : 'array[AxB]',
                tmpl : 'array[CxC]',
                imgRect,
                azimuth_gon : float,
                tmpl_halfSearchWinSideLen_px : int,
                rect_halfSearchWinSideLen_px : int,
                tmplCtrAP_rect : 'bool|array[2](float)',
                method = cv2.TM_CCOEFF_NORMED,
                plot : str = '' ) -> 'tuple(BruteForceResult,array[DxE],array[2](>-0.5))':
    if plot:
        import oriental.utils.pyplot as plt
        plotDir = r'D:\arap\data\Carnuntum_UAS_Geert\moscow'
        from os import path

    result = BruteForceResult()
    azimuth = azimuth_gon * np.pi / 200.
    #azimuth = azimuths[iAzimuth] / 200. * np.pi 
    result.M = np.array([ [ np.cos(azimuth), -np.sin(azimuth), 0. ],
                          [ np.sin(azimuth),  np.cos(azimuth), 0. ]] )
    # consider searchWinSideLen_px:
    # Scale the image, such that searchWinSideLen_px becomes the same size as in the template
    scale = float(tmpl_halfSearchWinSideLen_px) / rect_halfSearchWinSideLen_px
    result.M *= scale

    # relevant bug! http://code.opencv.org/issues/3212#note-1
    # it seems that cv2.warpAffine really only considers the minimal neighborhood in the source image,
    # as needed by the chosen interpolation method.
    # e.g. for INTER_LINEAR (bilinear interpolation),
    # it seems to use only the 4 nearest pixels in the source image.
    # For INTER_AREA, one might expect it to compute an area-weighted mean of all source pixels that are mapped onto the area of the target pixels.
    # However, it seems to use only the minimal number of neighboring pixels in the source image.
    # Unlike cv2.resize!
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) ? dst

    # What's a good std.dev. for a Gaussian kernel that does anti-aliasing for the given scale, before down-sampling during cv2.warpAffine?
    # With pyrDown, OpenCV down-scales images by a factor of 2
    # and uses a Kernel of size 5x5, with a std.dev. (close to) 1.1,
    # which corresponds to the default value of sigma in getGaussianKernel, if only the kernel size is given:
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 -> 1.1 for ksize==5
    #sigma=1.1*0.5/scale
    # skimage.pyramid_reduce (also for scale=0.5) proposes:
    sigma = 2. / scale / 6.
    # TODO: it's more CPU- (but less RAM-) efficient to pre-compute a pyramid of blurred images.
    rect_blurred = cv2.GaussianBlur( rect, ksize=(0,0), sigmaX=sigma )

    # transformiere die Ecken des Originalbildes im entzerrten Bild.
    # M: rect_warp_xy = M * [ rect_xy 1 ]
    corners_trafo = np.hstack(( imgRect.rc[:,::-1], np.ones((4,1)) )).dot(result.M.T)
    min_xy = np.min( corners_trafo, axis=0 )
    max_xy = np.max( corners_trafo, axis=0 )
    result.M[:,2] = -min_xy
    corners_trafo = np.hstack(( imgRect.rc[:,::-1], np.ones((4,1)) )).dot(result.M.T)

    # OpenCV wants image sizes as tuple(nCols,nRows) -> rect.shape[1::-1]
    #dsize = ( np.array(rect.shape[1::-1],dtype=np.float) * scale ).round().astype(np.int)
    dsize = np.ceil( max_xy - min_xy ).astype(np.int)
    rect_warp = cv2.warpAffine(
        src=rect_blurred,
        M=result.M,
        dsize=tuple(dsize.tolist()),
        flags=cv2.INTER_AREA ,# cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.,0.,0.,0.) )

    if plot:
        cv2.imwrite( path.join( plotDir, plot + '_rect_warp.png'), rect_warp )
        cv2.imwrite( path.join( plotDir, plot + '_opm.png' ), tmpl )

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
        image=rect_warp,
        templ=tmpl, 
        method=method
    )
    if method in ( cv2.TM_CCORR, cv2.TM_SQDIFF ):
        # we expect that rect_warp and tmpl have been reduced to their mean and normalized to their std.dev. beforehand. Hence, the Pearson ccoeff is:
        ccoeff /= tmpl.size - 1

    # Bildränder: evtl. am einfachsten mit Math. Morphologie am Rasterbild:
    mask = np.zeros( rect_warp.shape, dtype=np.uint8 )
    mask = cv2.fillConvexPoly( mask, corners_trafo.astype(np.int), (1,1,1) ) 
    mask = cv2.erode( src=mask,
                      kernel=np.ones( tmpl.shape ),
                      borderType=cv2.BORDER_CONSTANT,
                      borderValue=(0,0,0) )
    mask = mask[ tmpl_halfSearchWinSideLen_px:-tmpl_halfSearchWinSideLen_px,
                 tmpl_halfSearchWinSideLen_px:-tmpl_halfSearchWinSideLen_px ]
    
    ccoeff = ma.masked_array( ccoeff, mask==0 )   
    result.r = ccoeff.max()
    print( "Max. ccoeff: {:.2f}".format( result.r ) )

    rMax,cMax = np.unravel_index( ccoeff.argmax(), ccoeff.shape )

    # check if (rMax,cMax) is far enough away from the mask
    # 'far enough' depends on the kernel size used by LSM, as we need a margin of resp. width for computing derivatives
    # additionally, reserve 1px (arbitrarily)
    margin = 2 + 1
    loc_xy_bf = np.array([cMax,rMax])
    if np.any( loc_xy_bf-margin      < -.5 ) or \
       np.any( loc_xy_bf+margin - .5 > tuple(reversed(ccoeff.shape)) ) or \
       np.any( ccoeff.mask[ rMax-margin : rMax+margin+1,
                            cMax-margin : cMax+margin+1 ] ):
        print( "bf max is within a margin of {} to the border".format(margin) )
        result.error_type = 'bf-outside'
        return result,rect_warp,np.zeros(2)
                        
    warp_rc = np.array([rMax,cMax]) + tmpl_halfSearchWinSideLen_px
    rect_warp_cut = rect_warp[ warp_rc[0]-tmpl_halfSearchWinSideLen_px : warp_rc[0]+tmpl_halfSearchWinSideLen_px+1,
                               warp_rc[1]-tmpl_halfSearchWinSideLen_px : warp_rc[1]+tmpl_halfSearchWinSideLen_px+1 ]
    result.rw = ccoeffWeighted( tmpl, rect_warp_cut )
    print( "ccoeff weighted @detected: {:.2f}".format( result.rw ) )
    if plot:
        rTest = ccoeffNormed( tmpl, rect_warp_cut )
        plt.figure( 10, tight_layout=True ); plt.clf()
        plotWithDiffAndProduct( tmpl, rect_warp_cut )
        plt.savefig( path.join( plotDir, plot + '_corresp.jpg' ), bbox_inches='tight', dpi=150 )
        # Adjust brightness and contrast.

        #rect_warp_cut = rect_warp[ ctrSoll_warp_rc[0]-tmpl_halfSearchWinSideLen_px : ctrSoll_warp_rc[0]+tmpl_halfSearchWinSideLen_px+1,
        #                           ctrSoll_warp_rc[1]-tmpl_halfSearchWinSideLen_px : ctrSoll_warp_rc[1]+tmpl_halfSearchWinSideLen_px+1 ]
        #print( "ccoeff weighted @ground truth: {:.2f}".format( ccoeffWeighted( tmpl, rect_warp_cut ) ) )

    # transform from the warped AP to the rectified AP
    # M is the non-inverse affine transform from rect to rect_warp
    Minv = cv2.invertAffineTransform( result.M )
    rect_xy = Minv[:,:2].dot( np.array([cMax, rMax]) + tmpl_halfSearchWinSideLen_px ) + Minv[:,2]
    result.rect_xy = rect_xy

    if plot:
        plt.figure(6, tight_layout=True); plt.clf()
        
        maxAbs = ccoeff.max()#np.abs( ccoeff ).max()
        plt.imshow( ccoeff, cmap='RdBu', vmin=-maxAbs, vmax=maxAbs, interpolation='nearest' )
        plt.autoscale(False)
        plt.colorbar(format='%+.2f')

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

        plt.scatter( x=cMax, y=rMax, s=150, marker='o', edgecolors='m', facecolors='none' )

        if tmplCtrAP_rect is not None:
            # beschneiden auf gültigen Bereich: exklusive Bildränder, die durch Entzerrung entstanden sind.
            tmplCtrAP_rect_warp = result.M[:,:2].dot( tmplCtrAP_rect ) + result.M[:,2]
            soll_xy = tmplCtrAP_rect_warp - tmpl_halfSearchWinSideLen_px
            plt.scatter( *soll_xy, s=150, marker='x', color='g' )

        rowSum = mask.sum(axis=0)
        plt.xlim( rowSum.nonzero()[0][0]-0.5, rowSum.nonzero()[0][-1]+0.5 )
        colSum = mask.sum(axis=1)
        plt.ylim( colSum.nonzero()[0][-1]+0.5, colSum.nonzero()[0][0]-0.5 )
        plt.title('Corr.Coeff. x:true o:max +:$2^{nd}$max')
        #plt.savefig( os.path.join( plotDir, str(iUavParams) + '_' + uavParams.fn + "_corrcoeff.pdf" ), bbox_inches='tight' )

    # check: Nebenmaxima. Vielleicht ist das Verhältnis aus maximalem corrCoeff und zweitgrößtem corrCoeff ein geeignetes Maß für frühzeitiges continue?
    ccoeffFill = ccoeff.copy()
    retval = cv2.floodFill( ccoeffFill, np.empty( (0,), np.int8 ), ( cMax, rMax ), newVal=-2, loDiff=255, upDiff=0, flags=8 )
    result.r2nd = ccoeffFill.max()
    if result.r2nd == np.ma.maximum_fill_value( ccoeffFill ):
        result.r2nd = -np.inf # special case: all unmasked pixels have been filled.
    else:
        print("2nd maximum: {:.2f} is {:2.0f}%".format( result.r2nd, ( result.r2nd / result.r ) * 100. ) )

        rMax2,cMax2 = np.unravel_index( ccoeffFill.argmax(), ccoeffFill.shape )
        if plot:
            plt.figure(15); plt.clf()
            plt.imshow( ccoeffFill, cmap='RdBu', interpolation='nearest', vmin=-maxAbs, vmax=maxAbs )
            plt.colorbar(format='%+.2f')
            plt.scatter( x=cMax , y=rMax , s=150, marker='o', edgecolors='m', facecolors='none' )
            plt.scatter( x=cMax2, y=rMax2, s=150, marker='+', color='k' )
        if plot:
            plt.figure(6)
            plt.scatter( x=cMax2, y=rMax2, s=150, marker='+', color='k' )
            plt.savefig( path.join( plotDir, plot + '_ccoeffs.jpg' ), bbox_inches='tight', dpi=150 )

    return result,rect_warp,warp_rc
