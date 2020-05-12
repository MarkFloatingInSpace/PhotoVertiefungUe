# -*- coding: cp1252 -*-

import numpy as np
import cv2
# bug: missing constant
cv2.SCHARR =-1

from scipy import linalg
from contracts import contract

import oriental.utils.pyplot as plt
from oriental.utils import traitlets

@contract
def ccoeffNormed( img1 : 'array[NxM]',
                  img2 : 'array[NxM]' ):
    m1 = np.mean(img1)
    m2 = np.mean(img2)
    img1n = img1 - m1
    img2n = img2 - m2

    r =   np.sum( img1n * img2n ) \
        / ( np.sum(img1n**2) * np.sum(img2n**2) )**.5

    return r

@contract
def ccoeffWeighted( img1 : 'array[NxM]',
                    img2 : 'array[NxM]' ):
    diff = img1 - np.median(img1) # median returns float64 for integer input
    #std = ( np.sum( diff**2 ) / ( img1.size - 1 ) )**.5
    std = 1.4826 * np.median( np.abs(diff) )
    weights1 = np.pi/2. + np.arctan( diff / std )

    diff = img2 - np.median(img2)
    #std = ( np.sum( diff**2 ) / ( img1.size - 1 ) )**.5
    std = 1.4826 * np.median( np.abs(diff) )
    weights2 = np.pi/2. + np.arctan( diff / std )

    weights = ( weights1 * weights2 )**.5

    wMean1 = np.sum( weights * img1 ) / np.sum(weights)
    wMean2 = np.sum( weights * img2 ) / np.sum(weights)
    img1n = img1 - wMean1
    img2n = img2 - wMean2

    r =   np.sum( weights * img1n * img2n ) \
        / ( np.sum( weights*img1n**2) * np.sum( weights*img2n**2) )**.5

    return r

@contract
def plotWithDiffAndProduct( left  : 'array[NxM]',
                            right : 'array[NxM]' ):
    # Adjust brightness and contrast.
    imgNs = []
    for idx,img in enumerate((left,right)):
        if 1:
            imgN = img.astype(np.float) - np.median(img)
            sigma = 1.4826 * np.median( np.abs(imgN) )
        else:
            imgN = img.astype(np.float) - np.mean(img)
            sigma = np.std(imgN)
        imgN /= sigma
        imgNs.append( imgN )

        maxAbs = np.abs(imgN).max()
        plt.subplot( 2, 2, idx+1 )
        plt.imshow( imgN, interpolation='nearest', cmap='RdBu', vmin=-maxAbs, vmax=maxAbs )
        plt.grid()
        plt.title( 'OPM' if idx==0 else 'UP' )

    diff = imgNs[0] - imgNs[1]
    maxAbs = np.abs(diff).max()
    plt.subplot( 2,2,3 )
    plt.imshow( diff, interpolation='nearest', cmap='RdBu', vmin=-maxAbs, vmax=maxAbs )
    plt.grid(); plt.title('OPM - UP')
    #plt.colorbar()
    product = imgNs[0] * imgNs[1]
    maxAbs = np.abs(product).max()
    plt.subplot( 2,2,4 ); plt.imshow( product, interpolation='nearest', cmap='RdBu', vmin=-maxAbs, vmax=maxAbs )
    plt.grid(); plt.title('OPM * UP')

def setCurrentFigPos( x=50, y=50 ):
    #from ..utils.BlockingKernelManager import client
    # works for Tk. Other commands necessary for different backends. http://stackoverflow.com/questions/7802366/matplotlib-window-layout-questions
    # wm = plot.get_current_fig_manager()
    # wm.window.wm_geometry("800x900+50+50")
    # The IPython kernel must have been started with --pylab, such that plt and rcParams are available
    #cmd = 'plt.get_current_fig_manager().window.wm_geometry("{:.0f}x{:.0f}".format( *[ el * rcParams["figure.dpi"] for el in rcParams["figure.figsize"]  ] )'
    #cmd += '+"+{:.0f}+{:.0f}")'.format(x,y)
    
    #cmd = 'plt.get_current_fig_manager().window.wm_geometry("800x600+{:.0f}+{:.0f}" )'.format(x,y)
    #client.shell_channel.execute(cmd)
    pass

def plotTemplateOrCut( img, figNum, name ):
    import oriental.utils.pyplot as plt
    plt.figure(figNum, tight_layout=True); plt.clf()
    plt.imshow( img, interpolation='nearest', cmap='gray' )
    plt.autoscale(False)
    plt.scatter( x=(img.shape[1]-1)/2, y=(img.shape[0]-1)/2, s=150, marker='o', edgecolors='r', facecolors='none' )
    plt.title(name)
    plt.grid(color='r',linestyle='-')
    setCurrentFigPos()

class LSMResult(traitlets.HasStrictTraits):
    rect_xy    = traitlets.NDArray( dtype=np.float, shape=(2,) )
    M          = traitlets.NDArray( dtype=np.float, shape=(2,3) )
    niter      = traitlets.CInt()
    s0         = traitlets.CFloat()
    Rxx        = traitlets.NDArray( dtype=np.float, shape=(8,8) )
    stdDevs    = traitlets.NDArray( dtype=np.float, shape=(8,) )
    r          = traitlets.CFloat()
    rw         = traitlets.CFloat()
    error_m    = traitlets.CFloat()
    pos_obj    = traitlets.NDArray( dtype=np.float, shape=(3,) )
    error_type = traitlets.Unicode()

@contract
def lsm( template : 'array[AxB]', # A and B should be odd numbers. How to specify that?
         picture  : 'array[CxD],(C>A|C=A),(D>B|D=B)',
         picture_shift_rc : 'array[2](>-0.5)',
         estimateContrast : bool = False,
         plot : bool = False,
         weight : bool = False ) -> LSMResult:
    """Adaptive (?) Least-squares template matching
    The number of iterations may be a viable indicator of the signal strength / quality of results!
    """
    result = LSMResult()
    if 0:
        template = cv2.resize( template, dsize=tuple([ (el-1)//2+1 for el in template.shape ]) )
        picture  = cv2.resize( picture,  dsize=(0,0), fx=0.5, fy=0.5 )
        picture_shift_rc /= 2.

    # TODO: better adhere to the definitions of Lang & Förstner 1995 than to the ones of Grün.
    if plot:
        import oriental.utils.pyplot as plt
        plt.figure(101, tight_layout=True); plt.clf()
        plt.imshow( picture, interpolation='nearest', cmap='gray' )
        plt.autoscale(False)
        plt.scatter( x=picture_shift_rc[1], y=picture_shift_rc[0], s=150, marker='o', edgecolors='r', facecolors='none' )
        plt.title("whole picture")

        plotTemplateOrCut( template, 100, 'template' )

    # first-order derivatives
    # good kernel size?
    # For a kernel size of 3, the Scharr filter is recommended
    # cv2.Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) ? dst
    # TODO: Matthias Ockermüller hat Konvergenzgeschwindigkeit in Hinsicht auf Glättung der Ableitungen untersucht:
    # ist man weit von der Endposition entfernt, dann konvergiert LSM mit stark geglätteten Ableitungen schneller.
    # sonst umgekehrt.
    knlHSz = 2 # kernel half size
    shape = template.shape
    tmpl_halfSearchWinSideLen_px = ( ( np.array( shape, dtype=np.float ) - 1 ) / 2. ).round().astype(np.int)
    # For values exactly halfway between rounded decimal values, Numpy rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round to 0.0, etc.
    picture_ul = picture_shift_rc - tmpl_halfSearchWinSideLen_px
    picture_lr = picture_ul + shape # exclusive
    cut = picture[ picture_ul[0]-knlHSz:picture_lr[0]+knlHSz,
                   picture_ul[1]-knlHSz:picture_lr[1]+knlHSz ]

    if plot:
        plotTemplateOrCut( cut[knlHSz:-knlHSz,knlHSz:-knlHSz], 119, 'initial picture' )

    paramNames=['dx','mx','ax','dy','ay','my','b']

    if estimateContrast:
        x = np.zeros(8)
        x[1] = x[5] = x[6] = 1.
        paramNames = paramNames[:-1] + ['c'] + [paramNames[-1]]
    else:
        x = np.zeros(7)
        x[1] = x[5] = 1.

    for iIter in range(50):

        derivs = []
        for idx,(dx,dy,name) in enumerate( zip( (1,0),
                                                (0,1),
                                                ('x','y') ) ):
            # use the Scharr kernel, whose size is 3
            # -> pass a sub-image that is 1 row and column larger than the wanted image on top/bottom, left/right,
            # such that all wanted derivatives can be computed based on all needed neighbors.
            # Extract the wanted image area afterwards.
            # using cv2.Scharr, we get unnormalized derivatives!
            # Thus, use getDerivKernels with normalize=True -> meaningful derivatives.
            # cv2.getDerivKernels(dx, dy, ksize[, kx[, ky[, normalize[, ktype]]]]) ? kx, ky
            kx, ky = cv2.getDerivKernels(
                dx=dx, 
                dy=dy, 
                ksize=cv2.SCHARR if knlHSz==1 else 2*knlHSz+1,
                normalize=True,
                ktype=cv2.CV_64F )
            # cv2.sepFilter2D(src, ddepth, kernelX, kernelY[, dst[, anchor[, delta[, borderType]]]]) ? dst
            derivs.append(
                cv2.sepFilter2D( 
                    cut,
                    ddepth=cv2.CV_64F,
                    kernelX=kx,
                    kernelY=ky ) )

            derivs[idx] = derivs[idx][knlHSz:-knlHSz,knlHSz:-knlHSz]

            if 0:
                plt.figure(110+idx, tight_layout=True); plt.clf()
                maxAbs = abs(derivs[idx]).max()
                plt.imshow( derivs[idx], interpolation='nearest', cmap='RdBu', vmin=-maxAbs, vmax=maxAbs )
                plt.colorbar(format='%+.2f')
                plt.title("derivative {}".format(name) )
    
        cut = cut[knlHSz:-knlHSz,knlHSz:-knlHSz]

        assert cut.shape == template.shape
        if plot and False:#iIter < 10:
            plotTemplateOrCut( cut, 120+iIter, 'picture' )

        # ravel by default uses C-order (row-major). Construct c,r accordingly
        dx = np.ravel(derivs[0])
        dy = np.ravel(derivs[1])
        c = np.tile( np.arange(shape[1],dtype=np.float), shape[0] )
        r = np.repeat( np.arange(shape[0],dtype=np.float), shape[1] )
        if estimateContrast:
            A = np.column_stack(( dx, dx*c, dx*r, dy, dy*c, dy*r, np.ravel(cut), np.ones(len(dx)) ))
        else:
            A = np.column_stack(( dx, dx*c, dx*r, dy, dy*c, dy*r,                np.ones(len(dx)) ))

        l = np.ravel( template ).astype(np.float) - np.ravel( cut )

        if weight:
            med = np.median( template )
            diff = np.ravel( template ) - med
            #std = ( np.sum( diff**2 ) / ( template.size - 1 ) )**.5
            std = 1.4826 * np.median( np.abs(diff) )
            weights1 = np.pi/2. + np.arctan( diff / std )

            med = np.median( cut )
            diff = np.ravel( cut ) - med
            #std = ( np.sum( diff**2 ) / ( cut.size - 1 ) )**.5
            std = 1.4826 * np.median( np.abs(diff) )
            weights2 = np.pi/2. + np.arctan( diff / std )

            weights = ( weights1 * weights2 )**.5
            l *= weights
            A *= weights[:,np.newaxis]

        N = A.T.dot(A)
        try:
            C = linalg.cho_factor( N )
        except linalg.LinAlgError as ex:
            print(ex)
            result.error_type = 'lsm-singular'
            return result

        xhat = linalg.cho_solve( C, A.T.dot(l) )
        x += xhat
        # Note: while the grey values are non-linear in the geometric parameters (we use their 1st-order derivatives),
        # they are linear in the radiometric parameters.
        # While we apply the geometric parameters via M to derive the picture for the next iteration,
        # we don't apply the radiometric parameters.
        # Thus, x += xhat is not valid for the latter.
        if estimateContrast:
            # contrast has its actual meaning only if 1 is added.
            # Thus, contrast is plausible in the range [-1;1]: if less than -1, then that means that the image is inverted!
            x[-2:] = xhat[-2:]
        else:
            x[-1:] = xhat[-1:]
        v = A.dot(xhat) - l
        s0 = ( v.dot(v) / (shape[0]*shape[1] - A.shape[1]) )**.5
        #print( "iter {:2}: dx0/dy0: {:.2f}/{:.2f} x0/y0: {:.2f}/{:.2f} s0: {}".format(iIter,xhat[0],xhat[3],x[0],x[3],s0 ) )
        # M is a 2x3 matrix that defines the following affine transform, if WARP_INVERSE_MAP is used (otherwise, the inverse):
        # [ x_s, y_s ].T = M[:2,:2].dot( [ x_d, y_d ].T ) + M[:,2]
        # with _s ... source coordinates      i.e. rectified image
        #      _d ... destination coordinates i.e. warped image
        # i.e. the rotation is done around the origin of the destination system, and the offset is defined in the source coordinate system.
        M = np.array([ [ x[1], x[2], picture_ul[1]+x[0]-knlHSz ],
                       [ x[4], x[5], picture_ul[0]+x[3]-knlHSz ] ])
    
        # check if corners of warped image extend outside of picture.
        dsize = tuple( el+2*knlHSz for el in template.shape )
        for xy in ( (0,0), ( dsize[1], 0 ), tuple(reversed(dsize)), ( 0, dsize[0] ) ):
            xy_s = M[:2,:2].dot( xy ) + M[:,2]
            if np.any( xy_s < -0.5 ) or \
               np.any( xy_s - .5 > tuple(reversed(picture.shape)) ):
                print("Template extends outside picture")
                result.error_type = 'lsm-outside'
                return result

        # cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) ? dst
        # TODO: if scales deviate too much from 1., then we should resample picture again before calling cv2.warpAffine!
        cut = cv2.warpAffine( src=picture,
                              M=M,
                              dsize=dsize,
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
                            )
        #if np.abs(xhat[:-1]).max() < 0.01:
        if 's0_old' in locals() and (
            abs(s0 - s0_old) < 1.e-7 * s0_old or
            s0 < 1.e-13 # for synthetic images of differently offset details of the same image, s0 may be exactly 0.0!
           ):
            print( "LSM terminated after iteration #{}".format( iIter ) )
            break

        s0_old = s0
    else:
        print("LSM has reached the maximum number of iterations: {}".format( iIter+1 ) )
        result.error_type = 'lsm-maxiter'
        return result

    result.niter = iIter
    result.M = M
    result.s0 = s0

    if plot:
        plotTemplateOrCut( cut[knlHSz:-knlHSz,knlHSz:-knlHSz], 200, 'final picture' )
    
    Qxx = linalg.inv(N)

    result.stdDevs = result.s0 * np.diag(Qxx)**.5

    if plot:
        for name, wert, sigma in zip( paramNames, x, result.stdDevs ):
            print( "{:>3} {:=+8.4f} {:5.3f} {:4.0f}".format( name, wert, sigma, abs(wert)/sigma ) )

    # check correlations.
    result.Rxx = np.zeros(Qxx.shape)
    for r in range(result.Rxx.shape[0]):
        for c in range(r,result.Rxx.shape[1]):
            result.Rxx[c,r] = result.Rxx[r,c] = \
                Qxx[r,c] / ( Qxx[r,r]*Qxx[c,c] )**.5
    if plot:
        print("Correlations [%]:")
        for name in [''] + paramNames:
            print( "{:>3} ".format(name), end='' )
        print()
        for r in range(result.Rxx.shape[0]):
            print( "{:>3} ".format(paramNames[r]), end='' )
            for c in range(result.Rxx.shape[1]):
                if r==c:
                    print( "{:>3} ".format(''), end='' )
                else:
                    print( "{:=+3.0f} ".format( result.Rxx[r,c]*100 ), end='' )
            print( "{:>3} ".format(paramNames[r]), end='' )
            print()
        for name in [''] + paramNames:
            print( "{:>3} ".format(name), end='' )
        print()
    result.r = cv2.matchTemplate(
        image=cut[knlHSz:-knlHSz,knlHSz:-knlHSz],
        templ=template, 
        method=cv2.TM_CCOEFF_NORMED
    )[0,0]
    print( "Final ccoeff: {:.2f}".format( result.r ) )
    result.rw = ccoeffWeighted( cut[knlHSz:-knlHSz,knlHSz:-knlHSz], template )
    print( "Final ccoeff weighted: {:.2f}".format( result.rw ) )

    if plot:
        plt.figure( 124, tight_layout=True ); plt.clf()
        plotWithDiffAndProduct( template, cut[knlHSz:-knlHSz,knlHSz:-knlHSz] ) 

        plt.figure(123, tight_layout=True); plt.clf()
        for idx,img in enumerate( (template,cut[knlHSz:-knlHSz,knlHSz:-knlHSz]) ):
            plt.subplot( 1, 2, idx+1 )
            plt.imshow( img, cmap='gray', interpolation='nearest', vmin=0, vmax=255 )
            plt.autoscale(False)
            #plt.scatter( x=tmpl_halfSearchWinSideLen_px[1], y=tmpl_halfSearchWinSideLen_px[0], s=150, marker='o', edgecolors='r', facecolors='none' )
            if idx==0:
                plt.title('OPM')
            else:
                plt.title('UP r={:2.0f}% r_w={:2.0f}%'.format( result.r*100., result.rw*100. ) )
            plt.xticks( np.array([ 0.5, 1.0, 1.5]) * tmpl_halfSearchWinSideLen_px[1] )
            plt.yticks( np.array([ 0.5, 1.0, 1.5]) * tmpl_halfSearchWinSideLen_px[0] )
            plt.grid(color='r',linestyle='-')
        
    # Note: M depends on the kernel half size! M rotates about the origin of the destination coordinate system, which is shifted by -knlHSz
    return result

if __name__ == '__main__':
    from osgeo import gdal

    fn = r'D:\arap\data\Carnuntum_UAS_Geert\absOri\3240438_DxO_no_dist_corr.tif'
    imgOrig = np.rollaxis( gdal.Open( fn, gdal.GA_ReadOnly ).ReadAsArray(), 0, 3 )
    imgOrig = cv2.cvtColor( imgOrig, cv2.COLOR_RGB2GRAY )
    lu = (1110,1170)
    hWid = 100
    picExp = 100 # expand the picture in every direction w.r.t the template
    shift = 10
    img1 = imgOrig[ lu[0] : lu[0]+hWid*2+1,
                    lu[1] : lu[1]+hWid*2+1 ]
    img2 = imgOrig[ lu[0]-picExp+shift : lu[0]+picExp+hWid*2+1+shift,
                    lu[1]-picExp+shift : lu[1]+picExp+hWid*2+1+shift ]
    lsm( template=img1,
         picture=img2,
         picture_shift_rc=( picExp+hWid,
                            picExp+hWid ),
         estimateContrast=False, plot=True, weight=False )
