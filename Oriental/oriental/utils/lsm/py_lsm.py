# -*- coding: cp1252 -*-
from oriental import ori, log, adjust
import oriental.adjust.cost
import oriental.adjust.loss
import oriental.ori.transform
from oriental.utils import zip_equal, traitlets

import numpy as np
import cv2
# bug: missing constant
cv2.SCHARR = -1

from scipy import linalg, ndimage


from contracts import contract

import itertools
from enum import IntEnum

GeomTrafo = IntEnum('GeomTrafo','shift rigid similar affine')
RadiometricTrafo = IntEnum('RadiometricTrafo','none constant linear', start=0)
Solver = IntEnum('Solver','cholesky qr')

logger = log.Logger(__name__)

class LsmCost( adjust.cost.AnalyticDiff ):
    def __init__( self, tpl, img, x, knlHSz, geomTrafo, radiometricTrafo ):
        self.rows, self.cols = ( np.array( el.flat, float )
                                 for el in np.mgrid[ :tpl.shape[0], :tpl.shape[1] ] )
        self.tpl = tpl
        self.tplFloatFlat = np.ravel( tpl.astype( float ) )
        self.img = img
        self.xOrig = x
        self.knlHSz = knlHSz
        self.geomTrafo = geomTrafo
        self.radiometricTrafo = radiometricTrafo
        nGeomPars = 2 # shifts
        if geomTrafo == GeomTrafo.rigid:
            nGeomPars += 1 # angle
        elif geomTrafo == GeomTrafo.similar:
            nGeomPars += 2 # angle, scale
        elif geomTrafo == GeomTrafo.affine:
            nGeomPars += 4
        numResiduals = tpl.size
        parameterSizes = nGeomPars+radiometricTrafo,
        super().__init__( numResiduals, parameterSizes )

    def Evaluate( self, parameters, residuals, jacobians ):
        x = parameters[0]
        xFull = self.xOrig.copy()#np.r_[ np.eye(2).flat, np.zeros(3), 1 ]
        if self.geomTrafo == GeomTrafo.shift:
            xFull[4:6+self.radiometricTrafo] = x
        elif self.geomTrafo == GeomTrafo.rigid:
            angle = x[0]
            sAng = np.sin(angle)
            cAng = np.cos(angle)
            M = np.array([[ cAng, -sAng ],
                          [ sAng,  cAng ]])
            xFull[:4] = M.flat
            xFull[4:6+self.radiometricTrafo] = x[1:]
        elif self.geomTrafo == GeomTrafo.similar:
            angle, scale = x[:2]
            sAng = np.sin(angle)
            cAng = np.cos(angle)
            M = scale * np.array([[ cAng, -sAng ],
                                  [ sAng,  cAng ]])
            xFull[:4] = M.flat
            xFull[4:6+self.radiometricTrafo] = x[2:]
        else:
            xFull[:6+self.radiometricTrafo] = x

        try:
            cut = _getCut( self.img, self.tpl.shape, xFull, self.knlHSz, jacobians )
        except LsmOutside:
            # return False instead of throwing, so adjust will try another time with a smaller trust region radius,
            # until reaching max_num_consecutive_invalid_steps.
            return False
        if jacobians:
            cut, derivsXY = cut

        residuals[:] = self.tplFloatFlat - cut.flat

        if jacobians and jacobians[0] is not None:
            # 1 entry in jacobians for each parameter block
            # 1 column/ndarray in each entry for each parameter
            jacobian = jacobians[0]
            # derivs are cut out of larger arrays, and hence not contiguous. np.ravel would return a copy, so use flat here.
            derX, derY = ( derivs.flat for derivs in derivsXY )
            iCol = 0
            if self.geomTrafo == GeomTrafo.rigid:
                jacobian[:,0] =   derX * ( -sAng*self.cols - cAng*self.rows ) \
                                + derY * (  cAng*self.cols - sAng*self.rows ) # dangle
                iCol += 1
            elif self.geomTrafo == GeomTrafo.similar:
                sAngColsPluscAngRows  = sAng*self.cols + cAng*self.rows
                cAngColsMinussAngRows = cAng*self.cols - sAng*self.rows
                jacobian[:,0] = scale * (   derX * -sAngColsPluscAngRows
                                          + derY *  cAngColsMinussAngRows ) # dangle
                jacobian[:,1] =   derX * cAngColsMinussAngRows \
                                + derY * sAngColsPluscAngRows #dscale
                iCol += 2
            elif self.geomTrafo == GeomTrafo.affine:
                jacobian[:,0] = derX*self.cols; jacobian[:,1] = derX*self.rows
                jacobian[:,2] = derY*self.cols; jacobian[:,3] = derY*self.rows
                iCol += 4
            jacobian[:,iCol] = derX # shiftX
            iCol += 1
            jacobian[:,iCol] = derY # shiftY
            iCol += 1
            if self.radiometricTrafo >= RadiometricTrafo.constant:
                jacobian[:,iCol] = 1 # brightness
                iCol += 1
            if self.radiometricTrafo >= RadiometricTrafo.linear:
                jacobian[:,iCol] = cut.flat # contrast
                iCol += 1
            # the derivatives are:
            #  d_modelled_gray_value / d_unknown
            # while we compute the residuals as:
            # observer_gray_value - modelled_gray_value
            # Thus, invert the signs
            jacobian *= -1
        return True

class IterationCallback( adjust.IterationCallback ):

    def __init__( self ):
        self.logger = log.Logger('lsm')
        super().__init__()

    def callback( self, iterationSummary ):
        if not iterationSummary.step_is_successful:
            self.logger.info( "Iter #{0.iteration:02}: unsuccessful step. Cost:{0.cost} \N{GREEK SMALL LETTER MU}: {0.trust_region_radius}", iterationSummary )
        else:
            self.logger.info( 'Iter #{0.iteration:02} '
                                'Cost:{0.cost} '
                                '\N{GREEK CAPITAL LETTER DELTA} cost: {0.cost_change} '
                                '\N{GREEK SMALL LETTER MU}: {0.trust_region_radius:.5g} '
                                #'\N{GREEK SMALL LETTER ETA}: {0.eta} '
                                'Grad max: {0.gradient_max_norm:.9g} '
                                'Rel.decrease: {0.relative_decrease:.7g} '
                                '|\N{GREEK CAPITAL LETTER DELTA}x|_max: {0.step_max_norm:.4g}',
                                iterationSummary )
        return adjust.CallbackReturnType.SOLVER_CONTINUE

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
def plotWithDiffAndProduct( template  : 'array[NxM]',
                            picture   : 'array[NxM]' ):
    import oriental.utils.pyplot as plt
    # Adjust brightness and contrast.
    imgNs = []
    for idx,img in enumerate((template,picture)):
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
        plt.title( 'Tpl' if idx==0 else 'Pic' )

    diff = imgNs[0] - imgNs[1]
    maxAbs = np.abs(diff).max()
    plt.subplot( 2,2,3 )
    plt.imshow( diff, interpolation='nearest', cmap='RdBu', vmin=-maxAbs, vmax=maxAbs )
    plt.grid(); plt.title('Tpl - Pic')
    #plt.colorbar()
    product = imgNs[0] * imgNs[1]
    maxAbs = np.abs(product).max()
    plt.subplot( 2,2,4 ); plt.imshow( product, interpolation='nearest', cmap='RdBu', vmin=-maxAbs, vmax=maxAbs )
    plt.grid(); plt.title('Tpl * Pic')

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
    plt.imshow( img, interpolation='nearest', cmap='gray', vmin=0, vmax=255 )
    plt.autoscale(False)
    plt.scatter( x=img.shape[1]/2 -.5, y=img.shape[0]/2 -.5, s=150, marker='o', edgecolors='r', facecolors='none' )
    plt.title(name)
    plt.grid(color='r',linestyle='-')
    setCurrentFigPos()

def plotTemplateAndCut( tpl, cut, tplFullRes, cutFullRes, fignum=301 ):
    import oriental.utils.pyplot as plt
    plt.figure(fignum).clear()
    fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=True, num=fignum, tight_layout=True)
    mini = min( tpl.min(), cut.min() )
    maxi = max( tpl.max(), cut.max() )
    for name, img, ax in zip_equal( ('Tpl','Pic','TplFR','PicFR'), (tpl, cut, tplFullRes, cutFullRes), axes.flat ):
        im = ax.imshow( img, interpolation='nearest', cmap='gray', vmin=mini, vmax=maxi )
        ax.set_title( name )
        ax.grid(color='r',linestyle='-')
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.05, 0.05, .9, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

def plotDerivs( derivsXY ):
    import oriental.utils.pyplot as plt
    plt.figure(110).clear()
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=True, num=110, tight_layout=True)
    maxAbs = max( np.abs(el).max() for el in derivsXY )
    for name,derivs,ax in zip_equal( ('x','y'), derivsXY, axes ):
        im = ax.imshow( derivs, interpolation='nearest', cmap='RdBu', vmin=-maxAbs, vmax=maxAbs )
        ax.set_title( "derivative {}".format(name) )
        ax.grid(color='r',linestyle='-')
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.05, 0.05, .9, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

class LSMResult(traitlets.HasStrictTraits):
    rect_xy    = traitlets.NDArray( dtype=np.float, shape=(2,) )
    #M          = traitlets.NDArray( dtype=np.float, shape=(2,3) )
    niter      = traitlets.CInt()
    sigma0     = traitlets.CFloat()
    Rxx        = traitlets.NDArray( dtype=np.float, shape=(8,8) )
    stdDevs    = traitlets.NDArray( dtype=np.float, shape=(8,) )
    r          = traitlets.CFloat()
    rw         = traitlets.CFloat()
    error_m    = traitlets.CFloat()
    pos_obj    = traitlets.NDArray( dtype=np.float, shape=(3,) )

class LsmException(Exception):
    pass
class LsmMaxIter(LsmException):
    pass
class LsmSingular(LsmException):
    pass
class LsmOutside(LsmException):
    pass

def _getCut( image, tplShape, x, knlHSz, derivatives : bool = False ):#kernelsXY=None ):
    # M is a 2x3 matrix that defines the following affine transform, if WARP_INVERSE_MAP is used (otherwise, the inverse):
    # [ x_s, y_s ].T = M[:2,:2].dot( [ x_d, y_d ].T ) + M[:,2]
    # with _s ... source coordinates      i.e. `image`
    #      _d ... destination coordinates i.e. `cut`
    # i.e. the rotation is done about the origin of `cut`, and the offset is defined in `image`s coordinate system.

    # Pass a sub-image that is knlHSz rows and columns larger than the wanted image at top, bottom, left and right,
    # such that all wanted derivatives can be computed based on all needed neighbors.
    # Extract the wanted image area afterwards.

    # M rotates about the origin of the cut, which is extended by knlHSz
    # However, x defines a rotation about the non-extended cut.
    # Thus, we must adapt the offset vector.
    #M = np.array([ [ x[0], x[1], x[4]-knlHSz ],
    #               [ x[2], x[3], x[5]-knlHSz ] ])
    M = x[:4].reshape((2,2))
    M = np.c_[ M, M @ [-knlHSz,-knlHSz] + x[4:6] ]

    # check if corners of warped image extend outside of image.
    dsize = tuple( el+2*knlHSz for el in tplShape )
    corners = np.array( list( itertools.product( (0,dsize[1]),
                                                 (0,dsize[0]) ) ) ) - .5
    corners = ( M[:2,:2] @ corners.T ).T + M[:,2]
    if np.logical_or( corners.min(axis=0) + .5 < 0,
                      corners.max(axis=0) + .5 > image.shape[::-1] ).any():
        raise LsmOutside()

    useOpenCV = False

    # TODO: if scales deviate too much from 1., then we should blur `image` before calling cv2.warpAffine to avoid aliasing.
    # -> apply appropriate gaussian filter onto a copy of image (or template, depending on relative scale?)
    #    beforehand. Estimate the mean scale as linalg.det(M[:2,:2])
    if useOpenCV:
        # OpenCV's warpAffine's precision is limited, see: http://answers.opencv.org/question/62730/strange-warpaffine-precision-issue/
        # Thus, for slightly different transformations, warpAffine will return identical results.
        # It seems that using warpAffine, we can only get the precision of 0.02 pixel or worse.
        # This is a problem if LSM converges slowly: the cost_change will be exactly zero, and adjust.Solve will terminate prematurely!
        # Using float64 instead of float32 does not help. Seemingly, OpenCV will process with float32-precision in either case: https://github.com/Itseez/opencv/issues/4283
        cut = cv2.warpAffine( src=image,
                              M=M,
                              dsize=dsize,
                              flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP ) # INTER_LINEAR:8iter INTER_CUBIC:>20iter INTER_AREA:8iter
    else:
        # scipy.ndimage uses (rows,cols) as image coordinate system (not x,y). Thus, we need to flip M along its anti-diagonal, and reverse the order of the offset vector.
        cut = ndimage.affine_transform( input=image, matrix=M[:,:2][::-1,::-1], offset=M[:,2][::-1], output_shape=dsize, order=1 )

    # Apply radiometric correction.
    cut *= x[7]
    cut += x[6]

    if derivatives:
        kernelsXY = [ cv2.getDerivKernels(
                          dx=dx,
                          dy=dy,
                          ksize=cv2.SCHARR if knlHSz == 1 else 2*knlHSz+1,
                          normalize=True,
                          ktype=cv2.CV_32F )
                      for dx, dy in ( (1, 0), (0, 1) ) ]
        if useOpenCV:
            derivsXY = [ cv2.sepFilter2D( cut,
                                          ddepth=-1,# same as input
                                          kernelX=kernelX,
                                          kernelY=kernelY
                         )[ knlHSz:-knlHSz or None, knlHSz:-knlHSz or None ]
                         for (kernelX,kernelY) in kernelsXY ]
            # we could as well call cv2.Sobel directly, passing appropriate scales:
            #       for cv2.Scharr, pass scale=1/32
            #       for cv2.Sobel, pass scale=1/2^(ksize*2-dx-dy-2), as suggested in cv2.getDerivKernels
            #       Where ksize is the full side length of the kernel (3, 5, ...)
            #ksize = 2*knlHSz+1
            #if knlHSz == 1:
            #    scale = 1/32
            #else:
            #    scale = 1/2**(ksize*2-1-2)
            #    derivsXY2 = [ cv2.Sobel( cut,
            #                            ddepth=-1, # same as input
            #                            dx=dx,
            #                            dy=dy,
            #                            ksize = cv2.SCHARR if knlHSz==1 else ksize,
            #                            scale=scale )[ knlHSz:-knlHSz or None, knlHSz:-knlHSz or None ]
            #                 for dx, dy in ((1,0),(0,1)) ]
        else:
            # This only uses the 3x3 Sobel filter. No function for Scharr 3x3 filter, no way to use a larger Sobel filter.
            #derivsXY = [ ndimage.sobel( cut, axis=axis )[ knlHSz:-knlHSz or None, knlHSz:-knlHSz or None ]  * scale
            #             for axis in (1,0) ]
            derivsXY = []
            for axis,kernels in enumerate(kernelsXY):
                # again, note that scipy.ndimage uses the same image coordinate system as numpy i.e. (rows,cols) - and not (x,y).
                # Thus, reverse the order of the list of kernels.
                kernels = kernels[::-1]
                derRows = ndimage.correlate1d( cut, kernels[0].ravel(), axis=0 )
                derivsXY.append( ndimage.correlate1d( derRows, kernels[1].ravel(), axis=1 )[ knlHSz:-knlHSz or None, knlHSz:-knlHSz or None ] )

                #kernel2d = np.outer(*kernels)
                #assert kernel2d.sum() == 0. # we called cv2.getDerivKernels(normalize=True)
                #test = ndimage.correlate( cut, kernel2d )[ knlHSz:-knlHSz or None, knlHSz:-knlHSz or None ]
                # test == derivsXY[-1] (except for rounding errors - use float64 to reduce them)

    cut = cut[knlHSz:-knlHSz or None,knlHSz:-knlHSz or None]
    assert cut.shape == tplShape
    if derivatives:
        return cut, derivsXY
    return cut


@contract
def lsm( template : 'array[AxB]',
         image    : 'array[CxD],C>=A,D>=B',
         template2image : ori.transform.AffineTransform2D, # changed in-place
         geomTrafo : GeomTrafo = GeomTrafo.affine,
         radiometricTrafo : RadiometricTrafo = RadiometricTrafo.linear,
         solver : Solver = Solver.cholesky,
         plot : int = 0 ) -> LSMResult:
    """Least-squares template matching
    `template` is the 'observed' image, which is left unchanged.
    `image` gets transformed such that it optimally matches `template`.
    `image` may be large and will thus never be processed/copied as a whole.
    An initial affine transform may be known from a known orientation of images and a surface model.
    Is it adaptive? We adjust a fixed set of unknowns!
    The number of iterations may be a viable indicator of the signal strength / quality of results!
    Maybe think about color image matching. But that should surely be done in an appropriate color space.
    We minimize the sum of squared differences here, and hence use the normalized cross correlation coefficient as quality measure.
    Alternatively, we may minimize the sum of absolute differences, and measure quality as the sum of absolute differences.
    """
    # Kraus Band 2 C2.2.1 empfiehlt:
    # - radiometrische Anpassung im Vorhinein durchführen -> bessere numerische Stabilität, kleineres Gleichungssystem.
    # - template-Größen zw. 15 und 21 Pixeln
    # - Wiederholung von LSM mit einfacherem Transformationsmodell, falls einzelne Parameter nicht genau genug bestimmt werden können (z.B. Rotation)
    # - evtl. automat. Verkleinerung des templates, um genauere Ergebnisse zu erhalten, falls Bildinhalt dies zulässt.

    #assert all( el % 2 == 0 for img in (template,image) for el in img.shape ), 'Template and image must have even numbers of columns and rows, so we can easily create an image pyramid.'

    result = LSMResult()
    #image = cv2.GaussianBlur( image, ksize=(0,0), sigmaX=2. )
    #template = cv2.GaussianBlur( template, ksize=(0,0), sigmaX=2. )
    # TODO convert to float, so to avoid truncation errors in the image pyramid?
    # TODO if scale of template2image deviates too much from 1 then down-scale either template or image before computing the image pyramid, so to avoid aliasing in _getCut
    template = template.astype(np.float32)
    image = image.astype(np.float32)

    tmplCtr = ( np.array(template.shape[::-1],float) - 1 ) / 2.

    # Better adhere to the definitions of Lang & Förstner 1995 than to the ones of Grün.
    if plot:
        import oriental.utils.pyplot as plt
        plt.figure(101, tight_layout=True); plt.clf()
        plt.imshow( image, interpolation='nearest', cmap='gray', vmin=0, vmax=255 )
        plt.autoscale(False)
        xy = template2image.forward( tmplCtr * (1,-1) ) * (1,-1)
        plt.scatter( x=xy[0], y=xy[1], s=150, marker='o', edgecolors='r', facecolors='none' )
        plt.title("whole image")

        plt.figure(102, tight_layout=True); plt.clf()
        plt.imshow( template, interpolation='nearest', cmap='gray', vmin=0, vmax=255 )
        plt.autoscale(False)
        plt.scatter( x=tmplCtr[0], y=tmplCtr[1], s=150, marker='o', edgecolors='r', facecolors='none' )
        plt.title("whole template")

    # First-order derivatives
    # For a kernel size of 3, the Scharr filter is recommended
    # TODO: Matthias Ockermüller hat Konvergenzgeschwindigkeit in Hinsicht auf Glättung der Ableitungen untersucht:
    # ist man weit von der Endposition entfernt, dann konvergiert LSM mit stark geglätteten Ableitungen schneller, sonst umgekehrt.
    # np meint, dass sich diese Untersuchung darauf bezieht, dass nicht im transformierten image, sondern im template die Gradienten bestimmt und anschließend für die benötigten Positionen im image interpoliert werden.
    # Mit knlHSz==2 konvergiert das synthetische Bsp. in der kleinsten Pyramidenstufe nicht, sondern es oszilliert.
    # Should knlHSz be adapted to the image noise?
    # A plot of a central profile through the derivatives for a synthetic image (ellipse, hard edges) shows that
    # for kernel half sizes > 1, the derivatives are too much smoothed.
    # Instead of using a larger derivative kernel, it is probably better to smooth the image itself to cope with image noise.
    knlHSz = 1 # kernel half size

    # Work with image pyramids to increase the convergence radius.
    pyrTemplate = [template]
    pyrImage = [image]
    while 1:
        tpl = cv2.pyrDown( pyrTemplate[-1] )
        # TODO: what is a 'good' minimum resolution? 21x21px are typical.
        if np.min(tpl.shape) < knlHSz*2+9:
            break
        pyrTemplate.append( tpl )
        pyrImage   .append( cv2.pyrDown( pyrImage[-1] ) )

    # if initial shift is bad enough such that local 'hills' do not overlap, then LSM will diverge instantly.
    # TODO maybe define a preprocessing step/function that estimates a good enough initial trafo.
    # Thus, use cv2.matchTemplate initially to compute the discrete 2D cross-correlation function.
    # Use its maximum as initial shift instead of the passed shift.
    # That determines the shift with a resolution of 1px only. Thus, do it on a high-resolution pyramid level?
    # On the coarsest pyramid level, we might do an exhaustive search for a 2D-similarity transformation:
    # for all angles, for all scales: determine 2D-cross correlation function, and store its maximum.
    # While this may be seen as a preprocessing step, we better integrate that here,
    # so to avoid computing the smoothed, low-resultion image twice.
    if 0:
        ccorr = cv2.matchTemplate(
            image=pyrImage[-1],
            templ=pyrTemplate[-1],
            method=cv2.TM_CCOEFF_NORMED
        )
        # offset of upper/left pixel from pyrImage[-1] to pyrTemplate[-1]
        r, c = np.unravel_index( ccorr.argmax(),
                                 np.array(pyrImage[-1].shape) - np.array(pyrTemplate[-1].shape) + 1 )


    paramNames = 'mx', 'ax', 'ay', 'my', 'dx', 'dy', 'b', 'c'
    x = np.zeros( len(paramNames) )
    x[7] = 1.
    # Transform from OrientAL to cv image coordinates
    x[:4] = template2image.A.T.flat
    x[4:6] = template2image.t * (1,-1)
    # Down-scale the offset to the highest pyramid level.
    # We up-scale it at the start of each iteration, so we down-scale here by 1 power "too much"
    x[4:6] /= 2**len(pyrTemplate)

    if radiometricTrafo == RadiometricTrafo.linear:
        paramNames = paramNames[:-1]

    for pyrLvl in range( len(pyrTemplate)-1, -1, -1 ):
        tpl = pyrTemplate[pyrLvl]
        img = pyrImage   [pyrLvl]
        # up-scale the offset to the current pyramid level
        # TODO: the scale difference between levels is generally not exactly 2.
        x[4:6] *= 2

        sigma0 = None
        rows, cols = ( np.array( el.flat, float )
                       for el in np.mgrid[ :tpl.shape[0], :tpl.shape[1] ] )

        tplFloatFlat = np.array( tpl.flat, float )

        if 1: # use oriental.adjust
            if geomTrafo == GeomTrafo.shift:
                xLoc = x[4:6+radiometricTrafo]
            elif geomTrafo == GeomTrafo.rigid:
                M = x[:4].reshape((2,2))
                angle = ( np.arctan2( -M[0,1], M[0,0] ) + np.arctan2( M[1,0], M[1,1] ) ) / 2.
                xLoc = np.r_[ angle, x[4:6+radiometricTrafo] ]
            elif geomTrafo == GeomTrafo.similar:
                M = x[:4].reshape((2,2))
                angle = ( np.arctan2( -M[0,1], M[0,0] ) + np.arctan2( M[1,0], M[1,1] ) ) / 2.
                scale = linalg.det(M)
                xLoc = np.r_[ angle, scale, x[4:6+radiometricTrafo] ]
            elif geomTrafo == GeomTrafo.affine:
                xLoc = x[:6+radiometricTrafo]
            problem = adjust.Problem()
            # initilialize LsmCost with the full x, so we can consider fixed radiometric corrections
            cost = LsmCost( tpl, img, x, knlHSz, geomTrafo, radiometricTrafo )
            problem.AddResidualBlock( cost,
                                      adjust.loss.Trivial(),
                                      xLoc )
            solveOpts = adjust.Solver.Options()
            solveOpts.linear_solver_type = adjust.LinearSolverType.DENSE_QR
            iterationCallback = IterationCallback()
            solveOpts.callbacks.append( iterationCallback )
            solveOpts.max_num_iterations = 500
            summary = adjust.Solver.Summary()
            adjust.Solve(solveOpts, problem, summary)
            if geomTrafo == GeomTrafo.shift:
                x[4:6+radiometricTrafo] = xLoc
            elif geomTrafo == GeomTrafo.rigid:
                angle = xLoc[0]
                sAng = np.sin(angle)
                cAng = np.cos(angle)
                M = np.array([[ cAng, -sAng ],
                              [ sAng,  cAng ]])
                x[:4] = M.flat
                x[4:6+radiometricTrafo] = xLoc[1:]
            elif geomTrafo == GeomTrafo.similar:
                angle, scale = xLoc[:2]
                sAng = np.sin(angle)
                cAng = np.cos(angle)
                M = scale * np.array([[ cAng, -sAng ],
                                      [ sAng,  cAng ]])
                x[:4] = M.flat
                x[4:6+radiometricTrafo] = xLoc[2:]
            elif geomTrafo == GeomTrafo.affine:
                x[:6+radiometricTrafo] = xLoc

            if plot:
                cut, derivsXY = _getCut( img, tpl.shape, x, knlHSz, True )
                if plot > 1:
                    plotDerivs( derivsXY )
                    xFullRes = x.copy()
                    xFullRes[4:6] *= 2**pyrLvl
                    cutFullRes = _getCut( pyrImage[0], pyrTemplate[0].shape, xFullRes, knlHSz )
                    plotTemplateAndCut( tpl, cut, pyrTemplate[0], cutFullRes, 302 )
                else:
                    plotTemplateAndCut( tpl, cut )
        else:
            for iIter in range(500):
                cut, derivsXY = _getCut( img, tpl.shape, x, knlHSz, True )

                if plot:
                    def plotDerivProfiles():
                        plt.figure(105).clear()
                        fig, axes = plt.subplots(nrows=2, ncols=2, num=105, tight_layout=True, sharex=True, sharey=True)
                        for figNum,(dat,axs) in enumerate(zip((cut,tpl),axes)):
                            if not figNum:
                                ders = derivsXY
                            else:
                                ders = [ cv2.sepFilter2D( cut,
                                          ddepth=cv2.CV_32F,
                                          kernelX=kernelX,
                                          kernelY=kernelY )
                                         for (kernelX,kernelY) in kernelsXY ]
                            iRow = dat.shape[0]//2
                            axs[0].plot( dat[iRow,:], '-k' )
                            axs[0].plot( ders[0][iRow,:], '-g' )
                            axs[1].plot( dat[:,iRow], '-k' )
                            axs[1].plot( ders[1][:,iRow], '-g' )
                    #plotDerivProfiles()
                    if plot > 1:
                        plotDerivs( derivsXY )
                        xFullRes = x.copy()
                        xFullRes[4:6] *= 2**pyrLvl
                        cutFullRes = _getCut( pyrImage[0], pyrTemplate[0].shape, xFullRes, knlHSz )
                        plotTemplateAndCut( tpl, cut, pyrTemplate[0], cutFullRes, 302 )
                    else:
                        plotTemplateAndCut( tpl, cut )

                Acols = []
                derX, derY = ( derivs.flat for derivs in derivsXY )
                if geomTrafo == GeomTrafo.rigid:
                    M = x[:4].reshape((2,2))
                    angle = ( np.arctan2( -M[0,1], M[0,0] ) + np.arctan2( M[1,0], M[1,1] ) ) / 2.
                    sAng = np.sin( angle )
                    cAng = np.cos( angle )
                    Acols +=   derX * ( -sAng*cols - cAng*rows ) \
                             + derY * (  cAng*cols - sAng*rows ), # dangle
                elif geomTrafo == GeomTrafo.similar:
                    M = x[:4].reshape((2,2))
                    angle = ( np.arctan2( -M[0,1], M[0,0] ) + np.arctan2( M[1,0], M[1,1] ) ) / 2.
                    sAng = np.sin( angle )
                    cAng = np.cos( angle )
                    scale = linalg.det(M)
                    sAngColsPluscAngRows  = sAng*cols + cAng*rows
                    cAngColsMinussAngRows = cAng*cols - sAng*rows
                    Acols += scale * (   derX * -sAngColsPluscAngRows
                                       + derY *  cAngColsMinussAngRows ), # dangle
                    Acols +=  derX * cAngColsMinussAngRows \
                            + derY * sAngColsPluscAngRows, #dscale
                elif geomTrafo == GeomTrafo.affine:
                    Acols += derX*cols, derX*rows, \
                             derY*cols, derY*rows
                Acols += derX, # shiftX
                Acols += derY, # shiftY

                if radiometricTrafo >= RadiometricTrafo.constant:
                    Acols += np.ones(cut.size), # brightness
                if radiometricTrafo >= RadiometricTrafo.linear:
                    Acols += cut.flat, # contrast

                # avoid intermediate copies of flatiter->ndarray
                A = np.fromiter( itertools.chain(*Acols),
                                 dtype=float,
                                 count=len(Acols)*tpl.size
                               ).reshape( (-1,len(Acols)), order='F' )

                l = tplFloatFlat - cut.flat

                # TODO we might give central pixels a larger weight than pixels at the border of the search window.
                # Thus, perspective distortion, which is typically larger at the outer image areas, will have a smaller impact.
                # However, such weighting may make it harder to estimate scales.

                if solver == Solver.cholesky:
                    N = A.T @ A
                    try:
                        C = linalg.cho_factor( N )
                    except linalg.LinAlgError:
                        raise LsmSingular()

                    xhat = linalg.cho_solve( C, A.T @ l )
                elif solver == Solver.qr:
                    # Solution with QR decomposition does not square the condition number like the Cholesky solution does.
                    # Thus, this solution is numerically stabler. But it is also slower.
                    # http://www.netlib.org/lapack/lug/node42.html http://www.math.utah.edu/~pa/6610/20130927.pdf
                    try:
                        #Q, R = linalg.qr( A, mode='economic' )
                        Q, R, P = linalg.qr( A, mode='economic', pivoting=True )
                    except linalg.LinAlgError:
                        raise LsmSingular()
                    # with mode='economic', we don't need that:
                    #Q = Q[:,:A.shape[1]]
                    #R = R[:A.shape[1],:]
                    try:
                        #xhat = linalg.solve_triangular( R, Q.T @ l ) # without column pivoting
                        xhat = np.zeros(A.shape[1]) # with column pivoting
                        xhat[P] = linalg.solve_triangular( R, Q.T @ l )
                    except linalg.LinAlgError:
                        raise LsmSingular()

                if geomTrafo == GeomTrafo.shift:
                    x[4:6+radiometricTrafo] += xhat
                elif geomTrafo == GeomTrafo.rigid:
                    angle = xhat[0]
                    sAng = np.sin(angle)
                    cAng = np.cos(angle)
                    M = np.array([[ cAng, -sAng ],
                                  [ sAng,  cAng ]])
                    x[:4] = ( M @ x[:4].reshape((2,2)) ).flat
                    x[4:6+radiometricTrafo] += xhat[1:]
                elif geomTrafo == GeomTrafo.similar:
                    angle, scale = xhat[:2]
                    sAng = np.sin(angle)
                    cAng = np.cos(angle)
                    M = (1+scale) * np.array([[ cAng, -sAng ],
                                              [ sAng,  cAng ]])
                    x[:4] = ( M @ x[:4].reshape((2,2)) ).flat
                    x[4:6+radiometricTrafo] += xhat[2:]
                else:
                    x[:6+radiometricTrafo] += xhat

                v = A @ xhat - l
                vTv = v @ v
                lTl = l @ l
                #print( "iter {:2}: dx0/dy0: {:.2f}/{:.2f} x0/y0: {:.2f}/{:.2f} s0: {}".format(iIter,xhat[0],xhat[3],x[0],x[3],sigma0_ ) )
                #print( 'it{} x {}'.format( iIter, ' '.join('{:.2f}'.format(el) for el in x) ))
                # divide vTv by n, so the output is comparable between different pyramid levels
                print( 'it{:02} {} vTv/n {}'.format( iIter, ' '.join('{:+.2f}'.format(el) for el in xhat), vTv/l.size ))

                if vTv > lTl:
                    logger.warning('Solution is oscillating. Smooth image? Enlarge derivative kernel size?')
                    break
                if np.abs( lTl - vTv ) < 1.e-3 * lTl or \
                   lTl < 1.e-13:
                    print( "LSM @ pyr {} terminated after iteration #{}".format( pyrLvl, iIter ) )
                    break
            else:
                raise LsmMaxIter()

    # Transform back from cv to OrientAL image coordinates
    template2image.A.T.flat = x[:4]
    template2image.t = x[4:6] * (1, -1)

    sigma0 = ( vTv / (tpl.size - A.shape[1]) )**.5
    result.niter = iIter
    #result.M = M
    result.sigma0 = sigma0

    cut = _getCut( image, template.shape, x, knlHSz, False )
    if plot:
        plotTemplateOrCut( cut, 199, 'final image' )

    Qxx = linalg.cho_solve( C, np.eye( len(N) ) )

    result.stdDevs = result.sigma0 * np.diag(Qxx)**.5

    if plot:
        for name, wert, sigma in zip( paramNames, x, result.stdDevs ):
            print( "{:>3} {:=+8.4f} {:5.3f} {:4.0f}".format( name, wert, sigma, abs(wert)/sigma ) )

    # check correlations.
    sqrtDiagQxx = Qxx.diagonal() ** .5
    result.Rxx = ( Qxx / sqrtDiagQxx ).T / sqrtDiagQxx
    if plot:
        print('Correlations [%]:')
        print( ' '.join( '{:>3}'.format(el) for el in ('',) + paramNames ) )
        for r,row in enumerate(result.Rxx):
            paramName = '{:>3}'.format(paramNames[r])
            print( ' '.join( itertools.chain( [paramName],
                                              ( (' '*3 if r==c else '{:=+3.0f}'.format(result.Rxx[r,c]*100) ) for c in range(len(row)) ) ) ) )
    assert cut.shape == template.shape, 'we extract only 1 pixel below'
    result.r = cv2.matchTemplate(
        image=cut.astype(image.dtype),
        templ=template, 
        method=cv2.TM_CCOEFF_NORMED
    )[0,0]
    print( 'Final ccoeff: {:.2f}'.format( result.r ) )
    #result.rw = ccoeffWeighted( cut, template )
    #print( 'Final ccoeff weighted: {:.2f}'.format( result.rw ) )

    if plot:
        plt.figure( 300, tight_layout=True ); plt.clf()
        plotWithDiffAndProduct( template, cut )
        plotTemplateAndCut( template, cut )

    # return radiometric corrections?

    return result

if __name__ == '__main__':
    # TODO move to tests
    # TODO test with synthetic image showing a small circle, which proves convergence, even if only very local derivatives are present (coarse-to-fine).
    # TODO tests with rotated, synthetic image
    from oriental.utils import gdal
    import oriental.utils.pyplot as plt

    fn = r'D:\arap\data\Carnuntum_UAS_Geert\absOri\3240438_DxO_no_dist_corr.tif'
    imgOrig = gdal.imread( fn, bands=gdal.Bands.grey )
    hWid = 100  # template half width
    picExp = 100  # expand the image w.r.t the template in every direction
    lu = 1110, 1170
    ctr_rc = np.array(lu) + hWid -.5
    ctr_ori = ctr_rc[::-1] * (1,-1)
    template = imgOrig[ lu[0]: lu[0] + hWid * 2,
                        lu[1]: lu[1] + hWid * 2 ]
    if 0:
        shift = 10
        image = imgOrig[ lu[0]-picExp+shift : lu[0]+picExp+hWid*2+shift,
                         lu[1]-picExp+shift : lu[1]+picExp+hWid*2+shift ]
        image = ( image * .9 + 3 ).astype(image.dtype)
        template2image = ori.transform.AffineTransform2D( A=np.eye(2),
                                                          t=np.array([picExp,-picExp]) )
    elif 0:
        # rotate about the center by 30gon
        A = ori.r2d( 30. )
        t = -A @ np.array([hWid+picExp,-hWid-picExp]) + ctr_ori
        image2imgOrig = ori.transform.AffineTransform2D( A = A,
                                                         t = t )
        M = np.c_[ image2imgOrig.A.T, image2imgOrig.t * (1, -1) ]
        image = cv2.warpAffine( src=imgOrig,
                                M=M,
                                dsize=(picExp*2+hWid*2+1,)*2,
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP )
        A = ori.r2d( 20. )
        template2image = ori.transform.AffineTransform2D( A = A.T,
                                                          t = A.T @ np.array([-hWid,hWid]) + np.array([hWid+picExp,-hWid-picExp]) )
    else:
        # generate a synthetic image with an ellipsoid somewhere with a hard edge (no smearing),
        # generate a template from that
        # provide an initial transform that is offset so much that the ellipsoids do not overlap
        # -> check if coarse-to-fine approach works
        image = np.zeros( (picExp*2+hWid*2+1,)*2, np.uint8 )
        image = cv2.ellipse( img=image, center=(picExp+hWid,)*2, axes=(20,10),
                             angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1, lineType=cv2.LINE_8, shift=0 )
        image = cv2.ellipse( img=image, center=(picExp+hWid+20,)*2, axes=(10,20),
                             angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1, lineType=cv2.LINE_8, shift=0 )
        template = image[ picExp : -picExp,
                          picExp : -picExp ]
        # TODO with +10 instead of +5, LSM diverges, because the signal 'hills' won't overlap
        shift = 5
        template2image = ori.transform.AffineTransform2D( A = np.eye(2),
                                                          t = np.array([picExp+shift,-picExp+shift]) )
    lsm( template=template,
         image=image,
         template2image=template2image,
         geomTrafo=GeomTrafo.rigid,
         radiometricTrafo=RadiometricTrafo.none,
         solver = Solver.qr,
         plot=2 )


    pts = np.array([[hWid,-hWid],
                    [  74,-hWid],
                    [ 140,  -80]],float)
    plt.figure(501)
    plt.imshow( template, cmap='gray', interpolation='nearest', vmin=0, vmax=255 )
    plt.autoscale( False )
    plt.scatter( pts[:,0], -pts[:,1], s=150, marker='o', edgecolors='r', facecolors='none' )
    plt.title('template')
    pts = template2image.forward(pts)
    plt.figure( 502 )
    plt.imshow( image, cmap='gray', interpolation='nearest', vmin=0, vmax=255 )
    plt.autoscale( False )
    plt.scatter( pts[:,0], -pts[:,1], s=150, marker='o', edgecolors='r', facecolors='none' )
    plt.title('image')