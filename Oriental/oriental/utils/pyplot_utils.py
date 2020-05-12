# -*- coding: utf-8 -*-

from .. import config as _config
from .. import log
from . import openCV
from . import gdal as gdal_utils
import oriental.adjust.loss

import sys
if 'oriental.utils.pyplot' in sys.modules:
    from . import pyplot as plt
else:
    import matplotlib.pyplot as plt

# Some functions only work when importing matplotlib directly, not via oriental.utils.pyplot, because we forward only matplotlib.pyplot's functions, and e.g. not mpl_toolkits
# Still, instead of importing them during each function call, do the import on the module level (speed-up).
import matplotlib as mpl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import base64, contextlib, io, os, tempfile, urllib
from pathlib import Path

from contracts import contract
import numpy as np
from scipy import stats
import cv2

# for some reason, OpenCV does not export cv2.cv any more, and there does not seem to be available a replacement for cv2.cv.RGB!
class Cv(object):
    @staticmethod
    def RGB( r, g, b ):
        return ( b, g, r, 0. )
cv2.cv = Cv()

def _figureString( **kwargs ):
    """return the current matplotlib-figure as a str- or bytes-object, depending on the format"""
    format = kwargs.pop( 'format', None )
    if 'format' is None:
        raise Exception("'format' must be specified")
    transparent = kwargs.pop( 'transparent', True )
    bbox_inches = kwargs.pop( 'bbox_inches', 'tight' )

    text = format=='svg'

    if _config.redirectedPlotting:
        file = tempfile.NamedTemporaryFile(delete=False)
        file.close()
        arg = file.name
    else:
        # doesn't work with remote plotting!
        arg = buffer = io.StringIO() if text else io.BytesIO()

    plt.savefig( arg, format=format, transparent=transparent, bbox_inches=bbox_inches, **kwargs )

    if _config.redirectedPlotting:
        # must open e.g. png-files in binary mode!
        mode = 't' if text else 'b'
        with open( file.name, 'r' + mode ) as fin:
            contents = fin.read()
        try:
            os.remove(file.name)
        except:
            pass
    else:
        buffer.seek(0)
        contents = buffer.getvalue()

    return contents

@contract
def embedSVG( **kwargs ) -> str:
    """return the SVG-string of the current matplotlib-figure"""
    kwargs['format']='svg'
    xml = _figureString( **kwargs )

    # skip xml-declaration, processing instructions, etc.
    iStart = xml.find("<svg")
    # compress consecutive white spaces (blanks, newlines, etc.) into a single blank,
    # such that the log record will be a one-liner
    return ' '.join( xml[ iStart : ].split() )

@contract
def embedPNG( **kwargs ) -> str:
    """return the PNG-string of the current matplotlib-figure"""
    kwargs['format']='png'
    png = _figureString( **kwargs )
    png = urllib.parse.quote( base64.standard_b64encode( png ) )
    return '<img src="data:image/png;base64,{}" />'.format(png)

def plotEpipolar( imgFilePath, pt1, pt2, active, which, fundamentalMatrix ):
    # openCV-doc imread: In the case of color images, the decoded images will have the channels stored in B G R order.
    # Achtung: matplotlib nimmt eine andere Reihenfolge an: R-G-B!
    # Das fällt nicht sofort auf, da in den Photos kaum rot und blau vorkommen,
    #   sondern erst, wenn rot und blau mit openCV in Photos gezeichnet wird,
    #   und dann mit matplotlib geplottet UND mit openCV gespeichert:
    # rote Linien in matplotlib-figures sind dann in den Bilddateien blau u.u.!
    # cv2.cv.RGB( 255,0,0 ) erzeugt rot!
    # am besten immer erst unmittelbar vor der Anzeige mit imshow von BGR nach RGB konvertieren mit cv2.cvtColor( img_1, cv2.COLOR_BGR2RGB )

    img = cv2.imread( imgFilePath, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_IGNORE_ORIENTATION )
    if 1:
        plt.scatter(x=pt1[active,0],y=pt1[active,1], c='b')
        plt.scatter(x=pt1[~active,0],y=pt1[~active,1], c='r')
        img  = openCV.compDrawEpipolar( pt1[active,:], pt2[active,:], which, fundamentalMatrix, img )
        img  = openCV.compDrawEpipolar( pt1[~active,:], pt2[~active,:], which, fundamentalMatrix, img, cv2.cv.RGB( 255,0,0 ) )
    else: # check für die ersten Punkte
        for i in range(3):
            plt.scatter(x=pt1[i,0],y=pt1[i,1], c='k', marker="${}$".format(i), s=100, lw=0)
        img  = openCV.compDrawEpipolar( pt1[:3,:], pt2[:3,:], which, fundamentalMatrix, img, cv2.cv.RGB( 255,255,0 ) )
    plt.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB ) ) # , interpolation='None'
    plt.xlabel('x')
    plt.ylabel('y')

@contract
def plotImageObsAndBackProjections( ptObs : 'array[Nx2](float)',
                                    ptProj: 'array[Nx2](float)',
                                    active_: 'array[N](bool)',
                                    img   : 'array[AxBxC](uint8), (C=1|C=3)',
                                    title : str = "" ):
    plt.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB ) )
    plt.autoscale(False)
    for active,color in ((active_,'b'),(~active_,'r')):
        # original measurements
        #plt.scatter(x=ptObs[:,0],y=ptObs[:,1], c='b')
        plt.scatter( x=ptObs[ active,0], y=-ptObs[ active,1], c='k', marker="o", s=4, edgecolors=color )
        # backprojections with ORIENT-transform,
        # ptProj = ori.projection( Xori, t, omfika, ior )
        plt.plot( np.vstack( ( ptProj[ active,0], ptObs[ active,0]) ),
                  np.vstack( (-ptProj[ active,1],-ptObs[ active,1]) ), color=color, linestyle='-' )
    if np.all( active_ ):
        plt.title('{} obs.(b), resid.(b)'.format(title))
    else:
        plt.title('{} obs., inlier-resid.(b), outlier-resid.(r)'.format(title))

@contract
def logHisto( logger,
              data : 'array[AxB]',
              tag : str,
              dataName : str,
              cooNames : 'seq[3](str)',
              binEdges : 'seq[C]',
              cooNamesMath : 'seq[3](str)|None' = None
            ) -> None:
    cooNamesMath = cooNamesMath or cooNames
    hists = []
    # no easy way to simply add an overflow bin at the upper edge! http://stackoverflow.com/questions/26218704/matplotlib-histogram-with-collection-bin-for-high-values
    # alternative: use numpy to compute the histogram - numpy supports np.inf as bin edge; use pyplot.bar(.) to plot the pre-computed histogram.
    maxBin = binEdges[-1]
    overflow = data.max() > maxBin
    if overflow:
        delta = maxBin - binEdges[-2]
        binEdges = np.concatenate(( binEdges, [maxBin + delta] ))
        data[data>maxBin] = maxBin + delta/2
    plt.figure('histo')
    for iCoo,(label,color) in enumerate(zip(cooNamesMath,('r','g','b'))):
        plt.subplot( 3, 1, iCoo+1 )
        if iCoo == 0:
            plt.title(tag) # there is also plt.suptitle, which seems to be meant for a figure title in case of several axes. However, plt.suptitle seems to use its own settings, in our case with a smaller font!
        hists.append( plt.hist( data[:,iCoo], bins=binEdges, label=label, color=color )[0] )
        if overflow:
            if iCoo == 0:
                locs, labels = plt.xticks()
            # for some reason, labels has twice the size as locs! As we zip locs and labels, that doesn't matter, though.
            plt.xticks( locs, [ label.get_text() if loc <= 3. else '' for loc,label in zip(locs[:-1],labels[:-1]) ] + [r'$\infty$'] )
        plt.legend()
    ymax = max(( hist.max() for hist in hists ))
    for iCoo in range(3):
        plt.subplot( 3, 1, iCoo+1 )
        plt.ylim( top = ymax )
    plt.tight_layout()
    xml = embedSVG()
    plt.close('histo')
    #logger.infoRaw(xml)

    # A vertically printed histogram is hardly readable! As the number of bins is fixed, let's print it horizontally.
    #printWidth = max( math.floor( math.log10( hist.max() ) ) + 1, 3 ) # np.inf prints to 3 chars
    #fmtCounts = '\t{:^' + str(printWidth) + '}'
    #fmtBins   = '{:^4.2f}'
    #msg = [ ( fmtBins.format(edge), fmtCounts.format(count) ) for edge,count in itertools.zip_longest(binEdges,hist,fillvalue='') ]
    #logger.info(   'normalized std.dev.\t#PRC coords.\n'
    #             +  '\n'.join((e for t in msg for e in t if len(e))),
    #             tag='Histogram of normalized PRC coordinate standard deviations' )

    printWidth = max( int( np.floor( np.log10( ymax ) ) + 1 ),
                      4 ) # np.inf prints to 3 chars. We print floats with a precision of 2 in fixed format. Thus: 4
    if printWidth % 2 != 1:
        printWidth += 1
    fmtCounts = '{:^' + str(printWidth) + 'd}'
    fmtBins   = '{:^' + str(printWidth) + '.2f}'
    printWidthLeft = max( len(dataName), max(( len(cooName)+1 for cooName in cooNames )) )
    fmtCoo      =  '#{coo:'  + '{}'.format(printWidthLeft-1) + '} ' + ' ' * (printWidth//2+1) + '{counts}'
    fmtDataName = '\n{name:' + '{}'.format(printWidthLeft)   + '} {bins}'
    binNames = '|'.join(( fmtBins.format( e if not overflow or idx<len(binEdges)-1 else np.inf )
                          for idx,e in enumerate(binEdges) ))
    msg = '\n' + \
          '\n'.join(( fmtCoo.format( coo=coo, counts='|'.join(( fmtCounts.format(int(h)) for h in hist )) ) \
                      for coo,hist in zip(cooNames,hists) )) + \
          fmtDataName.format( name=dataName, bins=binNames )
    logger.infoScreen( msg, tag=tag )
    logger.infoRaw( xml + msg, tag=tag )

@contract
def resNormHisto( res : 'array[Nx2](float)',
                  fn : Path,
                  maxResNorm : 'float|None' = None, # pass float('inf') to draw all residuals.
                  rayleigh=True, # Since we estimate the std.dev. from the data, rayleigh makes sense NOT only for weighted residual norms i.e. residuals with mean~0 and sigma~1.
                  unit : str = 'px',
                  lossFunction = None # Plotting the loss function makes sense if the residuals are weighted, but still not having applied the loss function to them.
                ):
    # We better avoid the costly computation of the sqrt of all squared residuals.
    # Instead, let's compute the histogram of the squared residuals, for the squared bin edges.
    # Then plot the histogram with the un-squared bin edges.
    # Don't choke on extreme values!
    resNormSqr = np.sum(res**2, axis=1)
    resNormMax = resNormSqr.max()**.5
    if maxResNorm is None:
        maxResNorm = np.percentile(resNormSqr, 99)**.5
    else:
        maxResNorm = min(maxResNorm, resNormMax)

    pow2 = 8
    bins = np.linspace( 0, maxResNorm, 1 + 2**pow2 ) # hist will not have an item for the right-most edge, so add 1 to support downsampling by a factor of 2.
    hist, _ = np.histogram(resNormSqr, bins=bins**2)
    binCtrs = (bins[:-1] + bins[1:]) / 2.
    bins = bins[:-1] # plt.bar(.) wants the left bin edges only for align='edge'.
    # sum neighbor bin pairs until there is a meaningful maximum count of residual norms
    while pow2 >= 4 and hist.max() < 100:
        bins = bins[0::2]
        hist = hist[0::2] + hist[1::2]
        pow2 -= 1
    plt.figure(1, clear=True)
    nShown = hist.sum()
    nPresent = len(resNormSqr)
    if nShown < nPresent:
        barLabel = f'{nShown / nPresent:.0%} of {nPresent}; max:{resNormMax:.2f}'
    else:
        barLabel = f'{nPresent}'
    plt.bar(x=bins, height=hist, width=bins[1]-bins[0], align='edge', color='b', linewidth=0, label=barLabel)
    plt.xlabel( rf'$\|\mathbf{{r}}\| [{unit}]$' )
    plt.yticks([])
    plt.xlim( right=maxResNorm )

    if rayleigh:
        # The square root of the sum of squared errors of 2 independent, normally distributed random vectors
        # with variance 1 is distributed according to a chi-distribution with 2 degrees of freedom (chi unsquared!), which is the Rayleigh distribution for std.dev.=1.
        # https://en.wikipedia.org/wiki/Rayleigh_distribution#Parameter_estimation
        mad = np.median(np.abs(res.flat - np.median(res.flat)))
        # ppf: percent point function = quantile function = inverse of cumulative distribution function
        stdDev = mad / stats.norm.ppf(3/4)
        x = np.linspace( bins[0], bins[-1], 100 )
        y = stats.rayleigh.pdf( x, scale=stdDev )
        y *= (hist.sum() * (bins[1]-bins[0])) / (y.sum() * (x[1]-x[0]))
        plt.plot(x, y, '-r', label=f'Rayleigh($σ_{{MAD}}={stdDev:.2f})$')

    if lossFunction is not None:
        ylim = plt.ylim()
        # Evaluating and plotting the loss function makes sense only if res and hence resNormSqr are weighted residuals!
        # Additionally, plot the TrivialLoss, so one can compare it, unless lossFunction already is the TrivialLoss.
        # Loss functions expect as first argument the squared residual norm. The trivial loss just returns that squared residual norm unchanged.
        trivLoss = binCtrs**2
        if isinstance(lossFunction, oriental.adjust.loss.Trivial) or \
           isinstance(lossFunction, oriental.adjust.loss.Wrapper) and isinstance(lossFunction.wrapped(), oriental.adjust.loss.Trivial):
            lossScale = ylim[1] / trivLoss[-1]
        else:
            robLoss = np.empty_like(trivLoss)
            for idx, binCtrSqr in enumerate(trivLoss):
                robLoss[idx] = lossFunction.Evaluate( binCtrSqr )[0]
            lossScale = ylim[1] / robLoss[-1]
            plt.plot(binCtrs, robLoss * lossScale, 'y-', label=str(lossFunction))
        plt.plot(binCtrs, trivLoss * lossScale, 'y:', label='Squared loss')
        plt.ylim(ylim)

    #plt.legend(loc='upper right') # Matplotlib sorts the objects first by type, and second by plot order. Let's move the histogram to the top.
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [handles[-1]] + handles[:-1]
    labels  = [labels[-1 ]] + labels [:-1]
    plt.legend(handles, labels, loc='upper right')

    plt.savefig( str(fn), bbox_inches='tight', dpi=150 )
    plt.close(1)

@contract
def resHisto( res : 'array[N](float)',
              fn : Path,
              coordinate = 'x',
              minMax : 'array[2](float)|None' = None,
              gauss=True, # Since we estimate the mean and std.dev. (robustly) from the data, gauss makes sense NOT only for weighted residuals i.e. residuals with mean~0 and sigma~1.
              unit : str = 'px',
              lossFunction = None # Plotting the loss function makes sense if the residuals are weighted, but still not having applied the loss function to them.
            ):
    if minMax is None:
        minMax = np.array([np.percentile(res,  0.5),
                           np.percentile(res, 99.5)])

    pow2 = 8
    allBins = np.linspace( minMax[0], minMax[1], 1 + 2**pow2 ) # hist will not have an item for the right-most edge, so add 1 to support downsampling by a factor of 2.
    bins = allBins
    hist, _ = np.histogram(res, bins=bins)
    bins = bins[:-1] # plt.bar(.) wants the left bin edges only for align='edge'.
    # sum neighbor bin pairs until there is a meaningful maximum count of residual norms
    while pow2 >= 4 and hist.max() < 100:
        bins = bins[0::2]
        hist = hist[0::2] + hist[1::2]
        pow2 -= 1
    plt.figure(1, clear=True)
    nShown = hist.sum()
    nPresent = len(res)
    if nShown < nPresent:
        barLabel = f'{nShown / nPresent:.0%} of {nPresent} in [{res.min():.2f}; {res.max():.2f}]'
    else:
        barLabel = f'{nPresent}'
    plt.bar(x=bins, height=hist, width=bins[1]-bins[0], align='edge', color='b', linewidth=0, label=barLabel)
    plt.xlabel( rf'$\Delta {coordinate} [{unit}]$' )
    plt.yticks([])
    #plt.xlim( right=maxResNorm )

    if gauss:
        median = np.median(res)
        mad = np.median(np.abs(res - median))
        # ppf: percent point function = quantile function = inverse of cumulative distribution function
        stdDev = mad / stats.norm.ppf(3/4)
        x = np.linspace( bins[0], bins[-1], 100 )
        y = stats.norm.pdf( x, loc=median, scale=stdDev )
        y *= (hist.sum() * (bins[1]-bins[0])) / (y.sum() * (x[1]-x[0]))
        plt.plot(x, y, '-r', label=fr'$\mathcal{{N}}(median={median:.2f}, σ_{{MAD}}={stdDev:.2f})$')

    if lossFunction is not None:
        ylim = plt.ylim()
        # Evaluating and plotting the loss function makes sense only if res and hence res are weighted residuals!
        # Additionally, plot the TrivialLoss, so one can compare it, unless lossFunction already is the TrivialLoss.
        # Loss functions expect as first argument the squared residual norm. The trivial loss just returns that squared residual norm unchanged.
        trivLoss = allBins**2
        if isinstance(lossFunction, oriental.adjust.loss.Trivial) or \
           isinstance(lossFunction, oriental.adjust.loss.Wrapper) and isinstance(lossFunction.wrapped(), oriental.adjust.loss.Trivial):
            # Plotting the squared loss alone doesn't make sense.
            pass#lossScale = ylim[1] / trivLoss[-1]
        else:
            robLoss = np.empty_like(trivLoss)
            for idx, binSqr in enumerate(trivLoss):
                robLoss[idx] = lossFunction.Evaluate( binSqr )[0]
            lossScale = ylim[1] / robLoss[-1]
            plt.plot(allBins, robLoss * lossScale, 'y-', label=str(lossFunction))
            plt.plot(allBins, trivLoss * lossScale, 'y:', label='Squared loss')
            plt.ylim(ylim)

    #plt.legend(loc='upper right') # Matplotlib sorts the objects first by type, and second by plot order. Let's move the histogram to the top.
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [handles[-1]] + handles[:-1]
    labels  = [labels[-1 ]] + labels [:-1]
    plt.legend(handles, labels, loc='upper right')

    plt.savefig( str(fn), bbox_inches='tight', dpi=150 )
    plt.close(1)

@contract
def plotDistortion( fn : Path,
                    imgObsAndRes : 'array[Nx4](float)', 
                    nRowsSensor : 'int|None' = None, 
                    nColsSensor : 'int|None' = None, 
                    title : str ='Mean residuals', 
                    samplingDistancePx : 'int|None' = 100 ):
    #This function only works when importing matplotlib directly, not via oriental.utils.pyplot, because we don't forward mpl_toolkits
    #TODO: multi-page image files would be nice, with decreasing resolution, that better summarize clusters of similar distortion in the image plane.

    if nRowsSensor is None:
        nRowsSensor = -imgObsAndRes[:,1].min()
    if nColsSensor is None:
        nColsSensor = imgObsAndRes[:,0].max()
    if samplingDistancePx is None:
        # TODO consider #residuals, targetting at a minimum average count of residuals per cell to make the average precise, and at a maximum cell resolution, to keep the plot viewable at once on the screen.
        samplingDistancePx = int( np.mean([nColsSensor,nRowsSensor]) / 100. )
    with contextlib.suppress(FileExistsError):
        fn.parent.mkdir(parents=True)

    resolX = np.ceil( nColsSensor / ( nColsSensor // samplingDistancePx ) )
    resolY = np.ceil( nRowsSensor / ( nRowsSensor // samplingDistancePx ) )
    nX = int( np.ceil( nColsSensor / resolX ) )
    nY = int( np.ceil( nRowsSensor / resolY ) )
    avg = np.zeros( (nY,nX,2), float )
    counts = np.zeros( (nY,nX), int )

    for obsX,obsY,residX,residY in imgObsAndRes:
        col = int(  obsX // resolX )
        row = int( -obsY // resolY )
        avg[row,col] += residX, residY
        counts[row,col] += 1
    avg[counts>0,0] /= counts[counts>0]
    avg[counts>0,1] /= counts[counts>0]

    gridX = np.arange(resolX/2, resolX*nX, resolX)
    gridY = np.arange(resolY/2, resolY*nY, resolY)
    xv, yv = np.meshgrid(gridX, gridY)
    # median of mean of residual lengths per cell
    avgNormsSqr = np.sum(avg**2, axis=2 )
    medLen = np.median( avgNormsSqr )**.5
    # plot the residuals scaled, such that on average, they have a length of half the cell size, fitting into their own cell.
    resScale = samplingDistancePx / 2 / medLen

    plt.figure('distortion')
    # TODO: not optimal yet: we want only a few, well distinguishable colors, mapped from non-equidistant residual norm counts.
    # Probably, one would need to apply a mpl.colors.BoundaryNorm onto the cmap.
    #cmap = mpl.cm.get_cmap('YlGnBu',10) # doesn't work with ListedColormap. Use colorbar(boundaries,ticks) instead
    extent= -.5, nColsSensor-.5, -nRowsSensor+.5, .5
    # Calling mpl.cm.get_cmap('viridis',9) succeeds, but plt.imshow fails then.
    # It seems that the only way to use virids with other than the default 256 colors is to convert it to a LinearSegmentedColormap, as are all 'old' built-in colormaps (e.g. 'jet')
    # Actually, few colors distract from the residuals, while they highlight their quality. Thus, use many colors.
    N = 256
    if 1:
        # Instead of the residual count, we better show a normalized quality measure. But which one? Standard deviation? For angles, or magnitudes?
        # Let's use Helmert's Punktlagefehler, as we simply need to compute the std.devs for x and y separately (Werkmeister's would also require computing the covariances).
        # We show the ratio of the length of the average residual vector to the spread of residual vectors around their average.
        # If all residuals share similar length and direction, then this ratio will be large, independent of the amount of local distortion.
        # Thus, our image attributes the plotted mean residual vector with significance: if the color is low (e.g. < 3), then the residual vector is locally insignificant.
        # TODO: maybe fix the colortable to some range around 3, e.g. [1,2,3,4,5], and set cmap.under, cmap.over, so we can better compare repeated plots for different sets of ADPs.
        # s^{Helmert} = sqrt( s_x^2 + s_y^2 )
        helmert = np.zeros_like(avg)
        for obsX,obsY,residX,residY in imgObsAndRes:
            col = int(  obsX // resolX )
            row = int( -obsY // resolY )
            helmert[row,col] += ( np.array(residX, residY) - avg[row,col] )**2
        helmert[counts==1,:] = 0
        select = counts>1
        countsMinus1 = counts[select] - 1
        helmert[select,0] /= countsMinus1
        helmert[select,1] /= countsMinus1
        helmert = np.sum( helmert, axis=2 )**.5
        if 1:
            # even better than std.dev: normalized residual norm. Highlight areas where residuals are large w.r.t. their Helmert'scher Punktlagefehler
            cmap = mpl.colors.LinearSegmentedColormap.from_list( 'viridis_segmented', mpl.cm.get_cmap('viridis').colors, N )
            avgNorms = avgNormsSqr**.5
            plotData = np.zeros_like(helmert)
            select = helmert>0
            plotData[select] = avgNorms[select] / helmert[select]
        else:
            # Use the reversed version of viridis, to highlight the areas of high quality
            cmap = mpl.colors.LinearSegmentedColormap.from_list( 'viridis_r_segmented', mpl.cm.get_cmap('viridis_r').colors, N )
            plotData = helmert
        plt.imshow( plotData, cmap=cmap, interpolation='nearest', aspect='equal', extent=extent )
        # without passing ticks, the number of ticks will generally be different from the number of colors in the colormap.
        maxi, mini = plotData.max(), plotData.min()
        dist = ( maxi - mini ) / N
        ticks = np.arange( mini+dist/2, maxi, dist )
        # with many colors, don't force tick locations
        cb = plt.colorbar( ticks=ticks if N <= 9 else None, orientation='vertical' if nColsSensor < nRowsSensor else 'horizontal' )
        cb.set_label(r'$\|\bar{\mathbf{r}}\|/\sigma^{Helmert}_\mathbf{r}$')
    else:
        #cmap = mpl.cm.get_cmap('viridis')
        norm = mpl.colors.PowerNorm(gamma=.5,vmin=10,vmax=640)
        cmap = mpl.colors.LinearSegmentedColormap.from_list( 'viridis_segmented', mpl.cm.get_cmap('viridis').colors, N )
        plt.imshow( counts, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal', extent=extent )
        # while extend='max' draws a rectangle at the upper end of the colorbar (as documented), it does not consider cmap.set_over, if we use viridis (it works with YlGnBu)!
        #boundaries=np.array((0,10,20,40,80,160,320,640))
        #cb = plt.colorbar( extend='max', spacing='uniform', boundaries=boundaries, orientation='vertical' if nColsSensor < nRowsSensor else 'horizontal' )
        ticks = np.arange(1/N/2,1,1/N)
        cb = plt.colorbar(extend='max',ticks=norm.inverse(ticks).astype(int))
        cb.set_label('#imgPts')
    plt.plot( xv, -yv, '.w' )
    plt.plot( np.c_[    xv.flat,    xv.flatten() + resScale*avg[:,:,0].flatten() ].T,
              np.c_[ (-yv).flat, (-yv).flatten() + resScale*avg[:,:,1].flatten() ].T,
              '-r' )
    plt.xlim( extent[:2] )
    plt.ylim( extent[2:]  )
    plt.title( title )
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.add_artist( AnchoredSizeBar(ax.transData, size=samplingDistancePx, label='{:.2f}px'.format(samplingDistancePx/resScale), loc=4, color='red') )
    plt.savefig( str(fn), bbox_inches='tight', dpi=150 )
    plt.close('distortion')

@contract
def plotImgResiduals( fnIn : 'Path|str',
                      fnOut : 'Path|str',
                      scale : float,
                      imgObsAndResids : 'array[Nx4](float)',
                      imgObsAndResidsAndNames : 'array[Mx4](float)|seq(seq[4])|seq(seq[5])|None' = None,
                      px2µm : float = 1. ):
    # This may be called via multiprocessing.Pool. gdal_utils.imread emits logs, but our log system does not support access to the same XML-log by multiple processes!
    # We have 2 options:
    # - suppress output to file by setting an empty log file name
    # - log to arbitrary, temporary files.
    # Cleanest would be to collect and return the log messages here, and then log them in the main process.
    # There is no robust way to determine if this process is a child process created by multiprocessing. However, if child processes are not given an explicit name, then their names match the following:
    #if re.match(r'Process-\d+', multiprocessing.current_process().name):
    #    log.setLogFileName( '' )
    # better initialise multiprocessing.Pool with initializer=log.clearLogFileName
    fnIn = Path(fnIn)
    fnOut = Path(fnOut)
    img = gdal_utils.imread( str(fnIn) )
    fig=plt.figure(figsize=(10.0, 10.0/img.shape[1]*img.shape[0]), clear=True) # width, height in inches
    ax = plt.axes([0,0,1,1]) # left, bottom, width, height in normalized units
    plt.imshow( img, interpolation='gaussian', cmap='gray' if img.ndim==2 else None )
    plt.autoscale(False)
    plt.xticks([])
    plt.yticks([])
    # cast to int, for better readability.
    medLen = np.ceil( np.median( np.sum( imgObsAndResids[:,2:]**2, axis=1 ) )**.5 * px2µm ).astype(int)
    imgObsAndResids[:,2:] = imgObsAndResids[:,:2] - scale * imgObsAndResids[:,2:]
    plt.plot(  imgObsAndResids[:,0],
              -imgObsAndResids[:,1], '.m' )
    plt.plot(  imgObsAndResids[:,[0,2]].T,
              -imgObsAndResids[:,[1,3]].T, '-c' )
    if imgObsAndResidsAndNames is not None and len(imgObsAndResidsAndNames):
        posResCps = np.atleast_2d([posResNameCp[:4] for posResNameCp in imgObsAndResidsAndNames])
        posResCps[:,2:] = posResCps[:,:2] - scale * posResCps[:,2:]
        for posResNameCp in imgObsAndResidsAndNames:
            if len(posResNameCp) > 4:
                # TODO: ensure that the whole text is inside the image area, and thus visible in the plot.
                plt.text( posResNameCp[0], -posResNameCp[1], posResNameCp[4], color='y', clip_on=True )
        plt.plot(  posResCps[:,0],
                  -posResCps[:,1], '.c' )
        plt.plot(  posResCps[:,[0,2]].T,
                  -posResCps[:,[1,3]].T, '-m', clip_on=True )
    # while the scale is fixed, we adjust the size of the scale bar.
    # Thus, multiple plots with the same scale remain comparable, while the per-plot scale bar's length is the average length of scaled residuals.
    residualUnit = 'px' if px2µm==1. else 'µm'
    ax.add_artist( AnchoredSizeBar( ax.transData, size=scale*medLen/px2µm, label='{}{}'.format(medLen,residualUnit), loc=4, color='c', pad=0.3 ) )
    with contextlib.suppress(FileExistsError):
        fnOut.parent.mkdir(parents=True)
    plt.savefig( str( fnOut ), dpi='figure', bbox_inches='tight', pad_inches=0 )
    plt.close(fig)
