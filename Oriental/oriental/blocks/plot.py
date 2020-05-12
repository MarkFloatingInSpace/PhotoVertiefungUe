# -*- coding: cp1252 -*-
"block plot functions"

from pathlib import Path
import contextlib
import collections
import colorsys
import subprocess
import urllib
import re
import base64
from xml.dom import minidom
import multiprocessing

from oriental import config, adjust, log, utils
import oriental.adjust.cost
import oriental.utils.gdal
import oriental.utils.filePaths

from contracts import contract
import numpy as np
from scipy import spatial
import cv2
import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import PIL

mpl.rcParams['figure.dpi'] = 96 # the default would be 80, which is clearly too low for desktop monitors

logger = log.Logger('blocks.plot')

@contract
def _getLabelMeterForScaleLog10( legendScaleLog10 : int ):
    if legendScaleLog10 >= 6:
        return '10^{}km'.format(legendScaleLog10-3)
    if legendScaleLog10 >= 3:
        return '{}km'.format(10**(legendScaleLog10-3))
    if legendScaleLog10 >= 0:
        return '{}m'.format(10**legendScaleLog10)
    if legendScaleLog10 == -1:
        return '1dm'
    if legendScaleLog10 == -2:
        return '1cm'
    if legendScaleLog10 == -3:
        return '1mm'
    if legendScaleLog10 >= -6:
        return '{}\N{GREEK SMALL LETTER MU}m'.format(10**(legendScaleLog10+6))
    return '10^{}\N{GREEK SMALL LETTER MU}m'.format(legendScaleLog10+6)

@contract
def objResiduals2d( fn : 'Path|seq(Path)',                  # optionally, save in multiple formats, deduced from file extensions.
                    currentPositions : 'array[Nx3]',        # the current (estimated) object point positions
                    residuals        : 'array[Nx3]',        # the residuals of object point positions, following the OrientAL convention: residual := observed minus estimated
                    names            : 'seq[N](str|Path)',  # names of object points
                    plotScale        : 'number|None' = None, # scale the vectors from observed object point positions to their estimated positions
                    legendScaleLog10 : 'int|None' = None    # length that the legend's scale bar represents. Pass e.g. -3 to plot 1mm
                  ):
    "plot the 3d residuals of object points in 2d"
    obsCurrCps = np.array([ np.r_[ residual, currPos ]
                            for residual,currPos in utils.zip_equal(residuals,currentPositions) ])
    # compute observed position. Adhere to OrientAL-definition: residual:= observed minus estimated
    obsCurrCps[:,:3] = obsCurrCps[:,3:] + obsCurrCps[:,:3] # observed = estimated + residual

    if plotScale is None or legendScaleLog10 is None:
        horizResLensSqr = np.sum( ( obsCurrCps[:,:2] - obsCurrCps[:,3:5] )**2, axis=1 ) # horizontal residual lengths squared
        vertResLensSqr = ( obsCurrCps[:,2] - obsCurrCps[:,5] )**2 # vertical residual lengths squared
        medianResLen = np.median( np.r_[ horizResLensSqr, vertResLensSqr ] )**.5
    if plotScale is None:
        tree = spatial.cKDTree( obsCurrCps[:,:2] )
        dists = tree.query(obsCurrCps[:,:2], k=2, n_jobs=-1)[0]
        wantedResLen = np.median( dists[:,1] ) / 2 # half the median of 2D distances of nearest neighbors
        if medianResLen < 1.e-10:
            logger.warning('Residuals seem very small. Plot with scale=1.')
            plotScale = 1
        else:
            plotScale = wantedResLen / medianResLen
    if legendScaleLog10 is None:
        if medianResLen < 1.e-10:
            logger.warning('Residuals seem very small. Plot scale bar for 1m.')
            legendScaleLog10 = 0
        else:
            legendScaleLog10 = int(np.rint(np.log10(medianResLen)))

    # scale the vector to the current position
    obsCurrCps[:,3:] = obsCurrCps[:,:3] + plotScale * ( obsCurrCps[:,3:] - obsCurrCps[:,:3] )
    shortNames = utils.filePaths.ShortFileNames(names)
    plt.figure(1)
    for name,obsCurrCp in utils.zip_equal(names,obsCurrCps):
        plt.text( obsCurrCp[0], obsCurrCp[1], shortNames(name), color='k', zorder=1, fontsize = 'x-small' )
    plt.plot( obsCurrCps[:,0], obsCurrCps[:,1], '.g', zorder=2 )
    # default linewith is:
    # mpl.rcParams['lines.linewidth']
    # residuals in Z. Plot them thicker and before the residuals for X,Y, such that both will always be visible,
    # even if planar residual in object space is (almost) vertical in the plot
    lines = plt.plot( obsCurrCps[:,[0,0]].T, np.c_[ obsCurrCps[:,1], obsCurrCps[:,1] + obsCurrCps[:,5] - obsCurrCps[:,2] ].T, '-c', linewidth=2., zorder=3 )
    lines[0].set_label('Z')
    # residuals in X,Y
    lines = plt.plot( obsCurrCps[:,[0,3]].T, obsCurrCps[:,[1,4]].T, '-m', zorder=4 )
    lines[0].set_label('XY')
    plt.legend()
    ax = plt.gca()
    ax.add_artist( AnchoredSizeBar( ax.transData,
                                    size= 10**legendScaleLog10 * plotScale,
                                    label=_getLabelMeterForScaleLog10(legendScaleLog10),
                                    loc=4,
                                    color='m' ) )
    plt.axis('equal')
    plt.title( 'Control object point residuals')
    try:
        fns = list(fn)
    except TypeError:
        fns = [fn]
    for fn in fns:
        plt.savefig( str(fn), bbox_inches='tight', dpi=150 )
    plt.close(1)

@contract
def overlap( fn : Path,
             footprints : 'list[A]( $(array[Bx2](float)) )',
             gsd = 0.1,
             bbox : 'array[2x2](float)|None' = None,
             callback = None,
             nColors : 'int,>0' = 10 ):
    if bbox is not None:
        bboxMin, bboxMax = bbox
    else:
        bboxMin = np.full( 2,  np.inf )
        bboxMax = -bboxMin
        for footprint in footprints:
            for pt in footprint:
                np.minimum( pt, bboxMin, out=bboxMin )
                np.maximum( pt, bboxMax, out=bboxMax )

    resolution = np.ceil( ( bboxMax - bboxMin ) / gsd ).astype(int)
    luCorner = np.r_[ bboxMin[0], bboxMax[1] ]
    count = np.zeros( resolution[::-1], int )
    for footprint in footprints:
        count += cv2.fillConvexPoly( np.zeros_like(count), ( ( footprint - luCorner ) *(1,-1) / gsd ).astype(np.int), (1,1,1) )
    plt.figure(1)
    #cmap = mpl.cm.get_cmap('YlGnBu',10) # doesn't work with ListedColormap. Use colorbar(boundaries,ticks) instead
    #cmap = mpl.cm.get_cmap('viridis_r')
    cmap = mpl.colors.LinearSegmentedColormap.from_list( 'viridis_segmented', mpl.cm.get_cmap('viridis').colors, nColors )
    cmap.set_under('w')
    cmap.set_over('m')
    plt.imshow( count, cmap=cmap, interpolation='nearest', vmin=0.5, vmax=nColors+0.5, aspect='equal', extent=(bboxMin[0],bboxMax[0],bboxMin[1],bboxMax[1]) )
    # while extend='max' draws a rectangle at the upper end of the colorbar (as documented), it does not consider cmap.set_over, if we use viridis (it works with YlGnBu)!
    #plt.colorbar(extend='max',boundaries=np.arange(.5,11),ticks=np.arange(1,11))
    plt.colorbar( extend='max', ticks=np.arange(1,nColors+1).tolist() )
    if callback is not None:
        callback()
    plt.title('Overlap (gsd={})'.format(gsd))
    plt.savefig( str( fn ), bbox_inches='tight' )
    plt.close(1)

def _downScale( fnIn, fnOut ):
    img = utils.gdal.imread( str(fnIn), maxWidth=150 )
    #utils.gdal.imwrite( str(dotFn), img ) # GDAL only support CreateCopy() for PNG
    img = PIL.Image.fromarray(img)
    img.save(str(fnOut))

class ColoredBins:
    def __init__( self, nColors = 7 ):
        self.nColors = 7
        self.lowerBoundsExclusive = np.array( [0] + [ 10 * 2**exponent for exponent in range(nColors-1) ] )
        # Graphviz expects HSV-color values in the range [0;1]
        minHue = 0.17 # yellow
        maxHue = 1.   # red
        hues = np.linspace( minHue, maxHue, nColors )
        self.HSVs = np.empty( (hues.shape[0],3) )
        self.HSVs[:,0] = hues
        self.HSVs[:,1] = np.linspace( 0.4, 1., nColors )
        self.HSVs[:,2] = np.linspace( 1., 0.4, nColors )
        self.RGBs = np.array([ colorsys.hsv_to_rgb(*HSV) for HSV in self.HSVs ])
        minWidth = 0.5 # [pt]; 1pt == 1/72 inch == 0.35mm
        maxWidth = 3.0 # [pt]
        self.linewidths = np.linspace( minWidth, maxWidth, nColors )

    def getBin( self, nMatches ):
        return self.lowerBoundsExclusive.searchsorted( nMatches ).item() - 1
        
@contract
def connectivityGraph( fn : Path, imgId2nObj : dict, imgPairIds2nObj : dict, images : dict, embedSvgImgs : bool = True, cleanUp : bool = True ):
    # we need to provide low-resolution images to dot, or otherwise, it crashes, produces corrupt output, etc.
    tmpDir = fn.parent / "temp"
    with contextlib.suppress(FileExistsError):
        tmpDir.mkdir(parents=True)
    
    dotImgFns = {}
    with multiprocessing.Pool( initializer=log.clearLogFileName ) as pool:
        results = []
        for image in images.values():
            dotImgFns[image.id] = dotFn = tmpDir / ( Path(image.path).stem + '.png' )
            if dotFn.exists():
                continue
            results.append( pool.apply_async( func=_downScale, args=(image.path, dotFn) ) )
        # For the case of exceptions thrown in the worker processes:
        # - don't define an error_callback that re-throws, or the current process will hang. If the current process gets killed e.g. by Ctrl+C, the child processes will be left as zombies.
        # - collect the async results, call pool.close() and call result.get(), which will re-throw any exception thrown in the resp. child process.
        pool.close()
        for result in results:
            result.get()
        pool.join()

    #images = { key : image._replace(path=str(Path(image.path).stem)[:11]) for key, image in images.items() }
    shortFileNames = utils.filePaths.ShortFileNames([image.path for image in images.values()])

    # bei sehr vielen phos und Kanten sind die Kantenlabels praktisch nicht mehr lesbar.
    # Vorschlag von cb: statt Kantenlabels unterschiedliche Kantenfarben nutzen + Farbtabelle!
    # Farbgebung der Kanten über Quantile = robust gegen Verzerrung der Farbgebung durch "Ausreißer" in der Anzahl der Matches.

    # We want to visualize the expected quality of relative orientation of image pairs.
    # The number of matches per image pair has an absolute meaning in this context, independent of the data:
    # there is a minimum number required to orient an image pair, and beyond a high number of matches (e.g. 300), differences to not matter.
    # Thus, better use fixed bin edges.
    coloredBins = ColoredBins()
    
    # http://www.graphviz.org/doc/FAQ.html#Q32
    # For a large data set (300 phos, 34897 image connections), running dot.exe takes half an hour!
    # passing -v shows that its not the positioning of the nodes that takes long, but it is the layout of the edges.
    # splines=spline|polyline|ortho takes very long.
    # splines=line|curved takes seconds only
    # Thus, for large number of edges, use line instead of spline.
    dotFn = fn.with_suffix('')
    # If size ends in an exclamation point (!), then it is taken to be the desired size. In this case, if both dimensions of the drawing are less than size, the drawing is scaled up uniformly until at least one dimension equals its dimension in size. 
    # Do not specify size, but let graphviz derive it from the fixed size of nodes (images)
    #size="20.,20.!";
    # box, rect and rectangle are synonyms when specifying 'shape'
    # outputorder=edgesfirst -> make sure that nodes are printed after the edges, such that node labels will not be hidden under edges.
    with dotFn.open( "wt" ) as fout: 
        fout.write("""strict graph ImageConnectivity {
labelloc="t";
label="#objPts per image, #shared objPts per image pair";
node [ shape=rectangle, margin=0, regular=false, style="filled", fontsize=11, fillcolor="grey", width=0.5, height=0.3327, fixedsize=true, imagescale=true ];
edge [ fontcolor=blue, fontsize=8, penwidth=0.1 ];
layout=neato;
model=mds;
overlap=false;
outputorder=edgesfirst;
""" )
        fout.write( "splines={};\n".format( 'spline' if len(imgPairIds2nObj) < 1000 else 'line' )  )

        fout.write( 'imagepath="{}";\n'.format( tmpDir ) )
    
        fout.write( 'legend [ width=1.0, height={}, fillcolor=transparent, label=<<table border="0" cellborder="0" cellpadding="0" cellspacing="0">'.format( 0.2*(coloredBins.nColors+1)+0.1 ) )
        fout.write( '<tr><td>#shared objPts</td></tr>' )
        for idx in range(coloredBins.nColors-1,-1,-1):
            rgb = (coloredBins.RGBs[idx]*255).astype(int)
            fout.write( '<tr><td BGCOLOR="#{:02X}{:02X}{:02X}">&gt;{}</td></tr>'.format( rgb[0],
                                                                                         rgb[1],
                                                                                         rgb[2],
                                                                                         #"" if idx < len(quantiles) else "=",
                                                                                         coloredBins.lowerBoundsExclusive[idx]# if idx < len(quantiles) else nMatchesPerEdge[-1]#"&infin;"
                                                                                       ) )
        fout.write('</table>> ];\n')
            
        fout.write( 'filenames [ fillcolor=transparent, label="{}" fixedsize=false ];\n'.format(shortFileNames.commonName.replace('\\','/') ) )    
            
        for image in images.values():
            # dot does not accept node/edge IDs that e.g. start with a digit - unless the IDs are double-quoted!
            #fout.write( '"{}" [ image="{}" ];\n'.format( self.sfm.imgs[idx].shortName, os.path.basename(self.sfm.imgs[idx].fullPath) ) )
            fout.write( ('"{0}" [ label=<<table border="0" cellborder="0" cellpadding="0" cellspacing="0">' +
                                                '<tr><td align="left"><font color="magenta">{0}</font></td></tr>' +
                                                '<tr><td align="right"><font point-size="6" color="red">{1}</font></td></tr>' +
                                        '</table>>, ' +
                                        'tooltip="\\N: {1} imgPts",' +       
                                        'image="{2}" ];\n').format( shortFileNames(image.path),
                                                                    imgId2nObj[image.id],  
                                                                    dotImgFns[image.id].name ) )
            
        # The order in which Graphviz outputs the edges to svg cannot be influenced by the order in which they appear in the .dot-file :-(
        #for edge,nMatches in sorted( imgPairIds2nObj.items(), key=lambda x: x[1] ):
        for edge,nMatches in imgPairIds2nObj.items():
            # graphviz dot seems to render edges in unspecified order. It would be nice to render edges for many matches on top of edges for fewer matches. 
            # There may be many, many edges. Thus, don't plot the edge labels, and make the lines thinner. Otherwise, the plot is hardly readable.
            # Define edge tooltips instead of labels, which serve as mouse tooltips for svg's loaded in a browser.
            # graphviz dot does not support exponential (scientific) notation. Thus, for the edge length, use fixed-point notation with a large number of digits after the comma.
            iBin = coloredBins.getBin(nMatches)
            fout.write( '"{}" -- "{}" [len={:.14f}, tooltip="\\E: {} {}", penwidth={:.2f}, color="{:.2f} {:.2f} {:.2f}"];\n'.format(
                    shortFileNames(images[edge[0]].path),
                    shortFileNames(images[edge[1]].path),
                    1./nMatches,
                    nMatches,
                    "common objPts",
                    coloredBins.linewidths[iBin],
                    *coloredBins.HSVs[iBin] ) )
                            
            
        fout.write("}\n")

    args = [ str(config.dot), "-O", str(dotFn) ]
    args += [ "-T{}".format(fmt) for fmt in "svg png".split() ] # , "eps": cannot print &infin; contained in legend
    try:
        subprocess.check_call( args )
    except subprocess.CalledProcessError as exc:
        logger.warning(f'{exc.cmd} failed with exit status {exc.returncode}')
    else:
        if embedSvgImgs:
            # post-process the svg: embed the images instead of referencing them as external files.
            with minidom.parse( str(dotFn) + '.svg' ) as dom:
                for image in dom.getElementsByTagName('image'):
                    attributes = image.attributes
                    for ns, localName in attributes.keysNS():
                        if localName == 'href':
                            node = attributes[ns,localName]
                            with ( tmpDir / node.value ).open( 'rb' ) as fin:
                                encoded = urllib.parse.quote( base64.standard_b64encode( fin.read() ) )
                            node.value = 'data:image/png;base64,' + encoded

                # write the XML-tree in descending order of matches
                # "310197--310531: 6 matches"
                rex = re.compile(r'.*?:\s(?P<nmatches>\d+) (matches|common\sobjPts)')
                gs = dom.getElementsByTagName('g')
                edgesNmatches = []
                for g in gs:
                    if g.getAttribute('class') != 'edge':
                        continue
                    gNested, = g.getElementsByTagName('g')
                    a, = gNested.getElementsByTagName('a')
                    title = a.getAttribute('xlink:title')
                    assert( len(title) )
                    m = rex.match(title)
                    nMatches = int(m.group('nmatches'))
                    edgesNmatches.append( (g,nMatches) )

                edgesNmatchesSorted = sorted( edgesNmatches, key=lambda x: x[1] )
                old2newEdge = { old[0]:new[0] for old,new in utils.zip_equal(edgesNmatches,edgesNmatchesSorted) }

                def elementWriteXml( self, writer, indent="", addindent="", newl="" ):
                    node = old2newEdge.get(self, self)
                    origElementWriteXml( node, writer, indent, addindent, newl )

                def commentWriteXml(self, writer, indent="", addindent="", newl="" ):
                    pass

                origElementWriteXml = minidom.Element.writexml
                origCommentWriteXml = minidom.Comment.writexml
                minidom.Element.writexml = elementWriteXml
                minidom.Comment.writexml = commentWriteXml

                with open( str(dotFn) + '.svg', 'w', encoding='utf-8' ) as fout:
                    dom.writexml( fout, encoding='utf-8')

                # prevent side-effects in other code that uses minidom
                minidom.Element.writexml = origElementWriteXml#minidom.Element.oldwritexml
                minidom.Comment.writexml = origCommentWriteXml

                # dom.unlink() # done by context manager

    dotFn.unlink()  
    if cleanUp:
        for path in dotImgFns.values():
            path.unlink()

@contract
def connectivity2dMetric( fn : Path, imgId2nObj : dict, imgPairIds2nObj : dict, images : dict ):
    prcs = np.array([ img.prc[:2] for img in images.values() ])
    widthHeightPcs = prcs.ptp(axis=0)
    tree = spatial.cKDTree( prcs )
    dists, indxs = tree.query(prcs, k=2, n_jobs=-1)
    medMinimumInterPrcDist = np.median( dists[:,1] )
    targetMedMinimumInterPrcDist_inch = 20 / 72 # 20pt - 72pt == 1in
    figSize_inch = widthHeightPcs / medMinimumInterPrcDist * targetMedMinimumInterPrcDist_inch

    coloredBins = ColoredBins()

    plt.figure(1, figsize=figSize_inch )

    lineArtists = []
    group2lines = {}
    for (img1id,img2id),nObj in imgPairIds2nObj.items():
        prc1 = images[img1id].prc
        prc2 = images[img2id].prc
        group2lines.setdefault( coloredBins.getBin(nObj), [] ).append( [ prc1[0], prc2[0], prc1[1], prc2[1] ] )
    group2lines = sorted(group2lines.items(), key=lambda x: x[0])
    for group, lines in group2lines:
        lines = np.array(lines)
        lineArtist,*_ = plt.plot( lines[:,:2].T, lines[:,2:].T, linewidth=coloredBins.linewidths[group], color=coloredBins.RGBs[group] )
        lineArtist.set_label( '>{}'.format( coloredBins.lowerBoundsExclusive[group] ) )
        lineArtists.append(lineArtist)

    #group2Pts = {}
    #for imgId,nObj in imgId2nObj.items():
    #    group2Pts.setdefault( getBin(nObj), [] ).append( images[imgId].prc[:2] )
    #for group,pts in group2Pts.items():
    #    pts = np.array(pts)
    #    plt.plot( pts[:,0], pts[:,1], markersize=linewidths[group], color=RGBs[group] )

    shortFileNames = utils.filePaths.ShortFileNames([image.path for image in images.values()])
    for image in images.values():
        plt.text( image.prc[0], image.prc[1],
                  shortFileNames(image.path),
                  fontsize=6,#coloredBins.linewidths[-1], # image names are not the primary information, so make them small, but readable
                  horizontalalignment='center', verticalalignment='center' )
    plt.axis('equal')
    plt.title('#objPts per image pair')
    # we print the lines for many shared object points last, so they will dominate the plot. However, we want the legend entries in reverse order: many shared points at the top, few points at the bottom
    plt.gca().legend( reversed(lineArtists), reversed([el.get_label() for el in lineArtists]), loc='best')

    plt.savefig( str(fn), bbox_inches='tight' )
    plt.close(1)
