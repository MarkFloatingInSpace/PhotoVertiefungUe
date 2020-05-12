# -*- coding: cp1252 -*-
"log various aspects of a block"

from oriental import adjust, log, blocks, utils
import oriental.adjust.local_parameterization
import oriental.blocks.qxx

from contracts import contract
import numpy as np
from scipy import spatial

from itertools import chain, islice
import collections.abc
from pathlib import Path

logger = log.Logger(__name__)

@contract
def stdDevStats( title : str, parBlocks : collections.abc.Iterable, stdDevs : dict ):
    "extract the standard deviations for parBlocks, and log their statistics together with the parameter block names."
    # extract both names and stdDevs in a single iteration -> parBlocks can be an iterator!
    parNames, theStdDevs = zip( *( (parBlock.names, stdDevs[ parBlock.ctypes.data ])
                                    for parBlock in parBlocks ) )
    # ensure there is only 1 common name -> doesn't work for differently parameterized Euler angles, but then the stats wouldn't make sense, anyway.
    parNames, = frozenset(parNames)
    logger.info( '\N{GREEK SMALL LETTER SIGMA} {}\n'
                 'statistic\t{}\n'
                 'min' '\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'median\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'max' '\t{:.3f}\t{:.3f}\t{:.3f}',
                 title,
                 '\t'.join(chain(parNames,['norm'])),
                 *utils.stats.minMedMaxWithNorm(np.array(theStdDevs)).flat )


@contract
def parameters( parBlocks, stdDevs : 'dict|None' = None, logFunc = logger.info, nDigits : int = 3 ):
    """log parBlocks, with each parBlock on 1 line.
       if stdDevs are passed, then alternate value and stdDev lines.
       parBlocks may be a sequence or iterator"""
    namesVals = []
    for idx,parBlock in enumerate(parBlocks):
        if idx==0:
            name = parBlock.name
            staticNames = parBlock.staticNames
        else:
            if name != parBlock.name:
                raise Exception('All parBlocks must be of the same type')
        namesVals.append(( parBlock.names,
                           '\t'.join( chain( [str(parBlock.info.id)],
                                             ('{:.{}f}'.format(param, nDigits) for param in parBlock) ) ) ))
        if stdDevs is None:
            continue
        parBlockStdDevs = stdDevs.get( parBlock.ctypes.data, np.zeros(parBlock.size) )
        namesVals.append(( parBlock.names,
                           '\t'.join( chain( ['\N{GREEK SMALL LETTER SIGMA}'],
                                             # print with 1 decimal less, and append a non-breaking space.
                                             # Thus, value-lines are well distinguishable from stdDev-lines, while numbers will be aligned to the decimal point.
                                             ( '{:.{}f}\N{no-break space}'.format(parBlockStdDev, nDigits-1) for parBlockStdDev in parBlockStdDevs ) )) ))

    if len({el[0] for el in namesVals}) == 1:
        logFunc('{}s\n'
                'id\t{}\n'
                '{}',
                name,
                '\t'.join( namesVals[0][0] ),
                '\n'.join( nameVal[1] for nameVal in namesVals  ) )
    else:
        # parBlocks share the same static name, but not the same name - e.g. differently parameterized Euler angles.
        logFunc('{}s\n'
                'id\t{}\n'
                '{}',
                name,
                '\t'.join( chain( staticNames, [''] ) ),
                '\n'.join( '\t'.join(( nameVal[1], ''.join( el.capitalize() for el in nameVal[0] ) ))
                           for nameVal in namesVals  ) )

    #for idx,parBlock in enumerate(parBlocks):
    #    if idx==0:
    #        name = parBlock.name
    #        staticNames = parBlock.staticNames
    #    else:
    #        if name != parBlock.name:
    #            raise Exception('All parBlocks must be of the same type')
    #
    #    if stdDevs is None:
    #        namesVals.append(( parBlock.names,
    #                            '\t'.join( chain( [str(parBlock.info.id)],
    #                                                ('{:.{}f}'.format(el, nDigits) for el in parBlock) ) ) ))
    #    else:
    #        parBlockStdDevs = stdDevs.get( parBlock.ctypes.data, np.zeros(parBlock.size) )
    #        namesVals.append(( parBlock.names,
    #                           '\t'.join( chain( [str(parBlock.info.id)],
    #                                             ('{:.{}f}\t{:.{}f}'.format(param, nDigits, parBlockStdDev, nDigits-1)
    #                                               for param,parBlockStdDev in utils.zip_equal(parBlock,parBlockStdDevs) ) ) ) ))
    #if stdDevs is None:
    #    valFmt = '{}'
    #else:
    #    valFmt = '{}\t\N{GREEK SMALL LETTER SIGMA}'
    #
    #
    #if len({el[0] for el in namesVals}) == 1:
    #    logFunc('{}s\n'
    #            'id\t{}\n'
    #            '{}',
    #            name,
    #            '\t'.join( valFmt.format(el) for el in namesVals[0][0] ),
    #            '\n'.join( nameVal[1] for nameVal in namesVals  ) )
    #else:
    #    # parBlocks share the same static name, but not the same name - e.g. differently parameterized Euler angles.
    #    logFunc('{}s\n'
    #            'id\t{}\n'
    #            '{}',
    #            name,
    #            '\t'.join( chain( ( valFmt.format(el) for el in staticNames ),
    #                                [''] ) ),
    #            '\n'.join( '\t'.join(( nameVal[1], ''.join( el.capitalize() for el in nameVal[0] ) ))
    #                       for nameVal in namesVals  ) )


@contract
def objPtManifolds( block : adjust.Problem, objPts, ImgObsDataTypes : 'Type|seq(Type)', logFunc = logger.info, maxManifold : 'int,>1' = 14 ):
    """assumes that all objPts are currently observed in at least 2 images. Otherwise: IndexError"""
    histo = np.zeros( maxManifold-1, int )
    for objPt in objPts:
        nImgPts = sum( 1 for resBlock in block.GetResidualBlocksForParameterBlock(objPt)
                       if isinstance( getattr( block.GetCostFunctionForResidualBlock(resBlock), 'data', None ), ImgObsDataTypes ) )
        histo[ min( nImgPts-2, len(histo)-1 ) ] += 1
    logFunc('Views per object point statistics\n'
            '{}\n'
            '{}\n',
            '\t'.join( chain( ('{}'.format( el ) for el in range( 2, 1+len(histo) ) ),
                              ['{}+'.format( 1+len(histo) )] ) ),
            '\t'.join( '{}'.format( el ) for el in histo ) )

@contract
def imagePtsPerPho( block : adjust.Problem,
                    images : collections.abc.Iterable,
                    imgObsTypesAndNames : 'seq(tuple(Type,str))',
                    logStatsFunc = logger.info,
                    logFunc = logger.infoFile, shortImgFns = None ):
    imgPtsPerPho = []
    for image in images:
        nActivePts = np.zeros(len(imgObsTypesAndNames),int)
        xy = []
        for resBlock in block.GetResidualBlocksForParameterBlock( image.prc ):
            cost = block.GetCostFunctionForResidualBlock( resBlock )
            data = getattr( cost, 'data', None )
            for idx,(imgObsType,_) in enumerate(imgObsTypesAndNames):
                if isinstance( data, imgObsType ):
                    nActivePts[idx] += 1
                    break
            else:
                continue
            xy.append(( cost.x, cost.y ))
        if len(xy) > 2:
            # note: attribute 'volume' is the area of the 2-D convex hull polygon, not attribute 'area'!
            # alternative using OpenCV:
            # ch = cv2.convexHull( np.array(xy,np.float32) ).squeeze()
            # areaRatio = cv2.contourArea( ch ) / ( image.nCols * image.nRows )
            ch = spatial.ConvexHull(np.array(xy)) # TODO adapt for scanned images.
            if hasattr(image, 'pix2cam'):
                scale = image.pix2cam.meanScaleForward()
            else:
                scale = 1.
            imageArea = scale**2 * image.nCols * image.nRows
            areaRatio = ch.volume / imageArea
        else:
            areaRatio = 0.
        if shortImgFns is not None:
            path = shortImgFns(image.path)
        else:
            path = Path(image.path).name

        imgPtsPerPho.append(( path, nActivePts, areaRatio ))

    logStatsFunc( 'Image points per pho stats\n'
                  'statistic\t{}\n'
                  'min\t{}\n'
                  'med\t{}\n'
                  'max\t{}',
                  '\t'.join( '#{}'.format(el[1]) for el in imgObsTypesAndNames ),
                  *( '\t'.join( '{:.0f}'.format(el) for el in row )
                     for row in utils.stats.minMedMax( np.array([el[1] for el in imgPtsPerPho]) ) ) )

    logFunc( 'Image points per pho\n'
             'image\t{}\tconvHullArea\n'
             '{}',
             '\t'.join( '#{}'.format(el[1]) for el in imgObsTypesAndNames ),
             '\n'.join( '\t'.join( chain( ['{}'.format(el[0])],
                                          ( '{}'.format(count) for count in el[1] ),
                                          ['{:.0%}'.format(el[2])] ) )
                         for el in imgPtsPerPho ) )

@contract
def maxAbsCorrelations( block : adjust.Problem,
                        paramBlocks2print,
                        otherParamBlocks2consider = [] ):
    RxxSqr, varWantedParBlocks, nVarParBlocks1 = blocks.qxx.RxxSqr( block, paramBlocks2print, otherParamBlocks2consider )
    maxAbsCorrsSqr = RxxSqr.max( axis=1 )
    argmaxAbsCorrsSqr = RxxSqr.argmax( axis=1 )

    iRow = 0
    row2iParBlockiLocalPar = np.empty((maxAbsCorrsSqr.size,2),int)
    for iParBlock, parBlock in enumerate(varWantedParBlocks):
        if iParBlock == nVarParBlocks1:
            nPars2print = iRow
        for iLocalPar in range(block.ParameterBlockLocalSize(parBlock)):
            row2iParBlockiLocalPar[iRow] = iParBlock, iLocalPar
            iRow += 1

    def parBlockAndParNames( iRow ):
        iParBlock, iLocalPar = row2iParBlockiLocalPar[iRow]
        parBlock = varWantedParBlocks[iParBlock]
        try:
            blockName = '{} {}'.format( parBlock.name, parBlock.info.id )
        except AttributeError:
            blockName = iParBlock

        locParam = block.GetParameterization( parBlock )
        if locParam is None or isinstance( locParam, adjust.local_parameterization.Subset ):
            if locParam is None:
                iLocalPars = range(block.ParameterBlockLocalSize(parBlock))
            else:
                bVariable = np.logical_not(locParam.constancyMask)
                iLocalPars = np.flatnonzero(bVariable)
            parName = getattr( parBlock, 'names', range(parBlock.size) )[iLocalPars[iLocalPar]]
        else:
            # for any other local parametrization, let's not print a name, as the local parameter does not correspond to any.
            parName = 'loc{}'.format(iLocalPar)

        return blockName, parName


    def getCorrs():
        for iRow,(maxAbsCorrSqr,argmaxAbsCorrSqr) in enumerate( islice( utils.zip_equal(maxAbsCorrsSqr,argmaxAbsCorrsSqr), nPars2print )):
            yield parBlockAndParNames(iRow) + (maxAbsCorrSqr**.5,) + parBlockAndParNames(argmaxAbsCorrSqr)[::-1]

    if not nPars2print:
        logger.info('Maximum absolute correlations: all parameter blocks to print are constant!')
    else:
        logger.info('Maximum absolute correlations\n'
                    'block\telem\t\N{GREEK SMALL LETTER RHO}\tmax@elem\tblock\n'
                    '{}',
                    '\n'.join( '{}\t{}\t{:4.2%}\t{}\t{}'.format( *els ) for els in getCorrs() ) )