"""
Load a block from DB and adjust it.
"""

# <- baut Ausgleichung aus bestehener DB auf, gleicht diese neu aus (ggf. mit anderer Menge an freien Parametern, anderer Gewichtung) und
# gibt Statistiken und Plots aus.
# Evtl. Möglichkeit, um Passpunkte aufzuspalten in Pass- und Kontrollpunkte!
# Kontrollpunkte sollten z.B. gute Verteilung im Objekt- und im Bildraum haben.
# 
# TODO: Plot der Residuennormen, etc.
# 
# GSD ca. 35cm
# 
# Auto image point residuals a posteriori [mm]
# statistic      x      y  norm
# min       -0.116 -0.123 0.000
# median    -0.000 -0.000 0.011
# max        0.115  0.124 0.131
# -> Verteilungsmaß
# + mean, sigma, sigmaMAD
# 
# sigma auto tie img aus histogramm von relOri auslesen -> ca. 0.9px
# 
# Residuen von Passpunktgruppen sind hoch korreliert -> daraus lässt sich ca. die a priori Genauigkeit der ctrl obj abschätzen
# 
# The worst control object point observations [m]:
# name         dX     dY     dZ  norm
# 96111126 -0.177 -0.381  0.245 0.487
# 96061326  0.172 -0.319 -0.313 0.479
# 94190116 -0.337 -0.108  0.307 0.468
# 94140116  0.251  0.341 -0.182 0.461
# 96121326 -0.275  0.240  0.257 0.446
# 96041126  0.157 -0.253 -0.324 0.439
# 94131316 -0.110  0.256 -0.299 0.409
# 96171326 -0.301  0.215 -0.124 0.390
# 96171316 -0.346  0.148  0.089 0.387
# 96121316 -0.173  0.328  0.046 0.374
# -> residuen eher zu klein für a-priori gewichte

import argparse, itertools, sys
from collections import namedtuple
from contextlib import suppress
import multiprocessing
from pathlib import Path
from sqlite3 import dbapi2

import numpy as np
from scipy import linalg
import h5py

from oriental import absOri, adjust, blocks, config, log, ObservationMode, ori, Progress, utils
import oriental.absOri.main
import oriental.adjust.cost
import oriental.adjust.local_parameterization
import oriental.adjust.loss
import oriental.blocks.db
import oriental.blocks.deactivate
import oriental.blocks.export
import oriental.blocks.log
import oriental.blocks.plot
import oriental.utils.argparse
import oriental.utils.db
import oriental.utils.filePaths
import oriental.utils.pyplot_utils
import oriental.utils.stats

from contracts import contract

AutoImgObsData = namedtuple( 'AutoImgObsData', absOri.main.ImgObsData._fields + ('rgb',) )
CtrlImgObsData = namedtuple( 'CtrlImgObsData', absOri.main.ImgObsData._fields + ('name',) )
CtrlObjObsData = namedtuple( 'CtrlObjObsData', 'id name' )

logger = log.Logger("adjust")
intManuTypes = set(( int(ObservationMode.manual), int(ObservationMode.lsm) ))

def positiveFloat(string):
    value = float(string)
    if value <= 0.:
        raise argparse.ArgumentTypeError('{} must be positive'.format(string))
    return value

@contract
def parseArgs( args : 'list(str)|None' = None ):
    docList = __doc__.splitlines()
    parser = argparse.ArgumentParser( description=docList[0],
                                      epilog='\n'.join( docList[1:] ),
                                      formatter_class=utils.argparse.Formatter )

    parser.add_argument( '--outDir', default=Path.cwd() / "adjust", type=Path,
                         help='Store results in directory OUTDIR.' )
    parser.add_argument( '--outDb', type=Path,
                         help='Store the adjusted block in OUTDB. Default: OUTDIR/adjust.sqlite' )
    parser.add_argument( '--inDb', type=Path,
                         help='Adjust the block defined in INDB. Default: OUTDIR/../transform/transform.sqlite' )
    parser.add_argument( '--autoImgObsStdDev', type=positiveFloat, default=.5,
                         help='Standard deviation of automatic image observations.' )
    parser.add_argument( '--manuImgObsStdDev', type=positiveFloat, default=1.,
                         help='Standard deviation of manual image observations.' )
    parser.add_argument( '--manuObjObsStdDev', type=positiveFloat, default=[1.], nargs='+',
                         help='Standard deviation of manual image observations.' )
    parser.add_argument( '--checkPoints', default=[], nargs='*',
                         help='Do not introduce points with these names into the adjustment, but forward intersect them a posteriori.' )
    parser.add_argument( '--resetADP', action='store_true',
                         help='Reset the distortion parameters to zero a priori.' )
    parser.add_argument( '--sxx', action='store_true',
                         help='Store the full Sxx matrix a posteriori in OUTDIR/sxx.h5. Note that its computation fails for large adjustments.' )

    utils.argparse.addLoggingGroup( parser, "adjustLog.xml" )

    cmdLine = sys.argv[:]
    args = parser.parse_args( args=args )
    main( args, cmdLine, parser )


@contract
def main( args : argparse.Namespace, cmdLine : 'list(str) | None' = None, parser : 'ArgumentParser|None' = None ):

    with suppress(FileExistsError):
        args.outDir.mkdir(parents=True)
    args.outDir = args.outDir.resolve()
    utils.argparse.applyLoggingGroup( args, args.outDir, logger, cmdLine )

    if not args.outDb:
        args.outDb = args.outDir / 'adjust.sqlite'
    if not args.inDb:
        args.inDb = args.outDir / '..' / 'transform' / 'transform.sqlite'
    args.inDb = args.inDb.resolve(strict=True)

    if len(args.manuObjObsStdDev) == 1:
        args.manuObjObsStdDev = args.manuObjObsStdDev*3
    elif len(args.manuObjObsStdDev) == 2:
        args.manuObjObsStdDev = args.manuObjObsStdDev[:1]*2 + [args.manuObjObsStdDev[1]]
    elif len(args.manuObjObsStdDev) > 3:
        raise Exception('manuObjObsStdDev must contain 1 to 3 elements.')

    args.checkPoints = set(args.checkPoints)

    autoImgWeights = np.diag( 1. / np.full( 2, args.autoImgObsStdDev ) )
    manuImgWeights = np.diag( 1. / np.full( 2, args.manuImgObsStdDev ) )
    ctrlObjWeights = np.diag( 1. / np.array(args.manuObjObsStdDev))

    autoImgLoss = adjust.loss.Wrapper( adjust.loss.SoftLOne(1) )
    ctrlImgLoss = adjust.loss.Wrapper( adjust.loss.SoftLOne(1) )
    ctrlObjLoss = adjust.loss.Wrapper( adjust.loss.SoftLOne(1) )

    def getImgObsData(row):
        type = ObservationMode( row['type'] or ObservationMode.automatic )
        args = row['id'], row['imgId'], row['objPtId']
        if type == ObservationMode.automatic:
           klass = AutoImgObsData
           args = args + ((row['red'],row['green'],row['blue']),)
        else:
           klass = CtrlImgObsData
           args = args + (row['name'],)
        return klass( *args )

    def cbImagesLoaded( cameras, images ):
        newCameras = {}
        for imgId, image in images.items():
            camera = absOri.main.Camera( imgId,
                                         image.camera.ior.copy(),
                                         image.camera.s_ior.copy(),
                                         image.camera.adp.copy(),
                                         image.camera.s_adp.copy() )
            newCameras[imgId] = camera
            images[imgId] = absOri.main.Image( image.id,
                                               image.path,
                                               image.prc,
                                               image.omfika,
                                               image.nRows,
                                               image.nCols,
                                               image.pix2cam,
                                               image.mask_px,
                                               image.obsWeights,
                                               camera )
        return newCameras, images

    block, solveOpts, cameras, images, objPts = absOri.main.restoreRelOriBlock(
        args.inDb,
        getImgObsLoss = lambda row: ctrlImgLoss if row['type'] in intManuTypes else autoImgLoss,
        getImgObsData = getImgObsData,
        #cbImagesLoaded = cbImagesLoaded
    )
    logger.info('Block restored')

    if args.resetADP:
        for camera in cameras.values():
            camera.adp[:] = 0.
        logger.info(f'ADPs reset to zero.')

    # adapt to blocks.export.webGL and block.deactivate
    Image = namedtuple( 'Image', absOri.main.Image._fields + ('camId','rot') )
    for imgId in images:
        images[imgId] = Image( **images[imgId]._asdict(), camId = images[imgId].camera.id, rot=images[imgId].omfika )
    objPts = { key : value.pt for key, value in objPts.items() }

    ctrlObjPtIds = []
    checkPointIds = set()

    with dbapi2.connect(utils.db.uri4sqlite(args.inDb) + '?mode=ro', uri=True) as db:
        for objPtId, name in db.execute('''
            SELECT DISTINCT objPts.id, name
            FROM objPts
            JOIN imgObs ON objPts.id == imgObs.objPtId
            WHERE imgObs.type IN ( ?, ? ) ''', tuple( intManuTypes ) ):
            if name in args.checkPoints:
                objPt = objPts.pop(objPtId)
                block.RemoveParameterBlock(objPt)
                solveOpts.linear_solver_ordering.Remove(objPt)
                checkPointIds.add(objPtId)
                continue
            objPt = objPts[objPtId]
            ctrlObjPtIds.append(objPtId)
            cost = adjust.cost.ObservedUnknown( objPt, ctrlObjWeights )
            cost.data = CtrlObjObsData( objPtId, name )
            block.AddResidualBlock( cost, ctrlObjLoss, objPt )

        for resBlock in block.GetResidualBlocks():
            cost = block.GetCostFunctionForResidualBlock( resBlock )
            if isinstance( cost, adjust.cost.ObservedUnknown ):
                continue

            if isinstance( cost.data, CtrlImgObsData ):
                weights = manuImgWeights
                ctrlObjPtIds.append( cost.data.objPtId )
            else:
                weights = autoImgWeights
            cost.setWeights( weights )

    for camera in cameras.values():
        block.SetParameterBlockConstant(camera.ior)
        block.SetParameterBlockConstant(camera.adp)
    ctrlObjPtIds = set(ctrlObjPtIds)
    for objPtId in ctrlObjPtIds:
        block.SetParameterBlockConstant( objPts[objPtId] )


    def checkPointObjResiduals(titleSuffix):
        if not checkPointIds:
            return
        msgsObjAdj = []
        msgsObjRes = []
        msgsImgRes = []
        with dbapi2.connect(utils.db.uri4sqlite(args.inDb) + '?mode=ro', uri=True) as inDb:
            utils.db.initDataBase(inDb)
            hasRefPtsCol = utils.db.tableHasColumn(inDb, 'objPts', 'refPt')
            for checkPtId in checkPointIds:
                imgObservations = []
                for row in inDb.execute('''
                    SELECT name, x, y, imgId
                    FROM imgObs
                    WHERE objPtId == ?
                    ''', [checkPtId]):
                    image = images.get(row['imgId'])
                    if image is None:
                        continue # removed due to too few image observations, etc.
                    imgObservations.append((row['name'], np.array([row['x'], row['y']]), image))
                if len(imgObservations) < 2:
                    # If there is a refPt, then we might either use a local parameterization
                    # or a (lowly weighted) direct observation to keep adjPt from drifting away
                    # and get meaningful residuals with only a single image observation.
                    continue
                refPt = None
                if hasRefPtsCol:
                    X, Y, Z = inDb.execute('''
                        SELECT X(refPt), Y(refPt), Z(refPt)
                        FROM objPts
                        WHERE id == ?
                        ''', [checkPtId]).fetchone()
                    if (X or Y or Z) is not None:
                        refPt = np.array([X, Y, Z])
                if refPt is None:
                    continue # TODO: forward intersection
                problem = adjust.Problem()
                loss = adjust.loss.Trivial()
                adjPt = refPt.copy()
                for ptName, imgCoords, image in imgObservations:
                    cost = adjust.cost.PhotoTorlegard(*image.pix2cam.forward(imgCoords))
                    problem.AddResidualBlock(cost,
                                             loss,
                                             image.prc,
                                             image.rot,
                                             image.camera.ior,
                                             image.camera.adp,
                                             adjPt )
                    for par in (image.prc, image.rot, image.camera.ior, image.camera.adp):
                        problem.SetParameterBlockConstant(par)
                options = adjust.Solver.Options()
                options.linear_solver_type = adjust.LinearSolverType.DENSE_QR
                summary = adjust.Solver.Summary()
                adjust.Solve(options, problem, summary)
                if not adjust.isSuccess(summary.termination_type):
                    logger.warning(f'Adjustment of check point {ptName} has not converged')
                    continue
                msgsObjAdj.append(f'{ptName}\t' + '\t'.join(f'{el:.3f}' for el in adjPt))
                diff = refPt - adjPt
                msgsObjRes.append(f'{ptName}\t' + '\t'.join(f'{el:.3f}' for el in itertools.chain(diff, (linalg.norm(diff[:2]), linalg.norm(diff)))))
                imgRes = problem.Evaluate()[0].reshape((-1, 2)) # observations have been introduced without weights, so EvaluateOptions.weighted does not matter.
                for (ptName, _, image), res in utils.zip_equal(imgObservations, imgRes):
                    msgsImgRes.append(f'{ptName}\t{Path(image.path).stem}\t' + '\t'.join(f'{el:.3f}' for el in itertools.chain(res, [linalg.norm(res)])))
        msgsObjAdj.sort()
        logger.info('\n'.join([f'Check object points {titleSuffix} [m]',
                               'name X Y Z'.replace(' ', '\t')] + msgsObjAdj))
        msgsObjRes.sort()
        logger.info('\n'.join([f'Check object point residuals {titleSuffix} [m]',
                               'name ΔX ΔY ΔZ |ΔXY| |ΔXYZ|'.replace(' ', '\t')] + msgsObjRes))
        msgsImgRes.sort()
        logger.info('\n'.join([f'Check image point residuals {titleSuffix} [px]',
                               'name image Δx Δy |Δxy|'.replace(' ', '\t')] + msgsImgRes))
    checkPointObjResiduals('a priori')

    summary = adjust.Solver.Summary()
    solveOpts.max_num_iterations = 5000
    # There are many image observations, but only few control points. Control points thus contribute little to the objective function. Make sure we end-iterate, such that the sum of residuals of control points is small.
    # Appropriate weighting decreases the number of necessary iterations.
    solveOpts.parameter_tolerance = 1.e-14
    solveOpts.function_tolerance = 1.e-14
    adjust.Solve( solveOpts, block, summary)
    if not adjust.isSuccess( summary.termination_type ):
        raise Exception("adjustment failed: {}".format( summary.BriefReport() ) )
    logger.info('Adjustment with fixed ctrl obj pts finished')

    for objPtId in ctrlObjPtIds:
        block.SetParameterBlockVariable( objPts[objPtId] )

    if 1:
        for camera in cameras.values():
            param = camera.ior
            block.SetParameterBlockVariable(param)
            #iConstants = [ 0, 1 ]
            #iConstants = [ 2 ]
            #locPar = adjust.local_parameterization.Subset( param.size, iConstants )
            #block.SetParameterization( param, locPar )

    if 1:
        for camera in cameras.values():
            param = camera.adp
            iConstants = [ el for el in range(param.size) if el not in ( #adjust.PhotoDistortion.affinitySkewnessOfAxes, # Potsdam: affine ADP verkleinern Residuen v.a. an Passpunkten, aber nicht bei auto-Vkn
                                                                         #adjust.PhotoDistortion.affinityScaleOfYAxis,
                                                                         adjust.PhotoDistortion.optPolynomRadial3,
                                                                         #adjust.PhotoDistortion.optPolynomRadial5,
                                                                         #adjust.PhotoDistortion.optPolynomTangential1,
                                                                         #adjust.PhotoDistortion.optPolynomTangential2
                                                                        ) ]
            locPar = block.GetParameterization(param)
            if locPar is None:
                locPar = adjust.local_parameterization.Subset( param.size, iConstants )
                block.SetParameterization( param, locPar )
            else:
                for idx in range(param.size):
                    if idx in iConstants:
                        locPar.setConstant(idx)
                    else:
                        locPar.setVariable(idx)
                block.ParameterizationLocalSizeChanged(param)
            block.SetParameterBlockVariable(param)

    # There are many image observations, but only few control points. Control points thus contribute little to the objective function. Make sure we end-iterate, such that the sum of residuals of control points is small.
    # Appropriate weighting decreases the number of necessary iterations.
    adjust.Solve( solveOpts, block, summary)
    if not adjust.isSuccess( summary.termination_type ):
        raise Exception("adjustment failed: {}".format( summary.BriefReport() ) )
    logger.info('Adjustment with {} ctrl obj pts and free ior/adp finished'.format( 'constant' if block.IsParameterBlockConstant(objPts[next(iter(ctrlObjPtIds))]) else 'free') )

    if 1:
        #absOri.main.makeIorsAdpsVariableAtOnce( block, solveOpts, cameras, images, objPts,
        #                                        params = ( adjust.PhotoDistortion.optPolynomRadial3, 2, 0, adjust.PhotoDistortion.optPolynomRadial5 ) )

        deactivator = blocks.deactivate.DeactivateTiePts( block, solveOpts, cameras, images, objPts, AutoImgObsData )
        deactivator.deacBehindCam()
        deactivator.deacLargeResidualNorms( 3, weighted=True )
        #deactivator.deacFewImgResBlocks()
        #deactivator.deacSmallAngle(angleThreshGon=5)

        # Control object and image points are never automatically removed.
        # Log their residuals now, before switching to squared loss.

        autoImgLoss.Reset( adjust.loss.Trivial() )
        ctrlImgLoss.Reset( adjust.loss.Trivial() )
        ctrlObjLoss.Reset( adjust.loss.Trivial() )
        logger.info('Adjust with squared loss')

        adjust.Solve(solveOpts, block, summary)
        if not adjust.isSuccess( summary.termination_type ):
            raise Exception("adjustment failed: {}".format( summary.BriefReport() ) )
    else:
        logger.warning('Still using robust loss')

    redundancy = summary.num_residuals_reduced - summary.num_effective_parameters_reduced
    sigma0 = ( summary.final_cost * 2 / redundancy ) **.5 if redundancy > 0 else 0.
    #sigma0_check = ( np.sum(block.Evaluate()[0]**2) / redundancy ) **.5
    #logger.info(f'sigma0_check: {sigma0_check:.3f}')

    autoImgResids = blocks.residualsAndBlocks( block, AutoImgObsData )[0]
    ctrlImgResids, ctrlImgResBlocks = blocks.residualsAndBlocks( block, CtrlImgObsData )
    ctrlObjResids, ctrlObjResBlocks = blocks.residualsAndBlocks( block, CtrlObjObsData )
    logger.info( 'Adjustment overview\n'
                 'statistic\tvalue\n'
                 '{}',
                 '\n'.join( "{0}\t{1:{2}}".format(*els)
                         for els in (('#observations' , summary.num_residuals_reduced           , ''    ),
                                     ('#unknowns'     , summary.num_effective_parameters_reduced, ''    ),
                                     ('redundancy'    , redundancy                              , ''    ),
                                     ('σ_0'           , sigma0                                  , '.3f' ),
                                     ('#photos'       , len(images)                             , ''    ),
                                     ('#tieImgPts'    , len(autoImgResids)                      , ''    ),
                                     ('#tieObjPts'    , len(objPts)                             , ''    ),
                                     ('#controlImgPts', len(ctrlImgResids)                      , ''    ),
                                     ('#controlObjPts', len(ctrlObjResids)                      , ''    )) ) )

    ctrlObjCosts = [ block.GetCostFunctionForResidualBlock(resBlock) for resBlock in ctrlObjResBlocks ]
    currentPositions = np.array( [ objPts[ cost.data.id ] for cost in ctrlObjCosts ], float )
    blocks.plot.objResiduals2d( fn = [ args.outDir / ( 'controlObjectPointResiduals.' + ext ) for ext in ['png','svg'] ],
                                currentPositions = currentPositions,
                                residuals = ctrlObjResids,
                                names = [ cost.data.name for cost in ctrlObjCosts ],
                                #plotScale = 5000.,
                                #legendScaleLog10=-1 # auto plot scale estimation fails, because it is based on nearest neighbors, and most cpc's form groups of 2 or more.
                              )

    logger.info( 'Auto image point residuals a posteriori [px]\n'
                 'statistic\tΔx\tΔy\t|Δxy|\n'
                 'min' '\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'median\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'max' '\t{:.3f}\t{:.3f}\t{:.3f}',
                 *utils.stats.minMedMaxWithNorm(autoImgResids).flat )

    logger.info( 'Control image point residuals a posteriori [px]\n'
                 'statistic\tΔx\tΔy\t|Δxy|\n'
                 'min' '\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'median\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'max' '\t{:.3f}\t{:.3f}\t{:.3f}',
                 *utils.stats.minMedMaxWithNorm(ctrlImgResids).flat )
    ctrlImgResNormsSqr = np.sum( ctrlImgResids**2, axis=1 )
    ctrlImgCosts = [ block.GetCostFunctionForResidualBlock(resBlock) for resBlock in ctrlImgResBlocks ]
    logger.info( 'The worst control image point observations [px]:\n'
                'name\timg\tΔx\tΔy\t|Δxy|\n' +
                '\n'.join(
                   '{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                        ctrlImgCosts[idx].data.name,
                        Path(images[ctrlImgCosts[idx].data.imgId].path).stem,
                        *ctrlImgResids[idx],
                        linalg.norm(ctrlImgResids[idx])
                   ) for idx in ctrlImgResNormsSqr.argsort()[-10:][::-1] ) )

    logger.info( 'Control object point residuals a posteriori [m]\n'
                 'statistic\tΔX\tΔY\tΔZ\t|ΔXYZ|\n'
                 'min' '\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'median\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'max' '\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}',
                 *utils.stats.minMedMaxWithNorm(ctrlObjResids).flat )
    ctrlObjPtResNormsSqr = np.sum( ctrlObjResids**2, axis=1 )
    logger.info( 'The worst control object point observations [m]:\n'
                'name\tΔX\tΔY\tΔZ\t|ΔXYZ|\n' +
                '\n'.join(
                   '{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                        ctrlObjCosts[idx].data.name,
                        *ctrlObjResids[idx],
                        linalg.norm(ctrlObjResids[idx])
                   ) for idx in ctrlObjPtResNormsSqr.argsort()[-10:][::-1] ) )

    blocks.log.imagePtsPerPho(block, images.values(), imgObsTypesAndNames=[(AutoImgObsData,'auto'),(CtrlImgObsData,'manu')])
    blocks.log.objPtManifolds(block, objPts.values(), AutoImgObsData)

    logger.info('IORs\n'
                'id\tx0\ty0\tz0\n{}',
                '\n'.join('{}\t'.format(Path(images[camera.id].path).stem) + '\t'.join(f'{el:.3f}' for el in camera.ior) for camera in sorted(cameras.values(), key=lambda x: Path(images[x.id].path).stem)))
    logger.info('ADPs\n' +
                'id\t' + '\t'.join('a{}'.format(idx) for idx in range(9)) + '\n{}',
                '\n'.join('{}\t'.format(Path(images[camera.id].path).stem) + '\t'.join(f'{el:.3f}' for el in camera.adp) for camera in sorted(cameras.values(), key=lambda x: Path(images[x.id].path).stem)))

    autoImgResids = blocks.residualsAndBlocks(block, AutoImgObsData, weighted=True)[0]
    utils.pyplot_utils.resNormHisto( autoImgResids, args.outDir / 'autoImgResidualHistogram.png', maxResNorm=3. )

    blocks.export.webGL(args.outDir / 'reconstruction.html', block, cameras, images, objPts, type(None))

    if args.outDb.exists():
        args.outDb.unlink()
    blocks.db.save( args.outDb, block, cameras, images, objPts, stdDevs=None, TieImgObsData=AutoImgObsData, CtrlImgObsData=CtrlImgObsData )

    checkPointObjResiduals('a posteriori')

    def computeSaveSxx(sxxFn):
        parNames, parBlocks = [], []
        paramBlock2iStart = {}
        iStart = 0
        def addParBlock(parBlock, baseName, names):
            nonlocal iStart
            if block.IsParameterBlockConstant(parBlock):
                return
            loc = block.GetParameterization(parBlock)
            if loc is None:
                bFree = np.ones(parBlock.size, np.bool)
            else:
                bFree = np.logical_not(loc.constancyMask)
            paramBlock2iStart[parBlock.ctypes.data] = iStart
            iStart += sum(bFree)
            for name, free in utils.zip_equal(names, bFree):
                if free:
                    parNames.append(f'{baseName}_{name}')
            parBlocks.append(parBlock)

        for camera in cameras.values():
            addParBlock(camera.ior, f'cam_{camera.id}_ior', 'x0 y0 z0'.split())
        for camera in cameras.values():
            addParBlock(camera.adp, f'cam_{camera.id}_adp', [el[0] for el in sorted(adjust.PhotoDistortion.names.items(), key=lambda x: x[1])])
        for image in images.values():
            addParBlock(image.prc, f'image_{image.id}_prc', 'X0 Y0 Z0'.split())
        for image in images.values():
            addParBlock(image.rot, f'image_{image.id}_rot', 'r0 r1 r2'.split())
        if 0:
            for objPtId, objPt in objPts.items():
                addParBlock(objPt, f'obj_{objPtId}', 'X Y Z'.split())
            evalOpts = adjust.Problem.EvaluateOptions()
            evalOpts.weighted = True
            evalOpts.set_parameter_blocks(parBlocks)
            jacobian, = block.Evaluate(evalOpts, residuals=False, jacobian=True)
            N = (jacobian.transpose() @ jacobian).toarray()
            C, lower = linalg.cho_factor(N, overwrite_a=True)
            unitVecs = np.eye(C.shape[0], C.shape[1])
            Sxx = linalg.cho_solve((C, lower), unitVecs, overwrite_b=True)
        else:
            covOpts = adjust.Covariance.Options()
            if config.isGPL:
                covOpts.sparse_linear_algebra_library_type = adjust.SparseLinearAlgebraLibraryType.SUITE_SPARSE
            #covOpts.algorithm_type = adjust.CovarianceAlgorithmType.DENSE_SVD
            covariance = adjust.Covariance(covOpts)
            paramBlockPairs = list(itertools.combinations_with_replacement(parBlocks, 2))
            covariance.Compute( paramBlockPairs, block )
            Sxx = np.empty((len(parNames), len(parNames)))
            for paramBlockPair in paramBlockPairs:
                cofactorBlock = covariance.GetCovarianceBlockInTangentSpace( *paramBlockPair )
                iStarts = [paramBlock2iStart[el.ctypes.data] for el in paramBlockPair]
                Sxx[ iStarts[0] : iStarts[0] + cofactorBlock.shape[0],
                     iStarts[1] : iStarts[1] + cofactorBlock.shape[1]] = cofactorBlock
            Sxx = np.triu(Sxx, 0) + np.triu(Sxx, 1).T
        Sxx *= sigma0**2
        with h5py.File(sxxFn, 'w') as h5:
            dset = h5.create_dataset('Sxx', data=Sxx)
            dset.attrs.create('columnNames', parNames, shape=(len(parNames),), dtype=h5py.special_dtype(vlen=str))
            for image in images.values():
                dset.attrs[f'image_{image.id}'] = utils.filePaths.relPathIfExists(image.path, sxxFn.parent)
        logger.info(f'Sxx of all parameters except object points saved to {sxxFn}')

    if args.sxx:
        computeSaveSxx(args.outDir / 'Sxx.h5')

    # This may take long, so do it at the end.
    def plotResiduals():

        class Incrementor:
            def __init__(self, progress):
                self.progress = progress

            def __call__(self, arg):
                self.progress += 1

        residualsDir = args.outDir / 'residuals'

        logger.info('Plot residuals for each image into directory "{}"', residualsDir.name )
        progress = Progress(len(images))
        with multiprocessing.Pool( initializer = log.suppressCustomerClearLogFileName ) as pool:
            results = []
            for image in images.values():
                autoResiduals, autoResBlocks = blocks.residualsAndBlocks(block, AutoImgObsData, image.prc,
                                                                         weighted=False)
                ctrlResiduals, ctrlResBlocks = blocks.residualsAndBlocks(block, CtrlImgObsData, image.prc,
                                                                         weighted=False)
                autoObsAndRes = []
                for residuals, resBlock in utils.zip_equal(autoResiduals, autoResBlocks):
                    cost = block.GetCostFunctionForResidualBlock(resBlock)
                    obs = np.array([cost.x, cost.y])
                    obs = image.pix2cam.inverse(obs)
                    # proj = images[imgId].pix2cam.inverse( proj )
                    res = image.pix2cam.Ainv @ residuals
                    autoObsAndRes.append(np.r_[obs, res])

                ctrlObsAndResAndNames = []
                for residuals, resBlock in utils.zip_equal(ctrlResiduals, ctrlResBlocks):
                    cost = block.GetCostFunctionForResidualBlock(resBlock)
                    obs = np.array([cost.x, cost.y])
                    obs = image.pix2cam.inverse(obs)
                    # proj = images[imgId].pix2cam.inverse( proj )
                    res = image.pix2cam.Ainv @ residuals
                    ctrlObsAndResAndNames.append((*obs, *res, cost.data.name))

                kwds = dict( fnIn=image.path,
                             fnOut=residualsDir / ( Path(image.path).stem + '.jpg' ),
                             scale=100.,
                             imgObsAndResids=np.array(autoObsAndRes),
                             imgObsAndResidsAndNames=ctrlObsAndResAndNames,
                             #px2µm=image.pix2cam.meanScaleForward()*1000.
                           )
                results.append( pool.apply_async( func=utils.pyplot_utils.plotImgResiduals, kwds=kwds, callback=Incrementor(progress) ) )

            # For the case of exceptions thrown in the worker processes:
            # - don't define an error_callback that re-throws, or the current process will hang. If the current process gets killed e.g. by Ctrl+C, the child processes will be left as zombies.
            # - collect the async results, call pool.close() and call result.get(), which will re-throw any exception thrown in the resp. child process.
            pool.close()
            for result in results:
                result.get()
            pool.join()
    plotResiduals()