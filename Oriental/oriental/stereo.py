"""
Calibrate a stereo camera
"""

import argparse, shutil, sys
from collections import namedtuple
from contextlib import suppress
from pathlib import Path
from sqlite3 import dbapi2

from oriental import absOri, adjust, blocks, log, ori, utils
import oriental.absOri.main
import oriental.adjust.cost
import oriental.adjust.loss
import oriental.blocks.export
import oriental.blocks.db
import oriental.blocks.deactivate
import oriental.utils.argparse
import oriental.utils.db
import oriental.utils.stats

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from contracts import contract

logger = log.Logger("stereo")

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

    parser.add_argument( '--outDir', default=Path.cwd() / "stereo", type=Path,
                         help='Store results in directory OUTDIR.' )
    parser.add_argument( '--outDb', type=Path,
                         help='Store the calibrated block in OUTDB. Default: OUTDIR/stereo.sqlite' )
    parser.add_argument( '--inDb', type=Path,
                         help='Transform the block defined in INDB. Default: OUTDIR/../relOri/relOri.sqlite' )
    parser.add_argument('--cam0', required=True,
                        help='Image file paths that match this glob belong to camera 0, otherwise to camera 1.')
    parser.add_argument( '--imgObsStdDev', type=positiveFloat, default=.5,
                         help='Standard deviation of image observations.' )
    parser.add_argument( '--shiftStdDev', type=positiveFloat, default=.0005,
                         help="Standard deviation of zero-difference to mean relative shift of stereo images." )
    parser.add_argument( '--rotStdDev', type=positiveFloat, default=.01,
                         help="Standard deviation of zero-differences to mean relative rotation angles of stereo images." )
    parser.add_argument('--baselineLength', type=positiveFloat, default=0.5,
                        help='Baseline length, used to scale the block.')

    utils.argparse.addLoggingGroup( parser, "stereoLog.xml" )

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
        args.outDb = args.outDir / 'stereo.sqlite'
    if not args.inDb:
        args.inDb = args.outDir / '..' / 'relOri' / 'relOri.sqlite'
    args.inDb = args.inDb.resolve(strict=True)

    tempDbFn = args.outDb.parent / ( args.outDb.name + '_temp' + args.outDb.suffix )
    shutil.copyfile(args.inDb, tempDbFn)
    with dbapi2.connect(utils.db.uri4sqlite(tempDbFn) + '?mode=rw', uri=True) as db:
        utils.db.initDataBase(db)
        camIds = [ el[0] for el in db.execute('SELECT id FROM cameras') ]
        if not camIds:
            raise Exception('DB contains no cameras')
        if len(camIds) > 2:
            raise Exception(f'DB contains {len(camIds)} cameras, which is more than 2.')
        if len(camIds) == 2 and set(camIds) != {0,1}:
            raise Exception(f'DB contains 2 cameras, but their IDs are {camIds[0]}, {camIds[1]} instead 0, 1.')
        if len(camIds) == 1:
            if camIds[0] != 0:
                raise Exception(f'DB contains 1 camera, but its ID is {camIds[0]} instead of 0.')
            camTableColumnNames = utils.db.columnNamesForTable( db, 'cameras' )
            camTableColumnNames = [ el for el in camTableColumnNames if el != 'id' ]
            db.execute(f'''
                INSERT INTO cameras ({', '.join(camTableColumnNames)})
                SELECT {', '.join(camTableColumnNames)}
                FROM cameras
                WHERE id == 0
            ''')
        db.execute(f'''
            UPDATE images
            SET camId = CASE path GLOB "{args.cam0}"
                        WHEN 1 THEN 0
                        ELSE 1
                        END
        ''')
        phoCams0 = [el[0] for el in db.execute('SELECT id FROM images WHERE camId == 0').fetchall()]
        phoCams1 = [el[0] for el in db.execute('SELECT id FROM images WHERE camId == 1').fetchall()]
        logger.info(f'DB contains {len(phoCams0)+len(phoCams1)} images. {len(phoCams0)} images get assigned to camera 0, {len(phoCams1)} to camera 1.')
        if len(phoCams0) != len(phoCams1):
            raise Exception('Different numbers of images assigned to cameras.')

        imgIdPaths = db.execute('''
            SELECT id, path
            FROM images
        ''').fetchall()
        for imgId, path in imgIdPaths:
            repl = 'cam1' if 'cam1_' in path else 'cam2'
            path = Path('..') / '..' / 'A3D-IDS-Test01-Treppe' / 'Serie4' / path.replace('camX',repl)
            db.execute('''
                UPDATE images
                SET path = ?
                WHERE id == ?    
            ''', ( str(path), imgId ) )

    imgWeights = np.diag( 1. / np.full( 2, args.imgObsStdDev ) )
    loss = adjust.loss.Wrapper( adjust.loss.SoftLOne(1) )

    ImgObsData = namedtuple('ImgObsData', absOri.main.ImgObsData._fields + ('rgb',))
    def getImgObsData(row):
        args = row['id'], row['imgId'], row['objPtId'], (row['red'],row['green'],row['blue'])
        return ImgObsData( *args )

    block, solveOpts, cameras, images, objPts = absOri.main.restoreRelOriBlock(
        tempDbFn,
        getImgObsLoss = lambda row: loss,
        getImgObsData = getImgObsData )
    db.close()
    tempDbFn.unlink()

    logger.info('Block restored')
    assert(len(cameras)==2)
    # adapt to blocks.export.webGL and block.deactivate
    Image = namedtuple( 'Image', absOri.main.Image._fields + ('camId','rot') )
    for imgId in images:
        images[imgId] = Image( **images[imgId]._asdict(), camId = 0 if imgId in set(phoCams0) else 1, rot=images[imgId].omfika )
    objPts = { key : value.pt for key, value in objPts.items() }

    prcs1in0 = np.empty((len(phoCams0),3))
    totalRot1in0 = np.eye(3)
    for idx,(phoCam0, phoCam1) in enumerate(utils.zip_equal(phoCams0, phoCams1)):
        rot0, rot1 = [ ori.euler2matrix(images[phoCam].omfika) for phoCam in (phoCam0,phoCam1) ]
        prc1in0 = rot0.T @ ( images[phoCam1].prc - images[phoCam0].prc )
        prcs1in0[idx] = prc1in0

        rot1in0 = rot0.T @ rot1
        totalRot1in0 = rot1in0 @ totalRot1in0

    medPrc1in0 = utils.stats.geometricMedian(prcs1in0)
    #meanRot1in0 = linalg.fractional_matrix_power( totalRot1in0, 1./len(phoCams0) )
    meanOmFiKa1in0 = np.zeros(3)

    offsetCovInvSqrt = np.diag(1 / np.full(3, args.shiftStdDev))
    relrotCovInvSqrt = np.diag(1 / np.full(3, args.rotStdDev))
    resIdsRelOri = []
    for phoCam0, phoCam1 in utils.zip_equal(phoCams0, phoCams1):
        img0, img1 = [ images[phoCam] for phoCam in (phoCam0,phoCam1) ]
        cost = adjust.cost.CameraRig( offsetCovInvSqrRoot=offsetCovInvSqrt, omfikaCovInvSqrRoot=relrotCovInvSqrt )
        resIdsRelOri.append( block.AddResidualBlock( cost, loss,
                                                     medPrc1in0, meanOmFiKa1in0,
                                                     img0.prc, img0.omfika,
                                                     img1.prc, img1.omfika ) )

    solveOpts.linear_solver_ordering.AddElementToGroup(medPrc1in0, 1)
    solveOpts.linear_solver_ordering.AddElementToGroup(meanOmFiKa1in0, 1)

    constImgId0 = list(images)[np.argmin(np.array([linalg.norm(el.prc) for el in images.values()]))]
    constImgId1 = list(images)[np.argmin(np.abs(np.array([linalg.norm(el.prc - images[constImgId0].prc) for el in images.values()]) - 1))]
    block.SetParameterBlockConstant(images[constImgId0].prc)
    block.SetParameterBlockConstant(images[constImgId0].omfika)
    assert (abs(linalg.norm(images[constImgId1].prc) - 1) < 1.e-7)
    for img in images.values():
        img.prc[:] *= args.baselineLength
    for pt in objPts.values():
        pt[:] *= args.baselineLength
    parameterization = adjust.local_parameterization.Sphere(args.baselineLength)
    block.SetParameterization(images[constImgId1].prc, parameterization)

    summary = adjust.Solver.Summary()
    adjust.Solve(solveOpts, block, summary)
    assert adjust.isSuccess( summary.termination_type )

    plotResidualHistograms( block, ImgObsData, args.outDir, 'robust' )

    deactivator = blocks.deactivate.DeactivateTiePts(block, solveOpts, cameras, images, objPts, ImgObsData)
    deactivator.deacLargeResidualNorms(3, weighted=True)

    loss.Reset(adjust.loss.Trivial())
    logger.info('Adjust with squared loss')

    adjust.Solve(solveOpts, block, summary)
    if not adjust.isSuccess(summary.termination_type):
        raise Exception("adjustment failed: {}".format(summary.BriefReport()))

    plotResidualHistograms(block, ImgObsData, args.outDir, 'ls')

    #TODO nach stereo sind die Ergebnisse von dense noch immer schlechter!!
    #TODO plot residuals per image in image space

    blocks.export.webGL(args.outDir / 'reconstruction.html', block, cameras, images, objPts, type(None))
    blocks.export.ply(args.outDir / 'reconstruction.ply', cameras, images, objPts, block)
    if args.outDb.exists():
        args.outDb.unlink()
    blocks.db.save( args.outDb, block, cameras, images, objPts, TieImgObsData=ImgObsData )

@contract
def plotResidualHistograms( block : adjust.Problem, ImgObsData : type, outDir : Path, prefix : str ):
    imgRes, imgResBlocks = blocks.residualsAndBlocks( block, ImgObsData, weighted=False )
    allImgResNormsHisto( np.sum( imgRes**2, axis=1 ), fn=outDir / (prefix + '_imageResidualNorms.png'))

    relOriRes, relOriResBlocks = blocks.residualsAndBlocks( block, type(None), weighted=False )
    allImgResNormsHisto( np.sum( relOriRes[:,:3]**2, axis=1 ), fn=outDir / (prefix + '_shiftResidualNorms.png'))

    allImgResNormsHisto(relOriRes[:, 3] ** 2, fn=outDir / (prefix + '_omegaResidualNorms.png'))
    allImgResNormsHisto(relOriRes[:, 4] ** 2, fn=outDir / (prefix + '_phiResidualNorms.png'))
    allImgResNormsHisto(relOriRes[:, 5] ** 2, fn=outDir / (prefix + '_kappaResidualNorms.png'))

@contract
def allImgResNormsHisto( resNormSqr : 'array[N](float,>0)',
                         maxResNorm : 'float,>0|None' = None,
                         fn : Path = '' ):
    # we better not compute the sqrt of all squared residuals, but get the histogram of the squared residuals, for the squared bin edges. Then plot the histogram with the un-squared bin edges.
    # don't choke on extreme values!
    maxResNorm = maxResNorm or np.percentile(resNormSqr,99)**.5

    pow2 = 8
    bins = np.linspace( 0, maxResNorm, 1 + 2**pow2 ) # hist will not have an item for the right-most edge, so add 1 to support downsampling by a factor of 2.
    binsSqr = bins**2
    hist,_ = np.histogram( resNormSqr, bins=binsSqr )
    bins = bins[:-1] # plt.bar(.) wants the left bin edges only for it's 'left' argument.
    # sum neighbor bin pairs until there is a meaningful count of residual norms in the maximum bin
    while pow2 >= 4 and hist.max() < 100:
        bins = bins[0::2]
        hist = hist[0::2] + hist[1::2]
        pow2 -= 1
    plt.figure('residuals'); plt.clf()
    plt.bar( x=bins, height=hist, width=bins[1]-bins[0], color='b', linewidth=0 )
    plt.ylabel('residual count', color='b')
    plt.xlabel('residual norm')
    plt.xlim( right=maxResNorm )
    nShown = hist.sum()
    if nShown < len(resNormSqr):
        prefix = '{} of '.format( nShown )
    else:
        prefix = 'all '

    plt.title( prefix + "{} residual norms; max:{:.3f}".format(
                        len(resNormSqr), resNormSqr.max()**.5 ))
    if fn:
        plt.savefig( fn, bbox_inches='tight', dpi=150 )
        plt.close('residuals') #  for some reason, plt.close(fig) doesn't work