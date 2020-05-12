# -*- coding: cp1252 -*-
"""Dense image matching.

Use the output of `relOri`. 
Note that the dense matching software must be available on PATH.
"""

import oriental
from oriental import config, Progress, blocks, log, ori, adjust, utils
import oriental.adjust.parameters
import oriental.blocks.types
import oriental.utils.gdal
import oriental.utils.db
import oriental.utils.argparse
import oriental.utils.photo

import sys, os, subprocess, argparse, shutil, csv, struct, itertools, time
from pathlib import Path
from sqlite3 import dbapi2
from contextlib import suppress
import asyncio
import asyncio.subprocess

import numpy as np
import PIL.Image
from contracts import contract
from osgeo import ogr
ogr.UseExceptions()

DenseMatcher = utils.argparse.ArgParseEnum('DenseMatcher', 'pmvs')
Camera = blocks.types.createCompact( blocks.types.Camera, 'id' )
Image = blocks.types.createCompact( blocks.types.Image, 'nCols nRows pix2cam mask_px camId iorUndist'.split() )

logger = log.Logger("dense")

if oriental.config.isDvlp:
    _cmvsDir = Path( shutil.which( 'cmvs.exe' ) ).parent
else:
    _cmvsDir = Path( oriental.config.installDir ) / 'bin' / 'cmvs'

# http://www.di.ens.fr/pmvs/documentation.html

@contract
def parseArgs( args : 'list(str)|None' = None ):
    docList = __doc__.splitlines()
    parser = argparse.ArgumentParser( description=docList[0],
                                      epilog='\n'.join( docList[1:] ),
                                      formatter_class=utils.argparse.Formatter )

    parser.add_argument( '--outDir', default=Path.cwd() / "dense", type=Path,
                         help='Store results in directory OUTDIR.' )
    parser.add_argument( '--db', type=Path,
                         help='Use images found in DB. Default: OUTDIR/../relOri/relOri.sqlite' )
    parser.add_argument( '--denseMatcher', default=DenseMatcher.pmvs, choices=DenseMatcher, type=DenseMatcher,
                         help='Dense image matching external software to use.' )
    parser.add_argument( '--level', default=1, type=int,
                        help='Image pyramid level on which to operate. Choose high to get quick, sparse results. 0 for highest density.' )
    parser.add_argument( '--deleteTemporaries', action='store_true',
                         help='Delete temporary files. Otherwise, they might get re-used upon the next run: rectified images, orientation files, masks.' )
    utils.argparse.addLoggingGroup( parser, "denseLog.xml" )

    generalGroup = parser.add_argument_group('General', 'Other general settings')
    generalGroup.add_argument( '--no-progress', dest='progress', action='store_false',
                               help="Don't show progress in the console." )

    cmdLine = sys.argv[:]
    args = parser.parse_args( args=args )
    main( args, cmdLine, parser )


class Undistorter( utils.photo.Undistorter ):
    def __init__( self, refDataMtime, reduction ):
        self.refDataMtime = refDataMtime
        self.reduction = reduction
        super().__init__()

    @contract
    def __call__( self, distFn : Path, undistFn : Path, oriFn : Path, maskFn : Path, camera : Camera, image : Image ):
        def getResolution( imgFn ):
            info, = utils.gdal.imread( imageFilePath = str(imgFn), skipData = True )
            return info.nCols * info.nRows

        tempFiles = undistFn, oriFn, maskFn
        if all( p.exists() and p.stat().st_mtime >= self.refDataMtime for p in tempFiles ):
            return getResolution( undistFn ), camera.ior, True

        # Create undistorted images. Unfortunately, PMVS2 only supports P, but no other distortion parameters.
        pho_rect, pho_rect_mask, phoInfo, ior_rect = super().__call__( distFn, camera.ior, camera.adp, image.pix2cam, image.mask_px )
        
        img = PIL.Image.fromarray(pho_rect)
        # PMVS2 wants 3-channel images!
        if len(img.getbands()) < 3:
            img = img.convert('RGB')
        img.save( undistFn )

        #img = PIL.Image.fromarray(pho_rect_mask, mode='1') # Pillow bug: loading from boolean ndarray's does not work.
        img = PIL.Image.fromarray( pho_rect_mask.astype(np.uint8) * 255 )
        if maskFn.suffix.lower() == '.pbm':
            img = img.convert('1')
        img.save( maskFn )

        Rx200 = np.diag((1.,-1.,-1.))
        R = ori.euler2matrix(image.rot)
        P = np.empty( (3,4) )
        P[:,:3] =  Rx200 @ R.T
        P[:,3] = - Rx200 @ R.T @ ( image.prc - self.reduction )
        K = ori.cameraMatrix( ior_rect )
        P = K @ P

        with oriFn.open('wt') as oriFile:
            oriFile.write('CONTOUR\n')
            for row in P:
                oriFile.write( ' '.join(f'{el:.15f}' for el in row) + '\n' )

        return getResolution(undistFn), ior_rect, False


async def progress_async( progress, delay : float ):
    try:
        while True:
            await asyncio.sleep( delay )
            progress += 1
    except asyncio.CancelledError:
        # This exception is thrown below by calling progressTask.cancel()
        # Need to catch it, so the program does not abort.
        pass

async def check_output_async( *args, loop, **kwargs ):
    # bufsize, universal_newlines and shell should not be specified at all.
    create = asyncio.create_subprocess_exec( *args,
                                             stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.STDOUT,
                                             **kwargs )
    proc = await create

    if loop:
        progress = Progress(0)
        progressTask = loop.create_task( progress_async( progress, 0.5 ) )
        progressTask.add_done_callback( lambda future: progress.finish() )

    # PMVS2 may print "refinePatchBFGS failed!" many, many times. Let's compress identical, consecutive lines to a single one.
    class RepeatedOutput(str):
        def __init__( self, txt ):
            str.__init__(txt)
            self.count = 1

    output = []
    try:
        while 1:
            line = await proc.stdout.readline()
            if not line:
                break
            txt = line.decode('ascii').rstrip()
            if output and output[-1] == txt:
                output[-1].count += 1
            else:
                output.append( RepeatedOutput(txt) )
    except:
        proc.kill()
        raise
    finally:
        if loop:
            # make the coroutine return
            progressTask.cancel()
            await progressTask
        # Wait for the subprocess exit
        ret = await proc.wait()

    return ret, '\n'.join( '{}{}'.format( txt, f' <printed {txt.count} times>' if txt.count > 1 else '' ) for txt in output ).expandtabs(2)


@contract
def check_output( args : list, cwd : Path, progress : bool = True, **kwargs ):
    path = ';'.join([ str(_cmvsDir), str(cwd) ])
    env = os.environ.copy()
    env['PATH'] = path

    # pmvs.bat created by genOption.exe is not in the CMVS installation directory, but in args.outDir.
    # If we don't pass its absolute file path, then the subprocess will raise FileNotFoundError, even though the cwd is on its path!?
    exe = shutil.which( args[0], path=path )
    if exe is None:
        raise Exception( f'{args[0]} has not been found on PATH. Please install it and add its installation directory to PATH, e.g. by editing OrientALShell.bat' )
    args = [exe] + [ str(el) for el in args[1:] ]
    logger.info( 'Calling (cwd={}): {}', cwd, ' '.join( args ) )

    # pmvs2 creates lots of output, which may fill the buffer. Hence, read from it while the subprocess is running.
    # Also, we need to rotate the optional progress display while the subprocess is running.
    # Do so using asyncio.
    if os.name == 'nt':
        # https://docs.python.org/3/library/asyncio-platforms.html#subprocess-support-on-windows
        # "On Windows, the default event loop is SelectorEventLoop which does not support subprocesses. ProactorEventLoop should be used instead."
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    loop = asyncio.get_event_loop()
    try:
        # Return the Future’s result, or raise its exception.
        ret, output = \
            loop.run_until_complete( check_output_async( *args, cwd=str(cwd), env=env, loop=loop if progress else None ) )
    except FileNotFoundError as ex:
        raise Exception( f'{args[0]} has not been found on PATH. Please install {args[0]} and add its installation directory to PATH, e.g. by editing OrientALShell.bat\n{ex.strerror}' )
    finally:
        loop.close()
    if ret:
        raise Exception( 'Call (cwd={}) failed:\n{} exited with error code {}. Output is:\n{}'.format(
                            cwd, ' '.join( args ), ret, output ) )

    logger.infoFile( f'Output of {args[0]}:\n' + output )


@contract
def convertPlyText2Binary( plyTextFn : Path, reduction ):
    plyBinFn = plyTextFn.with_suffix('.bin.ply')
    headerEnded = False
    plyTypes = []
    plyTypes2StructTypes = {
        'char'  : 'b',
        'uchar' : 'B',
        'short' : 'h',
        'ushort': 'H',
        'int'   : 'i',
        'uint'  : 'I',
        'float' : 'f',
        'double': 'd' }
    plyType2Ctor = { 'float' : float,
                     'double': float }
    with plyTextFn.open(newline='') as fin, \
         plyBinFn.open('wb') as fout:
        reader = csv.reader( fin, delimiter=' ', skipinitialspace=True )
        for row in reader:
            if not headerEnded:
                txt = None
                if row[0].startswith('format'):
                    txt = 'format binary_little_endian 1.0'
                elif row[0].startswith('property'):
                    typ, name = row[1:]
                    assert len(plyTypes) != 0 or name == 'x'
                    assert len(plyTypes) != 1 or name == 'y'
                    assert len(plyTypes) != 2 or name == 'z'
                    if len(plyTypes) < 3:
                        typ = 'double' # We write unreduced coordinates, so let's use double, always.
                    txt = ' '.join(( row[0], typ, name ))
                    plyTypes.append( typ.strip() )
                else:
                    txt = ' '.join(row)
                fout.write( ( txt + '\n' ).encode('ascii') )
                if row[0].strip() == 'end_header':
                    headerEnded = True
                    stru = struct.Struct( '<' + ''.join( plyTypes2StructTypes[el] for el in plyTypes ) )
                continue
            args = itertools.chain( ( float(el) + reduc for el, reduc in utils.zip_equal( row[:3], reduction ) ),
                                    ( plyType2Ctor.get( plyType, int )( col ) for plyType, col in utils.zip_equal( plyTypes[3:], row[3:] ) ) )
            fout.write( stru.pack( *args ) )

    plyBinFn.replace( plyTextFn )


@contract
def main( args : argparse.Namespace, cmdLine : 'list(str) | None' = None, parser : 'ArgumentParser|None' = None ):
    if args.level < 0:
        raise Exception( f'--level must be >= 0, but is {args.level}' )
    with suppress(FileExistsError):
        args.outDir.mkdir(parents=True)
    args.outDir = args.outDir.resolve()
    utils.argparse.applyLoggingGroup( args, args.outDir, logger, cmdLine )

    if not args.progress:
        Progress.deactivate()

    if not args.db:
        args.db = ( args.outDir / '..' / 'relOri' / 'relOri.sqlite' ).resolve()

    logger.infoScreen( 'Output directory: {}', args.outDir )
    utils.argparse.logScriptParameters( args, logger, parser )

    undistDir  = args.outDir / 'visualize'
    imgOriDir  = args.outDir / 'txt'
    imgMaskDir = args.outDir / 'masks'
    modelsDir  = args.outDir / 'models' # If this directory does not exist beforehand, then PMVS2 silently creates no output!
    bundlerOutFn = args.outDir / 'bundle.rd.out'
    bundlerListFn = args.outDir / 'list.txt'

    try:
        shutil.rmtree(modelsDir, ignore_errors=True)
        for direc in undistDir, imgOriDir, imgMaskDir, modelsDir:
            with suppress(FileExistsError):
                direc.mkdir()

        imgId2Idx = {}
        undistImages = {}
        undistCameras = {}
        sumResolutions = 0
        undistFns = []
        # Must be opened read-only, so file modification time stays the same, which we compare to the modification times of temporary files.
        with dbapi2.connect( '{}?mode=ro'.format( utils.db.uri4sqlite(args.db) ), uri=True ) as db, \
             bundlerOutFn.open('wt') as bundlerOut, \
             bundlerListFn.open('wt') as bundlerList:
            utils.db.initDataBase( db )

            # Pass reduced coordinates to PMVS. Convert text-PLY-files generated by PMVS to binary-PLY-files and undo the coordinate reduction.
            reduction = np.array( db.execute( 'SELECT avg(X0), avg(Y0), avg(Z0) FROM images WHERE camID NOTNULL' ).fetchone(), float )
            logger.info( f'Reduction point: {reduction}' )

            # Count perspective images only.
            nImgs = db.execute( 'SELECT count(*) FROM images WHERE camID NOTNULL' ).fetchone()[0]
            logger.info( 'Undistort {} photos', nImgs )
            progress = Progress( nImgs )

            nPts = db.execute('''
                WITH perspectiveObjPtIds AS (
                    SELECT DISTINCT objPts.id
                    FROM objPts
                    JOIN imgObs ON imgObs.objPtId == objPts.id
                    JOIN images ON imgObs.imgId == images.id
                    WHERE images.camId NOT NULL
                )
                SELECT count(*)
                FROM perspectiveObjPtIds ''').fetchone()[0]
            bundlerOut.write( '# Bundle file v0.3\n'
                             f'{nImgs} {nPts}\n' )

            # Profiling shows that distortion_inplace executed on all pixels consumes 50% of the run-time of the whole script, if executed for each photo anew.
            # Thus, order the results according to the undistortion parameters and reuse them.
            # Check DB file modification time stamp. If older than existing undistorted images, then skip undistortion.
            undistorter = Undistorter( args.db.stat().st_mtime, reduction )
            reusedTemporaries = []

            fiducialMatAttrs = [ 'fiducial_{}'.format(el) for el in 'A00 A01 A10 A11'.split() ]
            fiducialVecAttrs = [ 'fiducial_{}'.format(el) for el in 't0 t1'          .split() ]

            rows = db.execute('''
                SELECT images.id, images.path,
                       images.X0, images.Y0, images.Z0,
                       images.r1, images.r2, images.r3, 
                       images.parameterization,
                       {},
                       AsBinary(images.mask) AS mask,
                       cameras.x0 as x0_, cameras.y0 as y0_, cameras.z0 as z0_,
                       cameras.reference, cameras.normalizationRadius,
                       {},
                       cameras.id as camId,
                       images.nCols,
                       images.nRows
                FROM images
                JOIN cameras ON images.camID == cameras.id
                ORDER BY cameras.id, images.nCols, images.nRows, images.path '''
                .format(
                    ',\n'.join( 'images.{}'.format(el) for el in fiducialMatAttrs + fiducialVecAttrs ),
                    ',\n'.join( 'cameras.{}'.format(key) for key in adjust.PhotoDistortion.names.keys() )
                )
            )
            for iRow, row in enumerate(rows):
                with utils.progressAfter(progress):
                    assert row['reference'] == 'principalPoint'
                    imgId2Idx[row['id']] = iRow
                    camera = Camera( ior=np.array([ row[el] for el in 'x0_ y0_ z0_'.split() ]),
                                     adp=adjust.parameters.ADP(
                                         normalizationRadius=row['normalizationRadius'],
                                         referencePoint=adjust.AdpReferencePoint.names[row['reference']],
                                         array=np.array([ row[str(val[1])] for val in sorted( adjust.PhotoDistortion.values.items(), key=lambda x: x[0] ) ]) ),
                                     id=row['camId'] )
                    A = np.array([row[el] for el in fiducialMatAttrs], float).reshape((2,2))
                    t = np.array([row[el] for el in fiducialVecAttrs], float)
                    pix2cam = None
                    if np.isfinite(A).all() and np.isfinite(t).all():
                        pix2cam = ori.transform.AffineTransform2D( A, t )
                    mask_px = None
                    if row['mask'] is not None:
                        polyg = ogr.Geometry( wkb=row['mask'] )
                        ring = polyg.GetGeometryRef(0)
                        nPts = ring.GetPointCount()
                        mask_px = np.empty( (nPts,2) )
                        for iPt in range(nPts):
                            mask_px[iPt,:] = ring.GetPoint_2D(iPt)

                    image = Image( prc = np.array([ row[el] for el in 'X0 Y0 Z0'.split() ]),
                                   rot = adjust.parameters.EulerAngles( adjust.EulerAngles.names[row['parameterization']],
                                                                        np.array([ row[el] for el in 'r1 r2 r3'.split() ]) ),
                                   nCols=row['nCols'],
                                   nRows=row['nRows'],
                                   pix2cam=pix2cam,
                                   mask_px=mask_px,
                                   camId=camera.id,
                                   iorUndist=camera.ior )

                    relOriImagePath = Path(row['path'])
                    if not relOriImagePath.is_absolute():
                        relOriImagePath = args.db.parent / relOriImagePath
                    undistFn = undistDir  / f'{iRow:08}.jpg'
                    oriFn    = ( imgOriDir  / f'{iRow:08}' ).with_suffix('.txt')
                    maskFn   = ( imgMaskDir / f'{iRow:08}' ).with_suffix('.pbm')

                    undistResolution, ior_rect, reused = undistorter( relOriImagePath, undistFn, oriFn, maskFn, camera, image )
                    sumResolutions += undistResolution
                    image.iorUndist[:] = ior_rect
                    if reused:
                        reusedTemporaries.append( relOriImagePath.name )
                    undistFns.append( undistFn )

                    bundlerList.write( 'visualize' + os.path.sep + undistFn.name + '\n' )
                    bundlerOut.write('{:.16e} 0 0\n'.format( ior_rect[2] )) # Note: bundler's format implicitly assumes that the principal point is at the image center.
                    R = ori.euler2matrix( image.rot )
                    for line in R.T:
                        bundlerOut.write( ' '.join( str(el) for el in line ) + '\n')
                    bundlerOut.write( ' '.join( str(el) for el in -R.T.dot( image.prc - reduction ) ) + '\n' )

                    undistCameras[ camera.id ] = camera
                    undistImages[ row['id'] ] = image

            if reusedTemporaries:
                logger.info( 'Temporary files for {} are up-to-date and will be reused.', 'all images' if len(reusedTemporaries)==nImgs else ', '.join(reusedTemporaries) )

            for objPtId, X, Y, Z, red, green, blue, nObs in db.execute('''
                SELECT objPts.id,
                       X(objPts.pt), Y(objPts.pt), Z(objPts.pt),
                       avg(red), avg(green), avg(blue),
                       count(*)
                FROM objPts
                JOIN imgObs ON imgObs.objPtId == objPts.id
                JOIN images ON imgObs.imgId == images.id
                WHERE images.camId NOT NULL
                GROUP BY objPts.id '''):
                bundlerOut.write( f'{X-reduction[0]} {Y-reduction[1]} {Z-reduction[2]}\n'
                                  f'{int(red*255)} {int(green*255)} {int(blue*255)}\n'
                                  f'{nObs}' )
                for row in db.execute('''
                    SELECT id, x, y, imgId
                    FROM imgObs
                    WHERE objPtId = ? ''', [objPtId] ):
                    image = undistImages[row['imgId']]
                    camera = undistCameras[image.camId]
                    xy = np.array([[ row['x'], row['y'] ]])
                    if image.pix2cam is not None:
                        xy = image.pix2cam.forward(xy)
                    xy = ori.distortion( xy, camera.ior, ori.adpParam2Struct(camera.adp),
                                         ori.DistortionCorrection.undist )[0]
                    xy = ( xy - camera.ior[:2] ) / camera.ior[2] * image.iorUndist[2] + image.iorUndist[:2]
                    #  The pixel positions are floating point numbers in a coordinate system where the origin is the center of the image, the x-axis increases to the right, and the y-axis increases towards the top of the image
                    xy -= (image.nCols - 1) / 2, -(image.nRows - 1) / 2
                    bundlerOut.write(' {} {} {} {}'.format( imgId2Idx[row['imgId']], row['id'], *xy))
                bundlerOut.write('\n')

        # Choose CMVS' 'maximage' based on the mean image resolution and the amount of locally available RAM.
        # Maximum number of images to be loaded at the same time.
        # Surely all those images need to fit into memory. But how much additional memory is consumed by PMVS2?
        # Let's reserve as much RAM for PMVS2 itself as for the images -> // 2
        maxImages = max( 1, int( oriental.MemoryInfo.installedPhysicalMemory // ( sumResolutions / nImgs * 3 ) // 2 ) ) # We assume all images to have 3 bands with 8-bit radiometric resolution.

        # https://www.di.ens.fr/cmvs/documentation.html
        # Reads bundle.rd.out, list.txt
        # Writes ske.dat, vis.dat, centers-000?.ply, centers-all.ply
        # CMVS never treats any images as 'other' images, but all images as 'target' images.
        check_output([ 'cmvs.exe', '.' + os.path.sep, maxImages, os.cpu_count() ], cwd=args.outDir)

        # Reads ske.dat
        # Writes option-000?, pmvs.bat
        check_output( [ 'genOption.exe', '.' + os.path.sep,
                        args.level, # level
                        2, # csize
                        0.7, # threshold
                        7, # wsize
                        3, # minImageNum
                        os.cpu_count() ],
                      cwd=args.outDir,
                      progress=False )

        # genOption prepares pmvs.bat to call pmvs2 - however, with the prefix 'pmvs'!?
        pmvsBatFn = args.outDir / 'pmvs.bat'
        #with pmvsBatFn.open('rt') as pmvsBat:
        #    lines = pmvsBat.readlines()
        #with pmvsBatFn.open('wt') as pmvsBat:
        #    for line in lines:
        #        if not line.strip():
        #            continue
        #        first,second,third = line.split()
        #        pmvsBat.write(' '.join(( first, '.' + os.path.sep, third )) )
        #
        ## http://www.di.ens.fr/pmvs/documentation.html
        ## In order to make PMVS2 export not only the PLY file that contains the coloured point cloud, but additionally the .patch and .pset (for PoissonRecon) files, arguments must end with 'PATCH' and 'PSET', resp.
        ##check_output( [ 'pmvs2.exe', '.' + os.path.sep, optionFn.name ], cwd=args.outDir, env=env )
        #check_output( [ 'pmvs.bat' ], cwd=args.outDir )

        # Don't edit and then call pmvs.bat, because executing a batch-file on Windows would start CMD.exe,
        # and CMD.exe fails to start if the CWD is a UNC path.
        # Instead, execute each line in pmvs.bat as a separate command.
        with pmvsBatFn.open('rt') as pmvsBat:
            for line in pmvsBat:
                if not line.strip():
                    continue
                first,_,third = line.split()
                # http://www.di.ens.fr/pmvs/documentation.html
                # In order to make PMVS2 export not only the PLY file that contains the coloured point cloud, but additionally the .patch and .pset (for PoissonRecon) files, arguments must end with 'PATCH' and 'PSET', resp.
                #check_output( [ 'pmvs2.exe', '.' + os.path.sep, optionFn.name ], cwd=args.outDir, env=env )
                check_output( [first, '.' + os.path.sep, third], cwd=args.outDir )


        centersAllPly = args.outDir / 'centers-all.ply'
        for fn in itertools.chain( args.outDir.glob('centers-????.ply'),
                                   [centersAllPly] if centersAllPly.exists() else [],
                                   modelsDir.glob('*.ply') ):
            convertPlyText2Binary( fn, reduction )
        logger.info( f'Point cloud files created in {modelsDir}' )

    finally:
        if args.deleteTemporaries:
            for pattern in 'bundle.rd.out', 'centers-????.ply', 'list.txt', 'option-????', 'pmvs.bat', 'ske.dat', 'vis.dat':
                for fn in args.outDir.glob(pattern):
                    fn.unlink()
            for direc in undistDir, imgOriDir, imgMaskDir:
                shutil.rmtree( direc, ignore_errors=True )
