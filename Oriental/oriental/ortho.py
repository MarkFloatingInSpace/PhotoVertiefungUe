# -*- coding: cp1252 -*-
"""Create an orthophoto mosaic based on geo-referenced aerial imagery.

Use the output of `absOri` or `relOri`. The orthophoto mosaic will be
based on a surface model created from the imagery itself by dense
image matching. Note that the dense matching software must be available
on PATH.
"""
import oriental
from oriental import config, Progress, blocks, log, ori, adjust, utils
import oriental.adjust.parameters
import oriental.blocks.types
import oriental.utils.gdal
import oriental.utils.db
import oriental.utils.dsm
import oriental.utils.argparse
import oriental.utils.filePaths
import oriental.utils.photo

import sys, os, subprocess, argparse, shutil, fnmatch, re, glob, itertools
from pathlib import Path
import sqlite3
from sqlite3 import dbapi2
from contextlib import suppress

import numpy as np
from contracts import contract
from osgeo import ogr
ogr.UseExceptions()

DenseMatcher = utils.argparse.ArgParseEnum('DenseMatcher', 'sure')
Camera = blocks.types.createCompact( blocks.types.Camera, 'id' )
Image = blocks.types.createCompact( blocks.types.Image, 'nCols nRows pix2cam mask_px'.split() )

logger = log.Logger("ortho")

@contract
def parseArgs( args : 'list(str)|None' = None ):
    docList = __doc__.splitlines()
    parser = argparse.ArgumentParser( description=docList[0],
                                      epilog='\n'.join( docList[1:] ),
                                      formatter_class=utils.argparse.Formatter )

    parser.add_argument( '--outDir', default=Path.cwd() / "ortho", type=Path,
                         help='Store results in directory OUTDIR.' )
    parser.add_argument( '--db', type=Path,
                         help='Rectify perspective images found in DB. Default: OUTDIR/../absOri/absOri.sqlite' )
    parser.add_argument( '--gsd', type=float,
                         help='Ground sampling distance.' )
    parser.add_argument( '--denseMatcher', default=DenseMatcher.sure, choices=DenseMatcher, type=DenseMatcher,
                         help='Dense matching external software to use.' )
    parser.add_argument( '--no-gpu', dest='gpu', action='store_false',
                         help='Do not use the GPU for processing.' )
    parser.add_argument( '--baseImages', default=['*'], nargs='*',
                         help='Process as base images only those that match these patterns. Supported wildcards: *, ?, [character set], [!character set]. Defaults to all perspective images in DB.' )
    parser.add_argument( '--minfold', type=int,
                         help='Each point must be observed in at least MINFOLD stereo models. Defaults to 2 for >3 images, and to 1 otherwise.' )
    parser.add_argument( '--maxmodels', default=5, type=int,
                         help='Maximum number of stereo models per image.' )
    parser.add_argument( '--no-separateTrueOrthos', dest='separateTrueOrthos', action='store_false',
                         help='Do not create a true orthophoto for each aerial.' )
    parser.add_argument( '--no-separateClassicOrthos', dest='separateClassicOrthos', action='store_false',
                         help='Do not create a classic (non-true) orthophoto for each aerial.' )
    parser.add_argument( '--externalDsm', default='dhm_lamb_10m.tif', type=Path,
                         help="Create classic orthophotos based on this external surface model. If a relative path, file is also searched in $ORIENTAL/data." )
    #parser.add_argument( '--area', nargs=4, default=None, type=float,
    #                     help='Area of interest: xmin xmax ymin ymax' )
    parser.add_argument( 'addArgsDenseMatcher', nargs='*',
                         help='Pass additional arguments to the dense matcher.' )
    parser.add_argument( '--keepTemporaries', action='store_true',
                         help='Do not delete temporary files, including those produced by SURE.' )
    parser.add_argument( '--reuseRectified', action='store_true',
                         help='Reuse already existing rectified aerials.' )
    utils.argparse.addLoggingGroup( parser, "orthoLog.xml" )

    generalGroup = parser.add_argument_group('General', 'Other general settings')
    generalGroup.add_argument( '--no-progress', dest='progress', action='store_false',
                               help="Don't show progress in the console." )

    cmdLine = sys.argv[:]
    args = parser.parse_args( args=args )
    main( args, cmdLine, parser )

# I've tried to parallelize the undistortion of all photos using multiprocessing in Python,
#   boost::interprocess::named_lock in utils::gdal to synchronize file system access, and
#   os.environ['OMP_NUM_THREADS']='1' to prevent ori and cv2 from being parallelized themselves.
# However, that even slowed down the whole process! Seemingly, everything except file access is well parallelized anyway.
# The most expensive operations are probably: decompression/compression in GDAL (which was synchronized by boost::interprocess, so no gain), and ori.undistort
class Undistorter( utils.photo.Undistorter ):
    def __init__(self):
        super().__init__()

    @contract
    def __call__( self, distFn : Path, undistFn : Path, oriFn : Path, camera : Camera, image : Image ) -> None:
        # Create undistorted aerials. Even better would be to pass transformed ADP to SURE, but it seems impossible to transform our model to theirs.
        # Check: IWITNESS (5 parameters – a variation of the Brown model)
        #        INPHOCOEFF (12 parameters).
        pho_rect, pho_rect_mask, phoInfo, ior_rect = super().__call__( distFn, camera.ior, camera.adp, image.pix2cam, image.mask_px )

        # SURE does not support explicit masks (as an alpha-channel or as TIFF 1-bit mask),
        # but SURE considers completely black pixels as invalid - see mail from Mathias Rothermel 2015-06-19
        # Thus, in order to make sure nothing outside the original image content gets matched,
        # we need to store the files with lossless compression (LZW),
        # and we explicitly set the invalid image area, as defined by `mask` to 0,0,0.
        pho_rect[ np.logical_not(pho_rect_mask) ] = 0
        
        phoInfo.description = f'{distFn.name}, undistorted.'
        # SURE considers (0,0,0)-pixels as background/invalid. If we use a lossy compression, then the pixels close to the foreground will not be considered as background.
        # Unfortunately, SURE does not consider TIFF bit masks.
        phoInfo.compression = utils.gdal.Compression.lzw
        #utils.gdal.imwrite( str(undistFn), pho_rect, self.mask, info=phoInfo, maskStorage=utils.gdal.MaskStorage.bitMask )
        utils.gdal.imwrite( str(undistFn), pho_rect, info=phoInfo )

        #pixelSize_mm = row['sensorWidth_mm'] /  row['nCols']
        #focal_mm = row['Z0']  * pixelSize_mm
        # According to ftp://ftp.ifp.uni-stuttgart.de/sure_public/SURE_Coordinate_Systems.pdf,
        # $IntOri_FocalLength and $IntOri_PixelSize are not used by SURE, and can be set to 1.

        # Only $IntOri_PrincipalPoint or $IntOri_CameraMatrix may be left out, as it seems. If any other entry is left out, then sure.exe -prep returns an error.

        # For the SURE Coordinate systems and their transformations, see:
        # ftp://ftp.ifp.uni-stuttgart.de/sure_public/SURE_Coordinate_Systems.pdf
        # i.e. the offset vector is the same as in ORIENTAL.
        # but the transpose of our rotation matrix needs to be post-rotated about the y-axis by 200gon
        # TODO: SURE's definition of the rotation matrix results in the camera-x-axis pointing rightwards, and the camera-y-axis pointing downwards,
        # and the image CS's origin is at the center of the top/left pixel. Thus, image coordinates are the same as with most libraries.
        # By using SURE's definitions internally, many annoying forward/backward transformations of 2D image coordinate systems could be skipped!

        # Create undistorted aerials. Even better would be to pass transformed ADP to SURE, but it seems impossible to transform our model to theirs.
        # Check: IWITNESS (5 parameters – a variation of the Brown model)
        #        INPHOCOEFF (12 parameters).

        Rx200 = np.diag([1.,-1.,-1.])
        R = Rx200.dot( ori.euler2matrix(image.rot).T )
        with oriFn.open('wt') as oriFile:
            oriFile.write("""$ImageID___________________________________________________(ORI_Ver_1.0)
{imgID}
$IntOri_FocalLength_________________________________________________[mm]
{focal_mm}
$IntOri_PixelSize______(x|y)________________________________________[mm]
{pixelSize_mm}\t{pixelSize_mm}
$IntOri_SensorSize_____(x|y)_____________________________________[pixel]
{sensorSizeX_px}\t{sensorSizeY_px}
$IntOri_PrincipalPoint_(x|y)_____________________________________[pixel]
{ppX_px}\t{ppY_px}
$IntOri_CameraMatrix_____________________________(ImageCoordinateSystem)
{K}
$ExtOri_RotationMatrix____________________(World->ImageCoordinateSystem)
{R}
$ExtOri_TranslationVector________________________(WorldCoordinateSystem)
{P0}
$IntOri_Distortion_____(Model|ParameterCount|(Parameters))______________
{distortion}
""".format(
    imgID=undistFn.name,
    focal_mm=1.,
    pixelSize_mm=1.,
    sensorSizeX_px=image.nCols, sensorSizeY_px=image.nRows,
    ppX_px=ior_rect[0], ppY_px=-ior_rect[1],
    K="""{z0}\t0.\t{x0}
0.\t{z0}\t{y0}
0.\t0.\t1.""".format( x0=ior_rect[0], y0=-ior_rect[1], z0=ior_rect[2] ),
        R='\n'.join( '\t'.join('{}'.format(el) for el in row ) for row in R ),
        P0='\t'.join( '{}'.format(el) for el in image.prc),
        distortion='NONE	  0'
        ) )

@contract
def check_output( args : list, cwd : Path, **kwargs ):
    logger.infoFile( 'Calling (cwd={}): {}', cwd, ' '.join(args) )
    try:
        subprocess.check_output( args,
                                 stderr=subprocess.STDOUT,
                                 cwd=str(cwd),
                                 universal_newlines=True,
                                 **kwargs )
    except subprocess.CalledProcessError as ex:
        raise Exception( "Call (cwd={}) failed with error code {}:\n{}\nOutput is: {}".format(
                         cwd, ex.returncode, ' '.join(args), ex.output.expandtabs(2) ) )


@contract
def main( args : argparse.Namespace, cmdLine : 'list(str) | None' = None, parser : 'ArgumentParser|None' = None ):

    with suppress(FileExistsError):
        args.outDir.mkdir(parents=True)
    args.outDir = args.outDir.resolve()
    utils.argparse.applyLoggingGroup( args, args.outDir, logger, cmdLine )

    if not args.progress:
        Progress.deactivate()

    if not args.db:
        args.db = ( args.outDir / '..' / 'absOri' / 'absOri.sqlite' ).resolve()

    if args.separateClassicOrthos:
        if not args.externalDsm:
            raise Exception('If separate non-true orthophotos shall be created, then --externalDsm must be specified.')
        infoExternalDsm = utils.dsm.info( args.externalDsm )

    logger.infoScreen( 'Output directory: {}', args.outDir )
    utils.argparse.logScriptParameters( args, logger, parser )

    undistDir = args.outDir / 'undistorted'
    with suppress(FileExistsError):
        undistDir.mkdir()
    oriDir = undistDir / 'ori' # SURE now does not accept any more image and .ori-files in the same folder. Hence, place the .ori's in a sub-folder
    with suppress(FileExistsError):
        oriDir.mkdir()
    baseImgFns = []
    undistOriFns = []
    objSpaceBbox2d = np.array([ np.inf, np.inf, -np.inf, -np.inf ])
    with dbapi2.connect( '{}?mode=ro'.format( utils.db.uri4sqlite(args.db) ), uri=True ) as absOriDb:
        utils.db.initDataBase( absOriDb )

        projCsWkt = absOriDb.execute(f"""
            SELECT value
            FROM config
            WHERE name == '{utils.db.ConfigNames.CoordinateSystemWkt}' """).fetchall()
        if not projCsWkt:
            logger.warning( 'DB defines no coordinate system.')
            projCsWkt = None
        else:
            projCsWkt = projCsWkt[0][0]

        # compute the median of z-coordinates of object points to help SURE
        terrainHeight = absOriDb.execute("""
            SELECT AVG(Z)
            FROM (
                SELECT Z(pt) as Z
                FROM objPts
                ORDER BY Z
                LIMIT 2 - (SELECT COUNT(*) FROM objPts) % 2    -- odd 1, even 2
                OFFSET (
                    SELECT (COUNT(*) - 1) / 2
                    FROM objPts
                )
            ) """).fetchone()[0]

        # don't count mosaic.tif and the orthophoto!
        nImgs = absOriDb.execute("SELECT count(*) FROM images WHERE camID NOTNULL").fetchone()[0]
        if args.minfold is None:
            args.minfold = 2 if nImgs > 3 else 1 # 2 is SURE's default.
        logger.info('Undistort {} photos', nImgs)
        progress = Progress( nImgs )

        # Profiling shows that distortion_inplace executed on all pixel consumed 50% of the run-time of the whole script, if executed for each photo anew.
        # Thus, order the results according to the undistortion parameters and reuse them.
        undistorter = Undistorter()

        fiducialMatAttrs = [ 'fiducial_{}'.format(el) for el in 'A00 A01 A10 A11'.split() ]
        fiducialVecAttrs = [ 'fiducial_{}'.format(el) for el in 't0 t1'          .split() ]

        rows = absOriDb.execute("""
            SELECT images.path,
                   images.X0, images.Y0, images.Z0,
                   images.r1, images.r2, images.r3, 
                   images.parameterization,
                   {},
                   AsBinary(images.mask) AS mask,
                   cameras.x0 as x0_, cameras.y0 as y0_, cameras.z0 as z0_,
                   cameras.reference, cameras.normalizationRadius,
                   {},
                   cameras.id as camId,
                   -- relOri.cameras.sensorWidth_mm,
                   -- relOri.cameras.sensorHeight_mm,
                   images.nCols,
                   images.nRows
            FROM images
                JOIN cameras
                    ON images.camID == cameras.id
            ORDER BY cameras.id, images.nCols, images.nRows, images.path """
            .format( 
                ',\n'.join( 'images.{}'.format(el) for el in fiducialMatAttrs + fiducialVecAttrs ),
                ',\n'.join( 'cameras.{}'.format(key) for key in adjust.PhotoDistortion.names.keys() )
            )
        )
        for row in rows:
            with utils.progressAfter(progress):
                assert row['reference'] == 'principalPoint'
                absOriImagePath = Path(row['path'])
                if not absOriImagePath.is_absolute():
                    absOriImagePath = args.db.parent / absOriImagePath
                undistFn = ( undistDir / absOriImagePath.name ).with_suffix('.tif')
                if any( fnmatch.fnmatch( row['path'], pattern ) for pattern in args.baseImages ):
                    baseImgFns.append( undistFn.stem ) # SURE ignores any file extensions.
                oriFn = ( oriDir / undistFn.name ).with_suffix('.ori')
                
                # if we didn't compute objSpaceBbox2d, we could continue already here.
                undistOriFns.append(( undistFn, oriFn ))

                camera = Camera( ior=np.array([ row['x0_'], row['y0_'], row['z0_'] ]),
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
                               mask_px=mask_px )

                def getCorners( nRows, nCols ):
                    # return the centers of pixels at the corners and the mid-points of the image borders, as these are the points where distortion is expected to be extreme.
                    r = -(nRows-1)
                    c =   nCols-1
                    return np.array([ [ 0  , 0   ],
                                      [ 0  , r/2 ],
                                      [ 0  , r   ],
                                      [ c/2, r   ],
                                      [ c  , r   ],
                                      [ c  , r/2 ],
                                      [ c  , 0   ],
                                      [ c/2, 0   ] ], float )
                def extendBbox():
                    corners = getCorners( image.nRows, image.nCols )
                    if image.pix2cam:
                        corners = image.pix2cam.forward(corners)
                    ori.distortion_inplace( corners, camera.ior, ori.adpParam2Struct( camera.adp ), ori.DistortionCorrection.undist )
                    R = ori.euler2matrix( image.rot )
                    cornersOnGround = []
                    for corner in corners:
                        ray = R.dot( np.r_[ corner, 0. ] - camera.ior )
                        factor = ( terrainHeight - image.prc[2] ) / ray[2]
                        cornersOnGround.append( (image.prc + factor*ray)[:2] )
                    cornersOnGround = np.array( cornersOnGround )
                    bboxOnGround = np.r_[ cornersOnGround.min(axis=0), cornersOnGround.max(axis=0) ]
                    objSpaceBbox2d[:2] = np.minimum( objSpaceBbox2d[:2], bboxOnGround[:2] )
                    objSpaceBbox2d[2:] = np.maximum( objSpaceBbox2d[2:], bboxOnGround[2:] )
                extendBbox()

                if args.reuseRectified and undistFn.exists() and oriFn.exists():
                    continue

                undistorter( absOriImagePath, undistFn, oriFn, camera, image )

    if not len(baseImgFns):
        raise Exception('Not even 1 photo selected as base image')
    logger.info( 'Use {} photo{} as base images', len(baseImgFns), 's' if len(baseImgFns)!=1 else '' )
    # call SURE
    # explicitly set the GSD. Maybe also the extents, such that they are rounded to meters or decimeters.
    # Can we show (and log) the output of SURE while SURE is running?
    sureWorkDir = args.outDir / 'SURE'
    with suppress(FileNotFoundError):
        shutil.rmtree( str( sureWorkDir ) )

    # mdoneus had problems starting sure.exe from within this script, while calling sure.exe with cwd=<sure install dir> succeeded.
    # -> log PATH in utils.argparse.applyLoggingGroup
    # -> use shutil.which to determine the absolute path to sure.exe, and log it.
    # Determine the absolute path to sure.exe, so
    # - users can check if the intended one is used, and
    # - we can clear PATH from all other directories
    sureExe = shutil.which('sure.exe')
    if sureExe is None:
        raise Exception( 'sure.exe has not been found on PATH. Please install SURE and add its installation directory to PATH, e.g. by editing OrientALShell.bat' )
    sureEnv = os.environ.copy()
    sureEnv['PATH'] = os.path.dirname( os.path.abspath(sureExe) )

    sureArgs = [ sureExe,
                 '-basepath', str(sureWorkDir.relative_to(args.outDir)),
                 '-no_gui', # does no_gui have any effect?
                 '-no_vis',
                 '-no_update' # suppress the automatic update
               ]
    try:
        def replaceInSureControlFile( fn : str, replacements : dict ):
            if not len(replacements):
                return
            with ( sureWorkDir / fn ).open('rt') as fin:
                lines = fin.readlines()
            rexControlEntry = re.compile(r'\s*\$(?P<key>\w+)\s*=\s*')
            replaced = []
            with ( sureWorkDir / fn ).open('wt') as fout:
                for line in lines:
                    match = rexControlEntry.match(line)
                    if match:
                        key = match.group('key')
                        value = replacements.get( key )
                        if value is not None:
                            fout.write( '\t${} = {}\n'.format(key,value) )
                            del replacements[key] # speedup
                            replaced.append((key,value))
                            continue
                    fout.write( line )
            logger.verbose('Changed following parameters in {}\n'
                           'parameter\tvalue\n'
                           '{}',
                           Path(fn).name,
                           '\n'.join( '{}\t{}'.format(*els) for els in replaced ) )

        sureArgsPrep = sureArgs + [
                        '-img', str(undistDir.relative_to(args.outDir)),
                        '-ori', str(oriDir.relative_to(args.outDir)),
                        '-fold', str(args.minfold),
                        '-pyr', '0', # full resolution
                        '-gpu' if args.gpu else '-cpu',
                        #'-no_interp', '-no_refine', # Interpolation of DSM holes and subsequent refinement of orthophotos does not consider the invalid image borders we have created during undistortion. So skip it.
                        '-interpiwd', '-refine', # Still, md wants orthophotos without holes. -interpiwd is slower, but yields nicer results than -interp
                        '-no_mesh',
                        #'-no_dsmmesh', # no effect?
                        #'-no_dense_cloud', # no effect?
                        # With -no_tile, 1 LAS file is stored for each base image: 3D_Points/pts_02110503_047.laz
                        # Without -no_tile, many consecutively numbered LAS files are stored instead : 3D_Points/0.laz
                        # During DSM generation, probably only the needed (smaller) tiles are read in, which is probably more efficient, especially for highly overlapping imagery.
                        #'-no_tile',
                        '-no_gtif', # turn off controlDsm.txt's $WriteGTif
                        '-laz',
                        '-dsm',
                        '-maxmodels', str(args.maxmodels),
                        '-initterrain', str(terrainHeight) # controlBootStrap.txt says: "Use image selection based on terrain height for manually set area? ( yes / no )" - what does that mean, really?
                       ]
        if args.gsd:
            sureArgsPrep.extend(['-gsd', str(args.gsd)])
        if nImgs != len(baseImgFns):
            baseImgsFn = args.outDir / 'baseImages.txt'
            with baseImgsFn.open('wt') as baseImgsFile:
                baseImgsFile.write( '\n'.join(baseImgFns))
            sureArgsPrep.extend(['-select', baseImgsFn.name])
        if args.addArgsDenseMatcher:
            #sureArgsPrep.append('-area')
            sureArgsPrep.extend(args.addArgsDenseMatcher)
        if not args.keepTemporaries:
            sureArgsPrep.append('-no_depth') # this turns off controlTriang.txt's $DepthImgOut and $AccDepthImgOut -> 3D_Points\pts_02110503_047.tif, 3D_Points\acc_02110503_047.tif
        sureArgsPrep.append('-prep')
        logger.info('Prepare for dense image matching.')
        check_output( args=sureArgsPrep, cwd=args.outDir, env=sureEnv )

        # Let's not edit SURE's control files, because the meanings of their entries change too frequently, and the preconditions to get valid DSM and orthophoto tiles are undocumented.
        # Anyway, we would edit the control files only in order to save temporary disk space and some file I/O time.

        ## for some reason, $AccPointsOut can only be turned off if -no_tile has been passed. Otherwise, no DSM will be generated (the SURE output then says 'skipping empty tile' for all tiles!)
        ## Anyway, if -no_tile is passed, then no 3D_Points/acc_DSC00632.tif - files are written, even if $AccPointsOut is turned on!
        #replacements = {}
        #if not args.keepTemporaries:
        #    if 0:#not args.keepTemporaries:
        #        # -no_cloud turns off controlTriang.txt's 'PointsOut', which controls if \3D_Points\pts_02110503_047.laz is written, which is necessary for generating a DSM afterwards!
        #        replacements.update( { 'AccPointsOut' : 0 } )
        #    replacements.update( { 'DepthImgOut'     : 0,    # don't write \3D_Points\pts_DSC00914.tif ? Very large files.
        #                           'AccDepthImgOut'  : 0 } ) # don't write \3D_Points\acc_DSC01170.tif ? Very large files.
        #replaceInSureControlFile( 'controlTriang.txt', replacements )
        #
        #replacements = {}
        ## I'm not sure whether this makes sense: process monitor shows that during the generation of DSM tiles, sure.exe reads the LAS-files in \3D_Points\
        ## Probably, this process is very much I/O bound, and thus it is inefficient to do it in multiple threads,
        ## especially if there is high overlap (many base images (or point clouds, resp.) available for each tile.
        ## Probably, that's also the reason why controlDsm.txt's $UseCUDA is turned off, even if -gpu is passed.
        ##replacements = { 'MaxThreadsDsm' : os.cpu_count() }
        #if not args.keepTemporaries:
        #    replacements.update( { 'MakeHeightImageVis'     : 0,
        #                           'MakeHeightImage'        : 0,
        #                           'SaveLasNotInterpolated' : 0 # don't write DSM/Cloud/DSM_Cloud_0001_-003.las
        #                          } )
        #replaceInSureControlFile( 'controlDsm.txt', replacements )

        logger.infoFile( 'Calling (cwd={}): {}', args.outDir, ' '.join(sureArgs) )
        if 0:
            subprocess.check_output( sureArgs,
                                     stderr=subprocess.STDOUT,
                                     cwd=str(args.outDir),
                                     env=sureEnv,
                                     universal_newlines=True )
        else:
            # http://stackoverflow.com/questions/2715847/python-read-streaming-input-from-subprocess-communicate/17698359#17698359
            outputs = []
            iStage = 0
            class Stage:
                def __init__( self, rexText, description ):
                    self.rex = re.compile(rexText)
                    self.description = description
                    self.progress = None
            # Note: SURE outputs to stdout and to its log file different texts.
            stages = [ Stage( r'\s*BASE\s+IMAGE\s+(?P<curr>\d+)\s+/\s+(?P<total>\d+)\s*$', # BASE IMAGE 1 / 3
                              'Dense image matching'  ),
                       Stage( r'\s*Thread\s+\d+\s+finished\s+tile\s+(?P<curr>\d+)\s+of\s+(?P<total>\d+)\s*', # Thread 1 finished tile 1 of 225   [(9,18) of (9,25)]
                             'DSM generation'        ),
                       Stage( r'\s*Refining\stile\s\w+\s\[(?P<curr>\d+)/(?P<total>\d+)\]', # Refining tile DSM_66186_47354 [1/60]
                             'Orthophoto refinement' ) ]
            with subprocess.Popen( sureArgs,
                                   bufsize=1, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   cwd=str(args.outDir),
                                   env=sureEnv,
                                   universal_newlines=True ) as proc:
                try:
                    match = None
                    currStage = stages[iStage]
                    nextStage = stages[iStage+1]
                    for line in proc.stdout:
                        while 1:
                            if not match:
                                match = currStage.rex.match( line )
                            if match:
                                if currStage.progress is None:
                                    logger.info(currStage.description)
                                    currStage.progress = Progress( int(match.group('total')) )
                                else:
                                    currStage.progress += 1
                            elif nextStage:
                                # Unfortunately, SURE's output is not consistent: having encountered "Thread 1 finished tile 1 of 225", the highest tile number that is output may be much lower than 255.
                                # Thus, we cannot know when one stage has ended, unless the next regex matches!
                                match = nextStage.rex.match(line)
                                if match:
                                    if currStage.progress is not None:
                                        currStage.progress.finish()
                                        currStage.progress = None
                                    currStage = nextStage
                                    iStage += 1
                                    nextStage = stages[iStage+1] if iStage+1 < len(stages) else None
                                    continue
                            match = None
                            break
                        outputs.append(line)
                    for stage in stages:
                        if stage.progress is not None:
                            stage.progress.finish() # SURE seems to skip one or more tiles without emitting respective messages to stdout.
                except:
                    proc.kill()
                    proc.wait()
                    raise
            if proc.returncode != 0:
                raise subprocess.CalledProcessError( proc.returncode, proc.args, ''.join(outputs) )
            logger.verbose( ''.join(outputs).expandtabs(2), tag='SURE output' )
    except FileNotFoundError as ex:
        raise Exception( 'sure.exe has not been found on PATH. Please install SURE and add its installation directory to PATH, e.g. by editing OrientALShell.bat\n{}'.format(ex.strerror) )
    except subprocess.CalledProcessError as ex:
        # In SURE version 1.2.0.644 (2015-09-16), sure.exe exits with code 1, even though its output seems okay.
        # Thus, issue a respective warning only.
        #raise Exception( 'SURE has exited with error code {}.\nOutput is:\n{}'.format( ex.returncode, ex.output ) )
        logger.warning( "SURE has exited with error code {}. However, its output might still be okay, and so we continue here. SURE's output is: {}", ex.returncode, ex.output.expandtabs(2) ) # ex.output may contain tabs. Avoid them being formatted as table cell separators by the logging system. 


    # The simple orthophoto-tiles seem okay, even if the blending could be better. Also, the light-falloff towards the image corners for long focal lengths is noticable.
    # However, the 'refined' orthophotos based on the interpolated DSM (no holes) are weird. Seemingly, invalid image areas are not considered as such during the 'refinement'.

    # args.gsd may be None. So read-out the actual GSD. We could parse the SURE control files. Instead, let's simply read the meta-data of one of the orthophoto tiles it has produced.
    dsmGsd = None
    logger.info('Merge DSM and orthophoto tiles.')
    # post-process the ortho-photos and DSMs: set the CS. ExifTool seems to be unable to do that, because it merely allows for copying en masse the block tags GeoTiffDirectory, GeoTiffDoubleParams and GeoTiffAsciiParams. 
    #for iDataset,(datasetName,listFn) in enumerate(( ('orthophoto','ortholist.txt'),
    #                                                 ('dsm','dsmlist.txt') )):
    for iDataset,(datasetName,direc) in enumerate(( ('orthophoto'        ,'Ortho'),
                                                    ('orthophoto_refined','Ortho_Refined'),
                                                    ('dsm'               ,'DSM'),
                                                    ('dsm_interpolated'  ,'DSM_Interpolated') )):
        logger.infoFile( 'Post-processing {}: set{} NODATA, merge tiles, write .prj sidecar file.',
                         datasetName,
                         ' coordinate system and' if projCsWkt is not None else '')
        tileFns = []
        # SURE writes dsmlist.txt, dsmextendedlist.txt, ortholist.txt; But it does not write orthorefinedlist.txt. So let's forget about the text files, and simply glob the output directories.
        #with ( sureWorkDir / listFn ).open() as fin:
        #    for relImgFn in fin:
        #        relImgFn = relImgFn.strip()
        #        if not relImgFn:
        #            continue
        #        imgFn = ( sureWorkDir / relImgFn ).resolve()
        if 1:
            for imgFn in glob.iglob( str( sureWorkDir / 'DSM' / direc / '*.tif' ) ):
                # even though these files may be deleted at the end of the script, we still set their projection, to make utils.gdal.mergeTiles create the mosaic with according metadata and world file.
                if projCsWkt is not None:
                    utils.gdal.setProjection( str(imgFn), projCsWkt )
                if iDataset < 2:
                    # SURE exports 3-channel images. Invalid pixels are not defined by a mask, but by values of (0,0,0)
                    # While GDAL now supports NO_DATA values for whole datasets (and not only single bands),
                    # the only way how to communicate to QGIS invalid pixels is an alpha-channel or a per-raster-band-no-data-value.
                    # We do not want to add such a channel, and we do not want to render all pixels invalid that have only 1 or 2 color bands set to zero!
                    # QGIS users may manually set dataset-wide NO_DATA values for each layer (image) separately. This is cumbersome, but we leave it this way for now:
                    # Layer properties -> Transparency -> User defined transparency -> transparent pixel list.
                    # Even if we delete these files right after merging, we need to tell mergeTiles how SURE defines NO_DATA
                    tileInfo, = utils.gdal.imread( str(imgFn), skipData=True )
                    utils.gdal.setDataSetNoData( str(imgFn), np.zeros( 1 if tileInfo.bands==utils.gdal.Bands.grey else 3 ) )
                    if dsmGsd is None:
                        dsmGsd = tileInfo.geotransform[0,1]
                else:
                    # SURE stores DSM images as 1-band float32, with invalid values being NaN
                    # As it is single-band, set a per-raster-band-no-data-value, which is interpreted by most graphics software (unlike per-dataset-no-data-value).
                    # The per-raster-band-no-data-value is set by mergeTiles for any raster files of type float32.
                    #utils.gdal.setDataSetNoData( str(imgFn), np.zeros( 1 ) * np.nan )
                    pass
                tileFns.append( str(imgFn) )

        if not len(tileFns):
            raise Exception("SURE has not produced any {} tiles.".format(datasetName))

        mosaicFn = args.outDir / '{}.tif'.format(datasetName)
        # TODO: need to set NODATA value for DSM to nan.
        # It seems that one cannot do this after calling mergeTiles, because mergeTiles creates overviews, and setting NODATA afterwards will not affect those overviews.
        # Thus, mergeTiles needs to consider the NODATA values of the input already.
        utils.gdal.mergeTiles( tileFns, str(mosaicFn) )

        # write a .prj sidecar-file, containing the projection as WKT-string. 
        # md wants this .prj for usage in Esri products. But what do they accept? http://support.esri.com/fr/knowledgebase/techarticles/detail/14056 seems to tell that it must be WKT without line breaks, as is exported here.
        if projCsWkt is not None:
            with mosaicFn.with_suffix('.prj').open('wt') as fout:
                fout.write( projCsWkt )
        logger.info( '{} mosaic saved to {}', datasetName, mosaicFn.relative_to(args.outDir) )

    def createSingleOrthos( trueOrthos : bool, outDir : Path ):
        # For each file in dsmSingle.txt that overlaps to at least some percent with the aerial in imgSingle.txt,
        # SURE produces an uncompressed TIFF-file in DSM/Ortho_Refined, with the resolution of the DSM.
        # Thus, we need to resample the external DTM to the wanted resolution of the orthophoto.
        # Also, it is thus not only more efficient to provide small, non-overlapping DTM tiles (instead of a single DTM-file),
        # but SURE seems to not at all be able to process really large TIFF files (e.g. dhm_lamb_10m.tif).
        # Note that the position and size of the tiles influence the result: image content that covers only a few percent of a DTM tile will not be re-mapped to an orthophoto tile!
        # Thus, small rectangles may be missing at the borders of the merged orthophoto.

        # Move original content of DSM_Extended and Ortho_Refined somewhere else, and produce the DTM tiles in DSM_Extended.
        # For each aerial, call SURE, merge the produced orthophoto tiles (store with projection, JPEG-compression and mask),
        # and clear the contents of Ortho_Refined before the next call.
        # Finally, move back to their original locations the original DSM and orthophoto-tiles produced by SURE for the whole image data set.
        gdalTranslateExe = shutil.which('gdal_translate.exe')
        if gdalTranslateExe is None:
            raise oriental.ExcInternal( "gdal_translate.exe has not been found, while it should have been shipped with OrientAL." )

        sureModuleTrueOrthoExe = shutil.which('ModuleTrueOrtho.exe')
        if sureModuleTrueOrthoExe is None:
            raise Exception( "SURE's ModuleTrueOrtho.exe has not been found on PATH. Please install SURE and add its installation directory to PATH, e.g. by editing OrientALShell.bat" )
        sureModuleTrueOrthoEnv = os.environ.copy()
        sureModuleTrueOrthoEnv['PATH'] = os.path.dirname( os.path.abspath(sureModuleTrueOrthoExe) )

        dsmExtendedDir  = sureWorkDir / 'DSM' / 'DSM_Extended'
        orthoRefinedDir = sureWorkDir / 'DSM' / 'Ortho_Refined'
        origDsmExtendedDir  = Path( str(dsmExtendedDir)  + '_orig' )
        origOrthoRefinedDir = Path( str(orthoRefinedDir) + '_orig' )
        with suppress(FileNotFoundError):
            shutil.rmtree( str(origDsmExtendedDir) )
        with suppress(FileNotFoundError):
            shutil.rmtree( str(origOrthoRefinedDir) )
        imgSingleTxt = sureWorkDir / 'imgSingle.txt'
        oriSingleTxt = sureWorkDir / 'oriSingle.txt'
        dsmSingleTxt = None if trueOrthos else sureWorkDir / 'dsmSingle.txt' 
        try:
            if not trueOrthos:
                os.rename( str(dsmExtendedDir) , str(origDsmExtendedDir) )
                dsmExtendedDir.mkdir()
                # Estimate the 2D bbox of the aerial image data set in Obj CS. We would actually need to do monoplotting for that.
                # For now, let's use a representative object plane and intersect the image rays through the image corners.
                # Subsequently, extend that bbox by a distance that should be enough for rather flat terrain.
                extension = 100. # [m]
                objSpaceBbox2d[:2] -= extension
                objSpaceBbox2d[2:] += extension
                tileSize = 50 # [m] # it seems that SURE is designed to manage small tiles. Otherwise, RAM consumption may be really high.
                # gdal_translate enlarges the output grid computed from -projwin such that the input and output rasters will be aligned. Consequently, the output tiles overlap a little. SURE does not complain about that.
                #dsmTileFns = utils.gdal.tile( str(infoExternalDsm.path), str(dsmExtendedDir), objSpaceBbox2d, tileSize, dsmGsd )
                logger.info('Prepare resampled DSM tiles.')
                dsmProgress = Progress( np.prod( np.ceil( ( objSpaceBbox2d[2:] - objSpaceBbox2d[:2] ) / tileSize ).astype(int) ).item() )
                dsmTileFns = []
                for ix in itertools.count():
                    minX = objSpaceBbox2d[0] + tileSize *  ix
                    maxX = objSpaceBbox2d[0] + tileSize * (ix+1)
                    if minX >= objSpaceBbox2d[2]:
                        break
                    for iy in itertools.count():
                        minY = objSpaceBbox2d[1] + tileSize *  iy
                        maxY = objSpaceBbox2d[1] + tileSize * (iy+1)
                        if minY >= objSpaceBbox2d[3]:
                            break
                        dsmTileFns.append( dsmExtendedDir / '{}_{:03}_{:03}.tif'.format( Path(infoExternalDsm.path).stem, ix, iy ) )
                        gdalTranslateArgs = [ gdalTranslateExe,
                                              '-co', 'TFW=YES',
                                              '-co', 'COMPRESS=LZW',
                                              '-co', 'BIGTIFF=NO',
                                              '-r' , 'cubic',
                                              '-tr', str(dsmGsd), str(dsmGsd),
                                              # ulx uly lrx lry
                                              '-projwin' ] + [ str(el) for el in (minX, maxY, maxX, minY) ] + [
                                              str(infoExternalDsm.path),
                                              str(dsmTileFns[-1]) ]
                        check_output( args=gdalTranslateArgs, cwd=sureWorkDir )
                        dsmProgress += 1

                with dsmSingleTxt.open('wt') as fout:
                    for dsmTileFn in dsmTileFns:
                        fout.write( '{}\n'.format( dsmTileFn.relative_to(dsmSingleTxt.parent) ) )

            os.rename( str(orthoRefinedDir), str(origOrthoRefinedDir) )
            logger.info( 'Create {} orthophotos for each aerial separately.', 'true' if trueOrthos else 'classic' )
            orthoProgress = Progress( len(undistOriFns) )
            for undistFn, oriFn in undistOriFns:
                with suppress(FileNotFoundError):
                    shutil.rmtree( str(orthoRefinedDir) )
                orthoRefinedDir.mkdir()
                with imgSingleTxt.open('wt') as fout:
                    fout.write( '{}\n'.format( str( utils.filePaths.relPathIfExists( undistFn, imgSingleTxt.parent) ) ) )
                with oriSingleTxt.open('wt') as fout:
                    fout.write( '{}\n'.format( str( utils.filePaths.relPathIfExists( oriFn, oriSingleTxt.parent ) ) ) )
                sureModuleTrueOrthoArgs = [ sureModuleTrueOrthoExe,
                                            '--ori', str(oriSingleTxt.relative_to(sureWorkDir)),
                                            '--img', str(imgSingleTxt.relative_to(sureWorkDir)) ] + (
                                            ['--dsm', str(dsmSingleTxt.relative_to(sureWorkDir))] if dsmSingleTxt else [] )
                check_output( args=sureModuleTrueOrthoArgs, cwd=sureWorkDir, env=sureModuleTrueOrthoEnv )

                # Merge the produced orthophoto tiles.
                orthoTileFns = []
                for imgFn in glob.iglob( str( orthoRefinedDir / '*.tif' ) ):
                    tileInfo, = utils.gdal.imread(str(imgFn), skipData=True)
                    utils.gdal.setDataSetNoData( str(imgFn), np.zeros( 1 if tileInfo.bands==utils.gdal.Bands.grey else 3 ) )
                    if projCsWkt is not None:
                        utils.gdal.setProjection( str(imgFn), projCsWkt )
                    orthoTileFns.append( str(imgFn) )

                if not len(orthoTileFns):
                    raise Exception("SURE has not produced any orthophoto tiles for aerial {}.".format(undistFn))

                mergedOrthoFn = outDir / '{}.tif'.format(undistFn.stem)
                utils.gdal.mergeTiles( orthoTileFns, str(mergedOrthoFn) )

                if projCsWkt is not None:
                    with mergedOrthoFn.with_suffix('.prj').open('wt') as fout:
                        fout.write( projCsWkt )
                logger.info( 'Separate orthophoto saved to {}', mergedOrthoFn.relative_to(args.outDir) )
                orthoProgress +=1
        finally:
            if origDsmExtendedDir.exists():
                with suppress(FileNotFoundError):
                    shutil.rmtree( str(dsmExtendedDir) )
                os.rename( str(origDsmExtendedDir) , str(dsmExtendedDir) )
            if origOrthoRefinedDir.exists():
                with suppress(FileNotFoundError):
                    shutil.rmtree( str(orthoRefinedDir) )
                os.rename( str(origOrthoRefinedDir), str(orthoRefinedDir) )

    # In addition to the orthophoto mosaic, md wants separate orthophotos for each aerial.
    # SURE now supports that. One needs to call ModuleTrueOrtho.exe in the SURE project directory,
    # passing custom text files for source images and their orientations, for only a single image (just select 1 line in orilist_level0.txt and imglist_texture_level0.txt):
    # D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\ortho\SURE>ModuleTrueOrtho.exe --ori oriSingle.txt --img imgSingle.txt
    # -> ModuleTrueOrtho.exe re-uses the DSM generated before, to create a true orthophoto for the single aerial specified in oriSingle.txt and imgSingle.txt.
    # ModuleTrueOrtho.exe will overwrite files in SURE\DSM\Cloud_Refined SURE\DSM\Ortho_Refined. To easily merge the single-image orthophoto tiles, one should empty those folders beforehand.
    # The orthophoto is limited to the area where the DSM from image matching is defined.
    # Thus, the generated orthophoto may not be what md really wants (he is not sure himself yet). He may prefer to have a classic (non-true) orthophoto, based on an external DTM!
    # That is also possible to produce with SURE, by replacing the DSM tiles in SURE\DSM\DSM_Extended generated by SURE with tiles from an external DTM.
    if args.separateTrueOrthos:
        createSingleOrthos( True, outDir = args.outDir / 'trueOrthos' )

    if args.separateClassicOrthos:
        createSingleOrthos( False, outDir = args.outDir / 'classicOrthos' )

    if not args.keepTemporaries:
        with suppress(FileNotFoundError):
            shutil.rmtree( str( sureWorkDir ) )


