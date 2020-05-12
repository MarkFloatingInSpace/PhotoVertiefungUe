# -*- coding: cp1252 -*-
"""Automatic incremental relative orientation of unordered image data sets based on image feature extraction and matching.

Workflow is as follows:

- For each image:
  + Determine approximate IORs based on LBA or Exif.
  + Detect and describe image features.

- For each image pair:
  + Determine homologous image points by matching feature descriptors.
  + Robustly compute the E-matrix and discard feature matches that disagree with that E-matrix.

- Compute the graph of object points (vertices) that are observed at image features (edges).
- Discard multiple projections i.e. object points that are observed multiply in the same image.
- Compute the graph of images (vertices) that are connected by feature matches (edges).

- Incremental reconstruction:
  + If no images have been oriented yet:
    * With the image pair with many feature matches and a large enough angle between the optical axes,
    * setting the baseline length to 1., and selecting the first image as datum,
    * triangulate object points.
   
  + Otherwise:
    * Select the image with most observed, already triangulated object points.
    * Compute the orientation of that image, based on those object points.
    * Triangulate new object points.
    * Introduce image observations for old and new object points.
    * Repeat this sub-step until N new images have been added.
   
  + Do a bundle block adjustment.
  + Introduce IOR, ADP as free parameters if they can be estimated well.
  + Detect outliers and remove them from the block.
  + Repeat until all images either form part of the block, or have failed to orient.

- Detect outliers and remove them from the block.
- Optionally, discard object points that have been observed only twice.
- Optionally, determine principal point positions.
"""

# oriental
from oriental import config, adjust, graph, utils, ori, match, log, Progress, relOri
import oriental.adjust.parameters
import oriental.relOri.SfMManager
import oriental.relOri.Printer
import oriental.utils.gdal
import oriental.utils.argparse
import oriental.utils.filePaths

# 3rd-party
import cv2
import numpy as np
from scipy import spatial
import h5py
from contracts import contract
from osgeo import osr
osr.UseExceptions()

# built-ins
import os, sys, argparse, traceback, contextlib
from itertools import chain
from glob import glob
from pathlib import Path


logger = log.Logger("relOri")

FeatureType = utils.argparse.ArgParseEnum('FeatureType', 'sift surf akaze')
ReuseMatches = utils.argparse.ArgParseEnum('ReuseMatches', 'none descriptors descriptorMatches descriptorAndEpipolarMatches')
StopAfter = utils.argparse.ArgParseEnum('StopAfter', 'none extract match')
SortBy = utils.argparse.ArgParseEnum('SortBy', 'strength size')
AdjustIorAdp = utils.argparse.ArgParseEnum('AdjustIorAdp', 'during atend never')
AtLeast3obs = utils.argparse.ArgParseEnum('AtLeast3obs', 'no before after')
IorAdpParams = utils.argparse.ArgParseEnum('IorAdpParams', 'pp f r3 r5')
iorAdpParams2Internal = { IorAdpParams.pp : 0,
                          IorAdpParams.f  : 2,
                          IorAdpParams.r3 : adjust.PhotoDistortion.optPolynomRadial3,
                          IorAdpParams.r5 : adjust.PhotoDistortion.optPolynomRadial5 }

@contract
def parseArgs( args : 'list(str)|None' = None ):
    docList = __doc__.splitlines()
    parser = argparse.ArgumentParser( description=docList[0],
                                      epilog='\n'.join(docList[1:]),
                                      formatter_class=utils.argparse.Formatter )

    parser.add_argument( 'photos', nargs='+',
                         help='(List of) file path pattern(s) that match(es) the names of image files to be processed. Supported wildcards: *, ?, [character set], [!character set]. '
                              'examples:'
                              r'  "D:\140115\*",'
                              r'  "D:\Carnuntum\*.jpg",'
                              r'  "D:\aerial\08?_orig.tif" "D:\aerial\09?_orig.tif" ' )
    parser.add_argument( '--outDir', default=Path.cwd() / "relOri", type=Path,
                         help='Store results into OUTDIR.' )
                         
    parser.add_argument('--reuseMatches', nargs='?', default=ReuseMatches.none, const=ReuseMatches.descriptorAndEpipolarMatches, choices=ReuseMatches, type=ReuseMatches,
                        help='Reuse results from a previous program run. '
                             'Choose the kind of results to reuse, and hence where to proceed from: '
                             'keypoints and descriptors only, or also their matches, or additionally their matches filtered by epipolar geometry.')
    parser.add_argument('--stopAfter', nargs='?', default=StopAfter.none, choices=StopAfter, type=StopAfter,
                        help='Stop after the given processing stage. '
                             'Use this to interrupt processing prematurely, e.g. to do the feature extraction on one machine, and the matching on another.')
    
    parser.add_argument('--plotIntermed', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--plotInterestingImages', nargs='*', type=str, default=[], help=argparse.SUPPRESS)

    cameraGroup = parser.add_argument_group('Camera', 'If photos have been taken with an analog (film-based) camera, then its parameters need to be specified. '
                                                      'For photos taken with a digital camera, this is necessary only if Exif meta data is wrong or missing.')
    cameraGroup.add_argument( '--analogModel', choices=relOri.SfMManager.AnalogCameras, type=relOri.SfMManager.AnalogCameras,
                              help='The camera manufacturer and model are the minimum amount of information needed to process scanned analog photos.' )
    cameraGroup.add_argument( '--focal', type=float,
                              help='The camera focal length. [mm] for analog cameras, [px] for digital cameras.' )
    cameraGroup.add_argument( '--filmFormat', type=float,
                              help='The width and height of the imaged area on film in mm.' )

    preprocessingGroup = parser.add_argument_group('Preprocessing', 'Preprocess the data')
    preprocessingGroup.add_argument( '--plotFiducials', action='store_true',
                                      help='Plot the result of fiducial mark detection' )
    preprocessingGroup.add_argument( '--histogramEqualization', action='store_true',
                                      help='Equalize image histograms before feature extraction.' )
    preprocessingGroup.add_argument( '--downSample', default=0, type=int,
                                      help='Halve the image resolution this many times. 0 will use full resolution; 1 will use half the width and half the height.' )

    extractionGroup = parser.add_argument_group('Feature Extraction', 'Affecting the extraction of features independently for each image')
    extractionGroup.add_argument( '--masks', nargs='*',
                                  help='(List of) file path pattern(s) that match(es) the names of masks for the image files to be processed. Image file name must be a substring of the mask file name' )
    extractionGroup.add_argument( '--featureType', default=FeatureType.sift, choices=FeatureType, type=FeatureType,
                              help='The feature point extractor to be used.' )
    extractionGroup.add_argument( '--affineTilts', default=6, type=int,
                                  help='No. of affine tilts to apply to images to make feature extractor fully affine invariant. 0 extracts features from non-tilted images only.' )
    extractionGroup.add_argument( '--surfHessianThreshold', type=float, default=100., 
                                  help='Detect SURF features with a hessian larger than SURFHESSIANTHRESHOLD' )
    extractionGroup.add_argument( '--siftContrastThreshold', type=float, default=0.04, 
                                  help='Discard SIFT features in low-contrast areas. The larger the threshold, the more features are discarded.' )
    extractionGroup.add_argument( '--siftEdgeThreshold', type=float, default=10., 
                                  help='Discard edge-like SIFT features. The smaller the threshold, the more features are discarded.' )
    extractionGroup.add_argument( '--akazeThreshold', type=float, default=0.001, 
                                  help='Discard AKAZE features in low-contrast areas. The larger the threshold, the more features are discarded.' )
    extractionGroup.add_argument( '--akazeDiffusivity', choices=match.AkazeOptions.Diffusivity.values.values(), type=lambda x: match.AkazeOptions.Diffusivity.names[x], default=match.AkazeOptions.Diffusivity.peronaMalikG2, 
                                  help='Diffusion filter function' )
    extractionGroup.add_argument( '--nMaxFeaturesPerImg', type=int, default=40000, 
                                  help='For each image, select at most NMAXFEATURESPERIMG of the extracted features' )
    extractionGroup.add_argument( '--subdivision', nargs='*', type=int, default=[1], 
                                  help='For each image, divide the image area into a grid with a resolution of SUBDIVISION. For each grid cell, '
                                       'select at most N features, with N = NMAXFEATURESPERIMG / <number of grid cells>. '
                                       'If SUBDIVISION has 2 elements, they are interpreted as #rows x #columns. '
                                       'If SUBDIVISION is a scalar, then the grid has SUBDIVISION columns and SUBDIVISION rows.' )
    extractionGroup.add_argument( '--sortBy', default=SortBy.strength, choices=SortBy, type=SortBy,
                                  help="Keep the NMAXFEATURESPERIMG strongest or largest features per image. "
                                  "STRENGTH: keep the strongest features; "
                                  "SIZE: keep the largest features; " )
    extractionGroup.add_argument( '--plotFeatures', action='store_true',
                                  help='Plot the extracted features over the original photos, and store the plots in outDir/features' )
    
    matchingGroup = parser.add_argument_group('Feature Matching', 'Affecting the matching of image features independently for each image pair')
    matchingGroup.add_argument( '--matchingPrecision', type=float, default=match.MatchingOpts().matchingPrecision,
                                help='Tune the parameters of approximate matching such that the exact nearest neighbour is found with a probability of matchingPrecision. If 1, use brute force matching on the GPU.' )
    matchingGroup.add_argument( '--maxDescriptorDistanceRatio', type=float, default=match.MatchingOpts().maxDescriptorDistanceRatio,
                                help='Keep only matches with a ratio of (distance to nearest neighbor) / (distance to second-nearest neighbor) smaller than MAXDESCRIPTORDISTANCERATIO. Default value depends on kVLD usage. Lowe proposes 0.8, bundler uses 0.6' )
    matchingGroup.add_argument( '--no-symmetricMatching', dest='symmetricMatching', action='store_false',
                                help="Don't check if matches are mutually closest. " )
     
    matchingGroup.add_argument( '--nMaxMatchesPerPair', type=int, default=match.MatchingOpts().minMaxMatchesPerPair[1], 
                                help='For each image pair, select the NMAXMATCHESPERPAIR feature matches with the smallest descriptor distance ratio')
    matchingGroup.add_argument( '--maxEpipolarDist', type=float,
                                help='The maximum distance between an image point and its epipolar line for a feature match to be considered an inlier during E-matrix estimation. '
                                     'Note that this threshold is used with the initial camera IORs/ADPs, so use a large enough value. Default: 2*MAXFINALRESIDUALNORM' )
    matchingGroup.add_argument( '--minPairInlierRatio', type=float, default=0.25,
                                help='The minimum ratio of inliers among all feature matches to be handled by RANSAC during E-matrix estimation' )
    matchingGroup.add_argument( '--minPairMatches', type=int, default=10,
                                help='Discard matches with less than MINPAIRMATCHES inlier matches' )
    matchingGroup.add_argument( '--thinOut', nargs=3, type=int,
                                help='Overlay the area of each image with a grid of ARG1 columns by ARG2 rows. Remove as many feature tracks as possible, '
                                     'such that each grid cell is covered with at least ARG3 features of largest multiplicity.' )
    matchingGroup.add_argument( '--plotMatches', action='store_true',
                                help='For each image pair, plot the feature matches over the original photos, and store the plots in outDir/matches' )    
    matchingGroup.add_argument( '--plotFilteredMatches', action='store_true',
                                help='For each image pair, plot the feature matches that remain after the E-matrix test over the original photos, and store the plots in outDir/matchesFiltered' )
    
    reconstructionGroup = parser.add_argument_group('Reconstruction', 'Affecting the reconstruction of relative camera orientations and object geometry')
    reconstructionGroup.add_argument('--initPair', nargs=2,
                                     help='File path substrings that match 2 images to begin the reconstruction with.')
    reconstructionGroup.add_argument( '--adjustIorAdp', default=AdjustIorAdp.atend, choices=AdjustIorAdp, type=AdjustIorAdp,
                                      help="Select if and when IOR and ADP of uncalibrated cameras shall be estimated."
                                      "DURING: estimate the parameters as early as possible, during reconstruction; "
                                      "ATEND: estimate the parameters after reconstruction; "
                                      "NEVER: keep the initial values" )
    reconstructionGroup.add_argument( '--adjustWhatDuring', nargs='+', default=[IorAdpParams.r3, IorAdpParams.f], choices=IorAdpParams, type=IorAdpParams,
                                      help='Select which IOR/ADP parameters may be adjusted during reconstruction' )
    reconstructionGroup.add_argument( '--IORgrouping', default=relOri.SfMManager.IORgrouping.sequence, choices=relOri.SfMManager.IORgrouping, type=relOri.SfMManager.IORgrouping,
                                      help="Form groups of photos that share the same IOR parameters. "
                                      "NONE: adjust a separate IOR for each image; "
                                      "SEQUENCE: adjust a separate IOR for each sequence of images taken consecutively with the same camera and nominal focal length; "
                                      "ALL: adjust 1 common IOR for all images;" )
    # Options/functionality needed by Livia Piermattei and for car's konya project:
    #reconstructionGroup.add_argument( '--EORgrouping', nargs='*', default=[], 
    #                                  help="Form groups of photos that share certain EOR parameters. May be a a single file path, or a list of strings. "
    #                                  "If a single file path, then the path must point to a text file with 1 image file per line, and blank lines separating EOR groups; "
    #                                  "If a list of strings, then each string specifies the image file name part by which a member of an EOR group differs from the group's common file name; " )
    #reconstructionGroup.add_argument( '--EORgroupsShare', default=EORgroupsShare[0], choices=EORgroupsShare,
    #                                  help="EOR groups share these EOR parameters. "
    #                                  "NOTHING: EOR groups share nothing - EORgrouping not used; "
    #                                  "PRC: all photos in each EOR group share the same projection centre position; "
    #                                  "ROT: all photos in each EOR group share the same rotation;" )

    # Deactivated: we cannot remove the UnitSphere parameterization of the second camera in order to replace the unconstrained datum with free-network-observations.
    # We would need to setup a new adjustment problem from scratch for that purpose.
    # So we cannot shift the block to the objPts' barycentre or a priori GPS-PRC-positions and re-adjust it.
    # Anyway, introducing observations for free network adjustment would prohibit us from using the Schur complement: at most 1 objPt in each observation
    # Thus, let's do what ORIENT does: adjust with arbitrary datum, compute S-Transform, and apply the S-Transform to Qxx-sub-blocks of OBJ, PRC, and RO
    #reconstructionGroup.add_argument( '--precision', action='store_true',
    #                                  help='Compute and store the precision of unknowns.' )

    reconstructionGroup.add_argument( '--minPnPInlierRatio', type=float, default=0.08,
                                      help='The minimum ratio of inliers among all observed, already reconstructed points to be handled by RANSAC during PnP-orientation' )
    reconstructionGroup.add_argument( '--maxIntermedResidualNorm', type=float,
                                      help='Maximum residual norm for each image point to be accepted as inlier during reconstruction. '
                                           'Note that this threshold may be used with the initial camera IORs/ADPs, so use a large enough value. Default: 4*MAXFINALRESIDUALNORM' )
    reconstructionGroup.add_argument( '--maxFinalResidualNorm', type=float, default=5.,
                                      help='Maximum residual norm for each image point to be accepted as inlier in the final non-robust adjustment' )
    reconstructionGroup.add_argument( '--minIntersectAngle', type=float, default=5.,
                                      help='Minimum intersection angle of observation rays at object point [gon]' )
    reconstructionGroup.add_argument( '--atLeast3Obs', nargs='?', default=AtLeast3obs.no, const=AtLeast3obs.after, choices=AtLeast3obs, type=AtLeast3obs,
                                      help='Discard or not object points that are observed in only 2 images before or after reconstruction.' )
    reconstructionGroup.add_argument( '--nImgs2OrientBeforeFullAdjustment', type=int, default=5,
                                      help='Number of images to add to the block before another full adjustment. Choose higher value to gain speed, lower value to gain stability.' )
    #reconstructionGroup.add_argument( '--plotReconstruction', action='store_true',
    #                                  help='Plot the result of reconstruction and store the plot in outDir/reconstruction' )

    geoTransformGroup = parser.add_argument_group('GeoTransform', 'Similarity transformation to a priori camera positions / rotations.')
    geoTransformGroup.add_argument( '--targetCS', default='UTM',
                                    help='Target coordinate system of the block (e.g.: "EPSG:31259"). Defaults to the UTM zone at the barycentre.' )
    geoTransformGroup.add_argument( '--stdDevPos', nargs='*', type=float, default=[10., 10., 20.],
                                    help='Standard deviations of a priori projection centre positions in target coordinate system.' )
    geoTransformGroup.add_argument( '--stdDevRot', nargs='*', type=float, default=[0.5, 0.5, 4.],
                                    help='Standard deviations of a priori projection centre rotation angles in target coordinate system.' )
    geoTransformGroup.add_argument( '--dsm', default='dhm_lamb_10m.tif', type=Path,
                                    help="Surface model used for LBA-footpoints. If a relative path, file is also searched in installation's 'data'-folder." )

    utils.argparse.addLoggingGroup( parser, "relOriLog.xml" )

    generalGroup = parser.add_argument_group('General', 'Other general settings')
    generalGroup.add_argument( '--no-plots', dest='plots', action='store_false',
                               help="Don't produce the standard graphical output." )
    generalGroup.add_argument( '--no-progress', dest='progress', action='store_false',
                               help="Don't show progress in the console." )

    cmdLine = sys.argv[:]
    args = parser.parse_args( args=args )
    main( args, cmdLine, parser )


@contract
def main( args : argparse.Namespace, cmdLine : 'list(str) | None' = None, parser : 'ArgumentParser|None' = None ) -> None:

    with contextlib.suppress(FileExistsError):
        args.outDir.mkdir(parents=True)
    args.outDir = args.outDir.resolve()
    utils.argparse.applyLoggingGroup( args, args.outDir, logger, cmdLine )

    if not args.progress:
        Progress.deactivate()

    # Argument defaults derived from other arguments
    args.maxEpipolarDist         = args.maxEpipolarDist         or 2 * args.maxFinalResidualNorm
    args.maxIntermedResidualNorm = args.maxIntermedResidualNorm or 4 * args.maxFinalResidualNorm
    # ----------- parameter checks
    
    if args.downSample < 0:
        raise Exception("downSample must not be negative.")
    # maximum distance between an image point and its epipolar line to be considered as inlier
    # We must consider here that the cameras are hardly calibrated, so use a large enough threshold
    if args.maxEpipolarDist <= 0.:
        raise Exception("maxEpipolarDist must be > 0, but is: {}".format(args.maxEpipolarDist))
    if args.maxIntermedResidualNorm <= 0.:
        raise Exception("maxIntermedResidualNorm must be > 0, but is: {}".format(args.maxIntermedResidualNorm))
    if args.maxFinalResidualNorm <= 0.:
        raise Exception("maxFinalResidualNorm must be > 0, but is: {}".format(args.maxFinalResidualNorm))
    # the argument for robust loss functions
    if not args.nImgs2OrientBeforeFullAdjustment > 0:
        raise Exception("nImgs2OrientBeforeFullAdjustment must be > 0, but is: {}".format(args.nImgs2OrientBeforeFullAdjustment))
    if any(( el<1 for el in args.subdivision)):
        raise Exception( "(elements of) subdivision must be > 0" )
    if len(args.subdivision) == 1:
        args.subdivision = args.subdivision * 2
    elif len(args.subdivision) > 2:
        raise Exception( "subdivision must have either 1 or 2 elements" )
    # check early if the target CS can be parsed.
    if args.targetCS != "UTM":
        try:
            cs = osr.SpatialReference()
            cs.SetFromUserInput( args.targetCS )
        except Exception as ex:
            raise Exception( "targetCS is not a valid coordinate system definition: {}".format(ex) )
        except:
            raise Exception( "targetCS is not a valid coordinate system definition." )
        if not cs.IsProjected():
            raise Exception("targetCS is not a projected coordinate system.")
        del cs
    for param, name in (( args.stdDevPos, 'stdDevPos'),
                        ( args.stdDevRot, 'stdDevRot') ):
        if any(( el<=0 for el in param )):
            raise Exception( "All elements of {} must be > 0".format(name) )
        if len(param) == 1:
            param.extend( param * 2 )
        elif len(param) != 3:
            raise Exception( "{} must have either 1 or 3 elements".format(name) )
    args.stdDevPos = np.array(args.stdDevPos,np.float)
    args.stdDevRot = np.array(args.stdDevRot,np.float)

    utils.argparse.logScriptParameters( args, logger, parser )

    # transform to internal representation after logging the script parameters, so cmdline-args and log share the same (external) representation.
    args.adjustWhatDuring = [ iorAdpParams2Internal[el] for el in args.adjustWhatDuring ]


    confidenceLevel = 0.999
    # -----------
        
    interestingImages = args.plotInterestingImages # plot even more for these images and subsequent ones 
        
    def getImageFilePathsMasks():
        # Check if for every pattern, at least 1 image file is matched. This makes sure that the cmdline has been parsed as intended.
        # Consider --subdivision=10 12 *.jpg
        # where "12" is parsed as image file pattern!
        #imageFilePaths = sorted([ os.path.abspath(item) for fnPattern in args.photos for item in glob( fnPattern ) ])
        imageFilePaths = []
        for fnPattern in args.photos:
            imageFilePaths_ = glob( fnPattern )
            if not imageFilePaths_:
                raise Exception('No file matched the photo file name pattern "{}".'.format(fnPattern))
            imageFilePaths.extend( os.path.abspath(item) for item in imageFilePaths_ )
        imageFilePaths.sort()
        if not args.masks:
            return [[el,None] for el in imageFilePaths]
        imageFilePathsMasks = []
        masks = []
        for fnPattern in args.masks:
            masks_ = glob( fnPattern )
            if not masks_:
                raise Exception('No file matched the mask file name pattern "{}".'.format(fnPattern))
            masks.extend( os.path.abspath(item) for item in masks_ )
        for imageFilePath in imageFilePaths:
            for iMask, mask in enumerate(masks):
                if Path(imageFilePath).stem in Path(mask).stem:
                    imageFilePathsMasks.append([ imageFilePath, mask ])
                    del masks[iMask]
                    break
            else:
                imageFilePathsMasks.append([ imageFilePath, None ])
        if len(masks):
            raise Exception('Following mask files could not be matched with an image file: {}'.format(', '.join(masks)))
        logger.verbose( 'Image and mask file correspondences\n'
                        'image file\tmask file\n'
                        '{}',
                        '\n'.join( '{}\t{}'.format(*els) for els in imageFilePathsMasks ) )
        return imageFilePathsMasks

    imageFilePathsMasks = getImageFilePathsMasks()
    if args.initPair:
        _matches = [[idx for idx, el in enumerate(imageFilePathsMasks) if pattern in el[0]] for pattern in args.initPair]
        for idx, _match in enumerate(_matches):
            if not _match:
                raise Exception(f'--initPair[{idx}]={args.initPair[idx]} does not match any input image file path.')
            if len(_match) > 1:
                raise Exception(f'--initPair[{idx}]={args.initPair[idx]} matches more than 1 input image file path: {", ".join(imageFilePathsMasks[el][0] for el in _match)}.')
        args.initPair = [el[0] for el in _matches]
    sfm = relOri.SfMManager.SfMManager( [ el[0] for el in imageFilePathsMasks ],
                      args.IORgrouping,
                      str(args.outDir),
                      args.minIntersectAngle,
                      (args.analogModel, args.focal, args.filmFormat),
                      args.plotFiducials )

    printer = relOri.Printer.Printer( sfm )

    logger.infoFile( len(sfm.imgs), tag='#images' )

    def getCameraMasks():
        maskDir = args.outDir / 'masks'
        for idx,((imgFn,maskFn),img) in enumerate(utils.zip_equal(imageFilePathsMasks,sfm.imgs)):
            if maskFn is None:
                if img.mask_px is None:
                    maskFn = ''
                else:
                    with contextlib.suppress(FileExistsError):
                        maskDir.mkdir(parents=True)
                    mask = np.zeros( (img.height,img.width), np.uint8 )
                    # mask_px may be non-convex!
                    cv2.fillPoly( mask, [img.mask_px.astype(np.int32)*(1,-1)], color=255 )
                    maskFn = str( maskDir / ( Path(img.fullPath).stem + '.tif' ) )
                    info = utils.gdal.ImageInfoWrite()
                    info.compression = utils.gdal.Compression.deflate
                    # Alternatively, we may store the mask as 1-bit TIFF. However, as we compress with deflate,
                    # this does't make much (any?) difference concerning the file size. 8bit is probably better supported by other software.
                    utils.gdal.imwrite( maskFn, mask, info=info, buildOverviews=False )
                imageFilePathsMasks[idx][1] = maskFn
    getCameraMasks()

    # adjust parameters for scanned image sets
    pix2camMeanScaleForward = np.median( [img.pix2cam.meanScaleForward() for img in sfm.imgs] )
    if abs( pix2camMeanScaleForward - 1. ) > 1.e-7:
        args.maxEpipolarDist         *= pix2camMeanScaleForward
        args.maxIntermedResidualNorm *= pix2camMeanScaleForward
        args.maxFinalResidualNorm    *= pix2camMeanScaleForward
        logger.info( 'Thresholds have been adapted to account for scanned images:\n'
                     'threshold\tadapted value\n' +
                     '\n'.join( ('{}\t{:.3f}'.format( name, getattr(args,name) ) for name in ['maxEpipolarDist', 'maxIntermedResidualNorm', 'maxFinalResidualNorm'] ) ) )

    resultsFn = args.outDir / 'relOri.sqlite'

    # simply delete the DB-file if existing. The alternative would be to drop all tables. However, that may still result in errors due to remaining triggers, etc.
    # raises if e.g. the file is opened in spatialite_gui
    with contextlib.suppress(FileNotFoundError):
        resultsFn.replace( args.outDir / 'relOri_backup.sqlite' )
    # ---------------------------
    fnFeatures = args.outDir / 'features.h5'
    #imagePaths=[ img.fullPath for img in sfm.imgs ]
    relImagePaths = [ utils.filePaths.relPathIfExists(img.fullPath, str(resultsFn.parent)) for img in sfm.imgs ]

    preprocessingOpts = match.PreprocessingOpts()
    preprocessingOpts.histogramEqualization = args.histogramEqualization
    preprocessingOpts.downSample = args.downSample
    def detectMatch():
        doExtract = extractAndDescribe = args.reuseMatches != ReuseMatches.descriptors
        doMatch = args.stopAfter != StopAfter.extract
        elems = []
        if doExtract and not doMatch:
            logger.info("Detect and describe feature points") 
        elif not doExtract and doMatch:
            logger.info("Match feature points")
        else:
            logger.info("Detect, describe, and match feature points")
        if args.featureType == FeatureType.surf:
            featureDetectOpts = match.SurfOptions()
            featureDetectOpts.hessianThreshold = args.surfHessianThreshold
        elif args.featureType == FeatureType.sift:
            featureDetectOpts = match.SiftOptions()
            featureDetectOpts.contrastThreshold = args.siftContrastThreshold
            featureDetectOpts.edgeThreshold     = args.siftEdgeThreshold
        elif args.featureType == FeatureType.akaze:
            featureDetectOpts = match.AkazeOptions()
            featureDetectOpts.threshold = args.akazeThreshold
            featureDetectOpts.diffusivity = args.akazeDiffusivity
        else:
            raise Exception( 'Feature type not supported: {}'.format(args.featureType) )
        featureDetectOpts.nAffineTilts = args.affineTilts

        featureFiltOpts = match.FeatureFiltOpts()
        featureFiltOpts.nMaxFeaturesPerImg = args.nMaxFeaturesPerImg
        featureFiltOpts.extractLargest = args.sortBy == SortBy.size
        featureFiltOpts.nRowCol = args.subdivision
        featureFiltOpts.plotFeatures = args.plotFeatures
            
        matchingOpts = match.MatchingOpts()
        matchingOpts.maxDescriptorDistanceRatio = args.maxDescriptorDistanceRatio
        matchingOpts.symmetricMatching = args.symmetricMatching
        matchingOpts.minMaxMatchesPerPair = (5,args.nMaxMatchesPerPair)
        matchingOpts.matchingPrecision = args.matchingPrecision
        matchingOpts.plotMatches = args.plotMatches

        allKeypoints, sfm.edge2matches = match.match(
            imagePaths = [ el[0] for el in imageFilePathsMasks ],
            masks = [ el[1] for el in imageFilePathsMasks ],#[ img.mask_px for img in sfm.imgs ],
            featureDetectOpts = featureDetectOpts,
            featureFiltOpts = featureFiltOpts,
            matchingOpts = matchingOpts,
            outDir = str(args.outDir),
            extractAndDescribe = doExtract,
            match = doMatch,
            preprocessingOpts = preprocessingOpts )
            
        for img,keypoints in zip(sfm.imgs,allKeypoints):          
            img.keypoints = keypoints
     
        logger.debug("Saving imagePaths to HDF5 file")
        with h5py.File( str(fnFeatures), 'r+' ) as features:
            dtype = h5py.special_dtype(vlen=str)
            features.attrs.create( 'imagePaths', relImagePaths, shape=(len(relImagePaths),), dtype=dtype )
        logger.verbose("imagePaths saved to HDF5 file")

    with contextlib.ExitStack() as stack:
        # as mentioned above, it seems not straight forward with Python's sqlite3 to rollback the insertion of the 'homographies' table in case of errors during the computation of its data.
        stack.callback( detectMatch )
        if args.reuseMatches not in ( ReuseMatches.none, ReuseMatches.descriptors ):
            with contextlib.suppress((OSError,KeyError)), \
                 h5py.File( str(fnFeatures), 'r' ) as features:
                # Support processing a whole block, followed by processing a sub-block, re-using the features/matches from the whole block.
                # Just try if all wanted datasets are present in the HDF5-file. Upon KeyError, we actually extract/match.
                #imagePathsSaved = features.attrs['imagePaths']
                if 0:#np.any( imagePathsSaved != relImagePaths ):
                    # We compare image file paths relative to relOri.sqlite.
                    # Thus, one can move an existing HDF5 file into a sibling directory and re-use it if the images are in a different directory.
                    # Or, one can move the HDF5 file together with the images to a different machine.
                    logger.warning( "Image file paths for existing feature points and matches don't match current image paths in '{}'", fnFeatures )
                else:
                    group = features['keypts']
                    for img in sfm.imgs:
                        img.keypoints = np.array( group[ os.path.basename( img.fullPath ) ] )
                    sfm.edge2matches = dict()
                    basename2idx = { os.path.basename( img.fullPath ) : idx for idx,img in enumerate(sfm.imgs) }
                    for key,value in features['matches'].items():
                        sfm.edge2matches[tuple( basename2idx[el] for el in key.split('?') )] = np.array(value)
                    stack.pop_all()
                    logger.info( "Loaded feature points and matches from file '{}'", fnFeatures )
    del detectMatch
    logger.info( sfm.countFeatures(), tag='#features matched' )

    if args.stopAfter in (StopAfter.extract, StopAfter.match):
        return

    edge2matches = {}
    def filterMatches():
        logger.info("Filter matches by E-matrix")
        edge2matches.clear()
        edge2matches.update(
            ori.filterMatchesByEMatrix( edge2matches=sfm.edge2matches,
                                        allKeypoints=[
                                            ori.distortion(
                                                img.pix2cam.forward( img.keypoints )[:,:2],
                                                img.ior,
                                                ori.adpParam2Struct(img.adp),
                                                ori.DistortionCorrection.undist )
                                            for img in sfm.imgs ],
                                        iors=[ img.ior for img in sfm.imgs ],
                                        maxEpipolarDist=args.maxEpipolarDist,
                                        nMinMatches=args.minPairMatches,
                                        inlierRatio=args.minPairInlierRatio,
                                        confidenceLevel=confidenceLevel ) )
        logger.info('Compute pairwise image orientation qualities')
        progress = Progress(len(edge2matches))
        with h5py.File( str(fnFeatures), 'r+' ) as features:
            if 'matchesFiltered' in features:
                del features['matchesFiltered']
            group = features.create_group('matchesFiltered')

            if 1:
                # Filter the candidates for the initial photo pair by a constant minimum #E-matrix-inliers (e.g. 100).
                # Among those candidates, choose the pair with the minimum #H-matrix-inliers. If multiple photos have the same number of H-matrix-inliers, choose the one with the highest #E-matrix-inliers
                #nHomog_nEssent = []
                #for edge,(matches,E,R,t,nHomographyInliers) in edge2matches.items():
                #    nEssentialInliers = matches.shape[0]
                #    if nEssentialInliers > 100:
                #        nHomog_nEssent.append(( nHomographyInliers, -nEssentialInliers, edge ))
                #    else:
                #        nHomog_nEssent.append(( 100000, -nEssentialInliers, edge ))
                #nHomog_nEssent.sort()
                #qualities = { el[2] : float( len(nHomog_nEssent)-idx ) for idx,el in enumerate(nHomog_nEssent) }
                qualities = {}
                for edge, (matches, E, R, t, nHomographyInliers) in edge2matches.items():
                    nEssentialInliers = matches.shape[0]
                    qualities[edge] = float( nEssentialInliers - nHomographyInliers + 100 if nEssentialInliers > 100 else nEssentialInliers )


            for edge,(matches,E,R,t,nHomographyInliers) in edge2matches.items():
                if 0:
                    # TODO: this method of estimating the expected quality of relative orientation is stupid, as it cannot consider the offset vector (its length is arbitrary):
                    # A pair of photos taken with the same PRC and pointing away from each other (with small overlap, such that there are feature matches),
                    # may result as having high quality, even though their object points are not reconstructible!
                    # See e.g. Piazza Bra, images 7, 8, 9.
                    rotVec,_ = cv2.Rodrigues(R)
                    angleGon = (rotVec**2).sum()**.5 * 200/np.pi
                    #quality = matches.shape[0] * angleGon
                    # Consider pairs with less than 15 matches as very unreliable. More than 100 matches do not add reliability.
                    # Intersection angles of less than 21 gon are very unreliable. 121 is the optimal intersection angle in the triangulated point, if the distances between the PRCs and the new point are equal.
                    # Within those bounds, 1 match counts as much as 1 gon.
                    quality = max( 0., min( 100, matches.shape[0] - 15 ) ) * max( 0., 121 - abs( 121 - angleGon ) - 21 )
                elif 0:
                    quality = sfm.imgPairQuality( edge[0], edge[1], matches, relOri.SfMManager.PairWiseOri( E,R,t ) )
                else:
                    quality = qualities[edge]
                compression = {}
                if config.isGPL:
                    if matches.size >= 8:
                        compression = dict( chunks=matches.shape, compression="szip", compression_opts = ('nn', 32 if matches.size>=32 else 16 if matches.size>=16 else 8 ) )
                else:
                    compression = dict( chunks=matches.shape, compression="gzip", compression_opts=6 )
                ds = group.create_dataset( '{}?{}'.format( *( os.path.basename( sfm.imgs[k].fullPath ) for k in edge ) ),
                                                            data=matches, **compression )
                ds.attrs['E'], ds.attrs['R'], ds.attrs['t'], ds.attrs['quality'] = E, R, t, quality
                edge2matches[edge] = matches, E, R, t, quality
                progress += 1

    with contextlib.ExitStack() as stack:
        stack.callback( filterMatches )
        if args.reuseMatches == ReuseMatches.descriptorAndEpipolarMatches:
            with contextlib.suppress((OSError,KeyError)), \
                 h5py.File( str(fnFeatures), 'r' ) as features:
                group = features['matchesFiltered']
                basename2idx = { os.path.basename( img.fullPath ) : idx for idx,img in enumerate(sfm.imgs) }
                for key,value in group.items():
                    edge2matches[tuple( basename2idx[el] for el in key.split('?') )] = np.array(value), value.attrs['E'], value.attrs['R'], value.attrs['t'], value.attrs['quality']
                stack.pop_all()
                logger.info( "Loaded filtered matches from file '{}'", fnFeatures )
    del filterMatches

    qualities = []
    nConnectionsOrig = len( sfm.edge2matches )
       
    sfm.edge2matches = dict() # filterMatchesByEMatrix may return less edges than were passed
    for edge,(matches,E,R,t,quality) in edge2matches.items():
        assert matches.shape[0] >= 5 # should be ensured by ori.pyd
        sfm.edge2PairWiseOri[edge] = relOri.SfMManager.PairWiseOri( E,R,t )
        qualities.append( (edge[0], edge[1], quality ) )
        sfm.edge2matches[edge] = matches
    edge2matches = None # Cython doesn't like this to be del'ed here.
                
    logger.info( sfm.countFeatures(), tag='#features after E-matrix test' )
    logger.info( "Removed {}({:.0%}) image connections due to E-matrix test, {} remain", nConnectionsOrig - len(sfm.edge2matches), float(nConnectionsOrig - len(sfm.edge2matches))/nConnectionsOrig, len(sfm.edge2matches) )

    def plotFilteredMatches():
        outDir = str( args.outDir / 'matchesFiltered' )
        logger.info( "Plot filtered matches to " + outDir )
        for edge, matches in sfm.edge2matches.items():
            imageMaskKeypts = []
            for iImg in edge:
                el = match.ImageMaskKeypts()
                el.imgFn = imageFilePathsMasks[iImg][0]
                el.maskFn = imageFilePathsMasks[iImg][1] or ''
                el.keyPts = sfm.imgs[iImg].keypoints
                imageMaskKeypts.append(el)
            match.plotMatches( matches, imageMaskKeypts, outDir, preprocessingOpts )
    if args.plotFilteredMatches:
        plotFilteredMatches()

    if 0:
        logger.info("Saving filtered matches to image files")
        printer.allMatches( "matches_inlier" )
            
    removedEdges = []

    if args.atLeast3Obs == AtLeast3obs.before:
        # Remove all feature tracks with less than 3 observations. In case of very high overlap, this may help to keep the reconstruction of manageable size.
        def remove2viewPts():
            sfm.buildFeatureTracks()
            sfm.featureTracks.computeComponentSizes()
            edge2matches = dict() # while we may change the values of a dict while iterating over it, we cannot remove whole items from it. Thus, create a local dict and replace the original with it in the end.
            removedEdges = []
            nBefore = 0
            nAfter = 0
            # TODO: the following can be parallelized easily in C++.
            for edge,old_matches in sfm.edge2matches.items():
                matches = []
                nBefore += old_matches.shape[0]
                for iMatch in range( old_matches.shape[0] ):
                    found,componentSize = sfm.featureTracks.componentSize( graph.ImageFeatureID( edge[0], old_matches[iMatch,0].item() ) )
                    assert found
                    if componentSize > 2:
                        matches.append( old_matches[iMatch,:] )
                if len(matches):
                    edge2matches[ edge ] = np.array( matches, old_matches.dtype )
                    nAfter += len(matches)
                else:
                    removedEdges.append(edge)
            logger.info('AtLeast3obs.before: #matches reduced from {} to {} ({:.0%}).\n'
                        '#edges reduced from {} to {} ({:.0%})', nBefore, nAfter, nAfter/nBefore, 
                                                                 len(sfm.edge2matches), len(edge2matches), len(edge2matches)/len(sfm.edge2matches)  )
            return edge2matches, removedEdges
        sfm.edge2matches, removedEdges_ = remove2viewPts()
        removedEdges.extend(removedEdges_)

    logger.info('Compute feature tracks')
    sfm.buildFeatureTracks()
    logger.info('Detect multiple projections') 
    msgs,removedEdges_ = sfm.removeMultipleProjections()
    removedEdges.extend(removedEdges_)
    if msgs:
        logger.log( log.Severity.info,
                    log.Sink.all if len(msgs) < 500 else log.Sink.file,
                    'Removed multiple projections\n'
                    'pho1\tpho2\t#removed\t%removed\t#remaining\n'
                    + '\n'.join( ( "{}\t{}\t{:4}\t{:2.0%}\t{:4}".format( 
                                    sfm.imgs[msg.iImg1].shortName,
                                    sfm.imgs[msg.iImg2].shortName,
                                    msg.nRemoved,
                                    msg.nRemoved / (msg.nRemoved+msg.nRemain),
                                    msg.nRemain ) for msg in msgs ) ) )

        logger.info('Re-compute feature tracks')
        sfm.buildFeatureTracks()
        
    nFeaturesTotal = sfm.countFeatures()
    logger.info( nFeaturesTotal, tag='#features after multiple projections removal' )

    if args.thinOut:
        removedEdges.extend( sfm.thinOutTracks( *args.thinOut ) )
        nFeaturesTotal = sfm.countFeatures()
        logger.info( nFeaturesTotal, tag='#features after thin out' )

    if 0:
        import pickle
        pickleFn = args.outDir / 'featureTracks.pickle'
        with pickleFn.open( 'wb' ) as fout:
            pickle.dump( sfm.featureTracks, fout, protocol=pickle.HIGHEST_PROTOCOL )
        logger.info( 'feature tracks dumped to {}.', pickleFn )
        #quit()

    logger.info("Compute image connectivity")
    removedEdges = set(removedEdges) # efficient lookup
    # map each edge in imageConnectivity as a tuple(iVertex0,iVertex1) -> element in matchesCoords
    for iImg1,iImg2,quality in qualities:
        # sfm.removeMultipleProjections() may delete all matches of an edge.
        # In that case, it removed that edge from sfm.edge2matches, while qualities remains unchanged
        if (iImg1,iImg2) in removedEdges:
            continue
        img1 = graph.ImageConnectivity.Image(iImg1)
        img2 = graph.ImageConnectivity.Image(iImg2)
            
        sfm.imageConnectivity.addEdge(
            graph.ImageConnectivity.Edge( img1=img1, img2=img2, quality=quality ) )
    del qualities
     
    if args.plots:          
        minCut = printer.connectivityGraph( "initial", countObs=False, minCut= len(sfm.imgs)<1000 )
        if minCut is not None:
            logger.info( "Minimum cut through image connectivity graph weighs {:.0f} image point observations and separates {} images from the other {} images.\n"
                         "Smaller set consists of: {}", 
                         minCut.sumOfWeights,
                         minCut.nImagesSmallerSet,
                         len(sfm.imgs) - minCut.nImagesSmallerSet,
                         ', '.join(sfm.imgs[iImg].shortName for iImg in minCut.idxsImagesSmallerSet) )
        
    if sfm.imageConnectivity.nVertices() < len(sfm.imgs):
        logger.warning("imageConnectivity contains only {} vertices, while {} image files were input", sfm.imageConnectivity.nVertices(), len(sfm.imgs) )
        
    nConnectedComponents = sfm.imageConnectivity.nConnectedComponents()
    if nConnectedComponents != 1:
        logger.warning( "imageConnectivity contains {} components!", nConnectedComponents )  
        # TODO: consider this during reconstruction
        # One way of considering it could be to reconstruct only the largest featureTrack, for example.
        # TODO: warn about articulation points
                 
    # Huber loss has the great advantage that its scale is directly interpretable.
    # However, Huber's derivatives are not smooth at the transition from squared to L1-loss.
    # SoftLOne instead has continuous 1st and 2nd derivatives (which are used by oriental.adjust), and its 0-th derivative is similar to Huber's.
    robustLossFunc = lambda: adjust.loss.SoftLOne( args.maxFinalResidualNorm / 2  )       
    globalLoss = adjust.loss.Wrapper( robustLossFunc() )
        
    # global bundle block
    #block = adjust.Problem()
    blockSolveOptions = adjust.Solver.Options()
    blockSolveOptions.max_num_iterations = 500
    blockSolveOptions.max_num_consecutive_invalid_steps = 15 # error for data set D:\arap\data\Carnuntum_UAS_Geert: 'Number of successive invalid steps more than Solver::Options::max_num_consecutive_invalid_steps: 5'
    blockSolveOptions.linear_solver_type = adjust.LinearSolverType.SPARSE_SCHUR
    blockSolveOptions.linear_solver_ordering = adjust.ParameterBlockOrdering()    

    summary = adjust.Solver.Summary()

    evalOpts = adjust.Problem.EvaluateOptions()
    evalOpts.apply_loss_function = False

    interestingImageHasPassed = False
        
    do_break = False
    while not do_break:
            
        #if args.plotIntermed:
        #    plt.close('all') # close figures to save memory
        imgsOrientedSinceLastFullAdjustment = []
        affectedImgs = set()
                
        if len(sfm.orientedImgs) == 0:
            ## initial state: no image has been oriented yet.
            block, globalLoss, addFeatureTracks = sfm.initBlock( args.maxIntermedResidualNorm, blockSolveOptions, robustLossFunc, args.initPair, printer if args.plotIntermed else None )
            affectedImgs = set(sfm.orientedImgs)
        
        else: # not the initial image pair
            def isInterestingImage():
                return img.idx in ( sfm.imageShortName2idx(elem) for elem in interestingImages if elem in sfm.imageShortName2idx )

            for iImg2OrientBeforeFullAdjustment in range(args.nImgs2OrientBeforeFullAdjustment):
                # Get the unoriented image that observes the most already reconstructed object points.
                # objPts with more than 2 imgObs are much more reliable. Do as now, but only for objPts that are observed in at least 3 images. Only if that result seems unreliable, repeat with objPts with 2 or more imgObs.
                candidates = sfm.candidatesPnP( exclude = { el[0] for el in imgsOrientedSinceLastFullAdjustment } )
                    
                #candidates.sort( key = lambda x: x.iKeyPts.shape[0], reverse=True )
                if len(candidates) == 0:
                    logger.info("No more candidate images left!")
                    do_break = True
                    break


                earlyExit, imgs = sfm.PnP( candidates, args.minPnPInlierRatio, confidenceLevel, args.maxIntermedResidualNorm, nInliersBreak=100, inlierRatioBreak=0.75 )

                if not len(imgs):
                    if not len(imgsOrientedSinceLastFullAdjustment):
                        logger.info("All candidate images have been tried to orient, but all failed!")
                        do_break = True
                    else:
                        logger.info("All candidate images have been tried to orient, but all failed! Try again after full adjustment.")
                    break
                    
                def areaRatio( img, bKeyPts, bInliers ):
                    image = sfm.imgs[img.idx]
                    keypts = image.keypoints[bKeyPts,:2][bInliers,:]
                    keyPtRows = np.ascontiguousarray( keypts ).view(np.dtype((np.void, keypts.dtype.itemsize * keypts.shape[1])))
                    _, uniqueIdxs = np.unique( keyPtRows, return_index=True )
                    keypts = keypts[uniqueIdxs]
                    if len(keypts) < 3:
                        return 0. # Otherwise, spatial.ConvexHull would crash, e.g. for: keypts = np.array([[ 546.76568604, -241.02688599], [ 547.59539795, -241.30935669]], dtype=float32)
                    try:
                        return spatial.ConvexHull( keypts ).volume / ( image.width * image.height )
                    except spatial.qhull.QhullError:
                        return 0. # too few points, or all (nearly) collinear

                logger.verbose(   'PnP candidates\n'
                                  'pho\t#inliers\t%inliers\t#objPts\timg area\n'
                                + '\n'.join( ( "{}\t{}\t{:.1%}\t{}\t{:4.0%}".format( sfm.imgs[img.img.idx].shortName,
                                                                                     nInliers,
                                                                                     nInliers/nKeyPts,
                                                                                     nKeyPts,
                                                                                     areaRatio( img.img, img.bKeyPts, img.bInliers ) ) 
                                             for img in imgs for [img,nKeyPts,nInliers] in [[ img, len( img.bInliers ), np.count_nonzero( img.bInliers ) ]] ) ) )
                if not earlyExit: # sfm.PnP appends items to its result list. If it exited early, then the promising item is the last one -> don't sort.
                    # sort by the number of inliers
                    #imgs.sort( key = lambda img: np.count_nonzero(img.bInliers) )
                    imgs.sort( key = lambda img: img.density )
                    
                img, bKeyPts, R, t, bInliers, representatives, density = imgs[-1]

                if isInterestingImage():
                    interestingImageHasPassed = True   
                    
                #if interestingImageHasPassed: # print the inliers/outliers for each tried image
                #    printer.candidates( imgs )
                
                nKeyPts = len(bInliers)
                nInliers = np.count_nonzero( bInliers )
                logger.info("pho {} PnP: {:4d}({:4.0%}) inliers of {:4d} objPts; {:4.0%} img area",
                    sfm.imgs[img.idx].shortName,
                    nInliers,
                    float(nInliers)/nKeyPts,
                    nKeyPts,
                    areaRatio( img, bKeyPts, bInliers ) )
                    
                if len(imgsOrientedSinceLastFullAdjustment) and not earlyExit:
                    logger.info('PnP results not too reliable. Try again after full adjustment.')
                    break

                if nInliers < 20:
                    logger.warning( "PnP: only {} inliers have been found. Results may be unreliable.", nInliers )
        
                camNew = sfm.imgs[img.idx]
                camNew.t = t
                camNew.R = R
        
                # Falls es nur sehr wenige inlier gibt,
                # img aber sehr viele matches mit einem bereits orientierten pho hat, die den E-matrix-Test bestanden haben,
                # dann könnte es besser sein, statt des Ergebnisses von solvePnPRansac basierend auf den wenigen bereits rekonstruierten objPts,
                # direkt die relative Orientierung zu berechnen zu jenem bereits orientieren Nachbarbild, das die meisten matches hat!
                # Denn: für die Berechnung der relativen Orientierung sind keine objPts nötig!
                # Allerdings muss trotzdem der Maßstab des Modells aus gemeinsamen, bereits rekonstruierten objPts bestimmt werden.
                # Deshalb gleich candidates weiterverwenden: enthält alle orientierten Bilder, die Nachbarbilder von noch nicht orientierten Bildern sind, deren E-Matrix-Filterung bestanden wurde,
                # und die zumindest 4 gemeinsame, bereits rekonstruierte objPts haben.
                # Am besten die E-matrix, die in getQuality berechnet wurde, abspeichern statt neu berechnen!
                #if inliers.shape[0] < 20:
                #    if iImg2OrientBeforeFullAdjustment > 0:
                #        break # do a full adjustment, and then try again.
                #    logger.warning("Result of solvePnPRansac seems unreliable with only {} inliers. Trying to orient via E-matrix.", inliers.shape[0] ) 
                #
                #    result = sfm.orientCandidatesByEMatrix( candidates, args.maxEpipolarDist, args.minPairInlierRatio, confidenceLevel )
                #    if len(result):
                #        logger.info("Orientation via E-matrix seems okay")
                #        img, camNew.R, camNew.t, iKeyPts, representatives, inliers = result
                            
                #hadIor = block.HasParameterBlock( camNew.ior )
                #hadAdp = block.HasParameterBlock( camNew.adp )
                #sfm.orientedImgs.append(img.idx)
                #
                ## tell sfm.imageConnectivity about newly oriented image
                #img.state = graph.ImageConnectivity.Image.State.oriented
                #sfm.imageConnectivity.setImageState( img )
        
                addFeatureTracks = dict.fromkeys( representatives[bInliers], relOri.SfMManager.ObjPtState.common )
                    
                    
                # After the final sample has been found, cv2.solvePnPRansac with flags=cv2.EPNP solves for all inliers (with useExtrinsicGuess=true),
                # and thus, the following might as well be skipped.
                # Note: sfm.addClosures unconditionally inserts image observations for already existing object points, no matter if sfm.adjustSingle changes the orientation that much that some or all image points would then be deemed outliers.
                summary = sfm.adjustSingle( img.idx, bKeyPts, representatives, bInliers, printer if args.plotIntermed else None )
                
                if not adjust.isSuccess( summary.termination_type ):
                    # this state could be handled by removing the added observations from the block again.
                    logger.info( summary.FullReport() )
                    sfm.logActiveFeaturesPerImage()
                    raise Exception("adjustment failed after additional observations have been introduced into the block")        
                    
                imgsOrientedSinceLastFullAdjustment.append(( img, camNew ))

                #msgs,affectedImgs = sfm.addClosures( img, args.maxIntermedResidualNorm, block, blockSolveOptions, globalLoss, addFeatureTracks )
                #logger.info( 'pho {} added: {} new imgPts, {} new objPts',
                #             sfm.imgs[img.idx].shortName,
                #             sum((el for msg in msgs for el in (msg.nObsAddedOther,msg.nObsAddedSelf))),
                #             sum((msg.nObjNew for msg in msgs)) )
                #msgs.sort( key = lambda x: x.nTotal, reverse=True )
                #logger.verbose( 'pho {} added: new imgPts and objPts per other pho\n'
                #                'other pho\t#valid\t%valid\t#total\t#addedSelf\t#addedOther\t#newObjPts\n'
                #                '{}',
                #                sfm.imgs[img.idx].shortName,
                #                '\n'.join(
                #                    ( '{}\t{:4}\t{:4.0%}\t{:4}\t{:4}\t{:4}\t{:4}'.format(
                #                            sfm.imgs[msg.iOther].shortName, msg.nValid, msg.nValid/msg.nTotal, msg.nTotal, msg.nObsAddedSelf, msg.nObsAddedOther, msg.nObjNew )
                #                        for msg in msgs ) ) )
                #
                #for el in ( camNew.t, camNew.omfika, camNew.ior, camNew.adp ):
                #    blockSolveOptions.linear_solver_ordering.AddElementToGroup( el, 1 )
                #
                #if not hadIor:
                #    block.SetParameterBlockConstant(camNew.ior)
                #if not hadAdp:
                #    block.SetParameterBlockConstant(camNew.adp)
                #
                #if isInterestingImage():
                #    printer.block( dict(), "reconstruction_with{}".format(sfm.shortFileNames(img.idx)) )

                # Even if nImgs2OrientBeforeFullAdjustment says that another image shall be oriented before the next adjustment,
                # we do so only if a minimum number of oriented images share their IOR with the new image.
                if args.adjustIorAdp == AdjustIorAdp.during:
                    nOrientedSameIOR = sum(( 1 for iImg in sfm.orientedImgs if iImg != img.idx and id(sfm.imgs[iImg].ior) == id(camNew.ior) ))
                    if nOrientedSameIOR < 5:
                        break
        
            nObsAddedTotal = 0
            nObjNewTotal = 0
            for img,camNew in imgsOrientedSinceLastFullAdjustment:
                hadIor = block.HasParameterBlock( camNew.ior )
                hadAdp = block.HasParameterBlock( camNew.adp )
                sfm.orientedImgs.append(img.idx)
            
                # tell sfm.imageConnectivity about newly oriented image
                img.state = graph.ImageConnectivity.Image.State.oriented
                sfm.imageConnectivity.setImageState( img )

                msgs,affectedImgs_ = sfm.addClosures( img, args.maxIntermedResidualNorm, block, blockSolveOptions, globalLoss, addFeatureTracks )
                affectedImgs.update(affectedImgs_)
                nObsAdded = sum((el for msg in msgs for el in (msg.nObsAddedOther,msg.nObsAddedSelf)))
                nObjNew = sum((msg.nObjNew for msg in msgs))
                logger.info( 'pho {} added: {:5d} imgPts, {:4d} objPts',
                             sfm.imgs[img.idx].shortName,
                             nObsAdded,
                             nObjNew )
                nObsAddedTotal += nObsAdded
                nObjNewTotal += nObjNew
                msgs.sort( key = lambda x: x.nTotal, reverse=True )
                logger.verbose( 'pho {} added: new imgPts and objPts per other pho\n'
                                'other pho\t#valid\t%valid\t#total\t#addedSelf\t#addedOther\t#newObjPts\n'
                                '{}',
                                sfm.imgs[img.idx].shortName,
                                '\n'.join(
                                    ( '{}\t{:4}\t{:4.0%}\t{:4}\t{:4}\t{:4}\t{:4}'.format(
                                            sfm.imgs[msg.iOther].shortName, msg.nValid, msg.nValid/msg.nTotal, msg.nTotal, msg.nObsAddedSelf, msg.nObsAddedOther, msg.nObjNew )
                                        for msg in msgs ) ) )
                
                for el in camNew.t, camNew.omfika, camNew.ior, camNew.adp:
                    blockSolveOptions.linear_solver_ordering.AddElementToGroup( el, 1 )
                
                if not hadIor:
                    block.SetParameterBlockConstant(camNew.ior)
                if not hadAdp:
                    block.SetParameterBlockConstant(camNew.adp)
                
                if isInterestingImage():
                    printer.block( dict(), "reconstruction_with{}".format( sfm.imgs[img.idx].shortName ) )
            if len(imgsOrientedSinceLastFullAdjustment) > 1:
                logger.info( 'total added: {:5d} imgPts, {:4d} objPts', nObsAddedTotal, nObjNewTotal )

        # end of distinction between inital image pair and subsequent images
            
        #assert( abs( np.linalg.norm( shouldHaveUnitNorm ) - 1 ) < 1.e-7 )
        #logger.info("Should have norm 1., has: {}".format( np.linalg.norm(shouldHaveUnitNorm) ) )
            
        if interestingImageHasPassed:
            printer.block( addFeatureTracks, "block before adj{}".format( '' if not imgsOrientedSinceLastFullAdjustment else sfm.imgs[imgsOrientedSinceLastFullAdjustment[0][0].idx].shortName ) )
            # check that the residuals are all small. Otherwise, transformations might be wrong.
            #printer.allImageResiduals( "bAdj", addFeatureTracks )
        
        if args.plotIntermed:
            residuals, = block.Evaluate(evalOpts)
            printer.residualHistAndLoss( robustLossFunc(), residuals )
                
        logger.verbose("Full adjustment ...")
        adjust.Solve(blockSolveOptions, block, summary)
        logger.info("Full adjustment done.")
        
        if interestingImageHasPassed:
            printer.block( addFeatureTracks, "block after adj{}".format( '' if not imgsOrientedSinceLastFullAdjustment else sfm.imgs[imgsOrientedSinceLastFullAdjustment[0][0].idx].shortName ) )
            #printer.allImageResiduals( "aAdj", addFeatureTracks )
            
        if not adjust.isSuccess( summary.termination_type ):
            # this state could be handled by removing the added observations from the block again.
            logger.info( summary.FullReport() )
            sfm.logActiveFeaturesPerImage()
            sfm.logCamParams( block, withIorAdp = args.adjustIorAdp == AdjustIorAdp.during )
            if args.plots:
                printer.block( dict(), "reconstruction" )
            raise Exception("adjustment failed after additional observations have been introduced into the block")

        if args.adjustIorAdp == AdjustIorAdp.during:
            if 1:
                sfm.makeIorsAdpsVariableAtOnce( block, blockSolveOptions,
                                                params = args.adjustWhatDuring,
                                                iorIds=tuple(affectedImgs),
                                                maxAbsCorr=0.9 )
            else:
                while 1:
                    msg = sfm.makeIorAdpVariable( block, maxAbsCorr=0.6, affectedImgs=affectedImgs ) # use a lower threshold on the maximum absolute correlation here, to keep the block stable
                    if not msg:
                        break
                    logger.info( msg )
                    logger.verbose("Full adjustment ...")
                    adjust.Solve(blockSolveOptions, block, summary)
                    logger.info("Full adjustment done.")
        
                    if not adjust.isSuccess( summary.termination_type ):
                        # this state could be handled by removing the added observations from the block again.
                        logger.info( summary.FullReport() )
                        sfm.logActiveFeaturesPerImage()
                        raise Exception("adjustment failed after additional parameters have been introduced into the block")
              
        # deactivate bad observations
        nRemovedObjPts, nRemovedImgObs, nObjPtsUnstable = sfm.deactivateOutliers( args.maxIntermedResidualNorm, block, robustLossFunc )
        if any( (nRemovedObjPts, nRemovedImgObs) ):
            logger.info( "Outliers: {} imgPts, {} objPts ({} angle); nMinObjPerPho {}", nRemovedImgObs, nRemovedObjPts, nObjPtsUnstable, min(sfm.imgs[iImg].nReconstObjPts for iImg in sfm.orientedImgs) )
            if 0:
                # a full re-adjustment after deactivation of bad observations costs time and may not be necessary:    
                adjust.Solve(blockSolveOptions, block, summary)
            
            if interestingImageHasPassed:
                printer.block( addFeatureTracks, "block after adj after remove{}".format( '' if not imgsOrientedSinceLastFullAdjustment else sfm.imgs[imgsOrientedSinceLastFullAdjustment[0][0].idx].shortName ) )
                #printer.allImageResiduals( "aAdjaRem", addFeatureTracks )
            
        if args.plotIntermed:
            residuals, = block.Evaluate(evalOpts)
            printer.residualHistAndLoss( robustLossFunc(), residuals, args.maxIntermedResidualNorm )

            costs = []
            counts = []
            resBlocks = []
            for resBlock in block.GetResidualBlocksForParameterBlock( sfm.imgs[sfm.orientedImgs[-1]].t ):
                cost = block.GetCostFunctionForResidualBlock( resBlock )
                if cost.isAct():
                    objPt = [ el for el in block.GetParameterBlocksForResidualBlock(resBlock) if isinstance( el, adjust.parameters.ObjectPoint ) ]
                    assert len(objPt)==1
                    nObs = sum( 1 for resBlock in block.GetResidualBlocksForParameterBlock( objPt[0] ) if block.GetCostFunctionForResidualBlock(resBlock).isAct() )
                    resBlocks.append(resBlock)
                    costs.append(cost)
                    counts.append(nObs)

            evalOpts_ = adjust.Problem.EvaluateOptions()
            evalOpts_.apply_loss_function = False
            evalOpts_.set_residual_blocks( resBlocks )
            residuals = block.Evaluate(evalOpts_)[0].reshape((-1,2))
            xys = np.fromiter( ( el for cost in costs for el in (cost.x,cost.y) ), dtype=float, count=len(costs)*2 ).reshape((-1,2))
            counts = np.array(counts,int)
            import oriental.utils.pyplot as plt
            plt.figure(394); plt.clf()
            plt.imshow( printer.imageRGB(sfm.orientedImgs[-1]), interpolation='none' )
            for idx, (sel, col) in enumerate(((counts==2,'m'), (counts==3,'c'), (counts>3,'g'))):
                plt.plot( xys[sel,0], -xys[sel,1], 'o' + col, markeredgecolor=col, markersize=idx+2 )
                plt.plot( np.vstack( ( xys[sel,0] - residuals[sel,0], xys[sel,0]) ),
                          np.vstack( (-xys[sel,1] + residuals[sel,1],-xys[sel,1]) ), color=col, linewidth=1., marker=None )
            
        sfm.logCamParams( block, severity=log.Severity.verbose )
        
        minObj,maxObj = sfm.objSpaceBbox()
                    
        logger.info('{} images oriented, {} left.\n'
                    'RefSys min/max: {:.2f}/{:.2f} {:.2f}/{:.2f} {:.2f}/{:.2f} nObjPts: {}',
            len(sfm.orientedImgs), len(sfm.imgs)-len(sfm.orientedImgs),
            minObj[0],maxObj[0],
            minObj[1],maxObj[1],
            minObj[2],maxObj[2],
            len(sfm.featureTrack2ObjPt) )   
            
        if not adjust.isSuccess( summary.termination_type ):
            logger.info( summary.FullReport() )
            sfm.logActiveFeaturesPerImage()
            raise Exception("adjustment failed after additional observations have been introduced into the block and bad tracks have been removed")

        # end of incremental reconstruction

    if args.atLeast3Obs == AtLeast3obs.after:
        logger.info("remove objPts with less than 3 img obs.")
        nRemovedObjPts, nRemovedImgObs = sfm.atLeast3obs( block )
        logger.info( "Removed {} objPts, removed {} imgPts", nRemovedObjPts, nRemovedImgObs )
        logger.info("block consists of {} phos, {} imgPts, and {} objPts", len(sfm.orientedImgs),len(sfm.imgFeature2costAndResidualBlockID),len(sfm.featureTrack2ObjPt) )
        adjust.Solve(blockSolveOptions, block, summary)
        if not adjust.isSuccess( summary.termination_type ):
            logger.info( summary.FullReport() )
            sfm.logActiveFeaturesPerImage()
            raise Exception("adjustment failed after removing objPts with less than 3 img obs")
        if args.plotIntermed:
            sfm.logCamParams( block )
            printer.block( dict(), "lstSq 3-img-obs" )
            printer.connectivityGraph( "atLeast3obs", True )

    if args.adjustIorAdp in ( AdjustIorAdp.atend, AdjustIorAdp.during ):
        if 1:
            sfm.makeIorsAdpsVariableAtOnce( block, blockSolveOptions,
                                            ( adjust.PhotoDistortion.optPolynomRadial3, 2, 0, adjust.PhotoDistortion.optPolynomRadial5 ),
                                            maxAbsCorr=0.95 )
        else:
            while 1:
                msg = sfm.makeIorAdpVariable( block, principalPoint=True )
                if not msg:
                    break
                logger.info( msg )
                logger.verbose("Full adjustment ...")
                adjust.Solve(blockSolveOptions, block, summary)
                logger.info("Full adjustment done.")
        
                if not adjust.isSuccess( summary.termination_type ):
                    # this state could be handled by removing the added observations from the block again.
                    logger.info( summary.FullReport() )
                    sfm.logActiveFeaturesPerImage()
                    raise Exception("adjustment failed after additional parameters have been introduced into the block")

                sfm.logCamParams( block, withIorAdp=True, severity=log.Severity.verbose )

    sfm.logCamParams( block ) 
    logger.verbose("robust block consists of {} phos, {} imgPts, and {} objPts", len(sfm.orientedImgs),len(sfm.imgFeature2costAndResidualBlockID),len(sfm.featureTrack2ObjPt) )

    # TODO estimate a residuals-dependent, sensible threshold to cut off outliers, to replace the user-specified args.maxFinalResidualNorm
    # the sum of squared residuals would follow the chi-squared distribution with 2 DoF, if we divided the residuals first by the (unknown) standard deviation: http://en.wikipedia.org/wiki/Chi-squared_distribution
    # the square root of the sum of squared residuals would follow the chi-distribution with 2 DoF, if we divided the residuals first by the (unknown) standard deviation: http://en.wikipedia.org/wiki/Chi_distribution
    # with a robust estimate of standard deviations of (the square root of) the sum of squared residuals, we may cut off the very tail of that distribution.
    # However, for robustly estimating the std.devs., we first need to know their distribution.
    # 3*sigmaMAD doesn't work, because while we compute sigmaMAD separately for the coordinate residuals (whose inliers we assume to be normal distributed),
    # we discard by residual norms. The square of the residual norms is chi square distributed (with 2 DoF, disregarding their non-unit-variance; defined for positive values only).
    # Also, even though sigmaMAD has the highest possible break down point (50%), it is very inefficient for estimating the standard deviation of a normally distributed random sample (only 37%),
    # see: 1993 Rousseeuw,Croux - Alternatives to the Median Absolute Deviation - J Amer.Stat.Assoc
    #residuals, = block.Evaluate( evalOpts )
    #printer.residuals( residuals, 3., fn='residualsRobust.png' )
    #medResidual = np.median( residuals )
    #sigmaMad = 1.4826 * np.median( np.abs( residuals - medResidual ) )
    #logger.info( 'sigmaMad: {}', sigmaMad )
    #sigmaMad2 = 1.4826 * np.median( np.abs( residuals ) )
    #logger.info( 'sigmaMad centered at zero: {}', sigmaMad2 )
    #q75, q25 = np.percentile(residuals, [75 ,25])
    #iqr = q75 - q25
    #sigmaIQR = iqr / 1.349
    #logger.info( 'sigmaIQR: {}', sigmaIQR )
    #printer.residualHistAndLoss( sfm.robustLoss(args.maxFinalResidualNorm), residuals, args.maxFinalResidualNorm, fn='residualNormsRobust.png' )
    #
    ## https://github.com/TheFrenchLeaf/Bundler/blob/master/src/Bundle.cpp (a modified version of Bundler)
    ## computes the 80% percentile p_80% of the image residual norms for each image.
    ## then uses 1.2 * 2 * p_80% as theshold for residual norms to discard outliers, after the theshold has been clamped to [8,16] (default bounds).
    ## computing the thresholds for each image separately, has the advantage of avoiding the risk to discard all image observations from an image, such that its parameters become undetermined
    #residualNormsSqr = residuals[0::2]**2 + residuals[1::2]**2
    #q80_normSqr = np.percentile(residualNormsSqr, 80)
    #logger.info( 'frenchThreshOnResidualNorms: {}', 1.2 * 2 * q80_normSqr**.5 )

    nRemovedObjPtsTot = 0
    nRemovedImgObsTot = 0
    while True:
        #adjust with robust loss and re-remove outliers, until no outliers remain with residual norms above maxFinalResidualNorm!
        nRemovedObjPts, nRemovedImgObs, nObjPtsUnstable = sfm.deactivateOutliers( args.maxFinalResidualNorm, block, robustLossFunc )
        nRemovedObjPtsTot += nRemovedObjPts
        nRemovedImgObsTot += nRemovedImgObs
        if not any( (nRemovedObjPts, nRemovedImgObs) ):
            break
        logger.verbose( "Outliers: {} imgPts, {} objPts ({} angle); nMinObjPerPho {}", nRemovedImgObs, nRemovedObjPts, nObjPtsUnstable, min(sfm.imgs[iImg].nReconstObjPts for iImg in sfm.orientedImgs) )
        logger.verbose("Full adjustment ...")
        adjust.Solve(blockSolveOptions, block, summary)
        logger.verbose("Full adjustment done.")
        
        if not adjust.isSuccess( summary.termination_type ):
            logger.info( summary.FullReport() )
            sfm.logActiveFeaturesPerImage()
            raise Exception( "Adjustment failed after {} outlier imgPts and {} objPts have been removed".format(nRemovedObjPts,nRemovedImgObs) )

    if any( (nRemovedObjPtsTot, nRemovedImgObsTot) ):
        logger.info( "Outliers: {} imgPts, {} objPts", nRemovedImgObsTot, nRemovedObjPtsTot )

    logger.info("Adjust with squared loss")
    globalLoss.Reset( adjust.loss.Trivial() )
    adjust.Solve(blockSolveOptions, block, summary)
    if not adjust.isSuccess( summary.termination_type ):
        logger.info( summary.FullReport() )
        sfm.logActiveFeaturesPerImage()
        raise Exception("adjustment failed after removal of outliers and resetting loss to squared")

    sfm.logCamParams( block, withIorAdp=True )

    if args.adjustIorAdp != AdjustIorAdp.never:
        def getCorrs():
            maxAbsCorrs = sfm.maxAbsCorrsIorAdp( block )
            maxAbsCorrs = sorted( maxAbsCorrs.items(), key=lambda x: x[0] )
            for iorId,(ior,adp) in maxAbsCorrs:
                pars = chain( ior, adp[[adjust.PhotoDistortion.optPolynomRadial3,adjust.PhotoDistortion.optPolynomRadial5]] )
                pars2 = ( ' '*3 if np.isnan(el) else '{:3.0%}'.format(el) for el in pars )
                yield chain( [str(iorId)], pars2 )
        try:
            logger.info( 'IOR/ADP max. abs. \N{GREEK SMALL LETTER RHO} with all other orientation parameters\n'
                         '{}\n'
                         '{}',
                         '\t'.join(['IOR/ADP ID','x0','y0','z0','r3','r5']),
                         '\n'.join( '\t'.join( row ) for row in getCorrs() ) )
        except:
            logger.warning( "Computation of correlations of IOR/ADP failed:\n{}", ''.join( traceback.format_exc() ) or 'Exception raised, but traceback unavailable' )

    targetCS = sfm.transformToAPrioriEOR( args.targetCS, args.stdDevPos, args.stdDevRot, args.dsm )
    sfm.logCamParams( block, withIorAdp=False ) # ior,adp remain unchanged, so don't log them

    # TODO: check if every photo orientation is actually determined. This may not be the case if after a photo had been oriented with PnP, all its observations (except <3) have been deactivated.
    # If we estimate IOR/ADP, and use a rank-revealing method for Qxx-estimation, then we anyway run into problems in that case. If we do not estimate IOR/ADP, then we do not reveal an underdetermined system!
    # Harder to identify, but as important, is revealing groups of phos that are not connected at all to the rest of photos, or only weakly connected.
    iUnoriented = np.setdiff1d( np.arange(len(sfm.imgs)), sfm.orientedImgs )
    logger.infoFile( len(iUnoriented), tag='#unorientedImgs' )
    if len(iUnoriented):
        logger.warning( "Could not orient all images: {}", ', '.join( sfm.imgs[iImg].shortName for iImg in iUnoriented ) )
    else:
        logger.info( "All images have been oriented" )

    redundancy = summary.num_residuals_reduced - summary.num_effective_parameters_reduced
    sigma0 = ( summary.final_cost * 2 / redundancy ) **.5 if redundancy > 0 else np.nan

    logger.info( 'Adjustment statistics\n'
                 'statistic\tvalue\n'
                 '{}',
                 '\n'.join([ "{0}\t{1:{2}}".format(*els) for els in [('#observations'                 , summary.num_residuals_reduced             , ''    ),
                                                                     ('#unknowns'                     , summary.num_effective_parameters_reduced  , ''    ),
                                                                     ('redundancy'                    , redundancy                                , ''    ),
                                                                     ('\N{GREEK SMALL LETTER SIGMA}_0', sigma0                                    , '.3f' ),
                                                                     ('#imgs'                         , len(sfm.orientedImgs)                     , ''    ),
                                                                     ('#imgPts'                       , len(sfm.imgFeature2costAndResidualBlockID), ''    ),
                                                                     ('#objPts'                       , len(sfm.featureTrack2ObjPt)               , ''    ) ] ]) )

    args.precision = False # See comments for parser. Implement S-Transform for that
    stdDevs = dict()
    if args.precision and redundancy > 0:
        logger.info("Compute the precision of unknowns")
        try:
            stdDevs = sfm.computePrecision( block, sigma0 )
        except:
            logger.warning( "Computation of the precision of unknowns failed:\n{}", ''.join( traceback.format_exc() ) or 'Exception raised, but traceback unavailable' )
    if not stdDevs:
        stdDevs = sfm.getConstancyAsPrecision( block )

    sfm.saveSQLite( str(resultsFn), targetCS, stdDevs )
    sfm.logStatistics( str(resultsFn) )

    if args.plots:
        logger.info( 'Generate graphics output' )
        printer.block( dict(), "reconstruction" )
        minCut = printer.connectivityGraph( "final", True, minCut = len(sfm.orientedImgs) < 1000 )    
        if minCut is not None:
            logger.info( "Minimum cut through image connectivity graph weighs {:.0f} image point observations and separates {} images from the other {} images:\n"
                         "{}", 
                         minCut.sumOfWeights,
                         minCut.nImagesSmallerSet,
                         len(sfm.orientedImgs) - minCut.nImagesSmallerSet,
                         ', '.join(sfm.imgs[iImg].shortName for iImg in minCut.idxsImagesSmallerSet) )
        # TODO: min-cut of graph with phos as vertices and edges for phos that are connected by threefold objPts (observed at least thrice), with edge weights=#threefold objPts
        # -> log where the block may be connected badly.
        resNormSqr = printer.allImageResiduals( '', dict(), residualScale=10., save=True )
        printer.allImgResNormsHisto( resNormSqr=resNormSqr, maxResNorm=args.maxFinalResidualNorm, fn='residuals.png' )
        printer.cleanUp()
