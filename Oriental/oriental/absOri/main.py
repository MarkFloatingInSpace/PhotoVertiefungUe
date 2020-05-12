# -*- coding: utf-8 -*-
"""Absolute orientation of aerial images by help of external reference data.


`absOri` uses as input:
  - The output of `relOri`, which may already be (coarsely) geo-referenced.
  - A reference orthophoto mosaic: `ortho`. This may be a web-mapping service
    or a local file.
  - A reference surface model: `dsm`. Use a local file.

Workflow is as follows:
  - Rectify each aerial w.r.t. the projection plane (if geo-referenced)
    or the best-fit plane through the object points (otherwise),
    by a homography, and stitch the rectified aerials to a mosaic.
  - Call MonoScope and let the user digitize control points in the prepared
    mosaic and in the reference orthophoto mosaic.
  - Interpolate terrain heights at the control points using the reference
    surface model.
  - Determine control point correspondences using a best-fit transformation.
  - Transfer the control image points from the prepared mosaic to the
    corresponding original aerial images.
  - Restore `relOri`'s bundle block,
    + introduce image point observations of the control points, and
    + direct observations of the control object points.
  - Extend the set of unknown IOR/ADP parameters, if feasible.
  - Store the results to the output database.

In the determination of correspondences between image points manually measured
in the prepared and in the reference orthophoto mosaics, `absOri` estimates
the best-fit 2D similarity transform whose rotation and scale are fixed
if `relOri` was able to geo-reference the data set.

TODO:
  - Transfer manual measurements in the prepared mosaic to all aerials
    using LSM. Forward intersect control points in model space and
    estimate best-fit 3D similarity transform.
  - Process single aerials via spatial resection.
  - Automatic control point measurements.
"""

# For now, let's assume that the a priori transformation from model to object space is sufficiently accurate,
# such that introducing image observations for the control points leads to convergence.
# Todo: interpolate some model surface, and project the image points into all other aerials.
#       Do LSM in respective areas, and compute an accurate 3-D position for each point observed in the mosaic.
#       Using the 3D-points, compute a robust inital transformation from model to object space.

# Otherwise, 
# interpolate coarse model 'heights', to be used only for a robust, initial transformation from model to object space.
# Let's simply use the 'height' of the nearest feature point in object space, which has been observed at least 3 times.
# While SpatiaLite would be exactly built for such searches, it only offers R*-trees (actually, SQLite does), which means that we can efficiently search in the DB only for intersecting rectangles / boxes.
# However, the density of feature points in object space noticably varies with location, and so it is difficult to make use of R*-tree searches here.
# Thus, let's create our own local KdTree.
#with dbapi2.connect( args.relOriDb.as_uri() + '?mode=rw', uri=True ) as relOriDb:
#    utils.db.initDataBase(relOriDb)
#    objPts = relOriDb.execute("""
#        SELECT X(objpts.pt),
#               Y(objpts.pt),
#               Z(objpts.pt)
#        FROM objpts
#        	JOIN    imgobs
#        	     ON imgobs.objPtID == objpts.id
#        GROUP BY objpts.id
#        HAVING COUNT(*)>2
#        """ ).fetchall()
#    objPts = np.array( objPts )
#kdTree = spatial_index.KdTree(objPts)
#idxs,dists = kdTree.knnSearch( queries=cpsOrtho_wrl, knn=1 )

# Mind that relOri transforms the block to a certain CS, if GPS PRC-positions are available, while the DSM CS is generally different.
# If no GPS PRC-positions are available, then relOri aligns the model CS to an average plane through the objPts, leaving the 1st PRC at the origin, and leaving scale and azimuth arbitrary: distance( 1st PRC, 2nd PRC ) == 1.
# Anyway,
# - we do not want to rely on GPS PRC-positions having been available, and hence the model being in a certain CS, and
# - even if GPS-positions have been available, those positions may be quite inaccurate, requiring many least-squares-iterations.
# So we have 2 options:
# (1) if at least 3 measurements were taken in the same rectified aerial, then estimate the pose of that aerial, and apply the resp. transformation to all other aerials.
# (2) compute a 2D similarity transform, assuming that the landscape is planar enough. Transform mosaic image points into the orthophoto CS.
#     How to apply this trafo to PRCs, ROTs, and objPts?
# (3) compute a 3D similarity transform, for which we need 3D points in the model CS.
#     I have found no lib to do monopolotting. PCL provides various surface reconstruction algorithms, but no method to intersect a ray with a surface.
#     PCL merely provides searching for voxels of an octree that intersect a ray.
#     We may compute the distances between the ray and all the points in all those voxels, select the point with minimum distance, and assign its projection onto the ray as depth of the control point
#     (i.e. we move the control point in model space along the observation ray).
# For simplicity, let's use (2) for now.
#
#@contract
#def loadOri( relOriDbFn : Path ):
#    if 1:
#        with dbapi2.connect( relOriDbFn.as_uri() + '?mode=ro', uri=True ) as relOriDb:
#            utils.db.initDataBase(relOriDb)
#            rows = relOriDb.execute("""
#                SELECT images.id, images.path, images.X0, images.Y0, images.Z0, images.r1, images.r2, images.r3, cameras.x0 as x0_,cameras.y0 as y0_, cameras.z0 as z0_,
#                       reference, normalizationRadius, {}
#                FROM images
#                    JOIN cameras
#                    ON images.camID == cameras.id
#                ORDER BY images.id """.format( ', '.join(adjust.PhotoDistortion.names.keys()) ) )
#            images = []
#            cameras = []
#            for row in rows:
#                images.append( Image( row['id'],
#                                      row['path'],
#                                      np.array([row['X0'], row['Y0'], row['Z0']]),
#                                      np.array([row['r1'],row['r2'],row['r3']]) ) )
#                adp = adjust.parameters.ADP( normalizationRadius=row['normalizationRadius'],
#                                             referencePoint     = adjust.AdpReferencePoint.names[ row['reference'] ] )
#                for name,value in adjust.PhotoDistortion.names.items():
#                    adp[int(value)] = row[name]
#                cameras.append( Camera( np.array([row['x0_'],row['y0_'],row['z0_']]),
#                                        # adp is already a subtype of np.array
#                                        adp ) )
#
#    else:
#        from oriental.utils.exif import phoInfo
#        fnAerials = list(Path(r'D:\arap\data\2014-12-04_Carnuntum_mit_Hardwareloesung').glob('*.jpg'))
#        imgIds = np.arange(len(fnAerials))
#        imgFns = [ str(el) for el in fnAerials ]
#        exifInfos = phoInfo( imgFns )
#
#        # geographic coordinates + height above ellipsoid
#        PRCs_wgs84 = [ exifInfo.prcWgs84 for exifInfo in exifInfos ]
#
#        PRCs = np.array([ np.array( wgs84_2_tgtCs.TransformPoint( *PRC_wgs84 ) ) for PRC_wgs84 in PRCs_wgs84 ])
#
#        # rotation matrices w.r.t. local system defined by north and gravity directions
#        omfikas = np.array([ exifInfo.prcWgsTangentOmFiKa for exifInfo in exifInfos ])
#
#        def focalFromCamDb():
#            with dbapi2.connect( Path(config.dbCameras).as_uri() + '?mode=ro', uri=True ) as cameras:
#                cameras.row_factory = dbapi2.Row
#                for info in exifInfos:
#                    rows = cameras.execute("""
#                        SELECT sensor_width_mm,
#                                sensor_height_mm
#                        FROM cameras
#                        WHERE     :exifMake  LIKE make || '%'
#                                AND :exifModel LIKE '%'  || model
#                    """, 
#                    { 'exifMake':info.make, 'exifModel':info.model } ).fetchall()
#                    assert len(rows)==1
#                    row = rows[0]
#                    info.ccdWidthHeight_mm = np.array([ row["sensor_width_mm"], row["sensor_height_mm"] ])
#                    info.focalLength_px = info.focalLength_mm * np.mean( info.ccdWidthHeight_px / info.ccdWidthHeight_mm )
#        focalFromCamDb()
#
#        iors = [ np.array([  info.ccdWidthHeight_px[0]/2,
#                            -info.ccdWidthHeight_px[1]/2,
#                                info.focalLength_px ]) for info in exifInfos ]
#        adps = [ adjust.parameters.ADP( normalizationRadius=100. ) for info in exifInfos ]
#        #nadirAngles = [ np.arccos( ori.omfika( omfika )[2,2] ) for omfika in omfikas ]
#
#
#    return images, cameras

# stdlib
import os, sys, subprocess, struct, itertools, argparse, traceback, functools, multiprocessing
from pathlib import Path
from contextlib import ExitStack, suppress
from datetime import datetime
from collections import OrderedDict, namedtuple

# oriental
from oriental import config, ObservationMode, Progress, adjust, blocks, ori, log, utils
adjust.importSubPackagesAndModules()
import oriental.blocks.footprint
import oriental.ori.transform
import oriental.utils
import oriental.utils.stats
import oriental.utils.gdal
import oriental.utils.db
import oriental.utils.crs
import oriental.utils.dsm
import oriental.utils.argparse
import oriental.utils.filePaths
import oriental.utils.pyplot_utils


# third-party
import sqlite3
from sqlite3 import dbapi2
import numpy as np
from scipy import linalg, spatial
import cv2
from osgeo import osr, ogr
osr.UseExceptions()
ogr.UseExceptions()
from contracts import contract, new_contract

new_contract('Path',Path)
new_contract('SpatialReference',osr.SpatialReference)
new_contract('CoordinateTransformation',osr.CoordinateTransformation)

logger = log.Logger("absOri")

Image = namedtuple('Image','id path prc omfika nRows nCols pix2cam mask_px obsWeights camera')
RectifiedImage = namedtuple('RectifiedImage', Image._fields + ('invHomography',))
Camera = namedtuple('Camera','id ior s_ior adp s_adp')
ObjPt = namedtuple('ObjPt','pt rgb nImgs')
ImgObsData = namedtuple('ImgObsData','id imgId objPtId')

MosaicGsd = utils.argparse.ArgParseEnum('GSD', 'min mean median max')
MosaicOrder = utils.argparse.ArgParseEnum('MosaicOrder', 'imageId nadirAngle flyingHeight')
MosaicResample = utils.argparse.ArgParseEnum('MosaicResample', 'direct preShrink')

new_contract('MosaicGsd',MosaicGsd)

@contract
def parseArgs( args : 'list(str)|None' = None ):
    docList = __doc__.splitlines()
    parser = argparse.ArgumentParser( description=docList[0],
                                      epilog='\n'.join(docList[1:]),
                                      formatter_class=utils.argparse.Formatter )

    parser.add_argument( '--outDir', default=Path.cwd() / "absOri", type=Path,
                         help='Store results in directory OUTDIR.' )
    parser.add_argument( '--relOriDb', type=Path,
                         help='Geo-reference an image set relatively oriented with relOri. Default: OUTDIR/../relOri/relOri.sqlite' )
    #parser.add_argument( '--photo', type=Path,
    #                     help='Geo-reference a single photo. Overrules RELORIDB.' )
    parser.add_argument( '--ortho', default='AustriaBasemap',
                         help='Ortho photo to use as reference. Either specify a file name or one of the configured WMS-services: {}'.format( ', '.join(utils.gdal.wms(utils.gdal.DataSet.ortho).keys()) )  )
    parser.add_argument( '--dsm', default='dhm_lamb_10m.tif',
                         help="Surface model to use as reference. Either specify a file name or one of the configured WMS-services: {}. If a relative path, file is also searched in $ORIENTAL/data.".format( ', '.join(utils.gdal.wms(utils.gdal.DataSet.dtm).keys()) ) )
    parser.add_argument( '--tgtCs', default='DSM',
                         help="Target coordinate system (e.g. 'EPSG:31253'). 'DSM' uses the DSM's system. 'MGI' chooses an appropriate MGI meridian strip. 'UTM' chooses an appropriate UTM zone. Note that the transformation between different vertical datums may be erroneous. "
                              "Thus, 'DSM' is always safe, as it makes sure that heights are left unchanged." )
    parser.add_argument( '--no-footprints', action='store_false', dest='footprints',
                         help='Do not store footprints in "OUTDIR/footprints/*.shp"' )
    parser.add_argument( '--no-plotResiduals', action='store_false', dest='plotResiduals',
                         help='Do not store residual plots in "OUTDIR/residuals/*.jpg"' )
    parser.add_argument( '--exportPly', action='store_true',
                         help='Store binary PLY-file of the geo-referenced reconstruction in "OUTDIR/reconstruction.ply"' )

    mosaicGroup = parser.add_argument_group('Mosaic')
    mosaicGroup.add_argument( '--gsd', default=str(MosaicGsd.median),
                              help="Ground sampling distance of mosaic. 'min'/'mean'/'median'/'max' compute the GSD based on the aerial images. 'ortho' gets replaced by the orthophoto's gsd and may be part of a numerical expression as in 'ortho*3/4'." )
    mosaicGroup.add_argument( '--order', default=MosaicOrder.nadirAngle, choices=MosaicOrder, type=MosaicOrder,
                              help='The order in which rectified aerials are printed into the mosaic. In overlapping areas, last printed aerials will finally be visible.' )
    mosaicGroup.add_argument( '--resample', default=MosaicResample.direct, choices=MosaicResample, type=MosaicResample,
                              help='Resampling of oblique images to verticals: '
                                   'DIRECT samples directly in the oblique image. '
                                   'PRESHRINK first shrinks the oblique image such that at least 1 of its corners is not undersampled when mapping to the vertical image, which is followed by an unsharp filter.' )
    mosaicGroup.add_argument( '--reuseMosaic', action='store_true',
                              help="Reuse an already existing mosaic, instead of recreating it. Saves time for repeated runs on the same dataset." )
    mosaicGroup.add_argument( '--saveSepRects', action='store_true',
                              help='Save rectified image for each aerial separately, in addition to the mosaic.' )

    measurementGroup = parser.add_argument_group('Measurements')
    measurementGroup.add_argument( '--no-fixedCorrespondences', action='store_false', dest='fixedCorrespondences',
                                   help='Corresponding points in the orthophoto and the mosaic have identical point names. Otherwise, correspondences are defined by best-fitting transformation.' )
                       
    adjustmentGroup = parser.add_argument_group('Adjustment')
    adjustmentGroup.add_argument( '--objStdDevs', nargs='*', type=float, default=[-1.],
                                  help="Standard deviations of control points in TGTCS. Image observations are always weighted with 1px (possibly multiplied by a scale factor for scanned images). Default: orthophotos's gsd/10 for X,Y and dsm's gsd/10 for Z." )
    adjustmentGroup.add_argument( '--fixedIorAdp', action='store_true',
                                  help='Do not adjust IOR and ADP parameters.' )
    adjustmentGroup.add_argument( '--fixedControlObjPts', action='store_true',
                                  help='Do not adjust control object point coordinates.' )
    adjustmentGroup.add_argument( '--no-precision', action='store_false', dest='precision',
                                  help='Do not compute the precision of unknowns.' )

    utils.argparse.addLoggingGroup( parser, 'absOriLog.xml' )

    generalGroup = parser.add_argument_group('General', 'Other general settings')
    generalGroup.add_argument( '--no-progress', dest='progress', action='store_false',
                               help="Don't show progress in the console." )

    cmdLine = sys.argv[:]
    args = parser.parse_args( args=args )
    main( args, cmdLine, parser )

@contract
def prepareMosaic( images : list,
                   terrainHeight : float,
                   mosaicId : int,
                   fnMosaic : Path,
                   fnMosaicMask : Path,
                   fnMosaicOverlap : Path,
                   mosaicCs : 'SpatialReference|None',
                   gsd : 'float|MosaicGsd|None',
                   absOriDbFn : Path,
                   order : MosaicOrder,
                   resample : MosaicResample,
                   outDir : 'Path|None' ) -> None:
    projectionWkt = mosaicCs.ExportToWkt() if mosaicCs else ''
    if order==MosaicOrder.imageId:
        permutation = range( len(images) )
    elif order==MosaicOrder.nadirAngle:
        # sort the data in order of decreasing nadir angle. Thus, oblique aerials are printed first into the mosaic, while vertical aerials are printed over them in the end.
        # compute nadir angles in target CS
        nadirAngles = [ np.arccos( ori.omfika( image.omfika )[2,2] ) for image in images ]
        permutation = np.argsort( nadirAngles )[::-1] # print most vertical photos last
    elif order==MosaicOrder.flyingHeight:
        flyingHeights = [ image.prc[2] for image in images ]
        permutation = np.argsort( flyingHeights )[::-1] # print lowest photos last
    else:
        raise Exception( "Mosaic order not implemented: {}".format(order) )
    images = [ images[idx] for idx in permutation ]

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
    # prepare an empty mosaic.
    # Determine its size first by intersecting the observation rays through the image corners with the ground plane,
    # and using the bounding rectangle of those intersection points
    cornersOnGround = []
    gsds = []
    for image in images:
        # Even though we consider ADP here to better estimate the mosaic extents,
        # the contents of rectified images may still be mapped to the outside of the mosaic for (nearly) axis-aligned images with barrel distortion.
        corners = getCorners( image.nRows, image.nCols )
        corners = image.pix2cam.forward(corners)
        ori.distortion_inplace( corners, image.camera.ior, ori.adpParam2Struct( image.camera.adp ), ori.DistortionCorrection.undist )
        R = ori.omfika( image.omfika )
        for corner in corners:
            ray = R.dot( np.r_[ corner, 0. ] - image.camera.ior )
            factor = ( terrainHeight - image.prc[2] ) / ray[2]
            cornersOnGround.append( (image.prc + factor*ray)[:2] )
        if not gsd or isinstance(gsd, MosaicGsd):
            # estimate GSD as the median pixel footprint where the optical axis intersects the ground plane.
            ray = -R[:,2] # observation ray along negative z-axis. Unit length!
            objDist = ( terrainHeight - image.prc[2] ) / ray[2]
            gsds.append( objDist / ( image.camera.ior[2] / image.pix2cam.meanScaleForward() ) )
    if isinstance(gsd, MosaicGsd):
        if gsd == MosaicGsd.min:
            gsd = np.min( gsds )
        elif gsd == MosaicGsd.mean:
            gsd = np.mean( gsds )
        elif gsd == MosaicGsd.median:
            gsd = np.median( gsds )
        elif gsd == MosaicGsd.max:
            gsd = np.max( gsds )
        else:
            raise Exception( "MosaicGsd enumerator not implemented: {}".format(gsd) )
    elif not gsd:
        gsd = np.median( gsds )
    # TODO: provide option to limit the spatial extents of the mosaic. Necessary for very oblique aerials whose footprint extends far out to the horizon.
    cornersOnGround = np.array( cornersOnGround )
    bboxOnGround = np.r_[ cornersOnGround.min(axis=0), cornersOnGround.max(axis=0) ]
    bboxOnGround[:2] -= bboxOnGround[:2] % gsd
    bboxOnGroundResol = np.ceil( ( bboxOnGround[2:] - bboxOnGround[:2] ) / gsd ).astype(int)
    bboxOnGround[2:] = bboxOnGround[:2] + gsd * bboxOnGroundResol

    logger.info( "Creating mosaic, corresponding mask, and overlap images: '{}', '{}', '{}', resolution: {}x{}, GSD: {:.3f}, {} in memory",
                 fnMosaic.name, fnMosaicMask.name, fnMosaicOverlap.name, bboxOnGroundResol[0], bboxOnGroundResol[1], gsd, utils.formatBytes( np.prod([bboxOnGroundResol[1],bboxOnGroundResol[0],3]) ) )
    mosaic = np.zeros( (bboxOnGroundResol[1],bboxOnGroundResol[0],3), np.uint8 )
    # we use OpenCV to reference np.ndarray on the C++ - side. OpenCV does not support uint32. So let's use int32. Besides, that allows us to have a nice NODATA value.
    mosaicMask = np.full( mosaic.shape[:2], -1, np.int32 ) # -1 is NODATA. DB is supposed to contain non-negative image IDs only.
    mosaicOverlap = np.zeros_like(mosaicMask)
    #mosaicVectorMask = ogr.Geometry(ogr.wkbMultiPolygon)
    progress = Progress(len(images))
    with dbapi2.connect( utils.db.uri4sqlite(absOriDbFn) + '?mode=rw', uri=True, isolation_level=None ) as absOriDb:
        absOriDb.execute("DELETE FROM homographies")
        # There seems to be no way to enclose the creation of a table and the insertion of data in a single transaction, such that both will be rolled back in case of an exception.
        # Setting isolation_level to None doesn't help either. http://stackoverflow.com/questions/15856976/transactions-with-python-sqlite3
        utils.db.initDataBase(absOriDb)
        for image in images:

            R = ori.omfika(image.omfika)

            flyingHeight = image.prc[2] - terrainHeight

            K = ori.cameraMatrix( image.camera.ior )
        
            # Achtung: OpenCV-Bild-KS ist rel. zu ORIENT-Bild-KS um x-Achse um 180° verdreht!
            #cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst
            # http://people.scs.carleton.ca/~c_shu/Courses/comp4900d/notes/homography.pdf
            # Bildausschnitt! -> linkes K so wählen, dass Bildecken des Originalphos auch im entzerrten pho vorhanden sind.
        
            # rotation about the x-axis by 200 gon
            Rx200 = np.diag([1.,-1.,-1.])
            Rocv = Rx200.dot( R ).dot( Rx200 )

            pho,infoIn = utils.gdal.imread( image.path, bands=utils.gdal.Bands.rgb, info=True )
            if image.nRows != pho.shape[0] or \
               image.nCols != pho.shape[1]:
                logger.warning( "Image resolution stored in data base does not correspond with actual image resolution - file has been changed {}: ({}x{}) vs. ({}x{})", 
                                image.path, image.nCols, image.nRows, pho.shape[1], pho.shape[0] )

            corners = getCorners( pho.shape[0], pho.shape[1] )
            corners = image.pix2cam.forward(corners)
            ori.distortion_inplace( corners, image.camera.ior, ori.adpParam2Struct( image.camera.adp ), ori.DistortionCorrection.undist )
            cornersOcv_h = np.c_[ corners*(1,-1), np.ones(corners.shape[0]) ]
            
            Kinv = linalg.inv(K)
            cornersOcvNorm_h = Kinv.dot( cornersOcv_h.T ).T # homogeneous, normalized image coordinates: referring to the principal point, for a focal length of 1.
            cornersOcvVertNorm_h = Rocv.dot( cornersOcvNorm_h.T ).T
            cornersOcvVertNorm = ( cornersOcvVertNorm_h[:,:2].T / cornersOcvVertNorm_h[:,2] ).T
        
            luVertNorm = cornersOcvVertNorm.min( axis=0 ) # normalized image coordinates of the upper/left corner of the rotated image (defined by the bounding window of the rotated corners of the original image)
            rlVertNorm = cornersOcvVertNorm.max( axis=0 )
            f2 = flyingHeight / gsd # wanted focal length, or the ratio of the distance between the rotation point and the image plane to 1 pixel.
            luVert = luVertNorm * f2 # image coordinates of the upper/left corner of the rotated image, scaled to the wanted focal length. Still referring to the principal point (=half-normalized?)
            rlVert = rlVertNorm * f2
            
            # shift the left,upper corner of the vertical image within its image plane, such that pixels result as aligned with the mosaic pixels.
            luBboxInVert = ( bboxOnGround[[0,3]] - image.prc[:2] )*(1,-1) / gsd # invert y-axis to transform from object coordinate system to OpenCV image system
            # make the difference vector from the upper,left corner of the mosaic to the resp. corner of the vertical image a multiple of the resolution (which is 1px -> make the coordinate differences whole numbers)
            luVert -= ( luVert - luBboxInVert ) % 1. # modulo 1: get the decimals

            if 0:
                ray = np.array([ luVert[0], -luVert[1], -f2 ])
                luVertGround = image.prc - flyingHeight / ray[2] * ray
                ( luVertGround[:2] - bboxOnGround[[0,3]] )*[1.,-1.] / gsd # should be integer numbers now!


            newSize = ( int( np.ceil( rlVert[0]-luVert[0] ) ),  # width of the vertical image
                        int( np.ceil( rlVert[1]-luVert[1] ) ) )
            K2 = np.array( [ [ f2,  0, -luVert[0] ], # camera matrix of the vertical image, un-normalizes homogeneous, normalized image coordinates (which refer to the upper/left image corner).
                             [  0, f2, -luVert[1] ],
                             [  0,  0,          1 ] ] )
        
            H = K2.dot( Rocv ).dot( Kinv ) # homography to transform homogeneous image coordinates in the oblique photo to image coordinates in the vertical photo.
        
            # direkte Umbildung: verwendet H direkt!
            cornersOcvVert = H.dot( cornersOcv_h.T ).T
            cornersOcvVert = ( cornersOcvVert[:,:2].T / cornersOcvVert[:,2] ).T

            considerAdp = True
            if 0:        
                considerAdp = False
                # indirekte Umbildung: verwendet (intern) inv(H) -> cv2.WARP_INVERSE_MAP nicht angeben!
                pho_rect = cv2.warpPerspective( src=pho, 
                                                M=H,
                                                dsize=newSize, 
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT )
            elif resample == MosaicResample.direct:
                # consider ADP. To do so, use cv2.remap (and cv2.convertMaps) instead of cv2.warpPerspective
                # We simply undistort every target grid position. For high resolution grids, it would make sense to distort only at a coarse resolution and interpolate the distortions then.
                # Unfortunately, cv2.remap does not support that: Destination image. the destination image has the same size as map1.
                # So we need to save RAM. Avoid temporaries, use in-place operations!
                Hinv = linalg.inv(H).astype(np.float32)

                #gridVert = np.mgrid[ :newSize[1], :newSize[0] ]
                #gridVert = np.c_[ gridVert[1].flat, gridVert[0].flat, np.ones(np.prod(gridVert.shape[1:])) ]
                #gridObl = Hinv.dot( gridVert.T ).T
                #gridOri = ( ( np.c_[ gridObl[:,0], -gridObl[:,1] ] ).T / gridObl[:,2] ).T
                #ori.distortion_inplace( gridOri, image.camera.ior, ori.adpParam2Struct( image.camera.adp ), ori.DistortionCorrection.dist )
                #gridOri = image.pix2cam.inverse(gridOri)
                #gridOri[:,1] *= -1.
                #mapx = gridOri[:,0].reshape( newSize[::-1] ).astype(np.float32)
                #mapy = gridOri[:,1].reshape( newSize[::-1] ).astype(np.float32)

                # Note: newSize[0] =^= nCols
                # grid in the (vertical) mosaic:
                # col/row-indices for each pixel, in row-major order.
                grid = np.empty( ( np.prod(newSize), 3 ), dtype=np.float32 )
                #grid[:,0] = np.tile( np.arange( newSize[0] ), newSize[1] )
                #grid[:,1] = np.repeat( np.arange( newSize[1] ), newSize[0] )
                oneRow = np.arange( newSize[0], dtype=np.float32 )
                for row in range(newSize[1]):
                    sel = np.s_[ row*newSize[0] : (row+1)*newSize[0] ]
                    grid[ sel, 0 ] = oneRow
                    grid[ sel, 1 ] = row
                grid[:,2] = 1

                # grid in the oblique, undistorted aerial camera CS
                grid = Hinv.dot( grid.T ).T
                # In-place dot products are unsupported: https://github.com/numpy/numpy/issues/610
                #np.dot( grid, Hinv.T, out=grid )
                #grid = ( ( np.c_[ grid[:,0], -grid[:,1] ] ).T / grid[:,2] ).T
                grid[:,0] /= grid[:,2]
                grid[:,1] *= -1
                grid[:,1] /= grid[:,2]
                grid = grid[:,:2]

                # grid in the oblique, distorted aerial camera CS
                ori.distortion_inplace( grid, image.camera.ior, ori.adpParam2Struct( image.camera.adp ), ori.DistortionCorrection.dist )

                # grid in the oblique, distorted aerial pixel CS
                grid = image.pix2cam.inverse(grid)

                # grid in the oblique, distorted aerial pixel CS, cv img coords
                grid[:,1] *= -1.

                mapx = grid[:,0].reshape( newSize[::-1] )
                mapy = grid[:,1].reshape( newSize[::-1] )
                del grid
                mapx, mapy = cv2.convertMaps( mapx, mapy, cv2.CV_16SC2 )
                if 1:
                    # simply use bilinear interpolation via inverse mapping from the vertical to the original (un-resized, unblurred) oblique
                    # this is the simplest solution, and it still seems to be the best one, as the results are sharpest. However, there may be aliasing artefacts, e.g. on grass.
                    pho_rect = cv2.remap( pho, mapx, mapy, cv2.INTER_LINEAR )

                # TODO: downsample the image before warping it.
                # Both cv2.warpPerspective and cv2.remap seem to not consider differences in resolution between source and destination images,
                # but they always consider the same neighborhood in the source image (3x3/4x4/8x8, depending on the interpolation method)
                # Hence, when downsampling an image during warping/remapping, the output image looks as if nearest-neighbor interpolation had been used.
                # Having downsampled the source image, ior and adp normalization radius need to be scaled accordingly.
                # However, downsampling the source image beforehand will not create optimal results either, because a single relative image scale must be chosen for the whole image,
                # while the relative image scales of source and destination images vary with image position.
                # Maybe GDAL can handle this? GDAL's warping API makes it easy to warp from one PCS to another PCS. However, one may supply a custom transformation function.
                # With GDAL, it is possible (actually, the default) to warp images without loading them all at once into memory.
                else:
                    # For a thoroughly antialiased perspective transform, we would need to apply a spatially variant filter, which is costly, see 2003 Khoshelham,Azizi - Kaiser filter for antialiasing in digital photogrammetry - PhoRec
                    # It seems that the better approach would be to locally approximate the perspective transform by affine transforms.
                    # The ideal filter for affine transforms is spatially invariant
                    # -> divide the image into cells, and process each cell with an appropriate filter. https://www.cs.cmu.edu/~ph/texfund/texfund.pdf
                    # E.g. choose the cell size such that the max. difference between perspective trafo and its local affine approximation is < 0.01px.
                    # TODO: use H and cv2.remap to map rectified's image scale onto the aerial's image space. smooth the aerial accordingly (using ROI) before actually mapping the aerial onto rectified.
                    # the scale is: gridObl[:,2]
                    gridScaleObl = np.mgrid[ :pho.shape[0], :pho.shape[1] ]
                    gridScaleObl = np.c_[ gridScaleObl[1].flat, gridScaleObl[0].flat, np.ones(np.prod(gridScaleObl.shape[1:])) ]
                    gridScaleVert = H.dot( gridScaleObl.T ).T
                    # let's forget about distortion here.
                    gridScaleObl2Vert = gridScaleVert[:,2].reshape(pho.shape[:2]) * image.camera.ior[2] / f2
                    phoBlurred = pho.copy()
                    #srcSigma = .5
                    #tgtSigma = .5
                    scaleStep = .1
                    for scale in np.arange( max( 1., gridScaleObl2Vert.min() ) + scaleStep/2, gridScaleObl2Vert.max() - scaleStep/2 + .001, scaleStep ):
                        sel = np.logical_and( gridScaleObl2Vert >= scale - scaleStep/2, gridScaleObl2Vert < scale + scaleStep/2 )
                        if not sel.any():
                            continue
                        #sigma = ( -8*scale**2*np.log(0.9) )**.5
                        #sigma = ( tgtSigma**2 * scale**2 - srcSigma**2 )**.5
                        sigma = 2 * scale / 6
                        phoBlurred[sel,:] = cv2.GaussianBlur( pho, ksize=(0,0), sigmaX=sigma )[sel,:]
                    pho_rect = cv2.remap( phoBlurred, mapx, mapy, cv2.INTER_LINEAR )
                del mapx, mapy

            else:
                assert resample == MosaicResample.preShrink
                # downscale the oblique as much as possible, such that no information is lost: the image scale in 1 corner of the downscaled oblique matches the image scale of the vertical, and in all other corners, the downscaled oblique is (a little) further downscaled during subsequent re-mapping
                # then remap
                # produces appealing rectifications
                
                # Make sure that H.dot(pho) and HDown.dot(phoDown) match exactly (mind that, of course, cv2.resize(.) saturate casts input.width*scaleX to an integer)!
                # Thus, do not recompute the observation rays of the corners of the resized pho, etc., but simply re-scale the oblique camera matrix

                downScale = f2 / cornersOcvVertNorm_h[:,2].min() / image.camera.ior[2]
                if downScale < 1:
                    phoDown = cv2.resize( src=pho, dsize=(0,0), fx=downScale, fy=downScale, interpolation=cv2.INTER_AREA )
          
                    iorDown = image.camera.ior * downScale
                    adpDown = adjust.parameters.ADP( normalizationRadius = image.camera.adp.normalizationRadius * downScale,
                                                     referencePoint  = image.camera.adp.referencePoint,
                                                     array           = image.camera.adp * downScale )
                    KDown = ori.cameraMatrix( iorDown )
                    KDownInv = linalg.inv(KDown)
                    if 0:
                        cornersOcv_hDown = cornersOcv_h.copy()
                        cornersOcv_hDown[:,:2] *= downScale

                        np.testing.assert_allclose( cornersOcvNorm_h, KDownInv.dot( cornersOcv_hDown.T ).T )

                    HDown = K2.dot( Rocv ).dot( KDownInv )
                    HDowninv = linalg.inv(HDown)
                else:
                    phoDown = pho
                    iorDown = image.camera.ior
                    adpDown = image.camera.adp
                    HDowninv = linalg.inv(H)

                gridVert = np.mgrid[ :newSize[1], :newSize[0] ]
                gridVert = np.c_[ gridVert[1].flat, gridVert[0].flat, np.ones(np.prod(gridVert.shape[1:])) ]
                gridObl = HDowninv.dot( gridVert.T ).T
                gridOri = ( ( np.c_[ gridObl[:,0], -gridObl[:,1] ] ).T / gridObl[:,2] ).T
                ori.distortion_inplace( gridOri, iorDown, ori.adpParam2Struct( adpDown ), ori.DistortionCorrection.dist )
                gridOri[:,1] *= -1.
                mapx = gridOri[:,0].reshape( newSize[::-1] ).astype(np.float32)
                mapy = gridOri[:,1].reshape( newSize[::-1] ).astype(np.float32)
                mapx, mapy = cv2.convertMaps( mapx, mapy, cv2.CV_16SC2 )
                pho_rect = cv2.remap( phoDown, mapx, mapy, cv2.INTER_AREA )
                if 1: # unsharp
                    amount = .5
                    if 0:
                        pho_rectBlurred = cv2.GaussianBlur( pho_rect, ksize=(0,0), sigmaX=3 )
                        pho_rect = cv2.addWeighted( pho_rect, 1+amount, pho_rectBlurred, -amount, 0 )
                    else:
                        # avoid introducing noise into low-contrast regions by excluding them from sharpening
                        pho_rectBlurred = cv2.GaussianBlur( pho_rect, ksize=(0,0), sigmaX=3 )
                        if pho_rect.squeeze().ndim==2:
                            pho_rectBlurredGray = pho_rectBlurred
                            pho_rectGray        = pho_rect
                        else:
                            pho_rectBlurredGray = cv2.cvtColor( pho_rectBlurred, cv2.COLOR_BGR2GRAY )
                            pho_rectGray        = cv2.cvtColor( pho_rect, cv2.COLOR_BGR2GRAY )
                        # differences may be negative, so cast to a signed type!
                        lowContrastMask = np.abs(pho_rectGray.astype(np.int16) - pho_rectBlurredGray.astype(np.int16)) < 5 # threshold
                        pho_rectSharpened = cv2.addWeighted( pho_rect, 1+amount, pho_rectBlurred, -amount, 0 )
                        pho_rectSharpened[lowContrastMask] = pho_rect[lowContrastMask]
                        pho_rect = pho_rectSharpened
                        # note: unsharpening has negative effects at the borders of the original image

            #else:
            #    # consider ADP
            #    # To avoid the nearest-neighbor-interpolation-effect:
            #    # - re-map the image to a resolution that guarantees that no down-sampling happens anywhere during re-mapping. Up-sampling is no problem, since we don't use nearest-neighbor-interpolation!
            #    # - then down-sample the re-mapped image
            #    # the following produces mathematically correct results, but is too memory intensive: the intermediate image may not fit into memory
            #    upScale = 1. / cornersOcvVertNorm_h[:,2].max()
            #    iorUpscaled = image.camera.ior * upScale
            #    adpUpscaled = adjust.parameters.ADP( normalizationRadius = image.camera.adp.normalizationRadius * upScale,
            #                                         referencePoint      = image.camera.adp.referencePoint,
            #                                         array               = image.camera.adp )
            #    f2Upscaled = iorUpscaled[2]
            #    luVertUpscaled = luVertNorm * f2Upscaled # normalized image coordinates of the upper/left corner of the rotated image, scaled to the wanted focal length. Still referring to the principal point.
            #    rlVertUpscaled = rlVertNorm * f2Upscaled
            #    newSizeUpscaled = ( int( np.ceil( rlVertUpscaled[0]-luVertUpscaled[0] ) ),  # width of the vertical image
            #                        int( np.ceil( rlVertUpscaled[1]-luVertUpscaled[1] ) ) )
            #    K2Upscaled = np.array( [ [ f2Upscaled,          0, -luVertUpscaled[0] ], # camera matrix of the vertical image, un-normalizes homogeneous, normalized image coordinates (which refer to the upper/left image corner).
            #                             [          0, f2Upscaled, -luVertUpscaled[1] ],
            #                             [          0,          0,                  1 ] ] )
            #
            #    HUpscaled = K2Upscaled.dot( Rocv ).dot( Kinv ) # homography to transform homogeneous image coordinates in the obique photo to image coordinates in the vertical photo.
            #
            #    HUpscaledinv = linalg.inv(HUpscaled)
            #    gridV = np.mgrid[ :newSizeUpscaled[1], :newSizeUpscaled[0] ]
            #    gridV = np.c_[ gridV[1].flat, gridV[0].flat, np.ones(np.prod(gridV.shape[1:])) ]
            #    gridV = HUpscaledinv.dot( gridV.T ).T
            #    gridV = ( ( np.c_[ gridV[:,0], -gridV[:,1] ] ).T / gridV[:,2] ).T
            #    ori.distortion_inplace( gridV, iorUpscaled, ori.adpParam2Struct( adpUpscaled ), ori.DistortionCorrection.dist )
            #    gridV[:,1] *= -1.
            #    mapx = gridV[:,0].reshape( newSizeUpscaled[::-1] ).astype(np.float32)
            #    mapy = gridV[:,1].reshape( newSizeUpscaled[::-1] ).astype(np.float32)
            #    mapx, mapy = cv2.convertMaps( mapx, mapy, cv2.CV_16SC2 )
            #    phoUpscaled_rect = cv2.remap( pho, mapx, mapy, cv2.INTER_LINEAR )    
            #    downScale =   f2 / f2Upscaled
            #    pho_rect = cv2.resize( src=phoUpscaled_rect, dsize=(0,0), fx=downScale, fy=downScale, interpolation=cv2.INTER_AREA )       
                      
            mask = np.zeros( pho_rect.shape[:2], np.uint8 )
            if considerAdp:
                if image.mask_px is None:
                    border = np.r_[
                        np.c_[ np.zeros(pho.shape[0]-1)                 , 0 : -pho.shape[0]+1 : -1 ],
                        np.c_[         : pho.shape[1]-1                 , np.full(pho.shape[1]-1, -pho.shape[0]+1. ) ],
                        np.c_[ np.full(pho.shape[0]-1, pho.shape[1]-1. ), -pho.shape[0]+1 : 0 ],
                        np.c_[ pho.shape[1]-1 : 0 : -1                  , np.zeros(pho.shape[1]-1) ]
                    ]
                else:
                    # if this is a scanned aerial, then use it's mask, as created/used for feature extraction.
                    border = []
                    segStart = image.mask_px[0]
                    for iEnd in range(1,len(image.mask_px)):
                        segEnd = image.mask_px[iEnd]
                        direc = segEnd - segStart
                        direcLen = linalg.norm(direc)
                        if not direcLen:
                            continue # duplicate point?
                        direc /= direcLen
                        for station in range( int(direcLen) ):
                            border.append( segStart + direc*station )
                            # end point will be appended as start point of the next segment
                        segStart = segEnd
                    border = np.array(border)

                border = image.pix2cam.forward(border)
                ori.distortion_inplace( border, image.camera.ior, ori.adpParam2Struct( image.camera.adp ), ori.DistortionCorrection.undist )
                border = np.c_[ border[:,0], -border[:,1], np.ones(border.shape[0]) ]
                border = H.dot( border.T ).T
                border = ( border[:,:2].T / border[:,2] ).T
                cornersOcvVert = border # due to casting to int, there may result consecutive duplicates. Is that a problem for OpenCV?
                # TODO: considering ADP for cornersOcvVert may shift the image corners outwards, and the mask may end up partially outside the mosaic!

            # If considerAdp, then cornersOcvVert may not be convex!
            mask = cv2.fillConvexPoly( mask, cornersOcvVert.astype(np.int), (255,255,255) ) 
            mask = cv2.erode( mask, np.ones((3,3)) )

            ray = np.array([ luVert[0], -luVert[1], -f2 ])
            luVertGround = image.prc - flyingHeight / ray[2] * ray
            warped2mosaic_rc = np.round( ( luVertGround[:2] - bboxOnGround[[0,3]] )*[1.,-1.] / gsd ).astype(np.int)[::-1]

            # TODO: intersect with image shapes. Otherwise, arrays on left and right sides of assignment may have different sizes!
            sel = np.s_[ warped2mosaic_rc[0] : warped2mosaic_rc[0] + pho_rect.shape[0],
                         warped2mosaic_rc[1] : warped2mosaic_rc[1] + pho_rect.shape[1] ]
            mosaic[ sel + (np.s_[:],) ][mask>0] = pho_rect[mask>0]

            mosaicMask[ sel ][mask>0] = image.id
            mosaicOverlap[ sel ][mask>0] += 1

            Hori = H.copy()
            Hori[:,1] *= -1
            Hori[1,:] *= -1
            Hori[0,:] += Hori[2,:] *  warped2mosaic_rc[1]
            Hori[1,:] += Hori[2,:] * -warped2mosaic_rc[0]
            absOriDb.execute("""
                INSERT INTO
                homographies
                VALUES( {} )""".format( ','.join( '?'*12 ) ) ,
                ( None, int(image.id), mosaicId, *Hori.flat ) )


            #ring = ogr.Geometry(ogr.wkbLinearRing)
            #for cornerOcvVert in cornersOcvVert:
            #    ring.AddPoint( cornerOcvVert[0], -cornerOcvVert[1] )
            #ring.FlattenTo2D() # even though we've constructed a 2D object, and call AddPoint(.) with only 2 arguments, AddPoint(.) makes ring a 2.5D object!
            #ring.CloseRings()
            #vectorMask = ogr.Geometry(ogr.wkbPolygon)
            #vectorMask.AddGeometry(ring)
            #mosaicVectorMask.AddGeometry( vectorMask )

            if outDir is not None:
                ray = np.array([ luVert[0], -luVert[1], -f2 ])
                # GDAL wants the upper/left corner, not the centre of the upper/left pixel
                ray[0] += .5
                ray[1] -= .5
                luVertGround = image.prc - flyingHeight / ray[2] * ray

                infoOut = utils.gdal.ImageInfoWrite()
                infoOut.geotransform = np.array([[ luVertGround[0], gsd,   0. ],
                                                 [ luVertGround[1],  0., -gsd ]])
                infoOut.projection = projectionWkt
                infoOut.description = "{}, orthorectified with respect to the ground plane of the coordinate system: {}.".format(image.path, projectionWkt or '<local CS>')
                # GDAL's GeoTIFF tags are limited in size. Better not store those thousands of vertices, or GDAL issues a warning about lost metadata during writing.
                #infoOut.vectorMask = vectorMask.ExportToWkt()

                utils.gdal.imwrite( str( outDir / Path(image.path).with_suffix('.tif').name ), pho_rect, mask, info=infoOut )

            progress += 1

    invPermutation = np.argsort(permutation) # store the file paths in correct order, matching the sequence of imgIds!
    infoOut = utils.gdal.ImageInfoWrite()
    infoOut.geotransform = np.array([[ bboxOnGround[0]-gsd/2, gsd,   0. ],
                                     [ bboxOnGround[3]+gsd/2,  0., -gsd ]])
    infoOut.projection = projectionWkt
    infoOut.description = "Mosaic of photos orthorectified with respect to the ground plane of the coordinate system: {}. " \
                          "Based on following photos: {}".format(projectionWkt or '<local CS>', ', '.join(( Path(el).name for el in [ images[idx].path for idx in invPermutation ] )) )
    #infoOut.vectorMask = mosaicVectorMask.UnionCascaded().ExportToWkt()
    utils.gdal.imwrite( str(fnMosaic), mosaic, ( mosaicMask >= 0 ).astype(np.uint8)*255, info=infoOut )

    infoOut.description = "Contains at each pixel the image ID that the content of the accompanying mosaic originates from. " \
                          "Image IDs are: {}".format( ', '.join(( '{}:{}'.format(image.id,Path(image.path).name)
                                                                  for image in [ images[idx] for idx in invPermutation ] )) )
    infoOut.vectorMask = ''
    # A TIFF bit mask for invalid areas would probably be best (TIFF bit masks are even compressed!), but QGIS still doesn't support them.
    # Let's set the dataset NODATA-vector instead.
    utils.gdal.imwrite( str(fnMosaicMask), mosaicMask, info=infoOut )
    utils.gdal.setDataSetNoData( str(fnMosaicMask), np.array([-1]) )

    # If possible, control points should be digitized in MonoScope in areas of large overlap.
    # It would be nice to store the overlap count as a sub-dataset inside fnMosaic. Unfortunately, GDAL does not support the creation of arbitrary sub-datasets (but only overviews and masks, using dedicated functions).
    infoOut.description = "Contains at each pixel the number of photos in which this pixel is visible. " \
                          "Based on following photos: {}".format( ', '.join(( Path(el).name for el in [ images[idx].path for idx in invPermutation ] )) )
    utils.gdal.imwrite( str(fnMosaicOverlap), mosaicOverlap, info=infoOut )

@contract
def determineCorrespondences( cpsOrtho_wrl  : 'array[Nx2](float)',
                              cpsMosaic_wrl : 'array[Nx2](float)',
                              permute : bool,
                              unknownRot : bool,
                              unknownScale : bool,
                              robust : bool ):
    leastAvgResidualLen = np.inf
    
    if robust:
        linearAverage = np.median
        linear2DAverage = functools.partial( utils.stats.geometricMedian, atol=1.e-3 )
        circularAverage = utils.stats.circular_median
    else:
        linearAverage = np.mean
        linear2DAverage = functools.partial( np.mean, axis=0 )
        circularAverage = utils.stats.circular_mean

    ctrOrtho  = linear2DAverage( cpsOrtho_wrl )
    redOrtho = cpsOrtho_wrl - ctrOrtho

    # the permutation does not affect the geometric median, so compute beforehand.
    ctrMosaic = linear2DAverage( cpsMosaic_wrl )
    redMosaic = cpsMosaic_wrl - ctrMosaic

    if unknownRot:
        redOrthoAngles  = np.arctan2( redOrtho [:,1], redOrtho [:,0] )
        redMosaicAngles = np.arctan2( redMosaic[:,1], redMosaic[:,0] )
    if unknownScale:
        redOrthoLenSqr  = np.sum( redOrtho **2, axis=1 )
        redMosaicLenSqr = np.sum( redMosaic**2, axis=1 )

    results = []

    if permute:
        permutations = itertools.permutations( range(len(cpsMosaic_wrl)) )
    else:
        permutations = [ range(len(cpsMosaic_wrl)) ]
    for permutation in permutations:
        permutation = np.array(permutation)
        redMosaic_permuted = redMosaic[permutation]
        if unknownRot:
            redMosaicAngles_permuted = redMosaicAngles[permutation]
            # TODO even better would be a circular average that is weighted by the distance from the origin.
            # Currently, angles of points close to the reduction point have the same weight, even though their angles are less precise.
            angle = circularAverage( redOrthoAngles - redMosaicAngles_permuted )
        else:
            angle = 0.

        if unknownScale:
            redMosaicLenSqr_permuted = redMosaicLenSqr[permutation]
            scale = linearAverage( redOrthoLenSqr / redMosaicLenSqr_permuted )**.5
        else:
            scale = 1.

        # Compute the sum of residual norms. Choose the trafo with the smallest such sum.
        sAng = np.sin(angle)
        cAng = np.cos(angle)
        # Rotate the object in a mathematically positive sense.
        R = np.array([[ cAng, -sAng ],
                      [ sAng,  cAng ]])
        est_cpsOrtho_wrl = ctrOrtho + scale * R.dot( redMosaic_permuted.T ).T
        def getAffineTrafo():
            return ori.transform.AffineTransform2D( A = scale * R,
                                                    t = -scale * R.dot( ctrMosaic ) + ctrOrtho )
        if False:
            cpsMosaic_wrl_permuted = cpsMosaic_wrl[permutation]
            est_cpsOrtho_wrl2 = getAffineTrafo().forward( cpsMosaic_wrl_permuted )

        avgResidualLen = linearAverage( np.sum( ( cpsOrtho_wrl - est_cpsOrtho_wrl )**2, axis=1 )**.5 )
        #logger.verbose( 'Permutation: {} mean residual norm: {} [obj units] scale: {} angle: {}gon', permutation, avgResidualLen, scale, angle*200/np.pi )
        results.append(( permutation, avgResidualLen, scale, angle*200/np.pi ))
        if avgResidualLen < leastAvgResidualLen:
            leastAvgResidualLen = avgResidualLen
            bestResult = ( permutation, getAffineTrafo() )

    logger.verbose( 'Analyzed correspondences\n'
                    'Permutation\tMean residual norm [obj units]\tScale\tAngle [gon]\n'
                    '{}',
                    '\n'.join( '\t'.join( '{}'.format(el) for el in result )
                                for result in results ) )
    return bestResult


@contract
def restoreRelOriBlock( relOriDbFn : Path,
                        getImgObsLoss,
                        getImgObsData = lambda row: ImgObsData( row['id'], row['imgId'], row['objPtId'] ),
                        cbImagesLoaded = None ):
    opts = adjust.Problem.Options()
    opts.enable_fast_removal = True
    block = adjust.Problem( opts )
    solveOpts = adjust.Solver.Options()
    solveOpts.linear_solver_ordering = adjust.ParameterBlockOrdering()    

    cameras = {}
    images = {}
    objPts = {}
    with dbapi2.connect( utils.db.uri4sqlite(relOriDbFn) + '?mode=rw', uri=True ) as relOriDb:
        utils.db.initDataBase(relOriDb)

        for row in relOriDb.execute('''
            SELECT * 
            FROM cameras '''):
            v_adp = []
            s_adp = []
            for val, enumerator in sorted( adjust.PhotoDistortion.values.items(), key=lambda x: x[0] ):
                v_adp.append( row[str(enumerator)] )
                s_adp.append( row['s_{}'.format(str(enumerator))] )
            adp = adjust.parameters.ADP( normalizationRadius=row['normalizationRadius'] or 1.,
                                         referencePoint     = adjust.AdpReferencePoint.names[ row['reference'] ],
                                         array              = np.array(v_adp) )
            cameras[row['id']] = Camera( row['id'],
                                         np.array([ row['x0']  , row['y0']  , row['z0']  ]),
                                         np.array([ row['s_x0'], row['s_y0'], row['s_z0']], float), # specify the type to avoid NULL being stored as None in an array of type object, such that: NULL -> None -> NaN
                                         adp,
                                         np.array(s_adp, float) )

        fiducialMatAttrs = [ 'fiducial_{}'.format(el) for el in 'A00 A01 A10 A11'.split() ]
        fiducialVecAttrs = [ 'fiducial_{}'.format(el) for el in 't0 t1'          .split() ]

        names = utils.db.columnNamesForTable( relOriDb, 'images')
        names[names.index('mask')] = 'AsBinary(mask) AS mask'
        for row in relOriDb.execute('''
            SELECT {}
            FROM images '''.format( ','.join(names) ) ):
            path = Path( row['path'] )
            if not path.is_absolute():
                path = Path( os.path.normpath( str(relOriDbFn.parent / path) ) )
            rot = adjust.parameters.EulerAngles( parametrization=adjust.EulerAngles.names[row['parameterization']],
                                                 array=np.array( [row['r1'], row['r2'], row['r3']], float ) )
            A = np.array([row[el] for el in fiducialMatAttrs], float).reshape((2,2))
            t = np.array([row[el] for el in fiducialVecAttrs], float)
            if np.isfinite(A).all() and np.isfinite(t).all():
                pix2cam = ori.transform.AffineTransform2D( A, t )
            else:
                pix2cam = ori.transform.IdentityTransform2D()
            mask_px = None
            if row['mask'] is not None:
                polyg = ogr.Geometry( wkb=row['mask'] )
                ring = polyg.GetGeometryRef(0)
                nPts = ring.GetPointCount()
                mask_px = np.empty( (nPts,2) )
                for iPt in range(nPts):
                    mask_px[iPt,:] = ring.GetPoint_2D(iPt)

            # for digital images, we weight observations with a std.dev. of 1px
            # for scanned images, we scale those 1px with the scanning resolution i.e. we assume that anaogue images are scanned with a meaningful resolution.
            obsWeights = np.eye(2)
            if not isinstance( pix2cam, ori.transform.IdentityTransform2D ):
                obsWeights /= pix2cam.meanScaleForward()
                
            images[row['id']] = Image( row['id'],
                                       str(path),
                                       np.array([ row['X0'], row['Y0'], row['Z0'] ]),
                                       rot,
                                       row['nRows'],
                                       row['nCols'],
                                       pix2cam,
                                       mask_px,
                                       obsWeights,
                                       cameras[row['camId']] )

        if cbImagesLoaded is not None:
            cameras, images = cbImagesLoaded( cameras, images )

        for row in relOriDb.execute("""
            SELECT objpts.id as id, X(pt) as X, Y(pt) as Y, Z(pt) as Z, avg(red) as red, avg(green) as green, avg(blue) as blue, count(*) as nImgs
            FROM objpts
            JOIN imgobs ON objpts.id == imgobs.objPtID
            GROUP BY objpts.id """ ):
            objPts[row['id']] = ObjPt( np.array([ row['X'], row['Y'], row['Z'] ]),
                                       np.array([ row['red'], row['green'], row['blue'] ]),
                                       row['nImgs'] )

        # ORDER BY image id.
        # Hence, when evaluating the block, sequences of observations that share the same ROT occur.
        # Hence, Photo may take advantage of that and cache and re-use rotation matrices already computed before.
        for row in relOriDb.execute("""
            SELECT *
            FROM imgObs
            ORDER BY imgId """ ):
            image = images[row['imgId']]
            imgCoos = np.array([ row['x'], row['y'] ], float)
            imgCoos = image.pix2cam.forward( imgCoos )
            cost = adjust.cost.PhotoTorlegard( imgCoos[0], imgCoos[1], image.obsWeights )
            cost.data = getImgObsData(row)
            block.AddResidualBlock( cost,
                                    getImgObsLoss(row),
                                    image.prc,
                                    image.omfika,
                                    image.camera.ior,
                                    image.camera.adp,
                                    objPts[row['objPtId']].pt )

    for image in images.values():
        solveOpts.linear_solver_ordering.AddElementToGroup( image.prc, 1 )
        solveOpts.linear_solver_ordering.AddElementToGroup( image.omfika, 1 )
    for camera in cameras.values():
        solveOpts.linear_solver_ordering.AddElementToGroup( camera.ior, 1 )
        solveOpts.linear_solver_ordering.AddElementToGroup( camera.adp, 1 )
        for param, s_param in ( (camera.ior,camera.s_ior), (camera.adp,camera.s_adp) ):
            assert len(param)==len(s_param)
            # 0.0 -> const
            # None -> variable; parameter value estimated, but not its standard deviation
            # > 0.0 -> variable
            iConstants = tuple( idx for idx, el in enumerate(s_param) if el==0. )
            if len(iConstants) == len(s_param):
                block.SetParameterBlockConstant( param )
            elif len(iConstants) > 0:
                locPar = adjust.local_parameterization.Subset( param.size, iConstants )
                block.SetParameterization( param, locPar )

    for objPt in objPts.values():
        solveOpts.linear_solver_ordering.AddElementToGroup( objPt.pt, 0 )

    if 0:
        summary = adjust.Solver.Summary()
        adjust.Solve(solveOpts, block, summary)
        assert adjust.isSuccess( summary.termination_type )
        assert len(summary.iterations) <= 1

    return block, solveOpts, cameras, images, objPts

@functools.singledispatch
def transformPoints( trafo, pts ):
    return np.array( trafo.TransformPoints( np.atleast_2d(pts).tolist() ) )

@transformPoints.register(ori.transform.ITransform2D)
def _( trafo, pts ):
    return trafo.forward( pts )

@contract
def transformEOR( srcCs_2_tgtCs : 'CoordinateTransformation|ITransform2D', images, objPts = None ) -> None:
    for image in images:
        # How to transform rotations? PROJ.4 only supports the transformation of points!
        R = ori.omfika(image.omfika)
        # Offset by only 1. [m?] may lead to inaccurate results?
        pts = np.r_[ '0,2',
                     image.prc,
                     image.prc + R[:,0], # R[:,0] is the direction of the camera's x-axis in object space, with the origin at the PRC.
                     image.prc + R[:,1],
                     image.prc + R[:,2] ]
        pts_tgtCs = transformPoints( srcCs_2_tgtCs, pts )
        image.prc[:] = pts_tgtCs[0,:]
        R_tgtCs = pts_tgtCs[1:,:] - pts_tgtCs[0,:]
        R_tgtCs = R_tgtCs.T # pts holds the columns of R as rows!
        for idx in range(3):
            R_tgtCs[idx,:] /= linalg.norm(R_tgtCs[idx,:])
        image.omfika[:] = ori.omfika( R_tgtCs )

    if objPts:
        for objPt in objPts:
            objPt.pt[:] = transformPoints( srcCs_2_tgtCs, objPt.pt )

@contract
def saveSQLite( absOriDbFn : Path, tgtCs : osr.SpatialReference, cameras : dict, images : dict, objPts : dict, stdDevs : dict, cpsMosaic_ori, cpsOrtho_ori, cpsAerials_ori, cpsOrtho_wrl ) -> None:
    logger.info( 'Saving results to "{}"', absOriDbFn.name )
    with dbapi2.connect( utils.db.uri4sqlite(absOriDbFn) + '?mode=rw', uri=True ) as absOriDb:
        # A table created using CREATE TABLE AS has no PRIMARY KEY and no constraints of any kind. The default value of each column is NULL.
        # Better use CREATE TABLE, followed by INSERT INTO!
        #absOriDb.execute("CREATE TABLE cameras AS SELECT * FROM relOri.cameras")
        utils.db.initDataBase(absOriDb)

        absOriDb.execute( f"""
            INSERT OR REPLACE INTO config ( name, value )
            VALUES ( '{utils.db.ConfigNames.CoordinateSystemWkt}', ? )""",
            ( tgtCs.ExportToWkt(), ) )

        pointZWkb = struct.Struct('<bIddd')
        def packPointZ(pt):
            # 1001 is the code for points with z-coordinate
            return pointZWkb.pack( 1, 1001, *pt )

        # AddGeometryColumn fails it the wanted column already exists as a regular SQLite-column and triggers a warning.
        # If the column exists already as a regular column (because absOri.sqlite is being re-used), then use RecoverGeometryColumn.
        # RecoverGeometryColumn requires that BLOBS already present in the existing column are valid geometries of the wanted type and SRID.
        # It seems that RecoverGeometryColumn chokes on NULL values, so set the pt-column already above.
        objPtsPtIsNew = 'pt' not in ( row['name'] for row in absOriDb.execute('PRAGMA table_info(objpts)') )
        absOriDb.execute("""
            SELECT {}(
                'objpts', -- table
                'pt',     -- column
                -1,       -- srid
                'POINT',  -- geom_type
                'XYZ'    -- dimension
                {}
                ) """.format( *( ('AddGeometryColumn','')#',1 -- NOT NULL') # Allow NULL values, so RecoverGeometryColumn will not choke on them if absOri.sqlite is re-used.
                                 if objPtsPtIsNew else
                                 ('RecoverGeometryColumn','') ) ) )

        absOriDb.executemany( """
            UPDATE objpts
            SET pt = GeomFromWKB(?, -1),
                s_X = ?,
                s_Y = ?,
                s_Z = ?
            WHERE id = ?""",
            ( ( packPointZ(objPt.pt), ) + tuple( stdDevs.get( objPt.pt.ctypes.data, (None,)*3 ) ) + ( iPt, )
            for iPt,objPt in objPts.items() ) )

        # There is still a flaw in our DB-layout: (lsm-) imgObs may reference (manual-) imgObs (in another image).
        # However, currently, it is possible that such imgObs reference different objPtId's!
        # As we don't use the following anywhere programmatically, let's deactivate it.
        assert len(np.unique([ cpsMosaic_ori.size,
                               cpsOrtho_ori.size,
                               #cpsAerials_ori.size,
                               cpsOrtho_wrl.size]))==1
        #for cpMosaic_ori, cpOrtho_ori, cpAerials_ori, cpOrtho_wrl in utils.zip_equal( cpsMosaic_ori, cpsOrtho_ori, cpsAerials_ori, cpsOrtho_wrl ):
        for cpMosaic_ori, cpOrtho_ori, cpOrtho_wrl in utils.zip_equal( cpsMosaic_ori, cpsOrtho_ori, cpsOrtho_wrl ):
            objPtId = absOriDb.execute("""
                INSERT INTO objpts (pt,s_X,s_Y,s_Z)
                VALUES( GeomFromWKB(?,-1), ?, ?, ? )""",
                ( ( packPointZ(cpOrtho_wrl['p']), ) + tuple( stdDevs.get( cpOrtho_wrl['p'].ctypes.data, (None,)*3 ) ) ) ).lastrowid
            for imgObs in ( cpMosaic_ori, cpOrtho_ori ): 
                absOriDb.execute("""
                    UPDATE imgobs
                    SET objPtId = ?
                    WHERE imgid = ? and name = ? """,
                    ( objPtId, int(imgObs['imgid']), imgObs['name'] ) )
            # We cannot simply use cpAerials_ori['name'] as name, because that may already be in use for this aerial.
            # But our trigger imgobs_default_name doesn't work!?
            #img = images[cpAerials_ori['imgid']]
            #imgObs = img.pix2cam.inverse( cpAerials_ori['p'] )
            #absOriDb.execute("""
            #    INSERT INTO imgobs (imgId,name,x,y,objPtId)
            #    VALUES( ?, ?, ?, ?, ? )""",
            #    ( int(cpAerials_ori['imgid']),
            #      'CP_{}'.format(cpAerials_ori['name']),
            #      float(imgObs[0]),
            #      float(imgObs[1]),
            #      objPtId ) )

        # TODO replace with https://www.gaia-gis.it/fossil/libspatialite/wiki?name=SpatialIndex
        absOriDb.execute(
            "SELECT {}( 'objpts', 'pt' )".format( 'CreateSpatialIndex' if objPtsPtIsNew else 'RecoverSpatialIndex' ) )

        absOriDb.executemany( """
            UPDATE images
            SET nCols = ?,
                nRows = ?,
                  X0 = ?,   Y0 = ?,   Z0 = ?,
                s_X0 = ?, s_Y0 = ?, s_Z0 = ?,
                  r1 = ?,   r2 = ?,   r3 = ?,
                s_r1 = ?, s_r2 = ?, s_r3 = ?
            WHERE id = ? """,
            ( ( img.nCols, img.nRows ) +
              tuple( img.prc ) +
              tuple( stdDevs.get( img.prc.ctypes.data, (None,)*3 ) ) +
              tuple( img.omfika ) +
              tuple( stdDevs.get( img.omfika.ctypes.data, (None,)*3 ) ) +
              ( img.id, )
              for img in images.values() ) )

        absOriDb.executemany( """
            UPDATE cameras
            SET   x0 = ?,   y0 = ?,   z0 = ?,
                s_x0 = ?, s_y0 = ?, s_z0 = ?,
                {}
            WHERE id = ? """.format(
            ','.join(   [        str(val[1]) + ' = ?' for val in sorted( adjust.PhotoDistortion.values.items(), key=lambda x: x[0] ) ]
                      + [ 's_' + str(val[1]) + ' = ?' for val in sorted( adjust.PhotoDistortion.values.items(), key=lambda x: x[0] ) ] )
            ),
            ( tuple( cam.ior ) +
              tuple( stdDevs.get( cam.ior.ctypes.data, (None,)*3 ) ) +
              tuple( cam.adp ) +
              tuple( stdDevs.get( cam.adp.ctypes.data, (None,)*9 ) ) +
              ( cam.id, )
              for cam in cameras.values() ) )

        absOriDb.execute("ANALYZE")

@contract
def prepareDb( absOriDbFn : Path, relOriDbFn : Path, fnMosaic : Path ) -> None:
    utils.db.createUpdateSchema( str(absOriDbFn) )
    with dbapi2.connect( utils.db.uri4sqlite(absOriDbFn) + '?mode=rwc', uri=True ) as absOriDb:
        utils.db.initDataBase( absOriDb )

        # don't drop imgobs or images, as they contain the manual measurements done in MonoScope!
        #absOriDb.execute("DELETE FROM imgobs WHERE type IS NOT {}".format(int(ObservationMode.manual)) ) # delete all non-manual image observations. Use IS NOT (instead of !=) to also select NULL entries.
        # Delete all non-manual, non-lsm image observations. Mind that users may have generated LSM-obs of GCPs in MonoScope.
        absOriDb.execute("DELETE FROM imgobs WHERE type NOT IN ({}) OR type ISNULL".format( ','.join([str(int(el)) for el in (ObservationMode.manual, ObservationMode.lsm) ])) )
        absOriDb.execute("UPDATE imgobs SET objPtId = NULL")
        #absOriDb.execute("DELETE FROM homographies WHERE NOT EXISTS ( SELECT * FROM imgobs WHERE homographies.id == imgobs.imgId )")
        absOriDb.commit() # By default, sqlite3 opens a transaction implicitly before a DML statement. Since Python 3.6, it does not implicitly commit open transactions before DDL statements any more.
        absOriDb.execute("PRAGMA foreign_keys = OFF") # This pragma is a no-op within a transaction!!
        # delete all perspective images with no observations left. Leave non-perspective images without obs, so mosaicId will remain unchanged, and eventually defined homographies will still refer to it.
        absOriDb.execute("DELETE FROM images WHERE camId NOTNULL AND NOT EXISTS ( SELECT * FROM imgobs WHERE images.id == imgobs.imgId )")
        absOriDb.commit()
        absOriDb.execute("PRAGMA foreign_keys = ON")
        assert absOriDb.execute("PRAGMA foreign_keys").fetchone()[0]
        absOriDb.execute("DELETE FROM cameras WHERE NOT EXISTS ( SELECT * FROM images WHERE cameras.id == images.camID )")
        if absOriDb.execute("""SELECT *
                               FROM geometry_columns
                               WHERE f_table_name == 'objpts' AND f_geometry_column == 'pt' """).fetchone():
            absOriDb.execute("SELECT DisableSpatialIndex( 'objpts', 'pt' )") # does this issue a warning, if no spatial index existed?
            absOriDb.execute("SELECT DiscardGeometryColumn( 'objpts', 'pt' )")
        absOriDb.execute("DELETE FROM objPts WHERE NOT EXISTS ( SELECT * FROM imgobs WHERE objPts.id == imgobs.objPtID )")
        if utils.db.tableHasColumn( absOriDb, 'objPts', 'pt' ):
            absOriDb.execute("UPDATE objPts SET pt = NULL") # otherwise, RecoverGeometryColumn will fail if we want to store points with a different SRID

        absOriDb.commit()
        absOriDb.execute("ATTACH DATABASE ? as relOri", [relOriDbFn.as_uri()] )
        absOriDb.execute("INSERT INTO main.objPts (id) SELECT id FROM relOri.objPts") # we need to insert objPts, because relOri.imgObs.objPtID is not NULL and it holds a foreign key on it. However, we do not want to re-use 'pt's object coordinate system. So just copy the ID.
        absOriDb.execute("INSERT OR REPLACE INTO main.cameras SELECT * FROM relOri.cameras")
        absOriDb.execute("INSERT OR REPLACE INTO main.images SELECT * FROM relOri.images")
        # Make paths relative to absOriDbFn.
        for row in absOriDb.execute("SELECT id,path FROM relOri.images").fetchall():
            path = Path(row['path'])
            if not path.is_absolute():
                path = relOriDbFn.parent / path
            path = str( utils.filePaths.relPathIfExists( path, start=absOriDbFn.parent ) ) # pathlib.Path's relative_to only supports files within subdirectories of its argument!!
            if path != row['path']:
                absOriDb.execute( "UPDATE main.images SET path = ? WHERE id = ? ", ( path, row['id'] ) )
        absOriDb.execute("INSERT INTO main.imgobs SELECT * FROM relOri.imgobs")

        fnRelMosaic = str( utils.filePaths.relPathIfExists( fnMosaic, absOriDbFn.parent ) )
        absOriDb.execute("INSERT OR IGNORE INTO main.images(path) VALUES(?)", [fnRelMosaic] ) # lastrowid seems to be undefined if row was already present
        mosaicId = absOriDb.execute("SELECT id FROM main.images WHERE path=?",[fnRelMosaic]).fetchone()[0]
    return mosaicId

@contract
def makeIorsAdpsVariableAtOnce( block : adjust.Problem,
                                solveOpts : adjust.Solver.Options,
                                cameras, images, objPts,
                                params = ( adjust.PhotoDistortion.optPolynomRadial3, 2, 0, adjust.PhotoDistortion.optPolynomRadial5 ) ):
    shortImgFns = utils.filePaths.ShortFileNames([ img.path for img in images.values() ])
    # extend the set of estimated ior/adp parameters
    # note: changes to the distortion model of aerials do not need to be propagated to the measurements in the mosaic.
    # However, when transforming between mosaic image CS and aerial image CS, the original homography and adp must be used!
    atOnce = True
    iorIds = list(cameras)
    iIorId = 0
    iLastSuccessParam=0
    iLastSuccessIorId=0
    iParam = 0
    while 1:
        if atOnce:
            msgs = makeIorAdpVariableAtOnce( block, cameras, images, objPts, shortImgFns, params[iParam] )
            if msgs:
                iLastSuccessParam = iParam
            else:
                iParam += 1
                if iParam == len(params):
                    iParam=0
                if iLastSuccessParam == iParam:
                    if len(iorIds) == 1:
                        break # There is only one camera anyway, so single-mode wouldn't change anything.
                    logger.verbose('Switching makeIorAdpVariableAtOnce to single-ior/adp mode')
                    atOnce = False
                    iParam = 0
                    iLastSuccessParam = 0
                continue
        else:
            msgs = makeIorAdpVariableAtOnce( block, cameras, images, objPts, shortImgFns, params[iParam], iorId=iorIds[iIorId] )
            if msgs:
                iLastSuccessParam=iParam
                iLastSuccessIorId=iIorId
            iIorId += 1
            if iIorId == len(iorIds):
                iIorId=0
                iParam += 1
                if iParam == len(params):
                    iParam=0
            if not msgs:
                if iLastSuccessParam==iParam and iLastSuccessIorId==iIorId:
                    break
                continue
        logger.info( '\n'.join(msgs) )
        logger.verbose("Full adjustment ...")
        summary = adjust.Solver.Summary()
        adjust.Solve(solveOpts, block, summary)
        logger.verbose("Full adjustment done.")
        
        if not adjust.isSuccess( summary.termination_type ):
            # this state could be handled by removing the added observations from the block again.
            logger.info( summary.FullReport() )
            raise Exception("adjustment failed after additional parameters have been introduced into the block")

@contract
def makeIorAdpVariableAtOnce( block : adjust.Problem,
                              cameras : dict,
                              images : dict,
                              objPts : dict,
                              shortImgFns : utils.filePaths.ShortFileNames,
                              param, # ior param index or adjust.PhotoDistortion; for ior, index 0 or 1 are both treated as 'principal point'
                              iorId : int = -1,
                              maxAbsCorr : float = 0.7
                            ) -> 'list(str)':
    # Try to set variable at once as many ior/adp parameters as possible and reasonable.
    # Need to pass all and only non-constant parameter blocks to adjust.sparseQxx

    # Introduce r3, z0, (x0,y0), r5 one after another, but for all cameras at the same time.
    # This may not yield the maximum set of variable ior/adp parameters. Thus, call makeIorAdpVariable after this function has 'converged'.
    msgs = []

    def isIor(par):
        return type(par) == int

    #paramName = str(param) if not isIor(param) else 'focal length' if param==2 else 'principal point'
    paramName = ( 'r3' if param==adjust.PhotoDistortion.optPolynomRadial3 else 'r5' ) if not isIor(param) else 'focal length' if param==2 else 'principal point'
    params = [0,1] if isIor(param) and param in (0,1) else [param]

    Intrinsic = namedtuple( 'Intrinsic', [ 'parBlock', 'wasConst', 'subset' ] )
    IntrinsicPair = namedtuple( 'IntrinsicPair', [ 'parBlock', 'wasConst', 'subset', 'anyParsSetFree', 'oParBlock', 'oIsConst', 'oSubset', 'imgIds' ] )

    def getIntrinsicPairs():
        intrinsicPairs = OrderedDict()
        for img in images.values():
            #if img.isCalibrated:
            #    continue # avoid zero-columns in jacobian!
            intrinsicPair = intrinsicPairs.get(img.camera.id)
            if intrinsicPair:
                intrinsicPair[-1].append( img.id )
                continue
            intrinsic = Intrinsic( img.camera.ior, block.IsParameterBlockConstant( img.camera.ior ), block.GetParameterization( img.camera.ior ) )
            oIntrinsic = Intrinsic( img.camera.adp, block.IsParameterBlockConstant( img.camera.adp ), block.GetParameterization( img.camera.adp ) )
            if not isIor(param):
                intrinsic, oIntrinsic = oIntrinsic, intrinsic

            anyParsSetFree = False
            if iorId in ( -1, img.camera.id ):
                if intrinsic.wasConst:
                    block.SetParameterBlockVariable(intrinsic.parBlock)
                    anyParsSetFree = True

                if not intrinsic.subset:
                    # set all parameters constant except for the first in coeffs
                    locPar = adjust.local_parameterization.Subset( intrinsic.parBlock.size, [ el for el in range(intrinsic.parBlock.size) if el not in params ] )
                    intrinsic = intrinsic._replace(subset=locPar)
                    block.SetParameterization( intrinsic.parBlock, locPar )
                    anyParsSetFree = True
                else:
                    if intrinsic.wasConst:
                        # We must wipe all other free parameters from subset: e.g. adjustment of ior's focal length has been tried before, but failed, and was set constant, again. Now, we may want to estimate the principal point
                        wantedConstancyMask = np.ones_like( intrinsic.subset.constancyMask )
                        wantedConstancyMask[params] = 0
                    else:
                        wantedConstancyMask = intrinsic.subset.constancyMask.copy()
                        wantedConstancyMask[params] = 0
                    iDiffs = np.flatnonzero( intrinsic.subset.constancyMask != wantedConstancyMask )
                    if iDiffs.size:
                        for iDiff in iDiffs:
                            if wantedConstancyMask[iDiff]:
                                intrinsic.subset.setConstant( int(iDiff) )
                            else:
                                intrinsic.subset.setVariable( int(iDiff) )
                        block.ParameterizationLocalSizeChanged( intrinsic.parBlock )
                        anyParsSetFree = True

            intrinsicPairs[img.camera.id] = IntrinsicPair( *intrinsic, anyParsSetFree, *oIntrinsic, [img.id] )

        return intrinsicPairs

    def logOrRevert( intrinsicPairs, maxAbsCorrsSqr ):
        # for all parameters/parameter blocks with correlations above maxAbsCorr, revert the changes from above. For the others, produce log messages.
        # Once a parameter has been set variable, it shall never be set constant, again.
        maxAbsCorrSqr = maxAbsCorr**2
        iPar = 0
        for currIorID,intrinsicPair in intrinsicPairs.items():
            # do not re-set parameters to constant that were set free in preceding function calls
            if iorId in ( -1, currIorID ):
                nVariable = intrinsicPair.subset.constancyMask.size - intrinsicPair.subset.constancyMask.sum()
                if intrinsicPair.anyParsSetFree:
                    # check only the parameter(s) under question, but not all of non-const subset
                    offsets = np.cumsum( np.logical_not(intrinsicPair.subset.constancyMask) ) - 1
                    idxs = offsets[ params ]
                    if maxAbsCorrsSqr is not None and maxAbsCorrsSqr[iPar + idxs].max() <= maxAbsCorrSqr:
                        msgs.append( 'Free parameter {} for photo{} {}'.format( paramName,
                                                                                '' if len(intrinsicPair.imgIds)==1 else 's',
                                                                                ', '.join(sorted( shortImgFns(images[imgId].path) for imgId in intrinsicPair.imgIds)) ) )
                    elif nVariable - len(params) == 0:
                        # Setting all parameters of a block to constant is illegal. Instead, set the whole block constant!
                        block.SetParameterBlockConstant( intrinsicPair.parBlock )
                    else:
                        for coeff in params:
                            intrinsicPair.subset.setConstant( coeff )
                        block.ParameterizationLocalSizeChanged( intrinsicPair.parBlock )
            else:
                if intrinsicPair.wasConst: # in this case, the par block has not been freed!
                    nVariable = 0
                elif intrinsicPair.subset:
                    nVariable = intrinsicPair.subset.constancyMask.size - intrinsicPair.subset.constancyMask.sum()
                else:
                    nVariable = intrinsicPair.parBlock.size

            iPar += nVariable

            if not intrinsicPair.oIsConst:
                if intrinsicPair.oSubset:
                    iPar += intrinsicPair.oSubset.constancyMask.size - intrinsicPair.oSubset.constancyMask.sum()
                else:
                    iPar += intrinsicPair.oParBlock.size

    intrinsicPairs = getIntrinsicPairs()
    if not any( el.anyParsSetFree for el in intrinsicPairs.values() ):
        return msgs # all cameras had the current parameters already set free, nothing to do.

    evalOpts = adjust.Problem.EvaluateOptions()
    evalOpts.apply_loss_function = True
    paramBlocks = [ *( par for el in intrinsicPairs.values() for par in (el.parBlock,el.oParBlock) if not block.IsParameterBlockConstant(par) ),
                    *( img.prc    for img in images.values() ),
                    *( img.omfika for img in images.values() ),
                    *( objPt.pt for objPt in objPts.values() ) ]
    evalOpts.set_parameter_blocks( paramBlocks )
    # jacobian contains columns only for paramBlocks
    # jacobian contains no columns for parameters that are set constant by way of a Subset-parameterization
    jacobian, = block.Evaluate( evalOpts, residuals=False, jacobian=True ) # we might ask for the cost, and compute sigmas!
    maxAbsCorrsSqr = None
    assert jacobian.shape[1] == np.unique( jacobian.nonzero()[1] ).size, "zero columns in jacobian"
    try:
        # A.T.dot(A) may not be invertible!
        # returns an upper triangular crs matrix
        # TODO: simplicial factorization (which is selected automatically for small problems) crashes with a segmentation fault
        QxxAll = adjust.sparseQxx( jacobian, adjust.Factorization.supernodal )
    except:
        pass
    else:
        nObjPts = len(objPts)
        # column slicing is very inefficient on csr matrices. Thus, convert to csc
        Qxx = QxxAll[:-nObjPts*3,:].tocsc()[:,:-nObjPts*3]
        # TODO: Not even the rows and columns concerning ior/adp seem to be dense, but only the whole diagonal.
        # Is that only the case for supernodal factorization?
        #if Qxx.nnz != Qxx.shape[0]*(Qxx.shape[0]+1)/2:
        #    import pickle
        #    with open('qxx.pickle','wb') as fout:
        #        pickle.dump( Qxx, fout, protocol=pickle.HIGHEST_PROTOCOL )
        #    raise Exception( 'sub-matrix for cameras is not dense! Qxx dumped to file: {}'.format('qxx.pickle') )
        Qxx = Qxx.toarray()
        diag = Qxx.diagonal().copy()
        Rxx = Qxx + Qxx.T
        np.fill_diagonal(Rxx,0)
        # we check the maximum correlation value only.
        # Thus, let's avoid the computation of square roots!
        #sqrtDiag = diag ** .5
        #Rxx = ( ( Rxx / sqrtDiag ).T / sqrtDiag ).T
        #maxAbsCorrs = np.abs(Rxx).max( axis=1 )
        RxxSqr = ( ( Rxx ** 2 / diag ).T / diag ).T
        maxAbsCorrsSqr = RxxSqr.max( axis=1 )
    finally:
        logOrRevert( intrinsicPairs, maxAbsCorrsSqr )
    return msgs

@contract
def exportPly( fn : Path, cameras, images, objPts ):
    prcs = np.array([ img.prc for img in images.values() ])
    tree = spatial.cKDTree(prcs)
    dists, indxs = tree.query(prcs, k=2, n_jobs=-1)
    medMinimumInterPrcDist = np.median( dists[:,1] )

    # .ply binary
    with fn.open( 'wb' ) as fout:
        fout.write("""ply
format binary_little_endian 1.0
comment generated by OrientAL
element vertex {nVertices}
property float64 x
property float64 y
property float64 z
property uint8 red
property uint8 green
property uint8 blue
property uint8 nImgPts
element face {nFaces}
property list uint8 uint32 vertex_indices
end_header
""".format( nVertices=len(objPts) + len(images)*5, nFaces=len(images)*6 ).encode('ascii') )
        stru = struct.Struct('<dddBBBB')
        for objPt in objPts.values():
            fout.write( stru.pack( *objPt.pt, *(objPt.rgb*255).astype(np.uint8), objPt.nImgs )  )

        for img in images.values():
            zLen = img.camera.ior[2] / img.pix2cam.meanScaleForward()
            pts = np.array([ [            0,             0,          0  ],
                             [  img.nCols/2,  img.nRows/2, -zLen ],
                             [ -img.nCols/2,  img.nRows/2, -zLen ],
                             [ -img.nCols/2, -img.nRows/2, -zLen ],
                             [  img.nCols/2, -img.nRows/2, -zLen ] ], float)
            # scale pyramids
            pts *= medMinimumInterPrcDist / 3 / zLen
            pts = ori.omfika( img.omfika ).dot( pts.T ).T + img.prc
            for pt in pts:
                fout.write( stru.pack( *pt, *np.array([255,0,255],dtype=np.uint8), 0 ) )

        stru = struct.Struct('<BIII')
        for iImg,img in enumerate(images.values()):
            # CloudCompare does not support:
            # - reading polylines from PLY files
            # - faces with a vertex count other than 3                          
            offset = len(objPts) + iImg * 5
            for iVtxs in np.array([[0, 1, 2],
                                   [0, 2, 3],
                                   [0, 3, 4],
                                   [0, 4, 1],
                                   [1, 2, 3],
                                   [3, 4, 1]], int):
                fout.write( stru.pack( 3, *(iVtxs+offset) ) )
    logger.info( 'Geo-referenced reconstruction exported to "{}"', fn.name )


@contract
def main( args : argparse.Namespace, cmdLine : 'list(str) | None' = None, parser : 'ArgumentParser|None' = None ) -> None:

    with suppress(FileExistsError):
        args.outDir.mkdir(parents=True)
    args.outDir = args.outDir.resolve()
    utils.argparse.applyLoggingGroup( args, args.outDir, logger, cmdLine )

    if not args.progress:
        Progress.deactivate()

    # ----------- parameter checks

    #if args.photo:
    #    raise Exception("Single photo geo-referencing not implemented")
    # elif not args.relOriDb:
    if not args.relOriDb:
        args.relOriDb = args.outDir / '..' / 'relOri' / 'relOri.sqlite'
    args.relOriDb = args.relOriDb.resolve()

    absOriDbFn = args.outDir / 'absOri.sqlite'
    # Subsequent runs of relOri may result in a different total number of image observations. Hence, the ids in relOri-db may now overlap with those in absOri-db.
    # Also, in that case, image point names from absOri and relOri may collide.
    # If relOri has been re-run, then image orientations will be different, making new manual measurements necessary.
    # Thus, instead of automatically trying to fix non-unique imgobs-ids and -names, let's create a new absOri.sqlite from scratch if relOri is newer than absOri (see above).
    def checkDbModTimes():
        relStatMTime = args.relOriDb.stat().st_mtime
        try:
            absStatMTime = absOriDbFn.stat().st_mtime
        except FileNotFoundError:
            pass
        else:
            if relStatMTime > absStatMTime:
                oldAbsOriDbFn = absOriDbFn.with_name( absOriDbFn.stem + '_old' + absOriDbFn.suffix  )
                absOriDbFn.replace( oldAbsOriDbFn )
                args.reuseMosaic = False
                logger.warning( "relOri.sqlite is newer than the existing absOri.sqlite. To avoid inconsistencies, absOri.sqlite has been renamed to {}. Please re-do your measurements.", oldAbsOriDbFn )
    checkDbModTimes(); del checkDbModTimes

    with suppress(KeyError):
        # support case-insensitive lookup
        lower2mixedCase = { el.lower() : el for el in utils.gdal.wms(utils.gdal.DataSet.ortho) }
        args.ortho = ''.join( el.strip() for el in utils.gdal.wms(utils.gdal.DataSet.ortho)[lower2mixedCase[args.ortho.lower()]].splitlines() )
    infoOrtho, = utils.gdal.imread( args.ortho, info=True, skipData=True )
    if not infoOrtho.projection:
        raise Exception("Ortho photo is not geo-referenced")
    infoOrtho.projection = utils.crs.fixCrsWkt( infoOrtho.projection )
    orthoCs = osr.SpatialReference( wkt = infoOrtho.projection )
    if orthoCs.IsGeographic():
        raise Exception( "Cannot use an ortho photo with a geographic coordinate system: '{}'", orthoCs.GetAttrValue('geogcs') )

    with suppress(KeyError):
        # support case-insensitive lookup
        lower2mixedCase = { el.lower() : el for el in utils.gdal.wms(utils.gdal.DataSet.dtm) }
        args.dsm = ''.join( el.strip() for el in utils.gdal.wms(utils.gdal.DataSet.dtm)[lower2mixedCase[args.dsm.lower()]].splitlines() )
    infoDsm = utils.dsm.info( Path(args.dsm) )
    #del args.dsm # don't delete the attribute, because utils.argparse.logScriptParameters wouldn't log it then. However, use infoDsm.path instead below here.
    if not infoDsm.projection:
        raise Exception("Surface model is not geo-referenced")
    infoDsm.projection = utils.crs.fixCrsWkt( infoDsm.projection )
    dsmCs = osr.SpatialReference( wkt = infoDsm.projection )
    if dsmCs.IsGeographic():
        raise Exception( "Cannot use a surface model with a geographic coordinate system '{}'", dsmCs.GetAttrValue('geogcs') )

    def checkTgtCs():
        if args.tgtCs.lower() not in ( 'dsm', 'utm', 'mgi' ):
            try:
                tgtCs = osr.SpatialReference( wkt = args.tgtCs ) # check this early. Otherwise, users may start to digitise, etc.
            except Exception as ex:
                raise Exception( "Cannot parse user-defined target coordinate system '{}':\n{}".format( args.tgtCs, ex ) )
            if tgtCs.IsGeographic():
                raise Exception( "Cannot use a geographic system as target coordinate system '{}'".format( tgtCs.GetAttrValue('geogcs') ) )
    checkTgtCs(); del checkTgtCs

    if len(args.objStdDevs) == 1:
        if args.objStdDevs[0] == -1.:
            args.objStdDevs = [ infoOrtho.geotransform[0,1]/10 ] * 2 + [ infoDsm.geotransform[0,1]/10 ]
        else:
            args.objStdDevs = args.objStdDevs * 3
    elif len(args.objStdDevs) != 3:
        raise Exception("objStdDevs must either have 1 or 3 values")
    args.objStdDevs = np.array(args.objStdDevs,float)
    if np.any( args.objStdDevs <= 0. ):
        raise Exception("All objStdDevs must be > 0.")

    # even though we only want to read relOriDb, let's open it read-write, as initDataBase may update it.
    utils.db.createUpdateSchema( str(args.relOriDb), utils.db.Access.readWrite )
    with dbapi2.connect( utils.db.uri4sqlite(args.relOriDb) + '?mode=rw', uri=True ) as relOriDb:
        utils.db.initDataBase(relOriDb)
        relOriCs = None
        for row in relOriDb.execute(f"""
            SELECT value from config
            WHERE name == '{utils.db.ConfigNames.CoordinateSystemWkt}' """):
            # relOriCSWkt is NULL in case of unknown georeference
            if row['value']:
                relOriCs = osr.SpatialReference( wkt = row['value']  )

    def getMosaicGsd( ortho, argsGsd ):
        return float( eval( argsGsd ) )

    try:
        args.gsd = MosaicGsd( args.gsd.strip().lower() )
    except ValueError:
        if 'ortho' in args.gsd.lower():
            args.gsd = getMosaicGsd( abs( infoOrtho.geotransform[0,1] ), args.gsd ) # [m] ground sampling distance = orthophoto pixel size
        else:
            args.gsd = float(args.gsd)

        if relOriCs is None:
            logger.warning("Passed mosaic GSD cannot be applied, as aerials are not geo-referenced")
            args.gsd = None
        elif relOriCs.GetAttrValue('PROJCS|SCALE_IS_VALID') == '0':
            logger.warning("Passed mosaic GSD cannot be applied, as the scale of aerials' geo-referencing is invalid.")
            args.gsd = None
        elif args.gsd <= 0.:
            raise Exception( "GSD must be > 0., but is {}".format(args.gsd) )

    utils.argparse.logScriptParameters( args, logger, parser )

    # The orthophoto CS is not a geodetic system in case of WMS, but pseudo-Mercator. Don't adjust the block in such a system.
    # Instead, always adjust in the CS of the surface model. Reason: When transforming 3-D points, PROJ.4 generally assumes ellipsoidal heights. PROJ.4 includes only a few geoid grids: https://trac.osgeo.org/proj/wiki/VerticalDatums
    # -> Never transform non-ellipsoidal heights using PROJ.4
    # NOTE: Transforming non-ellipsoidal heights with source and target systems sharing the same (horizontal) datum, ellipsoid, and vertical datum seems to work.
    # Tested on EPSG:31256 (MGI / Austria GK East) -> EPSG:31287 (MGI / Austria Lambert): both Bessel ellipsoid & orthometric heights above Triest
    # -> point heights (z-coordinates) are left unchanged.
    # However, the projection information of the Austrian-wide DTM dhm_lamb_10m.tif does not contain a shift to WGS84, and so the heights of transformed coordinates ARE affected!!
    # Probably, PROJ.4 assumes a zero-shift for CS that do not define a shift to WGS84.

    fnMosaic        = args.outDir / 'mosaic.tif'
    fnMosaicMask    = args.outDir / 'mosaicOrigin.tif'
    fnMosaicOverlap = args.outDir / 'mosaicOverlap.tif'

    mosaicId = prepareDb( absOriDbFn, args.relOriDb, fnMosaic )

    loss = adjust.loss.Trivial()
    #robLossArg = 10 / 2
    tiePtLoss = adjust.loss.Trivial() # adjust.loss.Wrapper( adjust.loss.SoftLOne( robLossArg  ) )
    block, solveOpts, cameras, images, objPts = restoreRelOriBlock( args.relOriDb, getImgObsLoss = lambda row: tiePtLoss )
    shortImgFns = utils.filePaths.ShortFileNames([ img.path for img in images.values() ])

    mosaicCs = None
    mosaicCsRotationUnknown = True
    mosaicCsScaleUnknown = True
    if relOriCs is not None:
        mosaicCs = dsmCs
        mosaicCsRotationUnknown = relOriCs.GetAttrValue('PROJCS|ROTATION_IS_VALID') == '0'
        mosaicCsScaleUnknown    = relOriCs.GetAttrValue('PROJCS|SCALE_IS_VALID'   ) == '0'
        transformEOR( osr.CoordinateTransformation( relOriCs, mosaicCs ), images.values(), objPts.values() )

    # optimal would be an average over the whole imaged area, excluding vegetation.
    terrainHeight = np.median( [ objPt.pt[2] for objPt in objPts.values() ] )

    if relOriCs is not None: # otherwise, the height is unit-less.
        logger.info( "Estimated average terrain height in mosaic CS: {:.2f}", terrainHeight )
    del relOriCs # not needed any more

    def maybePrepareMosaic():
        with ExitStack() as stack:
            # as mentioned above, it seems not straight forward with Python's sqlite3 to rollback the insertion of the 'homographies' table in case of errors during the computation of its data.
            stack.callback( prepareMosaic,
                            list(images.values()),
                            terrainHeight=terrainHeight,
                            mosaicId=mosaicId,
                            fnMosaic=fnMosaic,
                            fnMosaicMask=fnMosaicMask,
                            fnMosaicOverlap=fnMosaicOverlap,
                            mosaicCs=mosaicCs,
                            gsd=args.gsd,
                            absOriDbFn=absOriDbFn,
                            order=args.order,
                            resample=args.resample,
                            outDir = args.outDir if args.saveSepRects else None )
            if args.reuseMosaic and fnMosaic.exists() and fnMosaicMask.exists():
                def projOkay():
                    if mosaicCs is None:
                        return True
                    else:
                        infoMosaic, = utils.gdal.imread( str(fnMosaic), info=True, skipData=True )
                        infoMosaic.projection = utils.crs.fixCrsWkt( infoMosaic.projection )
                        return osr.SpatialReference( wkt = infoMosaic.projection ).ExportToProj4() == mosaicCs.ExportToProj4()
                if not projOkay():
                    logger.info('Mosaic exists, but has wrong CS')
                else:
                    with suppress(sqlite3.OperationalError), \
                         dbapi2.connect( utils.db.uri4sqlite(absOriDbFn) + '?mode=ro', uri=True ) as absOriDb:
                        nAerials, = absOriDb.execute("""
                            SELECT count(*)
                            FROM images
                            WHERE images.camId NOT NULL""").fetchone()
                        nHomos, = absOriDb.execute("""
                            SELECT count(*)
                            FROM images
                                JOIN homographies ON images.id == homographies.src
                            WHERE images.camId NOT NULL and homographies.dest == ? """, (mosaicId,) ).fetchone()
                        if nAerials==nHomos:
                            stack.pop_all()
                            logger.warning("Mosaic exists already and will be re-used. However, parameters may have changed and the mosaic image observations may hence be misplaced.")

    maybePrepareMosaic(); del maybePrepareMosaic

    def checkForeignKeys():
        with dbapi2.connect( utils.db.uri4sqlite(absOriDbFn) + '?mode=rw', uri=True ) as absOriDb:
            while 1:
                rows = absOriDb.execute("PRAGMA foreign_key_check").fetchall()
                if not rows:
                    break
                for tableName, rowid, referredTableName, iForeignKey in rows:
                    logger.warning("row {} of table {} violates a foreign key constraint referring to table {}. Entry will be deleted.", rowid, tableName, referredTableName )
                    absOriDb.execute("DELETE FROM ? WHERE rowid = ?", ( tableName, rowid ) )
    checkForeignKeys(); del checkForeignKeys

    # images & objPts are now in mosaic CS. However, mosaic CS may be a local CS.
    infoMosaic, = utils.gdal.imread( str(fnMosaic), info=True, skipData=True )
    infoMosaic.projection = utils.crs.fixCrsWkt( infoMosaic.projection )

    # We not only want the return code of calling MonoScope, but also its output to stderr as a string, which will be non-empty in case of errors.
    # Being a Windows GUI-program, MonoScope's stdin, stdout, and stderr are not assigned by default.
    # However, using subprocess.PIPE, the Python console's stderr is duplicated and inherited by MonoScope, so we can capture its output.
    # Note: boost.log does not support process forking: http://www.boost.org/doc/libs/1_58_0/libs/log/doc/html/log/rationale/fork_support.html
    def callMonoScope():
        if len(infoMosaic.projection):
            corners_gdal = np.array(
                [ (               0,               0),
                  (               0,infoMosaic.nRows),
                  (infoMosaic.nCols,infoMosaic.nRows),
                  (infoMosaic.nCols,               0) ], float )
            gdal2wrl = ori.transform.AffineTransform2D( A=infoMosaic.geotransform[:,1:], t=infoMosaic.geotransform[:,0] ) # Affine trafo pixel->CRS
            corners_wrl = gdal2wrl.forward( corners_gdal )
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for corner_wrl in corners_wrl:
                ring.AddPoint( *corner_wrl )
            ring.FlattenTo2D() # even though we've constructed a 2D object, and call AddPoint(.) with only 2 arguments, AddPoint(.) makes ring a 2.5D object!
            ring.CloseRings()
            polyg = ogr.Geometry(ogr.wkbPolygon)
            polyg.AddGeometry(ring)
            initialView = '{} {}'.format( polyg.ExportToWkt(), infoMosaic.projection )
        else:
            initialView = None

        logger.info('Calling MonoScope...')
        try:
            subprocess.run( [
                    'MonoScope{}.exe'.format('_d' if config.debug else '' ),
                    '--db', str(absOriDbFn),
                    '--noConfirmNames',
                    '--changeActiveViewOnly',
                    '--useLsm',
                    '--addOrLeaveImg', str(fnMosaic),
                    '--addOrLeaveImg', args.ortho
                ] +
                ( ['--initialView', initialView ] if initialView else [] ),
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True ) # No need to check the return value, as CalledProcessError will be thrown if non-zero.
        except OSError as ex:
            # Note: if MonoScope.exe is not found on PATH, then we get an OSError with appropriate error message.
            # However, if DLLs needed by MonoScope.exe cannot be loaded, then we get a CalledProcessError with no error message at all (stderr is empty),
            # and the return code is weird: 3221225781 or 3221225785 or ...?
            raise Exception('MonoScope not found on PATH:\n{}'.format(ex))
        # in case of subprocess.CalledProcessError, let's just fall through. The exception get's printed and logged by utils.argparse.exitCode

    callMonoScope(); del callMonoScope

    cpImgDt = np.dtype([('name', object), 
                        ('p',    float, (2,)),
                        ('lsmObs', object ) ]) # LSM image observations in aerials based on this img obs (in the mosaic)
    cpAerialDt = np.dtype([('imgid', int ),
                           ('name', object ), # use object instead of str, so we don't need to specify the maximum string length
                           ('p',    float, (2,) ) ])
    cpObjDt = np.dtype([('name', object ),
                        ('p',    float, (3,) ) ])
    def getCPs():
        with dbapi2.connect( utils.db.uri4sqlite(absOriDbFn) + '?mode=rw', uri=True ) as absOriDb:
            utils.db.initDataBase(absOriDb)
            imgIds = []
            for fnImg in ( fnMosaic, args.ortho ):
                if str(fnImg).lstrip().startswith('<'): # web mapping tile service
                    arg = str(fnImg)
                else:
                    # MonoScope assumes relative input paths to be relative to the cwd, but it stores all input paths relative to its DB.
                    arg = str( utils.filePaths.relPathIfExists( fnImg, absOriDbFn.parent ) )
                imgIds.append( absOriDb.execute("""
                    SELECT id
                    FROM images
                    WHERE ? == path """, [arg] ).fetchone()['id'] )
    
            allCps_ori = []
            for idx,imgId in enumerate(imgIds):
                rows = absOriDb.execute("""
                    SELECT id,name,x,y
                    FROM imgObs
                    WHERE imgid == ? 
                    ORDER BY name """, [imgId] ) # order by name, such that if --fixedCorrespondences, we do not need to sort!
                data = []
                for row in rows:
                    lsmObs = None
                    if idx==0: # for the mosaic, search for LSM img obs based on each CP
                        lsmRows = absOriDb.execute("""
                            SELECT imgId,name,x,y
                            FROM imgObs JOIN images ON imgObs.imgId == images.id
                            WHERE refId == ? AND images.camId NOTNULL""", [row['id']] ) # just to be sure: only select LSM-obs in perspective images.
                        if lsmRows:
                            lsmObs = []
                            for lsmRow in lsmRows:
                                lsmObs.append(( lsmRow['imgId'], lsmRow['name'], np.array([ lsmRow['x'],lsmRow['y'] ]) ))
                    data.append( ( row['name'], np.array([ row['x'],row['y'] ]), lsmObs ) )
                allCps_ori.append( np.array( data, dtype=cpImgDt ) )
            if len(allCps_ori[0]) != len(allCps_ori[1]):
                raise Exception( 'Numbers of control points in mosaic and orthophoto must match, while they are: {} vs. {}'.format( *( len(cps) for cps in allCps_ori ) ) )
            if len(allCps_ori[0]) < 3:
                raise Exception( 'At least 3 control points must be observed in both images, but only {} have been.'.format( len(allCps_ori[0]) ) )

            # collect information to transform mosaic image observations back to aerials. Unused if LSM-obs are present.
            for row in absOriDb.execute("""
                SELECT src, h00, h01, h02, h10, h11, h12, h20, h21, h22
                FROM homographies 
                WHERE dest = ? """, (imgIds[0],) ):
                Hori = np.array(tuple(row)[1:]).reshape((3,3))
                invHomography = linalg.inv( Hori )
                imgId = row['src']
                images[imgId] = RectifiedImage( invHomography=invHomography, *images[imgId] )

            return imgIds + allCps_ori
    mosaicId, orthoId, cpsMosaic_ori, cpsOrtho_ori = getCPs()

    def getIdAerials():
        """ Get the indices / imgIDs of the original aerials corresponding to the image points in the mosaic.
            Thereby, check that image points in the mosaic are in valid areas i.e. they are covered by a rectified aerial."""
        cpsMosaic_cv = cpsMosaic_ori['p'] * (1,-1)
        try:
            idAerialCpsMosaic = utils.gdal.interpolatePoints( str(fnMosaicMask), cpsMosaic_cv, interpolation=utils.gdal.Interpolation.nearest )
        except Exception as ex:
            raise Exception( 'Extraction of image Id for control points from "{}" failed: {}'.format( fnMosaicMask, ex ) )
        if np.count_nonzero(idAerialCpsMosaic>=0) != len(cpsMosaic_cv):
            raise Exception( 'Image observations outside the mosaic found. Maybe the image has changed since doing the measurements?' ) # this is prevented by MonoScope at the time of measurement, but the underlying image may have changed since then!
        return idAerialCpsMosaic
    idAerialCpsMosaic = getIdAerials()

    def getCpsOrthoWrl():
        # interpolate terrain heights for points in orthophoto
        ortho2wrl = ori.transform.AffineTransform2D( A=infoOrtho.geotransform[:,1:], t=infoOrtho.geotransform[:,0] ) # Affine trafo pixel->CRS
        cpsOrtho_gdal = cpsOrtho_ori['p'] * (1,-1) + .5
        cpsOrtho_wrl = np.empty( len(cpsOrtho_gdal), dtype=cpObjDt )
        cpsOrtho_wrl['name'] = cpsOrtho_ori['name']
        cpsOrtho_wrl['p'][:,:2] = ortho2wrl.forward( cpsOrtho_gdal )
        return cpsOrtho_wrl
    cpsOrtho_wrl = getCpsOrthoWrl()

    def getTgtCs():
        if args.tgtCs.lower() == 'dsm':
            return dsmCs
        tgtCs = osr.SpatialReference()
        if args.tgtCs.lower() in ( 'utm', 'mgi' ):
            wgs84 = osr.SpatialReference()
            wgs84.SetWellKnownGeogCS( 'WGS84' )
            ortho2Wgs84 = osr.CoordinateTransformation( orthoCs, wgs84 )
            cpsOrtho_wgs84 = np.array( ortho2Wgs84.TransformPoints( cpsOrtho_wrl['p'][:,:2].tolist() ) )[:,:2]
            meanLonLat = utils.stats.geometricMedian( cpsOrtho_wgs84 )
            if args.tgtCs.lower() == 'utm':
                tgtCs.SetWellKnownGeogCS( "WGS84" )
                tgtCs.SetUTM( utils.crs.utmZone( meanLonLat[0] ), int(meanLonLat[1] >= 0.) )
            else:
                assert args.tgtCs.lower() == 'mgi'
                meridian = utils.crs.lonGreenwichToMeridian( meanLonLat[0] )
                if meridian < 31:
                    tgtCs.ImportFromEPSG(31254)
                elif meridian == 31:
                    tgtCs.ImportFromEPSG(31255)
                else:
                    tgtCs.ImportFromEPSG(31256)
        else:
            tgtCs.SetFromUserInput( args.tgtCs )
        return tgtCs
    tgtCs = getTgtCs()
    logger.info("Target coordinate system: {}", tgtCs.GetAttrValue('projcs') )

    cpsOrtho_wrl['p'][:,:2] = np.array( osr.CoordinateTransformation( orthoCs, tgtCs ).TransformPoints( cpsOrtho_wrl['p'][:,:2].tolist() ) )[:,:2] # osr seems to always add the z-coordinate as 3rd column

    def getCorrespondences():
        # relOriCs may be NULL. Specifically in that case, we need to compute a trafo based on correspondences, and apply it to images & objPts.

        # Point names have no meaning (identical names do not define homologous points).
        # Identify corresponding points. Try all combinations and select the transform with the least SSE.
        # For robustness, use world coordinates, and compute an L1 2d transform with rotation and shift only.
        # TODO: Use the computed transformation as initial values for the adjustment.
        # What if the mosaic has no georeferencing at all yet? E.g. a single image! A single image would be an aerial, without a sparse object point cloud. Thus, a perspective transform (spatial resection) would seem more appropriate.
        # -> estimate scale, too.
        # One might take advantage of topology: select initial edges in both images, subsequent possible correspondences would need to lie on the same side of that edge then.
        # However, due to perspective distortions, measurement errors, etc., correct correspondences might be discarded that way!
        if mosaicCs is None:
            cpsMosaic_wrl = cpsMosaic_ori['p']
        else:
            mosaic2wrl = ori.transform.AffineTransform2D( A=infoMosaic.geotransform[:,1:], t=infoMosaic.geotransform[:,0] ) # Affine trafo pixel->CRS
            cpsMosaic_gdal = cpsMosaic_ori['p'] * (1,-1) + .5
            cpsMosaic_wrl = mosaic2wrl.forward( cpsMosaic_gdal )

            if mosaicCs is not tgtCs: # better rely on Python's 'is' for comparison than on OGRs IsSame() (may be buggy)
                mosaic2tgtCs = osr.CoordinateTransformation( mosaicCs, tgtCs )
                cpsMosaic_wrl = np.array( mosaic2tgtCs.TransformPoints( cpsMosaic_wrl.tolist() ) )[:,:2] # osr seems to always add the z-coordinate as 3rd column

        if args.fixedCorrespondences:
            orthoNames = set(cpsOrtho_wrl['name'])
            mosaicNames = set(cpsMosaic_ori['name'])
            namesSymmDiff = orthoNames.symmetric_difference(mosaicNames)
            if namesSymmDiff:
                raise Exception( 'Using fixedCorrespondences, the same point names must be observed in mosaic and orthophoto. '
                                 'However, points {} have been observed in only one of them.'.format( ', '.join(namesSymmDiff) ) )
        # if mosaic misses georeference, then do estimate rotation and scale, and apply the estimated transformation!
        mosaicPermutation, mosaic2ortho = determineCorrespondences( cpsOrtho_wrl['p'][:,:2],
                                                                    cpsMosaic_wrl,
                                                                    permute = not args.fixedCorrespondences,
                                                                    unknownRot  = mosaicCs is None or mosaicCsRotationUnknown,
                                                                    unknownScale= mosaicCs is None or mosaicCsScaleUnknown,
                                                                    robust=False )
        if not args.fixedCorrespondences:
            logger.info( 'Determined correspondences\n'
                         'Mosaic\tOrthophoto\n'
                         '{}'.format( '\n'.join(( '{}\t{}'.format( cpsMosaic_ori[mosaicPermutation[idx]]['name'], cpsOrtho_ori[idx]['name'] ) for idx in range(len(mosaicPermutation)) )) ) )
        return mosaicPermutation, mosaic2ortho

    mosaicPermutation, mosaic2ortho = getCorrespondences()

    cpsMosaic_ori = cpsMosaic_ori[mosaicPermutation]
    idAerialCpsMosaic = idAerialCpsMosaic[mosaicPermutation]
        
    # cpsOrtho_wrl already transformed to target CS, but target CS may be != dsm CS
    cpsOrtho_wrl['p'][:,2] = utils.gdal.interpolateRasterHeights( infoDsm.path, cpsOrtho_wrl['p'][:,:2], objPtsCS=tgtCs.ExportToWkt(), interpolation=utils.gdal.Interpolation.bilinear )
    # Consider locations where the DTM is undefined.
    valid = np.logical_not( np.isnan( cpsOrtho_wrl['p'][:,2] ) )
    idAerialCpsMosaic = idAerialCpsMosaic[valid]
    cpsMosaic_ori = cpsMosaic_ori[valid]
    cpsOrtho_ori = cpsOrtho_ori[valid]
    cpsOrtho_wrl = cpsOrtho_wrl[valid]
    if cpsMosaic_ori.size < 3:
        raise Exception( '{} points are discarded, because no terrain height could be interpolated. '
                            'Only {} points remain, while at least 3 control points must be observed in both images.'.format(
                            valid.size - cpsMosaic_ori.size, cpsMosaic_ori.size ) )

    # apply the estimated trafo!
    transformEOR( mosaic2ortho, images.values(), objPts.values() )

    logger.verbose( 'Control point coordinates in target CS\n'
                    'Name\tX\tY\tZ\n'
                    '{}',
                    '\n'.join( '{}\t{:.3f}\t{:.3f}\t{:.3f}'.format( cpOrtho_wrl['name'], *cpOrtho_wrl['p'] )
                               for cpOrtho_wrl in cpsOrtho_wrl ) )

    summary = adjust.Solver.Summary()
    # TODO: check change of omfikas. Transformed correctly?
    if 0: # test: re-adjust relOri block without mosaic/orthophoto obs.
        adjust.Solve(solveOpts, block, summary)
        if not adjust.isSuccess( summary.termination_type ):
            raise Exception( summary.FullReport() )

    cpWeights = np.diag( 1./args.objStdDevs )
    def introduceControl():
        # introduce orthophoto observations as observed unknowns
        orthoPhotoResidualIDs = []
        for cpOrtho_wrl in cpsOrtho_wrl:
            # ObservedUnknown respects EvaluateOptions.weighted :-)
            #cost = adjust.cost.NormalPrior( cpWeights, cpOrtho_wrl['p'] )
            cost = adjust.cost.ObservedUnknown( cpOrtho_wrl['p'], cpWeights )
            orthoPhotoResidualIDs.append(
                block.AddResidualBlock( cost,
                                        loss,
                                        cpOrtho_wrl['p'] ) )
            solveOpts.linear_solver_ordering.AddElementToGroup( cpOrtho_wrl['p'], 0 )

        mosaicResidualIDs = []
        cpsAerials_ori = []
        for imgId,mosaic_ori,cpOrtho_wrl in utils.zip_equal(idAerialCpsMosaic, cpsMosaic_ori, cpsOrtho_wrl):
            if not mosaic_ori['lsmObs']:
                image = images[imgId]
                # reverse-transform rect image coordinates -> aerial image coordinates!!
                ptRect_orih = np.array([ mosaic_ori['p'][0], mosaic_ori['p'][1], 1. ])
                ptAerialUndist_orih = image.invHomography.dot( ptRect_orih )
                ptAerialUndist_ori = ptAerialUndist_orih[:2] / ptAerialUndist_orih[2]
                # distort the results
                ptAerialdist_ori = ori.distortion( np.atleast_2d( ptAerialUndist_ori ),
                                                   image.camera.ior,
                                                   ori.adpParam2Struct( image.camera.adp ),
                                                   ori.DistortionCorrection.dist )[0]
                data = [( imgId, mosaic_ori['name'], ptAerialdist_ori )]
            else:
                data = [ ( imgId, name, images[imgId].pix2cam.forward( lsmObs ) )
                         for imgId, name, lsmObs in mosaic_ori['lsmObs'] ]
            cpsAerials_ori += data
            for imgId, name, ptAerialdist_ori in data:
                image = images[imgId]
                cost = adjust.cost.PhotoTorlegard( ptAerialdist_ori[0], ptAerialdist_ori[1], image.obsWeights )
                cost.data = ImgObsData( -1, image.id, -1 )
                mosaicResidualIDs.append(
                    block.AddResidualBlock( cost,
                                            loss,
                                            image.prc,
                                            image.omfika,
                                            image.camera.ior,
                                            image.camera.adp,
                                            cpOrtho_wrl['p'] ) )
        cpsAerials_ori = np.array( cpsAerials_ori, dtype=cpAerialDt )
        return orthoPhotoResidualIDs, mosaicResidualIDs, cpsAerials_ori

    orthoPhotoResidualIDs, mosaicResidualIDs, cpsAerials_ori = introduceControl()

    imgCsUnit = { 'px' if isinstance( image.pix2cam, ori.transform.IdentityTransform2D ) else 'mm' for image in images.values() }
    if len(imgCsUnit) > 1:
        logger.warning('Mixture of digital and scanned images detected. Will use the same weights for image observations in both image types. Will assume [px] as unit in all resp. output')
        imgCsUnit = 'px'
    else:
        imgCsUnit = list(imgCsUnit)[0]

    tiePointResidualIDs = block.GetResidualBlocks()[ : -len(orthoPhotoResidualIDs) -len(mosaicResidualIDs)]
    for rec in logger.recInfo():
        rec('Residuals a priori')

        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.weighted = False # compute residuals disregarding observation weights
        evalOpts.apply_loss_function = False
        evalOpts.set_residual_blocks( tiePointResidualIDs )
        residuals, = block.Evaluate(evalOpts)
        resNormsSqr = residuals[0::2]**2 + residuals[1::2]**2
        rec( 'Tie image point residual norm statistics a priori [{}]\n'
             'statistic\tvalue\n'
             'median\t{:.3f}\n'
             'max' '\t{:.3f}',
             imgCsUnit,
             np.median(resNormsSqr)**.5,
             np.max   (resNormsSqr)**.5 )

        # log residuals at control points a priori
        # Note: residuals may be quite large, even though we have applied our estimated transform, because estimation is 2D only - see the large residuals in y.
        evalOpts.set_residual_blocks( mosaicResidualIDs )
        residuals, = block.Evaluate(evalOpts)
        rec( 'Control image point residuals a priori [{}]\n'
             'image\tpoint\tx\ty\tnorm\n'
             '{}',
             imgCsUnit,
             '\n'.join( '{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}'.format( shortImgFns( images[ cpAerial_ori['imgid'] ].path ), cpAerial_ori['name'], *residual, linalg.norm(residual) )
                                                                 for cpAerial_ori,residual in utils.zip_equal(cpsAerials_ori,residuals.reshape((-1,2))) ) )

    #resNormsSqr = residuals[0::2]**2 + residuals[1::2]**2
    #logger.info( 'Aerial control point image residual norms a priori\n'
    #             'median\t{:.3f}px\n'
    #             'max' '\t{:.3f}px'.format( np.median(resNormsSqr)**.5,
    #                                        np.max   (resNormsSqr)**.5 ) )

    for camera in cameras.values():
        block.SetParameterBlockConstant( camera.ior )
        block.SetParameterBlockConstant( camera.adp )
    for image in images.values():
        block.SetParameterBlockConstant( image.omfika )
    for param in cpsOrtho_wrl['p']:
        block.SetParameterBlockConstant( param )

    solveOpts.parameter_tolerance = 1.e-16
    # There are many image observations, but only few control points. Control points thus contribute little to the objective function. Make sure we end-iterate, such that the sum of residuals of control points is small.
    # Appropriate weighting decreases the number of necessary iterations.
    solveOpts.function_tolerance = 1.e-16
    solveOpts.max_num_iterations = 500
    adjust.Solve(solveOpts, block, summary)
    if not adjust.isSuccess( summary.termination_type ):
        raise Exception( summary.FullReport() )

    for image in images.values():
        block.SetParameterBlockVariable( image.omfika )

    adjust.Solve(solveOpts, block, summary)
    if not adjust.isSuccess( summary.termination_type ):
        raise Exception( summary.FullReport() )

    if not args.fixedIorAdp:
        anyFreed = False
        for camera in cameras.values():
            for param,s_param in ( (camera.ior,camera.s_ior), (camera.adp,camera.s_adp) ):
                # 0.0 -> const
                # None -> variable; parameter value estimated by relOri, but not its standard deviation
                # > 0.0 -> variable
                if any( el != 0. for el in s_param ):
                    block.SetParameterBlockVariable( param )
                    anyFreed = True
        if anyFreed:
            adjust.Solve(solveOpts, block, summary)
            if not adjust.isSuccess( summary.termination_type ):
                raise Exception( summary.FullReport() )

    if not args.fixedControlObjPts:
        for param in cpsOrtho_wrl['p']:
            block.SetParameterBlockVariable( param )
        
        adjust.Solve(solveOpts, block, summary)
        if not adjust.isSuccess( summary.termination_type ):
            raise Exception( summary.FullReport() )

    if not args.fixedIorAdp:    
        # note: we do not need to apply our modifications of ior/adp to the image observations transferred from the mosaic to the aerials,
        # Because we have transferred them before modifying ior/adp. As we load ior/adp from relOriDB each time, it should be okay even for repeated runs of absOri.
        makeIorsAdpsVariableAtOnce( block, solveOpts, cameras, images, objPts )

    # Don't deactivate tiePoints: there is hardly any redundancy in the control points, and if they have large residuals, then there is most probably an outlier among them.
    # TODO: instead of fixed thresholds, we should use a robust estimate of the variance, or even histogram analysis of the residuals.
    #def deactTiePtOutliers( tiePointResidualIDs ):
    #    evalOpts = adjust.Problem.EvaluateOptions()
    #    evalOpts.weighted = True
    #    evalOpts.apply_loss_function = False
    #    msgs = []
    #    nRemovedImgObs = 0
    #    nRemovedObjPts = 0
    #    maxTieImgResNorms = robLossArg*2, robLossArg, 3
    #    for idx, maxTieImgResNorm in enumerate( itertools.chain( maxTieImgResNorms, itertools.repeat(maxTieImgResNorms[-1]) ) ):
    #        if idx > 0: # otherwise the block is unchanged.
    #            tiePtLoss.Reset( adjust.loss.SoftLOne( maxTieImgResNorm / 2  ) )
    #            adjust.Solve(solveOpts, block, summary)
    #            if not adjust.isSuccess( summary.termination_type ):
    #                raise Exception( summary.FullReport() )
    #
    #        evalOpts.set_residual_blocks( tiePointResidualIDs )
    #        residuals, = block.Evaluate(evalOpts)
    #        resNormsSqr = residuals[0::2]**2 + residuals[1::2]**2
    #        resBlocksToRemove = []
    #        for resBlock, resNormSqr in utils.zip_equal( tiePointResidualIDs, resNormsSqr ):
    #            if resNormSqr > maxTieImgResNorm**2:
    #                resBlocksToRemove.append( resBlock )
    #        resBlocksToRemoveSet = set(resBlocksToRemove)
    #        objPtIdsToRemove = set()
    #        for resBlock in resBlocksToRemove:
    #            cost = block.GetCostFunctionForResidualBlock( resBlock )
    #            objPt = objPts[cost.data.objPtId].pt
    #            resBlocks = set( block.GetResidualBlocksForParameterBlock( objPt ) )
    #            if len( resBlocks - resBlocksToRemoveSet ) < 2:
    #                resBlocksToRemoveSet.update( resBlocks )
    #                objPtIdsToRemove.add( cost.data.objPtId )
    #        keptTiePointResidualBlocks = []
    #        for resBlock in tiePointResidualIDs:
    #            if resBlock in resBlocksToRemoveSet:
    #                block.RemoveResidualBlock( resBlock )
    #            else:
    #                keptTiePointResidualBlocks.append( resBlock )
    #        for objPtId in objPtIdsToRemove:
    #            pt = objPts[objPtId].pt
    #            solveOpts.linear_solver_ordering.Remove( pt )
    #            assert len(block.GetResidualBlocksForParameterBlock( pt )) == 0
    #            block.RemoveParameterBlock( pt )
    #            del objPts[objPtId]
    #        msgs.append(( maxTieImgResNorm, len(resBlocksToRemoveSet), len(objPtIdsToRemove) ))
    #        nRemovedImgObs += len(resBlocksToRemoveSet)
    #        nRemovedObjPts += len(objPtIdsToRemove)
    #        tiePointResidualIDs = keptTiePointResidualBlocks
    #        if len(resBlocksToRemoveSet) + len(objPtIdsToRemove) == 0:
    #            break;
    #
    #    msgs.append(( 'total', nRemovedImgObs, nRemovedObjPts ))
    #    logger.info( 'Removed tie point outliers\n'
    #                 'Threshold [σ]\t#image obs\t#object points\n'
    #                 '{}',
    #                 '\n'.join( '\t'.join(str(el) for el in tup) for tup in msgs ) )
    #    return tiePointResidualIDs
    #if 1:
    #    tiePointResidualIDs = deactTiePtOutliers( tiePointResidualIDs )


    redundancy = summary.num_residuals_reduced - summary.num_effective_parameters_reduced
    sigma0 = ( summary.final_cost * 2 / redundancy ) **.5 if redundancy > 0 else 0.
    logger.info( 'Adjustment statistics\n'
                 'statistic\tvalue\n'
                 '{}',
                 '\n'.join( "{0}\t{1:{2}}".format(*els)
                            for els in (('#observations' , summary.num_residuals_reduced           , ''    ),
                                        ('#unknowns'     , summary.num_effective_parameters_reduced, ''    ),
                                        ('redundancy'    , redundancy                              , ''    ),
                                        ('σ_0'           , sigma0                                  , '.3f' ),
                                        ('#aerials'      , len(images)                             , ''    ),
                                        ('#tieImgPts'    , len(tiePointResidualIDs)                , ''    ),
                                        ('#tieObjPts'    , len(objPts)                             , ''    ),
                                        ('#controlImgPts', len(mosaicResidualIDs)                  , ''    ),
                                        ('#controlObjPts', len(orthoPhotoResidualIDs)              , ''    )) ) )

    objCsUnit= 'm' if tgtCs.GetAttrValue('UNIT')=='metre' else tgtCs.GetAttrValue('UNIT')

    for rec in logger.recInfo():
        rec('Residuals a posteriori')

        evalOpts.set_residual_blocks( tiePointResidualIDs )
        residuals, = block.Evaluate(evalOpts)
        resNormsSqr = residuals[0::2]**2 + residuals[1::2]**2
        rec( 'Tie image point residual norm statistics a posteriori [{}]\n'
             'statistic\tvalue\n'
             'median\t{:.3f}\n'
             'max' '\t{:.3f}',
             imgCsUnit,
             np.median(resNormsSqr)**.5,
             np.max   (resNormsSqr)**.5 )

        evalOpts.set_residual_blocks( mosaicResidualIDs )
        residuals, = block.Evaluate(evalOpts)
        rec( 'Control image point residuals a posteriori [{}]\n'
             'image\tpoint\tx\ty\tnorm\n'
             '{}',
             imgCsUnit,
             '\n'.join( '{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}'.format( shortImgFns( images[ cpAerial_ori['imgid'] ].path ), cpAerial_ori['name'], *residual, linalg.norm(residual) )
                        for cpAerial_ori,residual in utils.zip_equal(cpsAerials_ori,residuals.reshape((-1,2))) ) )
        evalOpts.set_residual_blocks( orthoPhotoResidualIDs )
        residuals, = block.Evaluate(evalOpts)
        rec( 'Control object point residuals a posteriori [{}]\n'
             'name\tX\tY\tZ\tnorm\n'
             '{}',
             objCsUnit,
             '\n'.join( '{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( cpOrtho_wrl['name'], *residual, linalg.norm(residual) )
                        for cpOrtho_wrl,residual in utils.zip_equal( cpsOrtho_wrl, residuals.reshape((-1,3)) ) ) )

    # estimate stochastic model a posteriori

    # Don't simply use id(arr) as key for stdDevs, because arr may be a temporary object, e.g.: cpOrtho_wrl['p']
    # Thus, it would be impossible to retrieve the std.devs. for mosaic points in saveSQLite!
    def getStdDevs():
        stdDevs = {}
        if not args.precision:
            for camera in cameras.values():
                for parBlock in ( camera.ior, camera.adp ):
                    if block.IsParameterBlockConstant( parBlock ):
                        stdDevs[parBlock.ctypes.data] = np.zeros(parBlock.size)
                        continue
                    locParam = block.GetParameterization( parBlock )
                    if locParam is None:
                        continue
                    # ndarray of dtype=np.float holds NaN for non-constant parameters. SQLite will consider those as NULL
                    # dtype=np.float must be passed, otherwise: ndarray.dtype==object
                    stdDevs[parBlock.ctypes.data] = np.array([ (0.0 if locParam.isConstant(idx) else None) for idx in range(parBlock.size) ], float )
            return stdDevs

        if 0:
            covOpts = adjust.Covariance.Options()
            covOpts.apply_loss_function = False
            covariance = adjust.Covariance( covOpts )
            paramBlocks = []
            for image in images.values():
                paramBlocks.append( image.prc )
                paramBlocks.append( image.omfika )
            for camera in cameras.values():
                paramBlocks.append( camera.ior )
                paramBlocks.append( camera.adp )
            for cpOrtho_wrl in cpsOrtho_wrl:
                paramBlocks.append( cpOrtho_wrl['p'] )
            paramBlockPairs = [ (el,el) for el in paramBlocks ]
            covariance.Compute( paramBlockPairs, block )
            for paramBlockPair in paramBlockPairs:
                cofactorBlock = covariance.GetCovarianceBlock( *paramBlockPair )
                stdDevs[ paramBlockPair[0].ctypes.data ] = sigma0 * np.diag(cofactorBlock)**0.5
        else:
            evalOpts = adjust.Problem.EvaluateOptions()
            evalOpts.apply_loss_function = False # all parameter blocks except for ior and adp are surely variable, so we only need to filter those.
            paramBlocks = [ *( image.prc    for image in images.values() ),
                            *( image.omfika for image in images.values() ),
                            *( camera.ior for camera in cameras.values() if not block.IsParameterBlockConstant(camera.ior) ),
                            *( camera.adp for camera in cameras.values() if not block.IsParameterBlockConstant(camera.adp) ),
                            *( objPt.pt for objPt in objPts.values() ),
                            *( cpOrtho_wrl['p'] for cpOrtho_wrl in cpsOrtho_wrl if not block.IsParameterBlockConstant(cpOrtho_wrl['p']) ) ]
            evalOpts.set_parameter_blocks( paramBlocks )
            # jacobian contains columns only for paramBlocks
            # jacobian contains no columns for parameters that are set constant by way of a Subset-parameterization
            jacobian, = block.Evaluate( evalOpts, residuals=False, jacobian=True )
            diagQxx = adjust.diagQxx( jacobian )
            iPar = 0
            for paramBlock in paramBlocks:
                locParam = block.GetParameterization( paramBlock )
                if locParam is None:
                    dQxx = diagQxx[ iPar : iPar+paramBlock.size ]
                    iPar += paramBlock.size
                else:
                    bVariable = np.logical_not(locParam.constancyMask)
                    dQxx = np.zeros( paramBlock.size )
                    dQxx[bVariable] = diagQxx[ iPar : iPar+bVariable.sum() ]
                    iPar += bVariable.sum()
                dQxx[dQxx<0.] = 0. # avoid warnings due to taking the square root of negative numbers.
                stdDevs[paramBlock.ctypes.data] = sigma0 * dQxx**.5

            # For all-constant ior/adp, store zero-std.devs instead of NULLs, because 0.0 means constant, while NULL means 'sigma not estimated'
            for camera in cameras.values():
                for paramBlock in (camera.ior, camera.adp):
                    stdDevs.setdefault( paramBlock.ctypes.data, np.zeros(paramBlock.size) )
            for cpOrtho_wrl in cpsOrtho_wrl:
                stdDevs.setdefault( cpOrtho_wrl['p'].ctypes.data, np.zeros(cpOrtho_wrl['p'].size) )

        # log stdDevs
        for rec in logger.recInfo():
            rec('Standard deviations of unknowns a posteriori')

            rec( 'σ tie object points [{}]\n'
                 'statistic\tX\tY\tZ\n'
                 'min' '\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'median\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'max' '\t{:.3f}\t{:.3f}\t{:.3f}',
                 objCsUnit,
                 *utils.stats.minMedMax( np.array([ stdDevs[ objPt.pt.ctypes.data ]  for objPt in objPts.values() ]) ).flat )

            rec( 'σ control object points [{}]\n'
                 'name\tX\tY\tZ\n'
                 '{}',
                 objCsUnit,
                 '\n'.join(( '{}\t{:.3f}\t{:.3f}\t{:.3f}'.format( name, *stdDevs[pt.ctypes.data] )
                             for name,pt in utils.zip_equal(cpsOrtho_wrl['name'],cpsOrtho_wrl['p']) )) )

            rec( 'σ image projection centres [{}]\n'
                 'statistic\tX0\tY0\tZ0\n'
                 'min' '\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'median\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'max' '\t{:.3f}\t{:.3f}\t{:.3f}',
                 objCsUnit,
                 *utils.stats.minMedMax( np.array([ stdDevs[ image.prc.ctypes.data ]  for image in images.values() ]) ).flat )


            rec( 'σ image rotation angles [gon]\n'
                 'statistic\tω\tφ\tκ\n'
                 'min' '\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'median\t{:.3f}\t{:.3f}\t{:.3f}\n'
                 'max' '\t{:.3f}\t{:.3f}\t{:.3f}',
                 *utils.stats.minMedMax( np.array([ stdDevs[ image.omfika.ctypes.data ]  for image in images.values() ]) ).flat )

            rec( 'σ camera IORs [{}] / ADPs []\n'
                 'camera\tx0\ty0\tz0\tr3\tr5\n'
                 '{}',
                 imgCsUnit,
                 '\n'.join( '\t'.join( (
                     str(cam.id),
                     *( 'c' if el==0. else '{:.3f}'.format(el)
                       for el in ( *stdDevs[ cam.ior.ctypes.data ],
                                    *stdDevs[ cam.adp.ctypes.data ][[ adjust.PhotoDistortion.optPolynomRadial3, adjust.PhotoDistortion.optPolynomRadial5]]
                                 ) ) ) ) for cam in cameras.values() ) )

        return stdDevs

    stdDevs = getStdDevs()

    cpsMosaicWithId_ori = np.array( [(mosaicId, el['name'], el['p']) for el in cpsMosaic_ori ], dtype=cpAerialDt )
    cpsOrthoWithId_ori = np.array( [(orthoId, el['name'], el['p']) for el in cpsOrtho_ori ], dtype=cpAerialDt )
    saveSQLite( absOriDbFn, tgtCs, cameras, images, objPts, stdDevs, cpsMosaicWithId_ori, cpsOrthoWithId_ori, cpsAerials_ori, cpsOrtho_wrl )

    def storeFootprints():
        """compute image footprints and store as 1 shape file.
        md needs the footprints only to store them in LBA as rough outlines of the image content, as quadrangles.
        Thus, do not use monoplotting of the undistorted image borders onto a DSM/DTM (in ortho), but simply backproject the image corners onto the ground plane.
        """
        terrainHeight = np.median([ el.pt[2] for el in objPts.values() ])
        outDir = args.outDir / 'footprints'
        logger.info( 'Store footprints for horizontal, flat terrain at height {:.3f} in directory "{}"', terrainHeight, outDir.name )
        for image in images.values():
            # adapt to oriental.blocks
            BlocksImage = namedtuple('BlocksImage', image._fields + ('rot',))
            blocksImage = BlocksImage( rot=image.omfika, *image )
            footprint = blocks.footprint.forConstantHeight( blocksImage.camera, blocksImage, terrainHeight )
            blocks.footprint.exportShapeFile( outDir / Path(blocksImage.path).with_suffix('.shp').name,
                                              [footprint],
                                              [Path(blocksImage.path).stem],
                                              tgtCs.ExportToWkt() )
    if args.footprints:
        storeFootprints()

    if args.exportPly:
        exportPly( args.outDir / 'reconstruction.ply', cameras, images, objPts )

    def plotResiduals():
        residualScale = 1.
        residualsDir = args.outDir / 'residuals'
        imgId2tiePtObsAndRes = { iImg : [] for iImg in images }
        imgId2cpPtObsAndRes =  { iImg : [] for iImg in images }
        for residualIds, imgId2PtObsAndRes in ( ( tiePointResidualIDs, imgId2tiePtObsAndRes ),
                                                ( mosaicResidualIDs  , imgId2cpPtObsAndRes ) ):
            evalOpts.set_residual_blocks( residualIds )
            residuals, = block.Evaluate(evalOpts)
            for iRes, residualID in enumerate(residualIds):
                cost = block.GetCostFunctionForResidualBlock( residualID )
                obsId, imgId, objPtId = cost.data
                obs = np.array([cost.x,cost.y])
                #proj = obs - residualScale * residuals[ iRes*2 : iRes*2+2 ]
                res = residuals[ iRes*2 : iRes*2+2 ]
                obs = images[imgId].pix2cam.inverse( obs )
                #proj = images[imgId].pix2cam.inverse( proj )
                res = images[imgId].pix2cam.Ainv @ res
                imgId2PtObsAndRes[imgId].append( np.r_[ obs, res ] )
            for imgId in imgId2PtObsAndRes:
                imgId2PtObsAndRes[imgId] = np.array( imgId2PtObsAndRes[imgId] )

        logger.info('Plot residuals for each image in directory "{}"', residualsDir.name )
        progress = Progress(len(imgId2tiePtObsAndRes))
        with multiprocessing.Pool( initializer = log.suppressCustomerClearLogFileName ) as pool:
            class Incrementor:
                def __init__( self, progress ):
                    self.progress = progress
                def __call__( self, arg ):
                    self.progress += 1
            results = []
            for imgId, tiePtObsAndRes in imgId2tiePtObsAndRes.items():
                cpPtObsAndRes = imgId2cpPtObsAndRes[imgId]
                kwds = dict( fnIn=images[imgId].path,
                             fnOut=residualsDir / ( Path(images[imgId].path).stem + '.jpg' ),
                             scale=residualScale,
                             imgObsAndResids=tiePtObsAndRes,
                             imgObsAndResidsAndNames=cpPtObsAndRes )
                results.append( pool.apply_async( func=utils.pyplot_utils.plotImgResiduals, kwds=kwds, callback=Incrementor(progress) ) )
            # For the case of exceptions thrown in the worker processes:
            # - don't define an error_callback that re-throws, or the current process will hang. If the current process gets killed e.g. by Ctrl+C, the child processes will be left as zombies.
            # - collect the async results, call pool.close() and call result.get(), which will re-throw any exception thrown in the resp. child process.
            pool.close()
            for result in results:
                result.get()
            pool.join()
    if args.plotResiduals:
        plotResiduals()
