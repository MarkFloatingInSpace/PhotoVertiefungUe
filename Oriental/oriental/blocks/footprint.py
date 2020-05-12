# -*- coding: cp1252 -*-
"get the footprint of images"

from oriental import ori, log, utils

from contracts import contract
import numpy as np
from scipy import spatial
from osgeo import ogr
ogr.UseExceptions()

from pathlib import Path
import contextlib

logger = log.Logger(__name__)

@contract
def _imageCorners( image ) -> 'array[4x2](float)':
    return np.array([ [           0,            0 ],
                      [           0, -image.nRows ],
                      [ image.nCols, -image.nRows ],
                      [ image.nCols,            0 ] ], float ) + ( -.5, .5 )


@contract
def forConstantHeight( camera, image, groundHeight : float, returnGsds : bool = False ) -> 'array[4x2]|tuple(array[4x2],array[4])|None':
    """assume an object plane of constant height
    """
    plane = ori.Plane()
    plane.offset = groundHeight
    plane.normal = np.array([0.,0.,1.])
    return forSlopedPlane( camera, image, plane, returnGsds )

@contract
def forSlopedPlane( camera, image, groundPlane : ori.Plane, returnGsds : bool = False ) -> 'array[4x2]|tuple(array[4x2],array[4])|None':
    pts = _imageCorners(image)
    pix2cam = getattr( image, 'pix2cam', None )
    if pix2cam is not None:
        pts = pix2cam.forward( pts )
    ori.distortion_inplace( pts, camera.ior, ori.adpParam2Struct( camera.adp ), ori.DistortionCorrection.undist )
    pts = np.c_[ pts, np.zeros(len(pts)) ]
    pts -= camera.ior
    R = ori.euler2matrix(image.rot)
    pts = R.dot(pts.T).T
    factors = ( image.prc.dot(groundPlane.normal) - groundPlane.offset ) /  -pts.dot(groundPlane.normal)
    if np.any( factors <= 0. ):
        return logger.warning( 'Back-projection of image corner is behind camera for image {}', image.path.name )
    ptsOnPlane = ( pts.T * factors ).T + image.prc
    if returnGsds:
        return ptsOnPlane[:,:2], factors
    return ptsOnPlane[:,:2]

@contract
def forObjPtsInView( camera, image, objPts : 'array[Ax3](float)' ) -> 'array[Bx2](float)|None':
    """transforms objPts into the image plane,
    intersects their convex hull with the image rectangle,
    reconstructs the depth of convex hull points,
    transforms them back into obj CS, and
    returns their convex hull in the X,Y-plane
    """
    # actually, we don't transform objPts into the image plane,
    # but we transform both the objPts and the image corners into the camera CS, and scale them to have z=-1
    ptsImg = _imageCorners(image)
    # we use a quadrangle, not the detailed outline. Undistorting the image corners doesn't make sense, but let's still do it.
    pix2cam = getattr( image, 'pix2cam', None )
    if pix2cam is not None:
        ptsImg = pix2cam.forward( ptsImg )
    ori.distortion_inplace( ptsImg, camera.ior, ori.adpParam2Struct( camera.adp ), ori.DistortionCorrection.undist )
    ptsImg = np.c_[ ptsImg, np.zeros(len(ptsImg)) ]
    ptsImg -= camera.ior
    ptsCam = ptsImg / camera.ior[2]
    ringCam = ogr.Geometry(ogr.wkbLinearRing)
    for pt in ptsCam:
        ringCam.AddPoint( *pt )
    ringCam.CloseRings()
    ringCam.FlattenTo2D()
    polygCam = ogr.Geometry(ogr.wkbPolygon)
    polygCam.AddGeometryDirectly(ringCam)
    assert polygCam.IsValid()

    R = ori.euler2matrix(image.rot)
    ptsObj = R.T.dot( (objPts - image.prc).T ).T
    ptsObj = ptsObj[ptsObj[:,2]<0] # discard points behind camera
    # scale the X,Y coordinates such that Z would be -1, but leave Z unchanged, so we can reconstruct it
    ptsObj[:,:2] = ( ptsObj[:,:2].T / ptsObj[:,2] ).T
    ch = spatial.ConvexHull( ptsObj[:,:2] ) # convex hull for x,y-coordinates only.
    # ConvexHull.vertices: Indices of points forming the vertices of the convex hull. For 2-D convex hulls, the vertices are in counterclockwise order.
    ptsObj = ptsObj[ch.vertices]
    ringObj = ogr.Geometry(ogr.wkbLinearRing)
    for pt in ptsObj:
        ringObj.AddPoint( *pt )
    ringObj.CloseRings()
    # don't flatten, but preserve the Z-coordinate. OGR will interpolate the Z-coordinates of the 2.5D polygon at intersection points of the 2 polygons
    polygObj = ogr.Geometry(ogr.wkbPolygon)
    polygObj.AddGeometryDirectly(ringObj)
    assert polygObj.IsValid()

    polygInterSect = polygObj.Intersection( polygCam )
    if polygInterSect is None or polygInterSect.IsEmpty():
        logger.warning( 'Sparse point cloud not at all visible in image {}',image.path.name )
        return
    assert polygInterSect.GetGeometryCount() == 1 # We expect a polygon without holes.
    ptsIntersect = []
    ringIntersect = polygInterSect.GetGeometryRef(0)
    for idx in range(ringIntersect.GetPointCount()):
        # GetPoint returns a tuple, not a Geometry
        ptsIntersect.append( ringIntersect.GetPoint(idx) )
    ptsIntersect = np.array(ptsIntersect, float)
    # use the preserved/interpolated Z-coordinate to reconstruct depth
    ptsIntersect[:,:2] = ( ptsIntersect[:,:2].T * ptsIntersect[:,2] ).T
    # rotate back into obj CS
    ptsIntersect = R.dot(ptsIntersect.T).T
    # compute the convex hull again, as different depths may affect it
    ch = spatial.ConvexHull( ptsIntersect[:,:2] )
    ptsIntersect = ptsIntersect[ch.vertices] + image.prc
    return ptsIntersect[:,:2].view(np.ndarray)

@contract
def exportShapeFile( fn : Path,
                     footprints : 'seq[A]($(array[Bx2](float)))',
                     names : 'seq[A](str)',
                     projCsWkt : str = '' ):
    with contextlib.suppress(FileExistsError):
        fn.parent.mkdir(parents=True)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource( str(fn) )
    layer = data_source.CreateLayer("footprints", geom_type=ogr.wkbPolygon)
    field_name = ogr.FieldDefn("ImageId", ogr.OFTString)
    field_name.SetWidth( max( len(name) for name in names ) )
    layer.CreateField(field_name)
    for footprint, name in utils.zip_equal( footprints, names ):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for pt in footprint:
            ring.AddPoint( *pt )
        ring.CloseRings()
        ring.FlattenTo2D() # even though we've constructed a 2D object, no matter if we call AddPoint(.) with 2 or 3 arguments, AddPoint(.) makes ring a 2.5D object!
        polyg = ogr.Geometry(ogr.wkbPolygon)
        polyg.AddGeometry(ring)
        assert polyg.IsValid()
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("ImageId", name )
        feature.SetGeometry(polyg)
        layer.CreateFeature(feature)
    data_source.SyncToDisk() # make sure that the file is okay, even if an exception is raised later on.

    # write a projection sidecar file. projCsWkt shall be the well-known text of a projected coordinate system. We don't check that here. 
    if projCsWkt:
        with fn.with_suffix('.prj').open('wt') as fout:
            fout.write( projCsWkt )
