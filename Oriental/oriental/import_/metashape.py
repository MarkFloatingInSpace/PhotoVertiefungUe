"""
Import a block from Agisoft MetaShape / PhotoScan
"""

import itertools, pkg_resources, sqlite3, struct, typing, zipfile
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from oriental import adjust, blocks, log, ori, ObservationMode, Progress, utils
import oriental.adjust.cost
import oriental.adjust.local_parameterization
import oriental.adjust.loss
import oriental.ori.transform
import oriental.utils.crs
import oriental.utils.db
import oriental.utils.filePaths

import numpy as np
from osgeo import ogr, osr

Sensor = namedtuple('Sensor', 'width height f cx cy b1 b2 k1 k2 k3 k4 p1 p2 p3 p4 make model')
Camera = namedtuple('Camera', 'sensorId path transform')

@dataclass
class ChunkTransform:
    scale : float
    rotation : np.ndarray
    translation : np.ndarray

    def toGlobal(self, ptLocal : np.ndarray) -> np.ndarray:
        return self.translation + self.scale * self.rotation @ ptLocal


logger = log.Logger('import')

def readPly( fileLike : typing.TextIO ):
    txt = fileLike.readline().decode().strip()
    assert txt == 'ply'
    txt = fileLike.readline().decode().strip()
    txt1, txt2, txt3 = txt.split()
    assert txt1 == 'format'
    if txt2 == 'binary_little_endian':
        endian = '<'
    elif txt2 == 'binary_big_endian':
        endian = '>'
    else:
        raise Exception(f'PLY format not understood: {txt2}')
    assert txt3 == '1.0'
    txt1, txt2, txt3 = fileLike.readline().decode().strip().split()
    assert txt1 == 'element' and txt2 == 'vertex'
    nPts = int(txt3)
    ply2struct = { 'uchar': 'B',
                   'int'  : 'i',
                   'uint' : 'I',
                   'float': 'f' } # char uchar short ushort int uint float double, or one of int8 uint8 int16 uint16 int32 uint32 float32 float64
    props = []
    while True:
        txt = fileLike.readline().decode().strip()
        if txt == 'end_header':
            break
        txt1, txt2, txt3 = txt.split()
        assert txt1 == 'property'
        fmt = ply2struct.get(txt2)
        if fmt is None:
            raise Exception(f'Format not understood: {txt2}')
        props.append(fmt)
    stru = struct.Struct( endian + ''.join(props) )
    for iPt in range(nPts):
        data = fileLike.read(stru.size)
        res = stru.unpack_from(data)
        yield res
    assert not len(fileLike.read(stru.size))  # end of file

def getDOM(zipFn : Path):
    with zipfile.ZipFile(zipFn) as zip_:
        with zip_.open('doc.xml') as doc:
            return ElementTree.parse(doc).getroot()

def getChunkFrame(pszFn: Path, chunkId):
    mainDOM = ElementTree.parse( pszFn ).getroot()
    projectFn = pszFn.parent / mainDOM.attrib['path'].format( projectname=pszFn.stem )
    projectDOM = getDOM(projectFn)
    allChunks = projectDOM.findall('chunks/chunk')
    if not allChunks:
        raise Exception('MetaShape project defines no chunks.')
    allChunkIds = ', '.join( el.attrib['id'] for el in allChunks )
    if chunkId is None:
        if len(allChunks) > 1:
            raise Exception(f'MetaShape project defines more than 1 chunk: {allChunkIds}. Explicitly specify the chunk to import using --chunk.')
        chunkId = int(allChunks[0].attrib['id'])
        logger.info(f'Importing chunk {chunkId}')
    myChunks = [ el for el in allChunks if int(el.attrib['id']) == chunkId ]
    if not myChunks:
        raise Exception(f'Chunk with ID {chunkId} not found. Available IDs: {allChunkIds}' )
    if len(myChunks) > 1:
        raise Exception(f'Chunk with ID {chunkId} matches {len(myChunks)} chunks. Available chunk IDs: {allChunkIds}' )
    chunkFn = projectFn.parent / myChunks[0].attrib['path']
    chunkDOM = getDOM(chunkFn)
    chunkTransform = chunkDOM.find('transform')
    trafo = ChunkTransform( rotation = np.array(chunkTransform.findtext('rotation').strip().split(), float).reshape((3,3)),
                            translation = np.array(chunkTransform.findtext('translation').strip().split(), float),
                            scale = float(chunkTransform.findtext('scale')) )

    frames = chunkDOM.findall('frames/frame')
    if not frames:
        raise Exception(f'{chunkFn} defines no frame.' )
    if len(frames)>1:
        raise Exception(f'{chunkFn} defines more than 1 frame, which is unsupported.')
    frameFn = chunkFn.parent / frames[0].attrib['path']
    frameDOM = getDOM(frameFn)

    return chunkDOM, frameDOM, frameFn, trafo

def getGeoc2Proj(chunkDOM : ElementTree.Element):
    geoc2proj = None
    referenceWkt = chunkDOM.findtext('reference')
    if referenceWkt:
        tgtCs = osr.SpatialReference()
        tgtCs.ImportFromWkt(referenceWkt)
        datum = tgtCs.GetAttrValue('DATUM')
        if datum is None:
            referenceWkt = None
        else:
            if datum.lower().replace('_',' ').replace('-',' ').startswith('militar geographische institut'):
                version = chunkDOM.get('version')
                if version and pkg_resources.parse_version(version) <= pkg_resources.parse_version('1.2.0'):
                    logger.info('At least until chunk document version 1.2.0, '
                                'PhotoScan used an outdated PROJ.4 version that ignores/strips TOWGS84 '
                                f'contained in the WKT for MGI. Hence, we will also assume for this chunk document {version} '
                                'that WGS84 and MGI datums are identical.')
                    tgtCs.SetTOWGS84( *([0.]*7) )
            wgs84 = osr.SpatialReference()
            wgs84.ImportFromEPSG(4978)  # WGS84 geocentric. Actually, this is an assumption, since I've found no documentation on which geocentric system PhotoScan specifically uses.
            geoc2proj = osr.CoordinateTransformation(wgs84, tgtCs)
        logger.info(f'Project CS is:\n{utils.crs.prettyWkt(tgtCs)}')

    return geoc2proj, referenceWkt

def getIor(s: Sensor) -> np.ndarray:
    ior = np.array([
        (s.width -1)/2 + s.cx,
        (s.height-1)/2 + s.cy,
        s.f])
    ior[1] *= -1
    return ior

def getSensors(chunkDOM : ElementTree.Element, frameDOM : ElementTree.Element) -> typing.Dict[int, Sensor]:
    for pc_sensor in chunkDOM.iterfind("sensors/sensor"):
        if pc_sensor.attrib['type'] != 'frame':
            logger.warning(f"Only frame cameras supported, but chunk defines a sensor of type {pc_sensor.attrib['type']}. Skipped." )
            continue
        calib = pc_sensor.find("calibration[@class='adjusted']") or pc_sensor.find("calibration")
        if calib is None:
            # Projects may contain sensors without calibration. P:\Projects\17_EINSICHT\07_Work_Data\20180618_Lavantal\Projekt\Lavanttal\Pfeiler1\Pfeiler1-Export.psx
            # contained 6 sensors, all with the same label and type="frame",
            # while all cameras only referenced 2 of those sensors.
            # The referenced sensors had a <calibration>, while the non-referenced had none.
            # Possibly, these non-referenced sensors (without <calibration>) end up in a PhotoScan project when a sub-set of a big block is exported.
            continue # No need to check here if this skipped sensor is referenced. Checking foreign keys below is sufficient.
        sensorId = int(pc_sensor.attrib['id'])
        firstCamera = chunkDOM.find(f"cameras/camera[@sensor_id='{sensorId}']")
        if firstCamera is None:
            continue
        firstCameraId = int(firstCamera.attrib['id'])
        pc_camera = frameDOM.find(f"cameras/camera[@camera_id='{firstCameraId}']")
        photo = pc_camera.find('photo')
        resolution = pc_sensor.find('resolution')
        args = {}
        for name in Sensor._fields:
            if name in 'width height'.split():
                value = int(resolution.attrib[name])
            elif name in 'make model'.split():
                elem = photo.find(f"meta/property[@name='Exif/{name.capitalize()}']")
                value = '' if elem is None else elem.attrib['value']
            else:
                value = calib.find(name)
                value = 0. if value is None else float(value.text)
            args[name] = value

        if calib.findtext('f') is None:
            # our PhotoScan-version 1.2.4: <chunk> without version-attribute
            # This version stores in XML fx, fy, skew,
            # but not: f, b1, b2.
            # Furthermore, cx and cy refer to the upper/left image corner, instead of to the image center.
            # Finally, p1 and p2 are swapped!
            # To get things right, compare XML contents to PhotoScan console, e.g.:
            # >>> chunk.sensors[0].calibration.f
            # I've double-checked this interpretation:
            # Setting (X, Y, Z) below in adpFromPhotoScan() to the marker coordinates in the camera CS,
            # yields the exact same image coordinates (u, v) as cam.project(chunk.markers[0].position) in the PhotoScan console.

            # Still, there is some flaw left: projecting a point from the camera system into the image
            # yields different results when done in the PhotoScan console and here:
            # P:\Projects\17_EINSICHT\07_Work_Data\20180625_FalkensteinII\b\Projekt_180623.psx

            # PhotoScan:
            # Bring marker position from chunk to camera CS:
            # ptCam=cam.transform.inv().mulp(chunk.markers[0].position)
            # # 2018-06-27 16:02:24 Vector([0.7594682471628621, -1.34290695302175, 5.734284254227696])
            # Project point from camera CS to image CS:
            # cam.sensor.calibration.project(ptCam)
            # # 2018-06-27 16:00:34 Vector([4645.39013671875, 1357.7900390625])
            # Doing it in 1 stop, yields the same image coordinates:
            # cam.project(chunk.markers[0].position)
            # # 2018-06-27 16:01:12 Vector([4645.39013671875, 1357.7900390625])

            # Now do it here:
            # Note: invert Y,Z of cam CS; invert x,y of image CS and add half a pixel, because PhotoScan's image CS origin is at the upper/left corner of the upper/left pixel.
            # ori.projection(np.array([0.7594682471628621, -1.34290695302175, 5.734284254227696]) * (1, -1, -1), np.zeros(3), np.zeros(3), np.array([x0, y0, z0]), adp) * (1, -1) + (.5, .5)
            # # array([4646.29925815, 1365.57414195])
            fx = float(calib.findtext('fx'))
            fy = float(calib.findtext('fy'))
            args['f'] = fy
            args['cx'] -= args['width']  / 2
            args['cy'] -= args['height'] / 2

            args['b1'] = fx - fy
            # P:\Projects\16_VOLTA\07_Work_Data\FBK_Dortmund_PS-project\Dortmund-PS_markers_matching.psx
            # lacks <skew>
            args['b2'] = float(calib.findtext('skew') or 0.)
            # These 2 are swapped! See XML vs. PhotoScan Console: >>> chunk.sensors[0].calibration.p1
            args['p1'], args['p2'] = args['p2'], args['p1']

        yield sensorId, Sensor(**args)

def getCameras(chunkDOM : ElementTree.Element, frameDOM : ElementTree.Element, frameFn : Path, sensors : typing.Dict[int, Sensor]):
    # A camera must both be enabled and must have been aligned. If it has been aligned, its node has a 'transform' child-node.
    # <chunk version="1.2.0" uses 0/1 for <camera enabled=?>,
    # while our PhotoScan-version 1.2.4 (older?) writes <chunk> without "version"-attribute, and it uses "true"/"false" for <camera enabled=?>
    # Simply query both (with one of the queries returning an empty list), and merge the returned lists.
    for pc_camera in itertools.chain.from_iterable(
            chunkDOM.iterfind(f"cameras/camera[@enabled='{enabled}'][transform]") for enabled in ('true', '1')):
        pc_cameraId = int(pc_camera.attrib['id'])
        sensorId = int(pc_camera.attrib['sensor_id'])
        photo = frameDOM.find(f"cameras/camera[@camera_id='{pc_cameraId}']/photo")
        path = (frameFn.parent / photo.attrib['path']).resolve()
        if sensorId not in sensors:
            logger.warning(f'Image with unsupported sensor found: {path}. Skipping.')
            continue
        transform = np.array(pc_camera.find('transform').text.split(), float).reshape((4, 4))
        assert (transform[3, :] == [0, 0, 0, 1]).all()
        yield pc_cameraId, Camera(
            sensorId=sensorId,
            path=path,
            transform=transform[:3, :])

def getMarkers(chunkDOM : ElementTree.Element):
    # Note: IDs of PhotoScan markers and tie points overlap!
    # i.e. a tie point may have the same ID as a marker (but with very different coordinates, generally).
    # PhotoScan does not seem to store adjusted positions for markers, but only their reference coordinates.
    # Hence, let SQLite choose a new ID when inserting a marker object point,
    # and use that new ID when inserting the according image points.
    markerId2Label_RefObjPt = {}
    markersWithMissingReference = []
    for marker in chunkDOM.iterfind('markers/marker'):
        label = marker.attrib['label']
        reference = marker.find('reference')
        if reference is None:
            markersWithMissingReference.append(label)
            continue
        assert bool(reference.attrib['enabled']) # we should use disabled reference coordinates as check points!
        coo = np.array([reference.attrib[coo] for coo in 'x y z'.split()], float)
        markerId2Label_RefObjPt[int(marker.attrib['id'])] = label, coo

    if markersWithMissingReference:
        logger.warning('Marker elements {} have no reference coordinates.'.format(', '.join(f'{label!r}' for label in markersWithMissingReference)))
    logger.info(f'Project defines {len(markerId2Label_RefObjPt)} markers with reference coordinates.')
    return markerId2Label_RefObjPt

def projected(sensor : Sensor, camPt : np.ndarray) -> np.ndarray:
    """
    From the manual of Agisoft MetaShape 1.5.1:

    Appendix C. Camera models
    Agisoft Metashape supports several parametric lens distortion models. Specific model which approximates best a real distortion field must be selected before processing. All models assume a central projection camera. Non-linear distortions are modeled using Brown's distortion model.

    A camera model specifies the transformation from point coordinates in the local camera coordinate system to the pixel coordinates in the image frame.

    The local camera coordinate system has origin at the camera projection center. The Z axis points towards the viewing direction, X axis points to the right, Y axis points down.

    The image coordinate system has origin at the top left image pixel, with the center of the top left pixel having coordinates (0.5, 0.5). The X axis in the image coordinate system points to the right, Y axis points down. Image coordinates are measured in pixels.

    Equations used to project a points in the local camera coordinate system to the image plane are provided below for each supported camera model.

    The following definitions are used in the equations:

    (X, Y, Z) - point coordinates in the local camera coordinate system,

    (u, v) - projected point coordinates in the image coordinate system (in pixels),

    f - focal length,

    cx, cy - principal point offset,

    K1, K2, K3, K4 - radial distortion coefficients,

    P1, P2, P3, P4 - tangential distortion coefficients,

    B1, B2 - affinity and non-orthogonality (skew) coefficients,

    w, h - image width and height in pixels.

    Frame cameras
    x = X / Z

    y = Y / Z

    r = sqrt(x^2 + y^2)

    x' = x(1 + K1 r^2 + K2 r^4 + K3 r^6 + K4 r^8) + (P1(r^2 + 2 x^2) + 2 P2 x y)(1 + P3 r^2 + P4 r^4)

    y' = y(1 + K1 r^2 + K2 r^4 + K3 r^6 + K4 r^8) + (P2(r^2 + 2 y^2) + 2 P1 x y)(1 + P3 r^2 + P4 r^4)

    u = w * 0.5 + cx + x' f + x' B1 + y' B2

    v = h * 0.5 + cy + y' f
    """
    X, Y, Z = camPt

    x = X / Z
    y = Y / Z

    x2 = x ** 2
    y2 = y ** 2
    r2 = x2 + y2
    r4 = r2 ** 2
    r6 = r2 * r4
    r8 = r4 ** 2

    s = sensor
    kFac = ( 1 +
             s.k1 * r2 +
             s.k2 * r4 +
             s.k3 * r6 +
             s.k4 * r8 )
    p34Fac = 1 + s.p3 * r2 + s.p4 * r4
    two_xy = 2 * x * y
    xp = x * kFac + (s.p1 * (r2 + 2 * x2) + s.p2 * two_xy) * p34Fac
    yp = y * kFac + (s.p2 * (r2 + 2 * y2) + s.p1 * two_xy) * p34Fac

    u = s.width  / 2 + s.cx + xp * s.f + xp * s.b1 + yp * s.b2
    v = s.height / 2 + s.cy + yp * s.f
    return np.array([u, v])

def importMetaShape( pszFn : Path, chunkId : typing.Union[int, None], db : sqlite3.Connection, dbDir : Path ):
    """ PhotoScan manual, appendix C:
    A camera model specifies the transformation from point coordinates in the local camera coordinate system
    to the pixel coordinates in the image frame.
    The local camera coordinate system has origin at the camera projection center. The Z axis points towards
    the viewing direction, X axis points to the right, Y axis points down.
    The image coordinate system has origin at the top left image pixel, with the center of the top left pixel
    having coordinates (0.5, 0.5). The X axis in the image coordinate system points to the right, Y axis points
    down. Image coordinates are measured in pixels.

    http://www.agisoft.com/forum/index.php?topic=1557.msg8152#msg8152
    The transformation matrices that are produced by using the Tools/"Export Cameras..." option will transform from camera local space to model global space.
    Doing a matrix multiply of the transformation matrix times the origin vector [0,0,0,1] results in the camera's location in model global space.
    """

    chunkDOM, frameDOM, frameFn, chunkTrafo = getChunkFrame(pszFn, chunkId)

    geoc2proj, referenceWkt = getGeoc2Proj(chunkDOM)
    if referenceWkt is not None:
        db.execute(f"""INSERT INTO config ( name, value )
                       VALUES ( '{utils.db.ConfigNames.CoordinateSystemWkt}', ? )""",
                   (referenceWkt.strip(),))

    sensors = dict(getSensors(chunkDOM, frameDOM))
    def importCameras(sensors):
        for sensorId, s in sensors.items():
            ior = getIor(s)
            normalizationRadius = (s.width**2 + s.height**2)**.5 / 3

            def adpFromPhotoScan():
                block = adjust.Problem()
                loss = adjust.loss.Trivial()
                prc = np.zeros(3)
                angles = np.zeros(3)
                adp = adjust.parameters.ADP(normalizationRadius=normalizationRadius)

                step = 20
                Z = ior[2]
                for X in range(int(-s.width // 2), int(s.width // 2 + 1), step):
                    for Y in range(int(-s.height // 2), int(s.height // 2 + 1), step):
                        u, v = projected(s, np.array([X, Y, Z]))
                        xOri = u - .5
                        yOri = -(v - .5)
                        cost = adjust.cost.PhotoTorlegard(xOri, yOri)
                        objPt = np.array([X, -Y, -Z])
                        block.AddResidualBlock(cost, loss, prc, angles, ior, adp, objPt)
                        block.SetParameterBlockConstant(objPt)

                block.SetParameterBlockConstant(prc)
                block.SetParameterBlockConstant(angles)
                block.SetParameterBlockConstant(ior)
                locPar = adjust.local_parameterization.Subset(adp.size, [int(adjust.PhotoDistortion.optPolynomRadial11)])
                block.SetParameterization(adp, locPar)
                solveOpts = adjust.Solver.Options()
                summary = adjust.Solver.Summary()
                adjust.Solve(solveOpts, block, summary)
                assert adjust.isSuccess(summary.termination_type)
                return adp

            adp = adpFromPhotoScan()
            #adp = adjust.parameters.ADP(normalizationRadius=normalizationRadius)
            # TODO either convert to OrientAL ADP,
            # or introduce a new ADP-model. Store covariance.
            adpEnumerators = [ enumerator for val, enumerator in sorted( adjust.PhotoDistortion.values.items(), key=lambda x: x[0] ) ]
            sql = '''
                INSERT INTO cameras (id, make, model, isDigital, x0, y0, z0, reference, normalizationRadius, {} )
                VALUES( ?, ?, ?, 1, ?, ?, ?, 'principalPoint', ?, {} )
                '''.format(','.join(str(el) for el in adpEnumerators),
                           ','.join(['?']*len(adpEnumerators)))
            db.execute(
                sql,
                ( sensorId, s.make, s.model, *ior, normalizationRadius, *adp ) )
        logger.info(f'{len(sensors)} cameras imported.')

    importCameras(sensors)

    cameras = dict(getCameras(chunkDOM, frameDOM, frameFn, sensors))

    def importImages():
        for cameraId, camera in cameras.items():
            photoPath = utils.filePaths.relPathIfExists( camera.path, start=dbDir )
            Rx200 = np.diag((1.,-1.,-1.))
            R = ( Rx200 @ camera.transform[:3,:3].T @ chunkTrafo.rotation.T ).T
            angles = ori.matrix2euler( R )
            prc = chunkTrafo.toGlobal(camera.transform[:3,3])
            if geoc2proj is not None:
                class Image:
                    pass
                image = Image()
                image.prc = prc
                image.rot = angles
                ori.transform.transformEOR(geoc2proj, [image])

            sensor = sensors[camera.sensorId]
            db.execute(f'''
                INSERT INTO images (id, camId, path, nCols, nRows, X0, Y0, Z0, r1, r2, r3, parameterization )
                VALUES( {','.join(['?']*12)} )''',
                ( cameraId,
                  camera.sensorId,
                  str(photoPath),
                  sensor.width,
                  sensor.height,
                  *prc,
                  *angles,
                  str(angles.parametrization)
                  ) )

        logger.info(f'{len(cameras)} images imported.')

    importImages()

    pointCloudFn = frameFn.parent / frameDOM.find('point_cloud').attrib['path']
    pointCloudDOM = getDOM(pointCloudFn)

    with zipfile.ZipFile(pointCloudFn) as pointCloudZip:
        pointZWkb = struct.Struct('<bIddd')
        with pointCloudZip.open(pointCloudDOM.find('points').attrib['path']) as points:
            # points.ply contains x/y/z/id only
            def getPoints():
                ptOgr = ogr.Geometry(ogr.wkbPoint)
                for X, Y, Z, ptId in readPly(points):
                    pChunk = np.array([X, Y, Z],float)
                    pGlobal = chunkTrafo.toGlobal(pChunk)
                    if geoc2proj is not None:
                        ptOgr.SetPoint(0, *pGlobal)
                        ptOgr.Transform(geoc2proj)
                        pGlobal[:] = ptOgr.GetPoint()
                    # 1001 is the code for points with z-coordinate
                    yield ptId, pointZWkb.pack(1, 1001, *pGlobal)

            nObjPtsInserted = db.executemany("""
                INSERT INTO objpts( id, pt )
                VALUES( ?, GeomFromWKB(?, -1) )""",
                ( getPoints()) ).rowcount

        logger.info(f'{nObjPtsInserted} tie object points imported.')

        projections = pointCloudDOM.findall('projections')

        progress = Progress(len(projections))
        nImgPtsInserted = 0
        for projection in projections:
            cameraId = int(projection.attrib['camera_id'])
            if cameraId not in cameras:
                progress += 1
                continue
            #nImgPts = int(projection.attrib['count'])
            #logger.info(f'Import {nImgPts} image points for image {cameraId}')
            with pointCloudZip.open(projection.attrib['path']) as imgObs:
                # each p??.ply contains x/y/size/id
                nImgPtsInserted += db.executemany("""
                    INSERT INTO imgObs(imgId,x,y,objPtId,diameter)
                    VALUES( ?,?,?,?,? )""",
                    ((cameraId, x-.5, -(y-.5), objPtId, size) for (x, y, size, objPtId) in readPly(imgObs))).rowcount
            progress += 1
        logger.info(f'{nImgPtsInserted} tie image points imported.')

        markerId2Label_RefObjPt = getMarkers(chunkDOM)

        def importMarkers():
            nInsertedCtrlObjPts = 0
            nInsertedCtrlImgPts = 0
            for marker in frameDOM.iterfind('markers/marker'):
                imgObsImgIds = []
                for location in marker.iterfind('location'):
                    # location.attrib['pinned'] only means that the image point position shall not be refined using LSM.

                    # mail M. Kölle 2020-02-21
                    # valid:   <location camera_id="132" pinned="1" x="747.148" y="1049.54"/>
                    # invalid: <location camera_id="133" pinned="0" valid="0"/>
                    if location.get('valid', '1') != '1':
                        continue
                    
                    xOri = float(location.attrib['x']) - .5
                    yOri = -(float(location.attrib['y']) - .5)
                    imgId = int(location.attrib['camera_id'])

                    if not db.execute('''
                        SELECT count(*)
                        FROM images
                        WHERE id = ? ''',
                        (imgId,)).fetchone()[0]:
                        continue  # Marker (image point of a control point) has been observed in an image that could not be oriented.

                    imgObsImgIds.append((xOri, yOri, imgId))

                if not imgObsImgIds:
                    continue

                markerId = int(marker.attrib['marker_id'])
                try:
                    label, refObjPt = markerId2Label_RefObjPt[markerId]
                except KeyError:
                    raise Exception(f'No reference coordinates defined for marker with ID {markerId}')

                if not nInsertedCtrlObjPts:
                    db.execute("""
                        SELECT AddGeometryColumn(
                            'objpts', -- table
                            'refPt',  -- column
                            -1,       -- srid -1: undefined/local cartesian cooSys
                            'POINT',  -- geom_type
                            'XYZ',    -- dimension
                            0         -- NOT NULL
                        )""")

                refObjPtWkb = pointZWkb.pack(1, 1001, *refObjPt)
                objPtId = db.execute("""
                    INSERT INTO objpts( pt, refPt )
                    VALUES( GeomFromWKB(?, -1), GeomFromWKB(?, -1) ) """,
                    (refObjPtWkb, refObjPtWkb)).lastrowid
                nInsertedCtrlObjPts += 1

                nInsertedCtrlImgPts += db.executemany(f"""
                    INSERT INTO imgObs(imgId, name, x, y, objPtId, type)
                    VALUES(?, ?, ?, ?, ?, {int(ObservationMode.manual)})""",
                           ((imgId, label, xOri, yOri, objPtId) for xOri, yOri, imgId in imgObsImgIds)).rowcount

            logger.info(f'{nInsertedCtrlObjPts} marker object points, {nInsertedCtrlImgPts} marker image points imported.')

        importMarkers()

        def deleteUnusedImgPts():
            nRowsBefore, = db.execute('SELECT count(*) from imgObs').fetchone()
            nRowsDeleted = db.execute("""
                DELETE
                FROM imgObs
                WHERE imgObs.objPtId NOT IN (SELECT id FROM objPts) """).rowcount
            logger.info(f'{nRowsDeleted} out of {nRowsBefore} image points deleted that do not reference an object point.')

        deleteUnusedImgPts()

        def assignColor():
            db.execute("""
                CREATE TEMP TABLE tracks(
                    id  INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    red INTEGER,
                    green INTEGER,
                    blue INTEGER ) """)
            # tracks.ply contains red/green/blue only
            with pointCloudZip.open(pointCloudDOM.find('tracks').attrib['path']) as tracks:
                nTracksImported = db.executemany("""
                    INSERT INTO tracks(id, red, green, blue)
                    VALUES( ?, ?, ?, ? )""",
                    ((trackId, red/255, green/255, blue/255) for (trackId,(red, green, blue)) in enumerate(readPly(tracks)))).rowcount

            nRowsUpdated = db.execute("""
                UPDATE imgObs
                SET (red, green, blue) = (
                  SELECT tracks.red, tracks.green, tracks.blue 
                  FROM tracks
                  WHERE tracks.id = imgObs.objPtId ) """).rowcount
            logger.info(f'Point colours assigned to {nRowsUpdated} image points.')
            db.execute('DROP TABLE tracks') # Not necessary, since it is a temp-table.

        assignColor()
