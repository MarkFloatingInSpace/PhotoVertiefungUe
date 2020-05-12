"""
Transform a block using control object and image points.
"""

import argparse, csv, os, shutil, struct, sys
from collections import abc, namedtuple
from contextlib import suppress
from pathlib import Path
from sqlite3 import dbapi2

from oriental import adjust, log, ObservationMode, ori, utils
import oriental.adjust.cost
import oriental.adjust.loss
import oriental.ori.transform
import oriental.utils.argparse
import oriental.utils.gdal
import oriental.utils.db
import oriental.utils.stats

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
gdal.UseExceptions()
ogr.UseExceptions()
import cv2
from contracts import contract

logger = log.Logger("transform")

@contract
def parseArgs( args : 'list(str)|None' = None ):
    docList = __doc__.splitlines()
    parser = argparse.ArgumentParser( description=docList[0],
                                      epilog='\n'.join( docList[1:] ),
                                      formatter_class=utils.argparse.Formatter )

    parser.add_argument( '--outDir', default=Path.cwd() / "transform", type=Path,
                         help='Store results in directory OUTDIR.' )
    parser.add_argument( '--outDb', type=Path,
                         help='Store the transformed block in OUTDB. Default: OUTDIR/transform.sqlite' )
    parser.add_argument( '--inDb', type=Path,
                         help='Transform the block defined in INDB. Default: OUTDIR/../relOri/relOri.sqlite' )
    parser.add_argument( '--objectPoints', type=Path, required=True,
                         help='Read the object point coordinates of the corresponding points from this file.' )
    parser.add_argument( '--imagePoints', type=Path, required=True,
                         help='Read the image point coordinates of the corresponding points from this file.' )
    parser.add_argument( '--cs',
                         help="Coordinate system of OBJECTPOINTS (e.g. 'EPSG:31253'). Taken over from OBJECTPOINTS, if unspecified. 'MGI' chooses an appropriate MGI meridian strip. 'UTM' chooses an appropriate UTM zone." )


    utils.argparse.addLoggingGroup( parser, "transformLog.xml" )

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
        args.outDb = args.outDir / 'transform.sqlite'
    if not args.inDb:
        args.inDb = args.outDir / '..' / 'relOri' / 'relOri.sqlite'
    args.inDb = args.inDb.resolve(strict=True)

    cameras, images, objPts = restore( args.inDb )
    if args.objectPoints.suffix.lower() == '.shp':
        cs, controlObjPts = controlObjFromOgr(args.objectPoints)
    else:
        controlObjPts = controlObjFromCsv( args.objectPoints )
    if args.imagePoints.suffix.lower() == '.sqlite':
        controlImgPts = controlImgFromDb(args.imagePoints, images.keys())
    else:
        controlImgPts = controlImgFromText( args.imagePoints, images.keys() )
    trafo = transform( controlImgPts, controlObjPts, images, objPts )
    logger.info('''Computed transformation: P_g = A P_l + t
with
P_g ... point in coordinate system of control points
P_l ... point in local/input coordinate system\v
Parameters
A\t\t\tt
''' + '\n'.join( '\t'.join( '{:.6f}'.format(el) for el in A_.tolist() + [t_] ) for (A_,t_) in utils.zip_equal( trafo.A, trafo.t ) ) )
    projections = project(images, controlImgPts, controlObjPts)
    shutil.copyfile( args.inDb, args.outDb )
    # TODO nicht beobachtete control object points wieder entfernen.
    # TODO CS speichern.
    saveDb( args.outDb, images, objPts, controlObjPts, projections )
    # Open outDb in MonoScope, and either delete the projected control object points (because they are not identifiable),
    # or move them into the right position.

ObjectPoint = namedtuple('ObjectPoint', 'name coords beschreibung')

@contract
def controlObjFromOgr( shapeFn : Path ):

    #driver = ogr.GetDriverByName('ESRI Shapefile')
    #dataSet = driver.Open( str(shapeFn), 0)  # 0 means read-only. 1 means writeable.
    dataSet = gdal.OpenEx( str(shapeFn), gdal.OF_VECTOR)
    layer = dataSet.GetLayer()
    if 0:
        layerDfn = layer.GetLayerDefn()
        for i in range(layerDfn.GetFieldCount()):
            name = layerDfn.GetFieldDefn(i).GetName()
            typeCode = layerDfn.GetFieldDefn(i).GetType()
            typeName = layerDfn.GetFieldDefn(i).GetFieldTypeName(typeCode)
            print(name, typeName)

    controlObjPts = {}
    for feature in layer:
        geom = feature.GetGeometryRef()
        #print( geom.ExportToWkt() )
        name = feature.GetField('pkt_nr')
        controlObjPts[name] = ObjectPoint(
            name,
            np.array( [ geom.GetX(), geom.GetY(), feature.GetField('hoehe') ], float ),
            feature.GetField('ppbeschrei'))
    return layer.GetSpatialRef().ExportToWkt(), controlObjPts

@contract
def controlObjFromCsv( csvFn : Path ):
    controlObjPts = {}
    with open( csvFn, newline='' ) as fin:
        dialect = csv.Sniffer().sniff(fin.read(1024))
        fin.seek(0)
        dialect.skipinitialspace = True
        reader = csv.reader(fin, dialect)
        for row in reader:
            if not row:
                continue
            if row[0].lstrip().startswith('#'):
                continue
            name, X, Y, Z = row
            controlObjPts[name] = ObjectPoint(
                name,
                np.array( [ X, Y, Z ], float ),
                beschreibung=None )
    return controlObjPts

ImagePoint = namedtuple('ImagePoint', 'ptName imgName coords')

@contract
def controlImgFromText( imgObsFn : Path, imagePaths : abc.Iterable ):
    imagePaths = {el.stem.lower() : el for el in imagePaths}
    controlImgPts = {}
    with open(imgObsFn, newline='') as fin:
        reader = csv.reader(fin, delimiter=';')
        for row in reader:
            if not row or row[0].lstrip().startswith('#'):
                continue
            if len(row) == 1:
                img = imagePaths[ Path(row[0]).stem.lower() ]
                info, = utils.gdal.imread( str(img), skipData=True )
                shape = info.nRows, info.nCols
            else:
                ptName, x, y = row
                # For AdV/Bonn, gm rotated all images by 180° for observing image point coordinates.
                if 0:
                    x = (shape[1]-1) - float(x)
                    y = (shape[0]-1) - float(y)
                    logger.warning('Control image points rotated by 180°!')
                controlImgPts.setdefault( ptName, [] ).append(
                    ImagePoint( ptName, img, np.array( [ float(x), -float(y) ] ) )
                )
    return controlImgPts

@contract
def controlImgFromDb( imgObsFn : Path, imagePaths : abc.Iterable ):
    controlImgPts = {}
    imagePaths = {el.stem.lower(): el for el in imagePaths}
    with dbapi2.connect( utils.db.uri4sqlite(imgObsFn) + '?mode=ro', uri=True ) as db:
        utils.db.initDataBase(db)
        rows = db.execute('''
            SELECT imgObs.name, imgObs.x, imgObs.y, images.path
            FROM imgObs
            JOIN images ON imgObs.imgId == images.id
            WHERE type IN ( ?, ? )
        ''', ( int(ObservationMode.manual), int(ObservationMode.lsm) ) )
        for row in rows:
            ptName = row['name']
            imgPath = imagePaths[Path(row['path']).stem.lower()]
            coords = np.array([row['x'], row['y']],float)
            imgPt = ImagePoint( ptName, imgPath, coords )
            controlImgPts.setdefault( ptName, [] ).append( imgPt )

    return controlImgPts


@contract
def restore( inDbFn : Path ):
    Camera = namedtuple('Camera',['id','ior','adp'])
    cameras = {}
    Image = namedtuple('Image','id path prc rot nRows nCols pix2cam mask_px camera')
    images = {}
    objPts = {}
    with dbapi2.connect( utils.db.uri4sqlite(inDbFn) + '?mode=ro', uri=True ) as db:
        utils.db.initDataBase(db)
        rows = db.execute("""
            SELECT id, x0, y0, z0,
                   reference, normalizationRadius, {}
            FROM cameras """.format(
            ', '.join((str(val[1]) for val in sorted(adjust.PhotoDistortion.values.items(), key=lambda x: x[0]))) )
        )
        for row in rows:
            v_adp = []
            for val, enumerator in sorted(adjust.PhotoDistortion.values.items(), key=lambda x: x[0]):
                v_adp.append(row[str(enumerator)])
            adp = adjust.parameters.ADP(normalizationRadius=row['normalizationRadius'],
                                        referencePoint=adjust.AdpReferencePoint.names[row['reference']],
                                        array=np.array(v_adp,float))
            cameras[row['id']] = Camera(row['id'],
                                        np.array([row['x0'], row['y0'], row['z0']],float),
                                        adp )
        fiducialMatAttrs = [ 'fiducial_{}'.format(el) for el in 'A00 A01 A10 A11'.split() ]
        fiducialVecAttrs = [ 'fiducial_{}'.format(el) for el in 't0 t1'          .split() ]
        for row in db.execute("""
            SELECT id, camId, path, X0, Y0, Z0, r1, r2, r3, parameterization, nRows, nCols, AsBinary(mask) AS mask, {}, {}
            FROM images """.format(','.join(fiducialMatAttrs),
                                   ','.join(fiducialVecAttrs))):
            A = np.array([row[el] for el in fiducialMatAttrs], float).reshape((2, 2))
            t = np.array([row[el] for el in fiducialVecAttrs], float)
            if np.isfinite(A).all() and np.isfinite(t).all():
                pix2cam = ori.transform.AffineTransform2D(A, t)
            else:
                pix2cam = ori.transform.IdentityTransform2D()
            mask_px = None
            if row['mask'] is not None:
                polyg = ogr.Geometry(wkb=row['mask'])
                ring = polyg.GetGeometryRef(0)
                nPts = ring.GetPointCount()
                mask_px = np.empty((nPts, 2))
                for iPt in range(nPts):
                    mask_px[iPt, :] = ring.GetPoint_2D(iPt)
            path = Path(row['path'])
            if not path.is_absolute():
                path = inDbFn.parent / path

            rot = adjust.parameters.EulerAngles( parametrization=adjust.EulerAngles.names[row['parameterization']],
                                                 array=np.array( [row['r1'], row['r2'], row['r3']], float ) )
            images[path] = Image(row['id'],
                                 path,
                                 np.array( [row['X0'], row['Y0'], row['Z0']], float),
                                 rot,
                                 row['nRows'],
                                 row['nCols'],
                                 pix2cam,
                                 mask_px,
                                 cameras[row['camId']])
        for row in db.execute('''
            SELECT id, X(pt) AS X, Y(pt) AS Y, Z(pt) AS Z
            FROM objPts '''):
            objPts[row['id']] = np.array([row[name] for name in 'X Y Z'.split()], float)
    return cameras, images, objPts


@contract
def transform( controlImgPts : dict, controlObjPts : dict, images : dict, objPts : dict ):
    CamParams = namedtuple('CamParams', 'R t ior adp')
    Correspondence = namedtuple('Correspondence', 'name glob loc')
    ResidualStat = namedtuple('ResidualStat', 'name min med max')
    correspondences = []
    residuals = []

    for ptName, imgObservations in controlImgPts.items():
        controlObjPt = controlObjPts[ptName]
        if controlObjPt.beschreibung == 'Bodenpunkt':
            continue
        if len(imgObservations) < 2:
            continue
        # TODO search for best intersection angle
        imgs = [ images[theImgObs.imgName] for theImgObs in imgObservations ]
        imgPts = [ img.pix2cam.forward(imgObs.coords) for (img,imgObs) in utils.zip_equal(imgs,imgObservations) ]
        camParams = [ CamParams( ori.euler2matrix(img.rot), img.prc, img.camera.ior, img.camera.adp ) for img in imgs ]
        X = ori.triangulatePoints( *[pt[np.newaxis,:] for pt in imgPts[:2]], *camParams[:2] ).ravel()
        problem = adjust.Problem()
        loss = adjust.loss.Trivial()
        for imgPt, img in utils.zip_equal( imgPts, imgs ):
            cost = adjust.cost.PhotoTorlegard(*imgPt)
            problem.AddResidualBlock(cost,
                                     loss,
                                     img.prc,
                                     img.rot,
                                     img.camera.ior,
                                     img.camera.adp,
                                     X )
            for par in (img.prc, img.rot, img.camera.ior, img.camera.adp):
                problem.SetParameterBlockConstant(par)
        options = adjust.Solver.Options()
        options.linear_solver_type = adjust.LinearSolverType.DENSE_QR
        summary = adjust.Solver.Summary()
        adjust.Solve(options, problem, summary)
        assert adjust.isSuccess(summary.termination_type)
        resids = problem.Evaluate()[0].reshape((-1,2))
        residuals.append( ResidualStat(ptName, *utils.stats.minMedMax( np.sum( resids**2, axis=1 )**.5 ) ) )
        correspondences.append( Correspondence(
            ptName,
            controlObjPt.coords,
            X ))

    residuals.sort( key=lambda el: el.name )
    logger.info('Control point triangulation image residual norms\n'
                'Name\tmin\tmedian\tmax\n' +
                '\n'.join( f'{el.name}\t{el.min:.3f}\t{el.med:.3f}\t{el.max:.3f}' for el in residuals ) )

    loc = np.array([el.loc  for el in correspondences], float)
    glob = np.array([el.glob for el in correspondences], float )
    trafo = ori.transform.AffineTransform3D.computeSimilarity( source=loc,
                                                               target=glob )
    resNorms = np.sum((glob - trafo.forward(loc)) ** 2, axis=1) ** .5
    logger.info('Transformation residual norms [object space units]:\n'
                'Name\tresidual norm\n' +
                '\n'.join( f'{corresp.name}\t{resNorm:.3f}' for corresp, resNorm in utils.zip_equal( correspondences, resNorms )  ) )
    #logger.info('Transformation residual norms [object space units]:\n' +
    #            'statistic\tvalue\n' +
    #            '\n'.join('{}\t{:.3}'.format(name, value) for name, value in [('median', np.median(resNorms)),
    #                                                                          ('mean', np.mean(resNorms)),
    #                                                                          ('max', np.max(resNorms))]))
    ori.transform.transformEOR(trafo, images.values(), objPts.values())
    return trafo


@contract
def project( images : dict, controlImgPts : dict, controlObjPts : dict ):
    # Project all controlObjPts into each image. If inside the image area and no controlImgPt exists yet, then insert it.
    Projection = namedtuple('Projection', 'ptName imgName gm coords')
    projections = {}
    plot = False
    nBodenpunkte = 0
    for imgFn, image in images.items():
        projs = []
        obs = []
        for ptName, controlObjPt in controlObjPts.items():
            if controlObjPt.beschreibung == 'Bodenpunkt':
                nBodenpunkte += 1
                continue
            ctrlImgPts = controlImgPts.get(ptName)
            observed = False
            if ctrlImgPts is not None:
                for ctrlImgPt in ctrlImgPts:
                    if ctrlImgPt.imgName == imgFn:
                        observed = True
                        obs.append((ptName, ctrlImgPt.coords ))
                        break
            if not observed:
                projs.append((ptName, controlObjPt.coords))
        proj = np.array( [el[1] for el in projs], float )
        proj = ori.projection( proj, image.prc, image.rot, image.camera.ior, image.camera.adp )
        proj = image.pix2cam.inverse( proj )
        sel = np.logical_and(
                np.logical_and( proj[:,0] >= -.5, proj[:,0] <=  image.nCols - .5 ),
                np.logical_and( proj[:,1] <=  .5, proj[:,1] >= -image.nRows + .5) )
        proj = proj[sel]
        names = [ el[0] for idx,el in enumerate(projs) if sel[idx] ]
        sel = np.zeros( len(proj), np.bool )
        for idx, p in enumerate(proj):
            sel[idx] = cv2.pointPolygonTest( image.mask_px.astype(np.float32), tuple(p), measureDist=False ) != -1 # 1:inside, 0:on edge
        proj = proj[sel]
        names = [el for idx, el in enumerate(names) if sel[idx]]
        if plot:
            plt.figure(1,clear=True)
            img = utils.gdal.imread(imgFn, depth=utils.gdal.Depth.u8 )
            plt.imshow( img, cmap='gray', interpolation='nearest' )
            plt.plot( proj[:,0], -proj[:,1], '.c' )
        for name, (x,y) in zip(names,proj):
            projections.setdefault( imgFn, [] ).append(
                Projection( name, imgFn, False, np.array([x,y],float) ))
            if plot:
                plt.text(x, -y, name)
        if len(obs):
            if plot:
                pts = np.atleast_2d( np.array([el[1] for el in obs],float))
                plt.plot( pts[:,0], -pts[:,1], '.r')
            for name, (x,y) in obs:
                projections.setdefault( imgFn, [] ).append(
                    Projection(name, imgFn, True, np.array([x,y],float) ))
                if plot:
                    plt.text(x, -y, name)
    if nBodenpunkte:
        logger.info('{} Bodenpunkte nicht exportiert, da vermutlich instabile Hoehe und schwer zu digitalisieren'.format(nBodenpunkte))
    return projections

@contract
def saveDb( outDbFn : Path, images : dict, objPts : dict, controlObjPts : dict, projections : dict ):
    with dbapi2.connect( utils.db.uri4sqlite(outDbFn) + '?mode=rw', uri=True ) as db:
        utils.db.initDataBase(db)
        db.executemany('''
            UPDATE images
            SET X0 = ?,
                Y0 = ?,
                Z0 = ?,
                r1 = ?,
                r2 = ?,
                r3 = ?
            WHERE id = ?
        ''', ( (*image.prc,*image.rot,image.id)
               for image in images.values() ) )

        pointZWkb = struct.Struct('<bIddd')
        def packPointZ(pt):
            # 1001 is the code for points with z-coordinate
            return pointZWkb.pack( 1, 1001, *pt )

        db.executemany('''
            UPDATE objPts
            SET pt = GeomFromWKB(?, -1)
            WHERE id = ?
        ''', ( ( packPointZ(objPt), objPtId )
               for objPtId, objPt in objPts.items() ) )

        controlObjPtName2Id = {}
        for ptName, objPt in controlObjPts.items():
            controlObjPtName2Id[ptName] = db.execute('''
                INSERT INTO objPts(pt)
                VALUES( GeomFromWKB(?,-1) )
            ''', (packPointZ(objPt.coords),) ).lastrowid

        imgPath2id = { img.path : img.id for img in images.values() }
        for imgFn, projs in projections.items():
            imgId = imgPath2id[imgFn]
            db.executemany('''
                INSERT INTO imgObs( imgId, name, x, y, objPtId, type )
                VALUES( ?, ?, ?, ?, ?, ? )
            ''', ( ( imgId, proj.ptName, *proj.coords, controlObjPtName2Id[proj.ptName],
                     int(ObservationMode.manual if proj.gm else ObservationMode.lsm) )
                   for proj in projs ) )

        db.execute("ANALYZE")