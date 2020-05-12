# -*- coding: cp1252 -*-
"communicate with SQLite databases"
from oriental import adjust, ori, utils, blocks
import oriental.adjust.parameters
import oriental.ori.transform
import oriental.utils.db
import oriental.utils.filePaths
import oriental.blocks.types

import contextlib, os
from pathlib import Path
from itertools import chain
import enum
import collections

from contracts import contract, new_contract
import numpy as np
from sqlite3 import dbapi2
from osgeo import ogr
ogr.UseExceptions()

What = enum.Enum('What', 'cameras images objPts imgObs')
new_contract('What',What)
Amount = enum.IntEnum('Amount', 'minimal id full')
new_contract('Amount',Amount)

InfoId = blocks.types.InfoId
InfoIdStdDevs = blocks.types.InfoIdStdDevs

@contract
def restore( dbFn : Path,
             restoreWhat : 'Amount|list(tuple(What,Amount))' = Amount.full,
             callback = None ) -> list:

    def allNull(db,tableName,cols):
        # note: distinct is for all tuples of the selection: a distinct selection of both columns of a table with contents
        # 0,0
        # 0,1
        # 1,0
        # 1,1
        # will return the whole contents, as no rows are equal.
        for row in db.execute("SELECT DISTINCT {} FROM {}".format(','.join(cols),tableName)):
            if any(el is not None for el in row):
                return False
        return True

    def restoreCameras( db, amount ):
        #iorFmt = [('coo',(float,3))]
        #adpFmt = [('coo',(float,9))]
        #iorAttrs, adpAttrs, cameraAttrs = [], [], []
        #iorGeom = 'x0 y0 z0'.split()
        #adpGeom = [ str(el[1]) for el in sorted( adjust.PhotoDistortion.values.items(), key=lambda x: x[0] ) ]
        #if amount >= Amount.id:
        #    IorInfo = IorInfoId
        #    AdpInfo = AdpInfoId
        #    iorFmt.append(('id',int))
        #    adpFmt.append(('id',int))
        #    for el in iorAttrs,adpAttrs,cameraAttrs:
        #        el.append('id')
        #    if amount == Amount.full:
        #        s_iorGeom = ['s_{}'.format(el) for el in iorGeom ]
        #        iorFmt.append(('sigma',(float,3)))
        #        adpFmt.append(('sigma',(float,9)))
        #        s_adpGeom = ['s_{}'.format(el) for el in adpGeom ]
        #        skippedCols = frozenset( chain( iorGeom,
        #                                         adpGeom,
        #                                         ['reference', 'normalizationRadius'],
        #                                         s_iorGeom,
        #                                         s_adpGeom,
        #                                         cameraAttrs,
        #                                         iorAttrs,
        #                                         adpAttrs ) )
        #        cameraAttrs.extend([ el for el in utils.db.columnNamesForTable(db,'cameras') if el not in skippedCols ])
        #        iorAttrs.append( 'stdDevs' )
        #        adpAttrs.append( 'stdDevs' )
        #
        #        IorInfo = IorInfoIdStdDevs
        #        AdpInfo = AdpInfoIdStdDevs
        #
        #data = []
        #addFormats = None
        #for row in db.execute("SELECT * FROM cameras"):
        #    ior = adjust.parameters.Ior([ row[el] for el in iorGeom ])
        #    if iorAttrs:
        #        ctorArgs = {}
        #        if amount == Amount.full:
        #            ctorArgs.update({ 'stdDevs' : np.array([row[el] for el in s_iorGeom],float) })
        #        ctorArgs.update({ name : row[name] for name in IorInfo._fields if name not in ctorArgs})
        #        ior = ior.view(adjust.parameters.InfoIor)
        #        ior.info = IorInfo(**ctorArgs)
        #
        #    adp = adjust.parameters.ADP( normalizationRadius= row['normalizationRadius'],
        #                                 referencePoint     = adjust.AdpReferencePoint.names[ row['reference'] ],
        #                                 array              = [ row[el] for el in adpGeom ] )
        #    if adpAttrs:
        #        ctorArgs = {}
        #        if amount == Amount.full:
        #            ctorArgs.update({ 'stdDevs' : np.array([row[el] for el in s_adpGeom],float) })
        #        ctorArgs.update({ name : row[name] for name in AdpInfo._fields if name not in ctorArgs})
        #        adp = adp.view(adjust.parameters.InfoADP)
        #        adp.info = AdpInfo(**ctorArgs)
        #    addDatum = tuple(row[name] for name in cameraAttrs)
        #    data.append(( row['id'], (ior, adp) + addDatum ))
        #    if addFormats is None:
        #        addFormats = [ np.min_scalar_type(datum) for datum in addDatum ]
        #    else:
        #        addFormats = [ np.promote_types(old,new) for old,new in utils.zip_equal( addFormats, addDatum ) ]
        #
        #dtype=np.dtype([('ior',iorFmt), ('adp',adpFmt)] + [(attr,fmt) for attr,fmt in zip(cameraAttrs,addFormats) ])
        ##dtype = np.dtype( {'names': 'ior adp'.split() + cameraAttrs, 'formats': [ (float,len(iorGeom)), (float,len(adpGeom)) ] + addFormats })
        ##dtype = np.dtype( {'names': 'ior adp'.split() + cameraAttrs, 'formats': [ np.object ]*2 + addFormats })
        #cameras = {}
        #for camId,datum in data:
        #    #cameras[camId] = np.rec.array( datum, dtype )
        #    camera = np.empty(1,dtype).view(np.recarray)[0]
        #    camera.ior.coo = datum[0]
        #    camera.ior.id = camId
        #    camera.ior.sigma = datum[0].info.stdDevs
        #    camera.adp.coo = datum[1]
        #    camera.adp.id = camId
        #    camera.adp.sigma = datum[1].info.stdDevs
        #    for attr,dat in utils.zip_equal(cameraAttrs,datum[2:]):
        #        camera[attr] = dat
        #    cameras[camId] = camera
        #    #camera = np.empty( 1, dtype )[0]
        #    #camera['ior'] = datum[0]
        #    #camera['adp'] = datum[1]
        #    #for attr,dat in utils.zip_equal(cameraAttrs,datum[2:]):
        #    #    camera[attr] = dat
        #    #cameras[camId] = camera.view(np.recarray)
        #return cameras

        iorAttrs, adpAttrs, cameraAttrs = [], [], []
        iorCols = 'x0 y0 z0'.split()
        adpCols = [ str(el[1]) for el in sorted( adjust.PhotoDistortion.values.items(), key=lambda x: x[0] ) ]
        s_iorCols = []
        s_adpCols = []
        if amount >= Amount.id:
            for el in iorAttrs,adpAttrs,cameraAttrs:
                el.append('id')
            IorInfo = InfoId
            AdpInfo = InfoId
            if amount == Amount.full:
                #skip stdDevs if all are NULL!
                s_iorCols = ['s_{}'.format(el) for el in iorCols ]
                s_adpCols = ['s_{}'.format(el) for el in adpCols ]
                skippedCols = frozenset( el.lower() for el in chain( iorCols,
                                                                     adpCols,
                                                                     ['reference', 'normalizationRadius'],
                                                                     s_iorCols,
                                                                     s_adpCols,
                                                                     cameraAttrs,
                                                                     iorAttrs,
                                                                     adpAttrs ) )
                for name in utils.db.columnNamesForTable(db,'cameras'):
                    if name.lower() not in skippedCols and not allNull(db,'cameras',[name]):
                        cameraAttrs.append(name)
                if allNull(db,'cameras',s_iorCols):
                    s_iorCols = []
                else:
                    iorAttrs.append( 'stdDevs' )
                    IorInfo = InfoIdStdDevs
                if allNull(db,'cameras',s_adpCols):
                    s_adpCols = []
                else:
                    adpAttrs.append( 'stdDevs' )
                    AdpInfo = InfoIdStdDevs

        Camera = blocks.types.createCompact(blocks.types.Camera,cameraAttrs)
        cameras = {}
        for row in db.execute("SELECT * FROM cameras"):
            ior = adjust.parameters.Ior([ row[el] for el in iorCols ])
            if iorAttrs:
                ctorArgs = {}
                if s_iorCols:
                    ctorArgs.update({ 'stdDevs' : np.array([row[el] for el in s_iorCols],float) })
                ctorArgs.update({ name : row[name] for name in IorInfo.allSlots if name not in ctorArgs})
                ior = ior.view(adjust.parameters.InfoIor)
                ior.info = IorInfo(**ctorArgs)

            adp = adjust.parameters.ADP( normalizationRadius= row['normalizationRadius'],
                                         referencePoint     = adjust.AdpReferencePoint.names[ row['reference'] ],
                                         array              = [ row[el] for el in adpCols ] )
            if adpAttrs:
                ctorArgs = {}
                if s_adpCols:
                    ctorArgs.update({ 'stdDevs' : np.array([row[el] for el in s_adpCols],float) })
                ctorArgs.update({ name : row[name] for name in AdpInfo.allSlots if name not in ctorArgs})
                adp = adp.view(adjust.parameters.InfoADP)
                adp.info = AdpInfo(**ctorArgs)

            cameras[row['id']] = Camera( ior=ior, adp=adp, **{ name : row[name] for name in cameraAttrs } )

        return cameras

    def restoreImages( db, amount, dbFn ):
        images = {}

        prcAttrs, rotAttrs, imgAttrs = [], [], []
        prcCols = 'X0 Y0 Z0'.split()
        rotCols = 'r1 r2 r3'.split()
        s_prcCols = []
        s_rotCols = []
        fiducialMatAttrs = [ 'fiducial_{}'.format(el) for el in 'A00 A01 A10 A11'.split() ]
        fiducialVecAttrs = [ 'fiducial_{}'.format(el) for el in 't0 t1'          .split() ]
        if amount >= Amount.id:
            for el in prcAttrs, rotAttrs, imgAttrs:
                el.append('id')
            imgAttrs.append('camId')
            PrcInfo = InfoId
            RotInfo = InfoId
            if amount == Amount.full:
                s_prcCols = ['s_{}'.format(el) for el in prcCols ]
                s_rotCols = ['s_{}'.format(el) for el in rotCols ]
                skippedCols = frozenset( el.lower() for el in chain( prcCols,
                                                                     rotCols,
                                                                     ['parameterization'],
                                                                     s_prcCols,
                                                                     s_rotCols,
                                                                     imgAttrs,
                                                                     fiducialMatAttrs,
                                                                     fiducialVecAttrs,
                                                                     prcAttrs,
                                                                     rotAttrs  ) )
                for name in utils.db.columnNamesForTable(db,'images'):
                    if name.lower() not in skippedCols and not allNull(db,'images',[name]):
                        imgAttrs.append(name)
                if allNull(db,'images',s_prcCols):
                    s_prcCols = []
                else:
                    prcAttrs.append( 'stdDevs' )
                    PrcInfo = InfoIdStdDevs
                if allNull(db,'images',s_rotCols):
                    s_rotCols = []
                else:
                    rotAttrs.append( 'stdDevs' )
                    RotInfo = InfoIdStdDevs

        Image = blocks.types.createCompact( blocks.types.Image, ['fiducialTrafo'] + imgAttrs )

        for row in db.execute("SELECT * FROM images"):
            prc = adjust.parameters.Prc([ row[el] for el in prcCols ])
            if prcAttrs:
                ctorArgs = {}
                if s_prcCols:
                    ctorArgs.update({ 'stdDevs' : np.array([row[el] for el in s_prcCols],float) })
                ctorArgs.update({ name : row[name] for name in PrcInfo.allSlots if name not in ctorArgs})
                prc = prc.view(adjust.parameters.InfoPrc)
                prc.info = PrcInfo(**ctorArgs)

            rot = adjust.parameters.EulerAngles( adjust.EulerAngles.names[row['parameterization']],
                                                 [ row[el] for el in rotCols ] )
            if rotAttrs:
                ctorArgs = {}
                if s_rotCols:
                    ctorArgs.update({ 'stdDevs' : np.array([row[el] for el in s_rotCols],float) })
                ctorArgs.update({ name : row[name] for name in RotInfo.allSlots if name not in ctorArgs})
                rot = rot.view(adjust.parameters.InfoEulerAngles)
                rot.info = RotInfo(**ctorArgs)
            A=np.array([row[el] for el in fiducialMatAttrs], float).reshape((2,2))
            t=np.array([row[el] for el in fiducialVecAttrs], float)
            fiducialTrafo = None
            if np.isfinite(A).all() and np.isfinite(t).all():
                fiducialTrafo = ori.transform.AffineTransform2D( A, t )
            ctorArgs = dict( prc=prc,
                             rot=rot,
                             fiducialTrafo=fiducialTrafo,
                             **{ name : row[name] for name in imgAttrs } )
            path = ctorArgs.get('path')
            if path is not None:
                if not path.lstrip().startswith('<'): # web mapping tile service
                    if not os.path.isabs(path):
                        path = os.path.normpath( str(dbFn.parent / path) )
                    ctorArgs['path'] = Path(path)
            images[row['id']] = Image( **ctorArgs )
        return images

    def restoreObjPts( db, amount ):
        objPts = {}

        objAttrs = []
        objCols = 'X Y Z'.split()
        s_objCols = []
        if amount >= Amount.id:
            objAttrs.append('id')
            ObjInfo = InfoId
            if amount == Amount.full:
                # this elegantly selects and returns all info, even user-defined columns
                s_objCols = ['s_{}'.format(el) for el in objCols ]
                skippedCols = frozenset( el.lower() for el in chain( ['pt'],
                                                                     s_objCols,
                                                                     objAttrs ) )
                for name in utils.db.columnNamesForTable(db,'objPts'):
                    if name.lower() not in skippedCols and not allNull(db,'objPts',[name]):
                        objAttrs.append(name)
                if allNull(db,'objPts',s_objCols):
                    s_objCols = []
                else:
                    objAttrs.append( 'stdDevs' )
                    ObjInfo = InfoIdStdDevs

        for row in db.execute("""
            SELECT X(pt) as X,
                   Y(pt) as Y,
                   Z(pt) as Z,
                   *
            FROM objPts """):
            objPt = adjust.parameters.ObjectPoint([ row[el] for el in objCols ])
            if objAttrs:
                ctorArgs = {}
                if s_objCols:
                    ctorArgs.update({ 'stdDevs' : np.array([row[el] for el in s_objCols],float) })
                ctorArgs.update({ name : row[name] for name in ObjInfo.allSlots if name not in ctorArgs})
                objPt = objPt.view(adjust.parameters.InfoObjectPoint)
                objPt.info = ObjInfo(**ctorArgs)
            objPts[row['id']] = objPt
        return objPts

    def restoreImgObs( db, amount ):

        # There may be a vast amout of image observations, and how to organize them is mostly up to the application.
        # Thus, let's return them in the most compact way, as a structured array without object references (all data embedded into the dtype)

        # union of red, green, blue as rgb?
        geomCols = 'x', 'y'
        fmts = [( ('image coordinates x, y','coo'), (float,2) )]
        s_geomCols = []
        addCols = []
        if amount >= Amount.id:
            addCols.extend( 'id imgId objPtId'.split() )
            if amount == Amount.full:
                s_geomCols = ['s{}'.format(el) for el in geomCols ]
                skippedCols = frozenset( el.lower() for el in chain( geomCols,
                                                                     s_geomCols,
                                                                     addCols ) )
                for name in utils.db.columnNamesForTable(db,'imgObs'):
                    if name.lower() not in skippedCols and not allNull(db,'imgObs',[name]):
                        addCols.append(name)
                if allNull(db,'imgObs',s_geomCols):
                    s_geomCols = []
                else:
                    fmts.append( ( ('image coordinate standard deviations','stdDevs'), (float,2) ) )
        
        affinities = { row['name'] : row['type'] for row in db.execute( "PRAGMA table_info('imgObs')" ) }
        addDtypes = []
        for addCol in addCols[:]:
            dbTypes = []
            for row in db.execute('SELECT DISTINCT typeof("{}") FROM imgObs'.format(addCol)):
                dbTypes.append(row[0])
            dbTypes = set(el.lower() for el in dbTypes)
            if dbTypes == {'null'}:
                addCols.remove(addCol) # an all-NULL column
            elif "blob" in dbTypes:
                addDtypes.append(np.object)
            elif "text" in dbTypes:
                if affinities[addCol] == 'DATETIME':
                    addDtypes.append('datetime64[ms]') # millisecond precision
                else:
                    maxTextLen = db.execute('SELECT max(length("{}")) FROM imgObs'.format(addCol)).fetchone()[0]
                    addDtypes.append( (np.str_,maxTextLen) )
            elif "real" in dbTypes:
                # For real types, let's always use float, because np.min_scalar_type checks their magnitude only, but not their precision
                # Thus, np.min_scalar_type(0.0000000000000001) -> float16, even though float16 has only 10 bits of mantissa.
                addDtypes.append( float )
            else:
                mini,maxi = db.execute('SELECT min("{col}"), max("{col}") FROM imgObs'.format(col=addCol)).fetchone()
                if {mini,maxi} <= {0,1}:
                    addDtypes.append(np.bool)
                else:
                    addDtypes.append( np.promote_types( np.min_scalar_type(mini), np.min_scalar_type(maxi) ) )

        def genData():
            for row in db.execute("SELECT * FROM imgObs"):
                datum = ( np.array([row[el] for el in geomCols],float) ,)
                if s_geomCols:
                    datum += ( np.array([row[el] for el in s_geomCols],float) ,)
                addDatum = tuple( row[name] for name in addCols )
                yield datum + addDatum
        addFmts = list(utils.zip_equal(addCols,addDtypes))
        dtype=np.dtype( fmts + addFmts )
        count = db.execute("SELECT count(*) FROM imgObs").fetchone()[0]
        imgObs = np.fromiter( genData(), dtype, count=count )
        return imgObs

        ## The following works, but is very slow. Thus, use sqlite3 above to determine the dtype for each additional column. Additionally, we can drop all-NULL columns beforehand!
        #data = []
        #addDtypes = None
        #for row in db.execute("SELECT * FROM imgObs"):
        #    datum = ( np.array([row[el] for el in geomCols],float) ,)
        #    if amount == Amount.full:
        #        datum += ( np.array([row[el] for el in s_geomCols],float) ,)
        #    addDatum = tuple( row[name] for name in addCols )
        #    data.append( datum + addDatum )
        #    if addDtypes is None:
        #        addDtypes = [ np.min_scalar_type(datum) if datum is not None else None for datum in addDatum ]
        #    else:
        #        for idx,(oldDtype,value) in enumerate(utils.zip_equal( addDtypes, addDatum )):
        #            if value is None:
        #                continue
        #            newDtype = np.min_scalar_type(value)
        #            if oldDtype is None:
        #                addDtypes[idx] = newDtype
        #            else:
        #                addDtypes[idx] = np.promote_types(oldDtype,newDtype)
        ## drop all-NULL columns
        #idxs = [ idx for idx,addDtype in enumerate(addDtypes) if addDtype is not None ]
        #addFmts = list(utils.zip_equal(addCols,addDtypes))
        #if len(idxs) < len(addDtypes):
        #    addFmts = [ addFmts[idx] for idx in idxs ]
        #    # This is slow, use np.fromiter and a generator instead.
        #    #data = [ datum[:len(fmts)] + tuple( datum[idx+len(fmts)] for idx in idxs ) for datum in data]
        #dtype=np.dtype( fmts + addFmts )
        #def genData():
        #    offset = len(fmts)
        #    for datum in data:
        #        yield datum[:offset] + tuple( datum[idx+offset] for idx in idxs )
        #imgObs = np.fromiter( genData(), dtype, count=len(data) )
        #return imgObs

    if type(restoreWhat) is Amount:
        restoreWhat = [ (what,restoreWhat) for what in What]
    ret = []
    if len(restoreWhat) > len(frozenset(restoreWhat)):
        raise Exception("Duplicates in restoreWhat")
    with dbapi2.connect( utils.db.uri4sqlite(dbFn) + '?mode=ro', uri=True ) as db:
        utils.db.initDataBase(db)

        for what,amount in restoreWhat:
            if what is What.cameras:
                ret.append( restoreCameras( db, amount ) )
            elif what is What.images:
                ret.append( restoreImages( db, amount, dbFn ) )
            elif what is What.objPts:
                ret.append( restoreObjPts( db, amount ) )
            elif what is What.imgObs:
                ret.append( restoreImgObs( db, amount ) )

        if callback:
            callback(db,ret)
    return ret

@contract
def save( absOriDbFn : Path,
          block : adjust.Problem,
          cameras : dict,
          images : dict,
          objPts : dict,
          stdDevs : 'dict|None' = None,
          TieImgObsData : type = type(None),
          callback = None,
          CtrlImgObsData : type = type(None)
        ) -> None:
    stdDevs = stdDevs or {}
    with contextlib.suppress(FileNotFoundError):
        absOriDbFn.unlink()
    utils.db.createUpdateSchema( str(absOriDbFn) )
    with dbapi2.connect( utils.db.uri4sqlite(absOriDbFn) + '?mode=rwc', uri=True ) as absOriDb:
        utils.db.initDataBase( absOriDb )
        photoDistortionSorted = [ str(el[1]) for el in sorted( adjust.PhotoDistortion.values.items(), key=lambda x: x[0] ) ]
        absOriDb.executemany( """
            INSERT INTO cameras(id,x0,y0,z0,s_x0,s_y0,s_z0,reference,normalizationRadius,{},{})
            VALUES             ( ?, ?, ?, ?,   ?,   ?,   ?,        ?,                  ?,{})""".format( ','.join(photoDistortionSorted),
                                                                                                        ','.join('s_{}'.format(el) for el in photoDistortionSorted),
                                                                                                        ','.join('?'*len(photoDistortionSorted)*2) ),
            ( tuple( chain(
                [camera.id],
                camera.ior,
                stdDevs.get( camera.ior.ctypes.data, (None,)*3 ),
                [str(camera.adp.referencePoint), camera.adp.normalizationRadius],
                camera.adp,
                stdDevs.get( camera.adp.ctypes.data, (None,)*camera.adp.size )
              ) ) for camera in cameras.values() ) )

        fiducialAttrs = [ 'fiducial_{}'.format(el) for el in 'A00 A01 A10 A11 t0 t1'.split() ]
        def genImages():
            for img in images.values():
                mask = None
                if img.mask_px is not None and len(img.mask_px):
                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    for pt in img.mask_px:
                        ring.AddPoint( *pt )
                    ring.CloseRings()
                    ring.FlattenTo2D()
                    polyg = ogr.Geometry(ogr.wkbPolygon)
                    polyg.AddGeometryDirectly(ring)
                    assert polyg.IsValid()
                    mask = polyg.ExportToWkb()

                yield tuple( chain(
                    [ img.id, img.camId, str( utils.filePaths.relPathIfExists( img.path, absOriDbFn.parent) ), img.nCols, img.nRows ],
                    img.prc,
                    stdDevs.get( img.prc.ctypes.data, (None,)*3 ),
                    img.rot,
                    stdDevs.get( img.rot.ctypes.data, (None,)*3 ),
                    [str(img.rot.parametrization)],
                    [None]*len(fiducialAttrs) if img.pix2cam is None else chain( img.pix2cam.A.flat, img.pix2cam.t ),
                    [mask]
                ) )
        absOriDb.executemany(
            'INSERT INTO images( id,camId,path,nCols,nRows,X0,Y0,Z0,s_X0,s_Y0,s_Z0,r1,r2,r3,s_r1,s_r2,s_r3,parameterization,{},mask )'.format( ','.join( fiducialAttrs ) ) +
            'VALUES            (  ?,    ?,   ?,    ?,    ?, ?, ?, ?,   ?,   ?,   ?, ?, ?, ?,   ?,   ?,   ?,               ?,{},GeomFromWKB(?, -1) )'.format( ','.join('?'*len(fiducialAttrs) ) ),
            genImages() )


        absOriDb.execute("""
            SELECT AddGeometryColumn(
                'objPts', -- table
                'pt',     -- column
                -1,       -- srid
                'POINT',  -- geom_type
                'XYZ',    -- dimension
                1         -- NOT NULL
                ) """)

        packPointZ = utils.db.packPointZ
        absOriDb.executemany( """
            INSERT INTO objpts(id,                 pt,s_X,s_Y,s_Z)
            VALUES            ( ?, GeomFromWKB(?, -1),  ?,  ?,  ?)""",
            ( tuple( chain( 
                [ objPtId, packPointZ(objPt) ],
                stdDevs.get( objPt.ctypes.data, (None,)*3 )
              ) ) for objPtId,objPt in objPts.items() ) )
            

        def genImgObs():
            for resBlock in block.GetResidualBlocks():
                cost = block.GetCostFunctionForResidualBlock( resBlock )
                costData = getattr(cost,'data',None)
                if costData is not None:
                    if isinstance( costData, TieImgObsData ):
                        yield (None          , cost.data.imgId, cost.x, cost.y, cost.data.objPtId) + tuple(cost.data.rgb)
                    elif isinstance( costData, CtrlImgObsData ):
                        yield (cost.data.name, cost.data.imgId, cost.x, cost.y, cost.data.objPtId) + (None,)*3
        absOriDb.executemany( """
            INSERT INTO imgobs(name,imgId,x,y,objPtId,red,green,blue)
            VALUES            (   ?,    ?,?,?,      ?,  ?,    ?,   ?)""", genImgObs() )

        if callback is not None:
            callback(absOriDb)