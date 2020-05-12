# -*- coding: cp1252 -*-

from os.path import basename
from collections import namedtuple

import numpy as np


import cv2

from osgeo import gdal
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

import sqlite3.dbapi2 as db
from contracts import contract

from oriental import adjust, ori
from oriental.adjust.parameters import EulerAngles, ADP
from oriental.adjust.cost import PhotoTorlegard, AutoDiff, omFiKaToRotationMatrix
from oriental.adjust.loss import Trivial

@contract
def checkSimilarityTrafo( objPtCoords : 'array[Nx3](float)',
                          modelPtCoords : 'array[Nx3](float)',
                          s : float,
                          R : 'array[3x3](float)',
                          x0 : 'array[3](float)' ):
    # check that ori.similarityTrafo really delivers the smallest mean squared error
    class SimTrafo3D( AutoDiff ):
        def __init__( self, x, y ):
            self.x = x
            self.y = y
            nResiduals = 3
            paramBlocks = (
                1, # s
                3, # omfika
                3 # x0
            )
            super().__init__( nResiduals, paramBlocks )

        def Evaluate( self, parameters, residuals ):
            s = parameters[0][0]
            omfika = parameters[1]
            x0 = parameters[2]
            typ = type(s)
            R = omFiKaToRotationMatrix( omfika )
            x = np.array([ typ(self.x[0]), typ(self.x[1]), typ(self.x[2]) ])
            y = np.array([ typ(self.y[0]), typ(self.y[1]), typ(self.y[2]) ])
            residuals[:] = s * R.dot( x - x0 ) - y
            return True

    s2 = np.array([s.copy()])
    omfika = ori.omfika(R)
    x02 = x0.copy()
    problem = adjust.Problem()
    loss = Trivial()
    for objPt, modelPt in zip( objPtCoords, modelPtCoords ):
        cost = SimTrafo3D( x=objPt, y=modelPt )
        problem.AddResidualBlock( cost,
                                    loss,
                                    s2,
                                    omfika,
                                    x02 )
    options = adjust.Solver.Options()
    summary = adjust.Solver.Summary()
    adjust.Solve( options, problem, summary )
    assert adjust.isSuccess( summary.termination_type )

    # there are slight differences in the offset and scale. Maybe enhance the computation?
    np.testing.assert_allclose( x0, x02, rtol=1e-3 )
    np.testing.assert_allclose( ori.omfika(R), omfika )
    np.testing.assert_allclose( np.array([s]), s2, rtol=1.e-3 )

@contract
def interpolateBilinear( grid   : 'array[2x2](float)',
                         pos_xy : 'array[2](float,(=0|>0),(<1|=1))' ) -> float: # restriction of values in [0;1] sometimes results in pycontracts-parsing error
    x,y = pos_xy
    ax = np.array( [ 1.-x, x ], dtype=np.float )
    ay = np.array( [ 1.-y, y ], dtype=np.float )
    return ax.dot( grid.T ).dot( ay )

@contract
def getObjPts( fnOrtho : str,
               fnDSM : str,
               dbManual : str ) -> 'array[N]':
    with db.connect( dbManual ) as manualOri:
        #manualOri.row_factory = db.Row

        cursor = manualOri.execute("""
            SELECT name,x,y
            FROM imgObs
                JOIN images
                  ON images.ID==imgObs.imgID
            WHERE images.path GLOB "*{}"
        """.format( basename(fnOrtho) ) )
        objPts = np.fromiter( ( (name,x,y,0.) for name,x,y in cursor ),
                              dtype=[ ( 'name', (np.str_,10) ),
                                      ( 'x'   , np.float     ),
                                      ( 'y'   , np.float     ),
                                      ( 'z'   , np.float     ) ] )
        

    dsOrtho = gdal.Open( fnOrtho, gdal.GA_ReadOnly )
    ortho2wrld = dsOrtho.GetGeoTransform()

    dsDSM = gdal.Open( fnDSM, gdal.GA_ReadOnly )
    dsm2wrld = dsDSM.GetGeoTransform()
    okay,wrld2dsm = gdal.InvGeoTransform( dsm2wrld )
    assert okay==1, "inversion of transform failed"

    for idx in range(objPts.shape[0]):
        # http://www.gdal.org/gdal_datamodel.html
        # pixel/line coordinates are from (0.0,0.0) at the top left corner of the top left pixel to (width_in_pixels,height_in_pixels) at the bottom right corner of the bottom right pixel
        x =  objPts[idx]['x'] + .5
        y = -objPts[idx]['y'] + .5
        x,y = gdal.ApplyGeoTransform( ortho2wrld, x, y )
        objPts[idx]['x'] = x
        objPts[idx]['y'] = y

        # we assume here that ortho and DSM share the same projection!

        # interpolate terrain heights
        posDsm = np.array( gdal.ApplyGeoTransform( wrld2dsm, x, y ) )

        # refer to center of upper/left pixel 
        posDsm -= 0.5

        # for some reason, there is no GDAL-functionality to interpolate rasters at specified positions
        iposDsm = np.floor( posDsm ).astype(np.int)
        heights = dsDSM.ReadAsArray( xoff=iposDsm[0].item(), yoff=iposDsm[1].item(), xsize=2, ysize=2 )

        z = interpolateBilinear( heights, (posDsm - iposDsm) )

        objPts[idx]['z'] = z

    return objPts

@contract
def getModelPts( fnOrtho : str,
                 dbRelOri : str,
                 dbManual : str ) -> 'array[N]':
    "extract imgObs in APs; forward intersect"

    with db.connect( dbManual ) as manualOri:
        imgObs = manualOri.execute("""
            SELECT imgObs.name,
                   images.path,
                   imgObs.x,
                   imgObs.y
            FROM imgObs
                JOIN images
                  ON images.ID==imgObs.imgID
            WHERE NOT images.path GLOB "*{}"
        """.format( basename(fnOrtho) ) ).fetchall()

    names = { obs[0] for obs in imgObs }
    name2obs = { name:[ (el[1],el[2],el[3]) for el in imgObs if el[0]==name ]
                 for name in names }

    modelPts = np.empty( len(name2obs),
                         dtype=[ ( 'name', (np.str_,10) ),
                                 ( 'x'   , np.float     ),
                                 ( 'y'   , np.float     ),
                                 ( 'z'   , np.float     ) ] )
    TriangulatePar = namedtuple( 'TriangulatePar', ( 'imgObs', 'R', 't', 'ior', 'adp' ) )  

    with db.connect( dbRelOri ) as relOri:
        for iName,(name,obss) in enumerate(name2obs.items()):
            problem = adjust.Problem()
            loss = Trivial()
            modelPt = np.zeros(3)
            if len(obss) < 2:
                raise Exception( "Point '{}' has been observed in only 1 aerial image ({})!".format(name,obss[0][0]) )
            triangulatePars = []
            for iObs,obs in enumerate(obss):
                cost = PhotoTorlegard( obs[1], obs[2] )
                X0,Y0,Z0,r1,r2,r3,parameterization,x0,y0,z0 = relOri.execute("""
                    SELECT images.X0,
                           images.Y0,
                           images.Z0,
                           images.r1,
                           images.r2,
                           images.r3,
                           images.parameterization,
                           cameras.x0,
                           cameras.y0,
                           cameras.z0
                    FROM images
                        JOIN cameras
                          ON images.camID==cameras.ID
                    WHERE images.path GLOB "*{}"
                """.format( basename(obs[0]) ) ).fetchone()

                P0 = np.array([X0,Y0,Z0])
                assert parameterization=='omfika'
                angles = EulerAngles( array=np.array([r1,r2,r3]) )
                ior = np.array([x0,y0,z0])
                adp = ADP( normalizationRadius = 1000. )
                problem.AddResidualBlock( cost,
                                          loss,
                                          P0, angles, ior, adp, modelPt )
                for el in (P0,angles,ior,adp):
                    problem.SetParameterBlockConstant(el)

                if iObs<2:
                    triangulatePars.append( TriangulatePar( np.array([ obs[1], obs[2] ]),
                                                            ori.omfika(angles),
                                                            P0,
                                                            ior,
                                                            adp ) )

            modelPt[:] = ori.triangulatePoints( triangulatePars[0].imgObs[np.newaxis,:],
                                                triangulatePars[1].imgObs[np.newaxis,:],
                                                triangulatePars[0],
                                                triangulatePars[1] )
            options = adjust.Solver.Options()
            summary = adjust.Solver.Summary()
            adjust.Solve( options, problem, summary )
            assert adjust.isSuccess( summary.termination_type )
            modelPts[iName]['name'] = name
            modelPts[iName]['x'] = modelPt[0]
            modelPts[iName]['y'] = modelPt[1]
            modelPts[iName]['z'] = modelPt[2]

    return modelPts

@contract
def manualTrafo( fnOrtho : str,
                 fnDSM : str,
                 dbRelOri : str,
                 dbManual : str,
                 plot : bool = False ) \
                 -> 'tuple( float, array[3x3](float), array[3](float) )':
    objPts = getObjPts( fnOrtho, fnDSM, dbManual )
    
    modelPts = getModelPts( fnOrtho, dbRelOri, dbManual )

    symmDiff = { el[0] for el in objPts } ^ { el[0] for el in modelPts }
    if len( symmDiff ):
        raise Exception( "Point(s) is/are observed in only 1 of objPts, modelPts: {}".format(symmDiff) )

    if len(modelPts) < 3:
        raise Exception( "At least 3 homologous points needed, but only {} given.".format( len(modelPts) ) )

    # 3D similarity trafo
    objPtCoords   = np.array([ (x,y,z) for name,x,y,z in sorted( objPts  , key=lambda x: x[0] ) ], dtype=np.float )
    modelPtCoords = np.array([ (x,y,z) for name,x,y,z in sorted( modelPts, key=lambda x: x[0] ) ], dtype=np.float )

    # y=s*R.dot(x-x0)
    s,R,x0,res = ori.similarityTrafo( x=objPtCoords, y=modelPtCoords )

    if 0:
        checkSimilarityTrafo( objPtCoords, modelPtCoords, s, R, x0 )

    objPtCoords_r = R.T.dot( modelPtCoords.T / s ).T + x0
    res_obj = np.sum( ( objPtCoords_r - objPtCoords )**2, axis=1 )**.5
    print("Residual norms[m] in object space:")
    for objPt,res in zip( sorted( objPts  , key=lambda x: x[0] ), res_obj ):
        print( "{:>4s}: {:02.3f}".format( objPt[0], res ) )


    if plot:
        import oriental.utils.pyplot as plt
        plt.figure(); plt.clf()
        plt.plot( objPtCoords[:,0], objPtCoords[:,1], 'ro' )
        plt.plot( np.vstack(( objPtCoords[:,0], objPtCoords_r[:,0] )),
                  np.vstack(( objPtCoords[:,1], objPtCoords_r[:,1] )),
                  '-m' )
        for name,X,Y,Z in sorted( objPts  , key=lambda x: x[0] ):
            plt.text( X, Y, name )
        plt.xlabel('Rechts')
        plt.ylabel('Hoch')
        plt.title('Manual abs. ori. residuals')
        plt.axis('equal')
        #lims = plt.axis()
        extents = ( objPtCoords[:,:2].min(axis=0) - 10,
                    objPtCoords[:,:2].max(axis=0) + 10 )
        dsOrtho = gdal.Open( fnOrtho, gdal.GA_ReadOnly )
        ortho2wrld = dsOrtho.GetGeoTransform()
        okay,wrld2ortho = gdal.InvGeoTransform( ortho2wrld )
        assert okay==1, "inversion of transform failed"
        # ortho is axis-aligned, so we don't need to consider a rotation about the vertical axis
        extents[0][:] = gdal.ApplyGeoTransform( wrld2ortho, *extents[0] )
        extents[1][:] = gdal.ApplyGeoTransform( wrld2ortho, *extents[1] )
        extents = np.round( extents[0] ), np.round(extents[1])
        cut = dsOrtho.ReadAsArray( xoff=int(extents[0][0]), yoff=int(extents[1][1]),
                                   xsize = int( extents[1][0] - extents[0][0] ),
                                   ysize = int( extents[0][1] - extents[1][1] ) )
        cut = np.rollaxis( cut, 0, 3 )
        # image coordinates of gdal have their origin in the upper/left corner of the upper/left pixel
        extents[0][:] = gdal.ApplyGeoTransform( ortho2wrld, *( extents[0] - .5 ) )
        extents[1][:] = gdal.ApplyGeoTransform( ortho2wrld, *( extents[1] + .5 ) )
        
        plt.imshow( cut, interpolation='none',
                    # left, right, bottom, top
                    extent=( extents[0][0], extents[1][0],
                             extents[0][1], extents[1][1] ) )
        #plt.axis(lims)

    return s,R,x0
