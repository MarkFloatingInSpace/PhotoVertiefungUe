# -*- coding: cp1252 -*-
import os, math

import cv2
import numpy as np
import sqlite3.dbapi2 as db

from .getObjPts import getObjPts

def rectifyAerialImages( dbFn, dirAbsOri ):
    """compute an adjusting plane through the object points,
       and rectify all aerial images w.r.t. that plane"""
    # bild.hoch ist wohl die Flughöhe[m] aus der Flugplanung
    
    from oriental import ori
    
    imgOriDt = np.dtype([('id',   np.int),
                         ('path', np.str_, 500),
                         ('X0',   np.float, (3,)),
                         ('R',    np.float, (3,3)),
                         ('x0',   np.float, (3,)) ])

    with db.connect( dbFn ) as relOri:
        relOri.row_factory = db.Row
        
        nPhos = relOri.execute("""
            SELECT COUNT(*)
            FROM images
            	JOIN    cameras
            	     ON images.camID=cameras.id
            """).fetchone()[0]
            
        imgOris = np.recarray( nPhos, dtype=imgOriDt )
        
        rows = relOri.execute("""
            SELECT images.id,
                   images.path,
                   images.X0,
                   images.Y0,
                   images.Z0,
                   images.r1,
                   images.r2,
                   images.r3,
                   images.parameterization,
                   cameras.x0 as x0_,
                   cameras.y0 as y0_,
                   cameras.z0 as z0_
            FROM images
            	JOIN    cameras
            	     ON images.camID=cameras.id
        """)
        for row,imgOri in zip(rows,imgOris):
            #imgOri = imgOris[idx
            imgOri.id   = row['id']
            imgOri.path = row['path']
            imgOri.X0[:] = np.array([ row['X0'], row['Y0'], row['Z0'] ])
            assert row['parameterization'] == 'omfika'
            # angles are stored in [gon]. ori.omfika considers that.
            imgOri.R[:]    = ori.omfika( np.array([ row['r1'], row['r2'], row['r3'] ]) )
            imgOri.x0[:]   = np.array([ row['x0_'], row['y0_'], row['z0_'] ])
    
    relObjPts = getObjPts( dbFn )
    
    plot = False
    
    if plot:
        from oriental.utils import pyplot, mlab, mlab_utils
        print("pyplot,mlab imported")
        mlab.figure(1); mlab.clf()
        mlab.points3d(  relObjPts[:,0],  relObjPts[:,1],  relObjPts[:,2], mode='sphere', scale_mode='none', scale_factor=0.01, reset_zoom=True, vmin=0, vmax=255 )
    
    normalPtcl = ori.fitPlaneSVD( relObjPts )
    if normalPtcl[2] < 0:
        normalPtcl *= -1
    if plot:
        # Normalvektor plotten
        cog = relObjPts.mean( axis=0 )
        mlab.quiver3d( cog[0],  cog[1],  cog[2], normalPtcl[0], normalPtcl[1], normalPtcl[2], mode='arrow', scale_mode='none', scale_factor=1., reset_zoom=False )
    
    imgRectDt = np.dtype([('id',   np.int),
                         ('path', np.str_, 500),
                         ('H',    np.float, (3,3)),
                         ('K',    np.float, (3,3)),
                         ('rc',   np.float, (4,2)) ])

    imgRects = np.recarray( nPhos, dtype=imgRectDt )
    for imgOri, imgRect in zip( imgOris, imgRects ):
        name = os.path.basename(imgOri.path)
        
        if plot:
            # mode = 'point' seems to be always 1px large, not influenced by scale_factor
            #scale_factor[drawing units]
            mlab_utils.camera( imgOri.x0, imgOri.R, imgOri.X0, "{}".format( name ) )
        
            mlab.quiver3d( imgOri.X0[0],  imgOri.X0[1],  imgOri.X0[2], normalPtcl[0], normalPtcl[1], normalPtcl[2], mode='arrow', color=(1.,1.,0.), scale_mode='none', scale_factor=.1, reset_zoom=False )
    
        K = ori.cameraMatrix( imgOri.x0 )

        # in the OpenCV camera CS, z points downwards in a vertical image
        #normalPtcl *= -1
        
        normalPtclLoc = imgOri.R.T.dot( normalPtcl[:3] )
        zAxis = np.array([0,0,1])
        cross = np.cross( zAxis, normalPtclLoc )
        cross /= np.linalg.norm(cross)
        angle = np.arccos( zAxis.dot(normalPtclLoc) )
        cross *= angle
        Rptcl = ori.rodrigues( cross )
        R = Rptcl.T#.dot( imgOri.R )
        
        # Achtung: OpenCV-Bild-KS ist rel. zu ORIENT-Bild-KS um x-Achse um 180° verdreht!
        #cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst
        # http://people.scs.carleton.ca/~c_shu/Courses/comp4900d/notes/homography.pdf
        # Bildausschnitt! -> linkes K so wählen, dass Bildecken des Originalphos auch im entzerrten pho vorhanden sind.
        
        # rotation about the x-axis by 200 gon
        Rx200 = np.array([[ 1.,  0.,  0. ],
                          [ 0., -1.,  0. ],
                          [ 0.,  0., -1. ]])
        Rocv = Rx200.dot( R.dot( Rx200 ) )
        
        pho = cv2.imread( imgOri.path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_IGNORE_ORIENTATION )
        if plot:
            pyplot.figure(1); pyplot.title(name)
            pyplot.imshow( cv2.cvtColor( pho, cv2.COLOR_BGR2RGB ) )
        
        # TODO: these are the centers of the corner pixels.
        # Use the outer pixel corners instead, i.e. -0.5, pho.shape[1]-0.5 ? 
        cornersOcv = np.array([ [              0,              0 ],
                                [ pho.shape[1]-1,              0 ],
                                [ pho.shape[1]-1, pho.shape[0]-1 ],
                                [              0, pho.shape[0]-1 ] ], dtype=np.float )
            
        Kinv = np.linalg.inv(K)
        cornersOcv_h = np.hstack( ( cornersOcv, np.ones( (4,1) ) ) )
        cornersOcvNorm_h = Kinv.dot( cornersOcv_h.T ).T
        cornersOcvVertNorm_h = Rocv.dot( cornersOcvNorm_h.T ).T
        cornersOcvVertNorm = cornersOcvVertNorm_h[:,:2] / cornersOcvVertNorm_h[:,2].reshape(-1,1)
        
        f2 = K[0,0] * cornersOcvVertNorm_h[:,2].max()   
        luVertNorm = cornersOcvVertNorm.min( axis=0 )
        rlVertNorm = cornersOcvVertNorm.max( axis=0 )
        luVert = luVertNorm * f2
        rlVert = rlVertNorm * f2
        newSize = ( int( math.ceil( rlVert[0]-luVert[0] ) ),
                    int( math.ceil( rlVert[1]-luVert[1] ) ) )
        K2 = np.array( [ [ f2,  0, -luVert[0] ],
                         [  0, f2, -luVert[1] ],
                         [  0,  0,          1 ] ] )
        
        H = K2.dot( Rocv.dot( Kinv ) )
        
        # direkte Umbildung: verwendet H direkt!
        cornersOcvVert = H.dot( cornersOcv_h.T ).T
        cornersOcvVert = cornersOcvVert[:,:2] / cornersOcvVert[:,2].reshape(-1,1)
        
        # indirekte Umbildung: verwendet (intern) inv(H) -> cv2.WARP_INVERSE_MAP nicht angeben!
        pho_rect3 = cv2.warpPerspective( src=pho, 
                                         M=H,
                                         dsize=newSize, 
                                         flags=cv2.INTER_LINEAR ,# | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0.,0.,0.,0.) )
                                         
        if plot:
            pyplot.figure(2); pyplot.title(name)
            pyplot.imshow( cv2.cvtColor( pho_rect3, cv2.COLOR_BGR2RGB ) )
        
        imgRect.path = os.path.join( dirAbsOri, name )
        cv2.imwrite( imgRect.path, pho_rect3 )
        
        imgRect.id = imgOri.id
        imgRect.H[:] = H
        imgRect.K[:] = K2
        imgRect.rc[:] = cornersOcvVert[:,1::-1]
        
    return imgRects, normalPtcl
