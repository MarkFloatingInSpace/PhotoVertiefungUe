# -*- coding: cp1252 -*-
import os, colorsys, subprocess, random, re, struct, urllib, base64
from collections import Counter
from xml.dom import minidom
from itertools import chain
import oriental
from oriental import Progress
from oriental.utils import ( traitlets as _traitlets,
                             gdal as gdal_utils,
                             zip_equal )
from contracts import contract

import numpy as np
from scipy import spatial
import cv2
# for some reason, OpenCV does not export cv2.cv any more, and there does not seem to be available a replacement for cv2.cv.RGB!
class Cv(object):
    @staticmethod
    def RGB( r, g, b ):
        return ( b, g, r, 0. )
cv2.cv = Cv()

from oriental import ori, graph, adjust, log

from oriental.utils import ( pyplot as plt,
                             pyplot_utils as plt_utils )
#if oriental.config.isDvlp:
#    from oriental.utils import mlab, mlab_utils
from oriental.relOri import web_gl
    
from oriental.relOri.SfMManager import SfMManager, ObjPtState

logger = log.Logger("relOri")

def createDirsOpen( fn, *args, **kwargs ):
    os.makedirs( os.path.dirname( fn ), exist_ok=True )
    return open( fn, *args, **kwargs )

class Printer(_traitlets.HasTraits):
    sfm = _traitlets.Instance(SfMManager)
    
    @contract
    def __init__( self,
                  sfmMgr : SfMManager ):
        super().__init__()
        self.sfm = sfmMgr
        self.cleanUpFns = set()

    @contract
    def imageBGR( self,
                  idx : int,
                  maxWidth : int = 0
                ) -> 'array[NxMx3]':
        "openCV reads images as BGR. For plotting with openCV, use img as is."
        return gdal_utils.imread( self.sfm.imgs[idx].fullPath, bands = gdal_utils.Bands.bgr, maxWidth = maxWidth )

    @contract
    def imageRGB( self, idx : int, maxWidth : int = 0 ):
        "For plotting with matplotlib, convert to RGB"
        return gdal_utils.imread( self.sfm.imgs[idx].fullPath, bands = gdal_utils.Bands.rgb, maxWidth = maxWidth )
        
    @contract
    def formatRanges( self, li : 'list(int)' ):
        ranges = []
        if len(li):
            l = sorted(li)
            beg = l[0]
            end = l[0]
            for i in range(1,len(l)):
                db = l[i]-l[i-1]
                if db==1:
                    end=l[i]
                else:
                    ranges.append( (beg,end) )
                    beg=l[i]
                    end=l[i]
            ranges.append( (beg,l[-1]) )
        r = ""
        for ra in ranges:
            if len(r):
                r = r + ", "
            if ra[0]==ra[1]:
                r = r + "{}".format(ra[0])
            else:
                r = r + "{}-{}".format(ra[0],ra[1])
        return r
    
    @contract
    def single( self,
                figNumber : 'int|str',
                title : str,
                iImg : int,
                ptsPix : 'array[Nx2](float)',
                objPts : 'array[Nx3](float)',
                bInliers : 'array[M](bool)',
                cam ):
        plt.figure(figNumber); plt.clf()
        plt.imshow( self.imageRGB(iImg), interpolation='none' )
        plt.autoscale(False)
        plt.title("img{}: {} inliers of {}".format( self.sfm.imgs[iImg].shortName, np.count_nonzero(bInliers), ptsPix.shape[0] ))    
        
        xProj = ori.projection( objPts, cam.t, cam.omfika, cam.ior, cam.adp )

        plt.scatter( x=ptsPix[~bInliers,0], y=-ptsPix[~bInliers,1], marker="o", s=4, color='r', edgecolors='m', linewidths = (1.,) )
        plt.plot( np.vstack( ( xProj[~bInliers,0], ptsPix[~bInliers,0]) ),
                  np.vstack( (-xProj[~bInliers,1],-ptsPix[~bInliers,1]) ), color='r', linewidth=2., marker=None )

        plt.scatter( x=ptsPix[bInliers,0], y=-ptsPix[bInliers,1], marker="o", s=4, color='w', edgecolors='m', linewidths = (1.,) )
        plt.plot( np.vstack( ( xProj[bInliers,0], ptsPix[bInliers,0]) ),
                  np.vstack( (-xProj[bInliers,1],-ptsPix[bInliers,1]) ), color='w', linewidth=2., marker=None )

    
    @contract
    def block( self,
               addFeatureTracks : dict,
               title : str,
               iImgs = None,
               objPts = None,
               save : bool = False ) -> str:
        # Visualize the block in 3D
        if 1:
            # WebGL
            #if not len(addFeatureTracks):
            # Use colors from photos to set the hue of plotted points.
            # TODO: consider addFeatureTracks
            iImgs = iImgs or self.sfm.orientedImgs
            nObjPts = len( objPts or self.sfm.featureTrack2ObjPt )
            objPts_  = np.ones( nObjPts,
                               dtype=[ ('X', np.float, 3), ('RGB', np.float, 3) ] )
            objPtsNImgPts = []
            if objPts is None:
                #objPts_[:]['X'] = np.array(self.sfm.featureTrack2ObjPt.values()).reshape((-1,3))
                # a NumPy-bug: np.array(.) does not handle dict views correctly: https://github.com/numpy/numpy/issues/573
                # A work-around would be to use an intermediate list. However, it may be huge!

                # get objPts and their colors
                for idx,(featureTrack,objPt) in enumerate(self.sfm.featureTrack2ObjPt.items()):
                    objPts_[idx]['X'][:] = objPt
                    imgFeatures = self.sfm.featureTrack2ImgFeatures[featureTrack]
                    objPtsNImgPts.append( len(imgFeatures) )
                    (iImg,iFeat) = imgFeatures[0] # simply use the first image
                    objPts_[idx]['RGB'][:] = self.sfm.imgs[iImg].keypoints[iFeat,2:5].reshape((1,1,3))

            else:
                objPts_[:]['X'][:] = objPts
     
            imgs = np.empty( len(iImgs), dtype=[ ( 'X0'      , np.float     , (3, ) ),
                                                 ( 'angles'  , np.float     , (3, ) ),
                                                 ( 'rotParam', (np.str,10) ),
                                                 ( 'whf'     , np.float     , (3,)  ), # extents of the pyramid representing the camera: image width / height / focal length in arbitrary units
                                                 ( 'name'    , (np.str,500)         ) ] )
            for idx in range(len(iImgs)):
                iImg = iImgs[idx]
                img = self.sfm.imgs[iImg]
                imgs[idx]['X0'][:] = img.t
                imgs[idx]['angles'][:] = img.omfika / 200. * np.pi
                imgs[idx]['rotParam'] = 'XYZ'
                imgs[idx]['whf'][:] = ( img.width, img.height, img.ior[2] / img.pix2cam.meanScaleForward() )
                imgs[idx]['name'] = "{}{}".format( img.shortName, "D" if iImg==iImgs[0] else "" )
            
            prcCpy = imgs[:]['X0'].copy()
            #tree = KdTree( prcCpy )
            #indxs,distsSqr = tree.knnSearch( prcCpy, 2 )
            #medMinimumInterPrcDist = np.median( distsSqr[:,1] )**.5
            tree = spatial.cKDTree( prcCpy )
            dists, _ = tree.query( prcCpy, k=2, n_jobs=-1)
            medMinimumInterPrcDist = np.median( dists[:,1] )

            if 0: # .ply ascii text
                with open( os.path.join( self.sfm.outDir, title + ".ply" ), 'w' ) as fout:
                    fout.write("""ply
format ascii 1.0
comment generated by OrientAL
element vertex {nVertices}
property double x
property double y
property double z
property uchar red
property uchar green
property uchar blue
property uchar nImgPts
element face {nFaces}
property list uchar int vertex_indices
end_header
""".format( nVertices=len(objPts_) + len(imgs)*5, nFaces=len(imgs)*6 ) )
                    for objPt,nImgPts in zip_equal(objPts_,objPtsNImgPts):
                        fout.write( '\t'.join(['{}']*7).format(*( objPt['X'].tolist() + (objPt['RGB']*255).astype(np.uint8).tolist() + [nImgPts] )) + '\n' )
                    for iImg in iImgs:
                        img = self.sfm.imgs[iImg]
                        pts = np.array([ [            0,             0,          0  ],
                                         [  img.width/2,  img.height/2, -img.ior[2] ],
                                         [ -img.width/2,  img.height/2, -img.ior[2] ],
                                         [ -img.width/2, -img.height/2, -img.ior[2] ],
                                         [  img.width/2, -img.height/2, -img.ior[2] ] ], dtype=np.float)
                        # scale pyramids
                        pts *= medMinimumInterPrcDist / 5 / img.ior[2]
                        pts = ori.omfika( img.omfika ).dot( pts.T ).T + img.t
                        for pt in pts:
                            fout.write( '\t'.join(['{}']*7).format( *(pt.tolist() + np.array([255,0,255],dtype=np.uint8).tolist() + [0]) ) + '\n' )
                    for idx,iImg in enumerate(iImgs):
                        img = self.sfm.imgs[iImg]
                        # CloudCompare does not support:
                        # - reading polylines from PLY files
                        # - faces with a vertex count other than 3                          
                        offset = len(objPts_) + idx * 5
                        for iVtxs in np.array([[0, 1, 2],
                                               [0, 2, 3],
                                               [0, 3, 4],
                                               [0, 4, 1],
                                               [2, 1, 3],
                                               [4, 3, 1]], dtype=np.int):
                            fout.write( ('3\t' + '\t'.join(['{}']*3) + '\n').format(*(iVtxs+offset)) )

            # .ply binary
            # Because CloudCompare does not support the import of lines from PLY-files,
            # let's display the coordinate axes as cuboids.
            nObjPts = len(objPts_)
            # title may actually contain directory separators, pointer to an inexistent directory. Hence, use openCreateDirs to create it, if necessary.
            with createDirsOpen( os.path.join( self.sfm.outDir, title + ".ply" ), 'wb' ) as fout:
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
""".format( nVertices=nObjPts + len(imgs)*(5+8*3), nFaces=len(imgs)*(6+6*2*3) ).encode('ascii') )
                stru = struct.Struct('<dddBBBB')
                for objPt,nImgPts in zip_equal(objPts_,objPtsNImgPts):
                    fout.write( stru.pack( *chain( objPt['X'], (objPt['RGB']*255).astype(np.uint8), [nImgPts] ) )  )

                for iImg in iImgs:
                    img = self.sfm.imgs[iImg]
                    pts = np.array([ [  0,  0,  0 ],
                                     [  1,  1, -1 ],
                                     [ -1,  1, -1 ],
                                     [ -1, -1, -1 ],
                                     [  1, -1, -1 ] ], float)
                    pts[:,0] *= img.width/2
                    pts[:,1] *= img.height/2
                    pts[:,2] *= img.ior[2] / img.pix2cam.meanScaleForward()
                    # scale pyramids
                    scale = medMinimumInterPrcDist / 3 / ( img.ior[2] / img.pix2cam.meanScaleForward() )
                    pts *= scale
                    R = ori.omfika( img.omfika )
                    pts = R.dot( pts.T ).T + img.t
                    for pt in pts:
                        fout.write( stru.pack( *chain( pt, np.array([255,0,255],dtype=np.uint8), [0] ) ) )
                    
                    # axes as cuboids
                    axLen = ( img.width + img.height ) / 4.
                    axWid = axLen / 20.
                    pts = np.array([ [ 0.,  axWid,  axWid ],
                                     [ 0., -axWid,  axWid ],
                                     [ 0., -axWid, -axWid ],
                                     [ 0.,  axWid, -axWid ] ] )
                    pts = np.r_[ pts, pts + np.array([ axLen, 0., 0. ]) ] * scale
                    for iAx in range(3):
                        if iAx==0:
                            R2 = np.eye(3)
                            col = np.array([255,0,0],np.uint8)
                        elif iAx==1:
                            R2 = np.array([[  0, 1, 0 ],
                                           [ -1, 0, 0 ],
                                           [  0, 0, 1 ]],float).T
                            col = np.array([0,255,0],np.uint8)
                        else:
                            R2 = np.array([[  0, 0, 1 ],
                                           [  0, 1, 0 ],
                                           [ -1, 0, 0 ]],float).T
                            col = np.array([0,0,255],np.uint8)
                        pts_ = R.dot(R2).dot( pts.T ).T + img.t
                        for pt in pts_:
                            fout.write( stru.pack( *chain( pt, col, [0] ) ) )

                stru = struct.Struct('<BIII')
                for idx,iImg in enumerate(iImgs):
                    img = self.sfm.imgs[iImg]
                    # CloudCompare does not support:
                    # - reading polylines from PLY files
                    # - faces with a vertex count other than 3                          
                    offset = nObjPts + idx * (5+8*3)
                    for iVtxs in np.array([[0, 1, 2],
                                           [0, 2, 3],
                                           [0, 3, 4],
                                           [0, 4, 1],
                                           [1, 2, 3],
                                           [3, 4, 1]], int):
                        fout.write( stru.pack( *chain( [3], iVtxs+offset ) ) )
                    for iAx in range(3):
                        offset = nObjPts + idx * (5+8*3) + 5 + 8*iAx
                        for iVtxs in np.array([[0, 3, 2],
                                               [2, 1, 0],
                                               [0, 1, 5],
                                               [5, 4, 0],
                                               [0, 4, 7],
                                               [7, 3, 0],
                                               [2, 3, 7],
                                               [7, 6, 2],
                                               [1, 2, 6],
                                               [6, 5, 1],
                                               [4, 5, 6],
                                               [6, 7, 4]], int):
                            fout.write( stru.pack( *chain( [3], iVtxs+offset ) ) )

            # TODO: export in Snavely's Bundler format for CloudCompare: bundle.out

            axisLen = medMinimumInterPrcDist / 5
            cameraPositionZ = medMinimumInterPrcDist * 10
            cameraFarPlane= ( objPts_[:]['X'].max() - objPts_[:]['X'].min() ) * 2
            if 1:
                # reduce by the coordinate-wise median, and scale uniformly in all directions by sigma MAD of X,Y-coordinates
                median = np.median( imgs[:]['X0'], axis=0 )
                sigma = 1.4826 * np.median( np.abs( imgs[:]['X0'] - median )[:,:2] )
                imgs[:]['X0']   = ( imgs[:]['X0']   - median ) / sigma
                objPts_[:]['X'] = ( objPts_[:]['X'] - median ) / sigma
                imgs[:]['whf'] /= sigma
                axisLen /= sigma
                cameraPositionZ /= sigma
                cameraFarPlane /= sigma


            fn = os.path.join( self.sfm.outDir, title + ".html" )
            web_gl.plotBlockWebGL( fn, objPts_, imgs, axisLen, cameraPositionZ, cameraFarPlane )

            return fn
        
        if 0:
            # Mayavi.mlab
            # speed up?:
            # http://docs.enthought.com/mayavi/mayavi/tips.html#accelerating-a-mayavi-script
            
            mode = 'sphere' # 'point' seems to be always 1px large, not influenced by scale_factor
            scale_factor = 0.01 # in drawing units
            
            # don't plot extreme coordinates, as that would make navigating in the plot difficult
            def skipme( pts ):
                return np.any( np.logical_or( np.absolute(pts) < -100, np.absolute(pts) > 100  ), axis=1 )
                
            fig = mlab.figure(title); mlab.clf()
            
            if not len(addFeatureTracks):
                # final plot for publication.
                # Use colors from photos to set the hue of plotted points.
                # Save figure to file.
                # mayavi unterstützt RGB-Werte nur, um ganze VTK-Objekte einzufärben - aber nicht einzelne Punkte, die gemeinsam via points3d geplottet werden.
                # Einzelne Punkte zu plotten ist extrem langsam.
                # Aber: mayavi hat eine eingebaute 'hsv' Farbtabelle! Damit lässt sich zumindest der Farbton ausgeben.
                #       Das erzeugt sinnvolle Farben für Punkte, deren Farbton gut bestimmt ist (Vegetation), aber zufällige Farben für Punkte 'ohne' Farbton (schwarz/weiß, Straße, etc.)
                # for some reason, mayavi wants the scalars to be an array of floats, not dtype=np.uint8
                # okay - scalars are interpolated (and can be used to scale the geometry)
                scalars = np.empty( (len(self.sfm.featureTrack2ObjPt)) )
                rgb = np.empty( (len(self.sfm.featureTrack2ObjPt),3) )
                normals = np.zeros( (len(self.sfm.featureTrack2ObjPt),3) )
                idx = 0
                for featureTrack in self.sfm.featureTrack2ObjPt.keys():
                    (iImg,iFeat) = self.sfm.featureTrack2ImgFeatures[featureTrack][0] # just use the first image
                    keyPt = self.sfm.imgs[iImg].keypoints[iFeat,:2]
                    #col = images[iImg][keyPt[1],keyPt[0],:].reshape((1,1,3))
                    col = self.sfm.imgs[iImg].keypoints[iFeat,2:5].astype( dtype=np.uint8 ).reshape((1,1,3))
                    colHsv = cv2.cvtColor( col, cv2.COLOR_RGB2HSV )
                    scalars[idx] = colHsv[0,0,0] # hue
                    rgb[idx,:] = col
                    
                    for iImg,iFeat in self.sfm.featureTrack2ImgFeatures[featureTrack]:
                        normal = self.sfm.imgs[iImg].t - self.sfm.featureTrack2ObjPt[featureTrack]
                        normal /= np.linalg.norm( normal )
                        normals[idx,:] += normal
                
                    normals[idx,:] /= np.linalg.norm( normals[idx,:] )
                        
                    idx += 1
                if 1:
                    pts = np.array(self.sfm.featureTrack2ObjPt.values()).reshape((-1,3))
                    skip = skipme( pts )
                    mlab.points3d(  pts[~skip,0],  pts[~skip,1],  pts[~skip,2], scalars[~skip], colormap='hsv', mode=mode, scale_mode='none', scale_factor=scale_factor, reset_zoom=True, vmin=0, vmax=255 )
                else:
                    # plot each point separately with its RGB-colours. Works, but is prohibitively slow.
                    for idx,pt in enumerate(self.sfm.featureTrack2ObjPt.values()):
                        if np.sum( np.any( np.logical_or( np.absolute(pt) < -100, np.absolute(pt) > 100  ) ) ):
                            continue
                        # mayavi wants RGB-values in the range [0.,1.]
                        mlab.points3d( pt[0], pt[1], pt[2], color=(rgb[idx,0]/255.,rgb[idx,1]/255.,rgb[idx,2]/255.), mode=mode, scale_mode='none', scale_factor=scale_factor, reset_zoom=True )
                        if idx > 100:
                            break
                        
                if 0: # matplotlib kann 3d mit RGB-Farben plotten!                                                     
                    pts = np.array(self.sfm.featureTrack2ObjPt.values()).reshape((-1,3))
                    skip = skipme( pts )
                    plt_utils.plotPointCloud( pts[~skip,:], rgb[~skip,:], title )
                        
                if save:
                    # save refsys to text file
                    representative2lineIdx = {}
                    with open( "{}/pointcloud_{}.txt".format(self.sfm.outDir,title), 'wt' ) as fout:
                        fout.write( "# idx X Y Z nX nY nZ R G B\n" )
                        for idx,(repr,pt) in enumerate(self.sfm.featureTrack2ObjPt.items()): 
                            representative2lineIdx[repr]=idx
                            fout.write( "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format( idx,
                                                                                           pt[0], pt[1], pt[2],
                                                                                           normals[idx,0], normals[idx,1], normals[idx,2],
                                                                                           rgb    [idx,0], rgb    [idx,1], rgb    [idx,2] ) )
                    with open( "{}/cameras_{}.txt".format(self.sfm.outDir,title), 'wt' ) as fout:
                        fout.write( "# name PRC_X PRC_Y PRC_Z omega[gon] phi[gon] kappa[gon] x_0[px] y_0[px] c[px]\n" )
                        for iImg in self.sfm.orientedImgs:
                            img = self.sfm.imgs[iImg]
                            fout.write( "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format( img.shortName + ( "D" if iImg==self.sfm.orientedImgs[0] else "" ),
                                                                                           img.t[0], img.t[1], img.t[2],
                                                                                           img.omfika[0], img.omfika[1], img.omfika[2],
                                                                                           img.ior[0], img.ior[1], img.ior[2] ) )
                    imgObsFouts = []
                    for idx in range(len(self.sfm.orientedImgs)): 
                        imgObsFouts.append( open( "{}/imgObs_{}_{}.txt".format(self.sfm.outDir,self.sfm.imgs[idx].shortName,title), 'wt' ) )
                        imgObsFouts[-1].write( "# objPtIdx x y\n" )
                                     
                    for repr,features in self.sfm.featureTrack2ImgFeatures.items():    
                        lineIdx = representative2lineIdx[repr]
                        for feature in features:   
                            iImg   = feature[0]
                            iKeyPt = feature[1]
                            keyPt = self.sfm[iImg].keypoints[iKeyPt,:2]
                            imgObsFouts[iImg].write( "{:6}\t{:7.2f}\t{:7.2f}\n".format( lineIdx, keyPt[0], keyPt[1] ) )    
                                                  
                    for idx in range(len(imgObsFouts)): 
                        imgObsFouts[idx].close()
                                  
                
            else:
                oldPts     = [ item[1] for item in self.sfm.featureTrack2ObjPt.items() if item[0] not in addFeatureTracks ]
                commonPts  = [ item[1] for item in self.sfm.featureTrack2ObjPt.items() if item[0]     in addFeatureTracks and addFeatureTracks[item[0]] == ObjPtState.common ]
                newPts     = [ item[1] for item in self.sfm.featureTrack2ObjPt.items() if item[0]     in addFeatureTracks and addFeatureTracks[item[0]] == ObjPtState.new    ]
                oldPts     = np.array(oldPts).reshape((-1,3))
                commonPts  = np.array(commonPts).reshape((-1,3))
                newPts     = np.array(newPts).reshape((-1,3))
            
                # most of all, we are interested in common points - so print them last.
                skip = skipme( oldPts )
                mlab.points3d(     oldPts[~skip,0],     oldPts[~skip,1],     oldPts[~skip,2], color=(.1,.1,.1), mode=mode, scale_mode='scalar', scale_factor=scale_factor*1./3., reset_zoom=False )
                skip = skipme( newPts )
                mlab.points3d(     newPts[~skip,0],     newPts[~skip,1],     newPts[~skip,2], color=(1,1,1)   , mode=mode, scale_mode='scalar', scale_factor=scale_factor*2./3., reset_zoom=False )
                skip = skipme( commonPts )
                mlab.points3d(  commonPts[~skip,0],  commonPts[~skip,1],  commonPts[~skip,2], color=(1,0,0)   , mode=mode, scale_mode='scalar', scale_factor=scale_factor, reset_zoom=False )
            
            for iImg in self.sfm.orientedImgs:
                img = self.sfm.imgs[iImg]
                mlab_utils.camera( img.ior, img.R, img.t, "{}{}".format( img.shortName, "D" if iImg==self.sfm.orientedImgs[0] else "" ) )
            if len(addFeatureTracks):
                mlab_utils.legend( ["old","common","new"], [(.1,.1,.1),(1,0,0),(1,1,1)] )
            mlab.view( azimuth=90, elevation=-45, distance=5, focalpoint=(0,0,0) )
            
            if not len(addFeatureTracks) and \
               self.sfm.outDir:
                # png, jpg, bmp, tiff, ps, eps, pdf, rib (renderman), oogl (geomview), iv (OpenInventor), vrml, obj (wavefront)
                # why is the VTK File Format not supported?
                # vtkRIBExporter reports an error in the vtkOutputWindow: 'Bad representation sent'
                # Wavefront obj is saved without colors!
                # ParaView
                # - does not load VRML colors as colors,
                # - crashes on import of .obj if cameras have been plotted, cannot import other 3D formats.
                # - VTK-files may be saved using the mayavi plot window: 'View the Mayavi pipeline'->
                # Instant Player correctly loads VRML files, but seems not handy.
                # Display with Mayavi2 Application:
                # - needs additional Python package 'envisage', install with 'pip install envisage'
                # - start with 'python C:\Python27-x64\Scripts\mayavi2-script.pyw'
                # - cannot read formats .iv, .rib, .oogl
                # - reads format .obj, but does not import/display anything!?
                # - reads and imports VRML files, but only with extension .wrl, and does not correctly display point colours (InstantPlayer does!)
                # - the orientation of camera tretrahedron surfaces seems to be wrong (inside out -> rendered as black)
                # Meshlab crashes on import of .obj, reports error on import of VRML: 'file without geometry'
                # -> Just make a snapshot!
                for fmt in [ 'vrml' ]: 
                   mlab.savefig( "{}/{}.{}".format(self.sfm.outDir,title,fmt) )
                   
            return fig
    
    @contract
    def imageResiduals( self,
                        figureNumber,
                        iImg : int,
                        features : 'array[N](int)',
                        addFeatureTracks : dict = {},
                        residualScale : 'float,>0' = 1.,
                        save : bool = True ):
        scale = False#True # save memory
        x = []
        X = []
        tracks = []
        img = self.sfm.imgs[iImg]  
        for iFeat in features:
            found,featureTrack = self.sfm.featureTracks.component( graph.ImageFeatureID( iImg, int(iFeat) ) )
            assert found
            x.append( img.keypoints[iFeat,:2] )
            X.append( self.sfm.featureTrack2ObjPt[featureTrack] )

            track = addFeatureTracks.get( featureTrack, -1 )
            tracks.append(track)

        x = np.array(x)
        X = np.array(X)
        xProj = ori.projection( X, img.t, img.omfika, img.ior, img.adp )

        # For plotting, we want pixels. But the returned residualNormsSqr must be in camera coordinates!
        residualNormsSqr = np.sum( (img.pix2cam.forward(x) - xProj)**2, axis=1 )

        xProj = img.pix2cam.inverse(xProj)

        if scale:
            x /= 2.
            xProj /= 2.
        tracks = np.array(tracks)

        if residualScale != 1.:
            xProj = x + ( xProj - x ) * residualScale
            
        tit = "{} {}{}".format( figureNumber, self.sfm.imgs[iImg].shortName, " residual scale:{}".format(residualScale) if residualScale!=1. else "" )
        # with tight_layout=True, matplotlib issues a user warning when calling savefig( ..., bbox_inches='tight' )
        # savefig( ..., bbox_inches='tight' ) alone seems to do the job anyway, so skip tight_layout here.
        fig = plt.figure( tit )
        plt.clf()
        if scale:
            plt.imshow( self.imageRGB(iImg)[0::2,0::2,:], interpolation = None if save else 'none' )
        else:
            plt.imshow( self.imageRGB(iImg), interpolation = None if save else 'none' ) # nn-interpolation does not speed up plotting significantly!
        plt.autoscale(False)

        for val,col in zip_equal( (-1        ,ObjPtState.common,ObjPtState.new),
                                  ((0.,1.,1.),(1,0,0)          ,(1,1,1)) ):
            sel = tracks==val
            if not sel.any():
                continue
            #plt.scatter( x=x[sel,0],y=-x[sel,1], color=col, marker="o", s=2, edgecolors=(1.,0.,1.), linewidths = (1.,) )
            plt.plot( x[sel,0], -x[sel,1], 'om', markeredgecolor='m', markersize=2 )
            
            plt.plot( np.vstack( ( xProj[sel,0], x[sel,0]) ),
                      np.vstack( (-xProj[sel,1],-x[sel,1]) ), color='c', linewidth=1., marker=None )
    
        plt.title(tit)
        if save:
            residualsDir = os.path.join( self.sfm.outDir, 'residuals' )
            direc, fn = os.path.split(self.sfm.imgs[iImg].shortName + '.jpg')
            if figureNumber:
                fn = str(figureNumber) + '_' + fn
            fn = os.path.join( residualsDir, direc, fn )
            os.makedirs( os.path.dirname(fn), exist_ok=True )
            # saving as JPEG requires PIL to be installed!
            plt.savefig( fn, bbox_inches='tight', dpi=150 )
            plt.close(tit) #  for some reason, plt.close(fig) doesn't work 

        return residualNormsSqr

    @contract
    def allImageResiduals( self,
                           figureNumber,
                           addFeatureTracks : dict,
                           residualScale : float = 10.,
                           save : bool = True ) -> 'array[N](float,>0)':
        "image residuals for every image in the block"
        logger.info('Plot residuals for each image')
        iImg2features = {}
        for (iImg, iFeat) in self.sfm.imgFeature2costAndResidualBlockID:
            iImg2features.setdefault( iImg, [] ).append( iFeat )
        allImgResNormsSqr = []
        imgMsgs = []
        fmt = '{}\t' + '\t'.join(['{:.2f}']*3)
        progress = Progress(len(iImg2features))
        for iImg,features in iImg2features.items():
            features = np.array( features, dtype=int )
            residualNormsSqr = self.imageResiduals( figureNumber, iImg, features, addFeatureTracks, residualScale, save )
            allImgResNormsSqr.extend(residualNormsSqr.tolist())
            imgMsgs.append( fmt.format(
                self.sfm.imgs[iImg].shortName,
                np.median(residualNormsSqr)**.5,
                np.mean  (residualNormsSqr)**.5,
                np.max   (residualNormsSqr)**.5  ) )
            progress += 1
        allImgResNormsSqr = np.array(allImgResNormsSqr)
        header = ['Image residual norms',
                  'pho\tmed\tmean\tmax',
                  fmt.format( 'all',
                              np.median(allImgResNormsSqr)**.5,
                              np.mean  (allImgResNormsSqr)**.5,
                              np.max   (allImgResNormsSqr)**.5 ) ]
        logger.info( '\n'.join( header + imgMsgs) )
        return allImgResNormsSqr

    def allImgResNormsHisto( self,
                             resNormSqr : 'array[N](float,>0)',
                             maxResNorm : 'float,>0|None',
                             fn : str = '' ):
        # we better not compute the sqrt of all squared residuals, but get the histogram of the squared residuals, for the squared bin edges. Then plot the histogram with the un-squared bin edges.
        # don't choke on extreme values!
        maxResNorm = maxResNorm or np.percentile(resNormSqr,99)**.5

        pow2 = 8
        bins = np.linspace( 0, maxResNorm, 1 + 2**pow2 ) # hist will not have an item for the right-most edge, so add 1 to support downsampling by a factor of 2.
        binsSqr = bins**2
        hist,_ = np.histogram( resNormSqr, bins=binsSqr )
        bins = bins[:-1] # plt.bar(.) wants the left bin edges only for it's 'left' argument.
        # sum neighbor bin pairs until there is a meaningful count of residual norms in the maximum bin
        while pow2 >= 4 and hist.max() < 100:
            bins = bins[0::2]
            hist = hist[0::2] + hist[1::2]
            pow2 -= 1
        plt.figure('residuals'); plt.clf()
        plt.bar( x=bins, height=hist, width=bins[1]-bins[0], color='b', linewidth=0 )
        plt.ylabel('residual count', color='b')
        plt.xlabel('residual norm')
        plt.xlim( right=maxResNorm )
        nShown = hist.sum()
        if nShown < len(resNormSqr):
            prefix = '{} of '.format( nShown )
        else:
            prefix = 'all '

        plt.title( prefix + "{} img residual norms; max:{:.2f}".format(
                            len(resNormSqr), resNormSqr.max()**.5 ))
        if fn:
            plt.savefig( os.path.join( self.sfm.outDir,  fn ), bbox_inches='tight', dpi=150 )
            plt.close('residuals') #  for some reason, plt.close(fig) doesn't work 

    
    #def allMatches( self, prefix ):
    #    # save plots of matches for each image pair
    #    outdir_ = os.path.join( outdir, 'matches' )
    #    if not os.path.isdir(outdir_):
    #        os.mkdir(outdir_)
    #    
    #    for edge,matches in self.sfm.edge2matches.items():
    #        cvMatches = []
    #        keypoints1 = []
    #        keypoints2 = []
    #        for iMatch in range(matches.shape[0]):
    #            # need to provide unused Ctor-arg distance
    #            #dMatch = cv2.DMatch( matches[iMatch,0], matches[iMatch,1], 1. )
    #            cvMatches.append( cv2.DMatch( iMatch, iMatch, 1. ) )
    #            
    #            iFeature = matches[iMatch,0]
    #            x = allKeypoints[edge[0]][iFeature,0]
    #            y = allKeypoints[edge[0]][iFeature,1] 
    #            keypoints1.append( cv2.KeyPoint( x, y, 1. ) )
    #            
    #            iFeature = matches[iMatch,1]
    #            x = allKeypoints[edge[1]][iFeature,0]
    #            y = allKeypoints[edge[1]][iFeature,1] 
    #            keypoints2.append( cv2.KeyPoint( x, y, 1. ) )
    #                        
    #        img_match = cv2.drawMatches( images[edge[0]], keypoints1,
    #                                     images[edge[1]], keypoints2,
    #                                     cvMatches,
    #                                     matchColor=(0.,0.,0.,0.) ) # , matchColor=(0.,0.,255.,0.)
    #        cv2.imwrite( "{}/{}_{}-{}.jpg".format( outdir_, prefix, self.sfm.imgs[edge[0]].shortName, self.sfm.imgs[edge[1]].shortName ), img_match )     
    
    def connectivityMatrix( self, connMat, title ):
        # create a masked array from connMat, where zeros are masked
        # for masked values, colormap does not lookup its regular color table, but it returns a specific color
        # colormap.bad        
        mx = np.ma.masked_array( connMat, mask=connMat==0)                    
        from matplotlib import colors
        plt.imshow( mx, aspect='equal', interpolation='none', norm=colors.LogNorm() )
        #plt.jet()
        plt.spectral()
        plt.colorbar()
        plt.title(title)
    

    def cleanUp( self ):
        for fn in self.cleanUpFns:
            try:
                os.remove( fn )
            except:
                logger.warning( 'File could not be deleted: {}', fn )

    @contract
    def connectivityGraph( self, name : str, countObs : bool, minCut : bool = False ):
        # we need to provide low-resolution images to dot, or otherwise, it crashes, produces corrupt output, etc.
        dotImgFns = []
        for iImg in range(len(self.sfm.imgs)):
            fn = os.path.join( self.sfm.outDir, os.path.splitext( os.path.basename(self.sfm.imgs[iImg].fullPath) )[0] + '.png' )
            dotImgFns.append(fn)
            if os.path.exists( fn ):
                continue
            img = self.imageBGR(iImg,150)
            #if 0: # much faster
            #    step = img.shape[0] // 150
            #    theImg = img[::step,::step,:]
            #else:
            #    width = 150.
            #    fac = width / img.shape[1]
            #    theImg = cv2.resize( img, dsize=( int(width), int(fac * img.shape[0]) ), interpolation=cv2.INTER_LANCZOS4 )
            cv2.imwrite( fn, img )
        
        self.cleanUpFns.update( dotImgFns )

        iImg2nObs = Counter()
        if countObs:
            edge2nMatches = Counter()
            for features in self.sfm.featureTrack2ImgFeatures.values():
                for i1 in range(len(features)):
                    iImg1 = features[i1][0]
                    iImg2nObs[iImg1] += 1
                    for i2 in range(i1+1,len(features)):
                        iImg2 = features[i2][0]
                        if iImg1 < iImg2:
                            edge2nMatches[(iImg1,iImg2)] += 1
                        else:
                            edge2nMatches[(iImg2,iImg1)] += 1
        
        else:    
            edge2nMatches = { edge if edge[0]<edge[1] else (edge[1],edge[0]) : matches.shape[0] for edge,matches in self.sfm.edge2matches.items() }                 
            for edge,matches in self.sfm.edge2matches.items():
                iImg2nObs[edge[0]] += matches.shape[0]
                iImg2nObs[edge[1]] += matches.shape[0]
            
        if len(edge2nMatches)==0:
            logger.info("No matches defined, nothing to print")
            return
        #nMatchesPerEdge = sorted( edge2nMatches.values() )
        # bei sehr vielen phos und Kanten sind die Kantenlabels praktisch nicht mehr lesbar.
        # Vorschlag von cb: statt Kantenlabels unterschiedliche Kantenfarben nutzen + Farbtabelle!
        # Farbgebung der Kanten über Quantile = robust gegen Verzerrung der Farbgebung durch "Ausreißer" in der Anzahl der Matches.

        # We want to visualize the expected quality of relative orientation of image pairs.
        # The number of matches per image pair has an absolute meaning in this context, independent of the data: there is a minimum number required to orient an image pair, and beyond a high number of matches (e.g. 300), differences to not matter.
        # Thus, better use fixed bin edges.
        nColors = 7
        lowerBoundsExclusive = [0] + [ 10*2**exponent for exponent in range(nColors-1) ]

        #nColors = min( 7, len(set(nMatchesPerEdge)) )
        #quantiles = [ nMatchesPerEdge[ int(round(float(idx)/nColors*len(nMatchesPerEdge))) ] for idx in range(1,nColors) ]
        
        def getBin( nMatches ):
            for idx in range(nColors-1,-1,-1):
                if nMatches > lowerBoundsExclusive[idx]:
                    return idx
            assert False
        
        # Graphviz expects HSV-color values in the range [0;1]
        #minHue = 0.17 # yellow
        #maxHue = 1.   # red
        #hues = np.linspace( minHue, maxHue, nColors )
        #HSVs = np.empty( (hues.shape[0],3) )
        #HSVs[:,0] = hues
        #HSVs[:,1] = np.linspace( 0.4, 1., nColors )
        #HSVs[:,2] = np.linspace( 1., 0.4, nColors )
        saturation = 100 # constant, so the various hues are maximally perceivable
        angles = np.linspace( 0, 2*np.pi, nColors, dtype=np.float32 ) + np.pi/2 # start at light yellow -> green -> blue -> red -> magenta ->  very dark yellow (brown)
        lab = np.c_[ np.linspace( 100, 0, nColors, dtype=np.float32 ), # from full luminance at yellow to complete darkness at brown
                     np.cos(angles)*saturation,
                     np.sin(angles)*saturation
                   ].reshape((1,-1,3)) # for float32, cv2 expects luminance (L) in [0;100]. Similar for a & b, but the radius of representable colours depends on the luminance.
        HSVs = cv2.cvtColor( cv2.cvtColor( lab, cv2.COLOR_LAB2RGB ), cv2.COLOR_RGB2HSV ).squeeze()
        HSVs[:,0] /= 360 # for float32, OpenCV's Hue is in [0;360]
        minWidth = 0.5 # [pt]; 1pt == 1/72 inch == 0.35mm
        maxWidth = 3.0 # [pt]
        penwidths = np.linspace( minWidth, maxWidth, nColors )
    
        # http://www.graphviz.org/doc/FAQ.html#Q32
        # For a large data set (300 phos, 34897 image connections), running dot.exe takes half an hour!
        # passing -v shows that its not the positioning of the nodes that takes long, but it is the layout of the edges.
        # splines=spline|polyline|ortho takes very long.
        # splines=line|curved takes seconds only
        # Thus, for large number of edges, use line instead of spline.
        fn = "{}/{}ConnectivityGraph.dot".format( self.sfm.outDir, name )
        # If size ends in an exclamation point (!), then it is taken to be the desired size. In this case, if both dimensions of the drawing are less than size, the drawing is scaled up uniformly until at least one dimension equals its dimension in size. 
        # Do not specify size, but let graphviz derive it from the fixed size of nodes (images)
        #size="20.,20.!";
        # box, rect and rectangle are synonyms when specifying 'shape'
        # outputorder=edgesfirst -> make sure that nodes are printed after the edges, such that node labels will not be hidden under edges.
        with open( fn, "wt" ) as fout: 
            fout.write("""strict graph ImageConnectivity {
    node [ shape=rectangle, margin=0, regular=false, style="filled", fontsize=11, fillcolor="grey", width=0.5, height=0.3327, fixedsize=true, imagescale=true ];
    edge [ fontcolor=blue, fontsize=8, penwidth=0.1 ];
    layout=neato;
    model=mds;
    overlap=false;
    outputorder=edgesfirst;
    """ )
            fout.write( "splines={};\n".format( 'spline' if len(edge2nMatches) < 1000 else 'line' )  )

            fout.write( 'imagepath="{}";\n'.format( os.path.abspath(self.sfm.outDir) ) )
    
            fout.write( 'legend [ width=0.9, height={}, fillcolor=transparent, label=<<table border="0" cellborder="0" cellpadding="0" cellspacing="0">'.format( 0.2*(nColors+1)+0.1 ) )
            fout.write( '<tr><td>#matches</td></tr>' )
            for idx in range(nColors-1,-1,-1):
                rgbBackground = [ int(round(col*255)) for col in colorsys.hsv_to_rgb( *HSVs[idx] ) ]
                rgbFont = [0]*3 if HSVs[idx,2] > .5 else [255]*3
                fout.write( '<tr><td BGCOLOR="#{:02X}{:02X}{:02X}">'
                            '<font COLOR="#{:02X}{:02X}{:02X}">&gt;{}</font>'
                            '</td></tr>'.format( *rgbBackground,
                                                 *rgbFont,
                                                 #"" if idx < len(quantiles) else "=",
                                                 lowerBoundsExclusive[idx]# if idx < len(quantiles) else nMatchesPerEdge[-1]#"&infin;"
                                                ) )
            fout.write('</table>> ];\n')
            
            fout.write( 'filenames [ fillcolor=transparent, label="{}" fixedsize=false ];\n'.format(self.sfm.shortFileNames.commonName.replace('\\','/') ) )    
            
            for idx in range(len(self.sfm.imgs)):
                # dot does not accept node/edge IDs that e.g. start with a digit - unless the IDs are double-quoted!
                #fout.write( '"{}" [ image="{}" ];\n'.format( self.sfm.imgs[idx].shortName, os.path.basename(self.sfm.imgs[idx].fullPath) ) )
                fout.write( ('"{0}" [ label=<<table border="0" cellborder="0" cellpadding="0" cellspacing="0">' +
                                                 '<tr><td align="left"><font color="magenta">{0}</font></td></tr>' +
                                                 '<tr><td align="right"><font point-size="6" color="red">{1}</font></td></tr>' +
                                            '</table>>, ' +
                                     'tooltip="\\N: {1} imgPts",' +       
                                     'image="{2}" ];\n').format( self.sfm.imgs[idx].shortName,
                                                                 iImg2nObs[idx],  
                                                                 os.path.basename( dotImgFns[idx] ) ) )
            
            # The order in which Graphviz outputs the edges to svg cannot be influenced by the order in which they appear in the .dot-file :-(
            #for edge,nMatches in sorted( edge2nMatches.items(), key=lambda x: x[1] ):
            for edge,nMatches in edge2nMatches.items():
                # graphviz dot seems to render edges in unspecified order. It would be nice to render edges for many matches on top of edges for fewer matches. 
                # There may be many, many edges. Thus, don't plot the edge labels, and make the lines thinner. Otherwise, the plot is hardly readable.
                # Define edge tooltips instead of labels, which serve as mouse tooltips for svg's loaded in a browser.
                # graphviz dot does not support exponential (scientific) notation. Thus, for the edge length, use fixed-point notation with a large number of digits after the comma.
                iBin = getBin(nMatches)
                fout.write( '"{}" -- "{}" [len={:.14f}, tooltip="\\E: {} {}", penwidth={:.2f}, color="{:.2f} {:.2f} {:.2f}"];\n'.format(
                        self.sfm.imgs[edge[0]].shortName,
                        self.sfm.imgs[edge[1]].shortName,
                        1./nMatches,
                        nMatches,
                        "common objPts" if countObs else "matches",
                        penwidths[iBin],
                        *HSVs[iBin] ) )
                            
            
            fout.write("}\n")
            
        args = [ oriental.config.dot, '-O', fn, '-q' ] # -q: suppress warning messages like "Warning: some nodes with margin (3.20,3.20) touch - falling back to straight line edges"
        args += [ '-T{}'.format(fmt) for fmt in [ 'png', 'svg' ] ] # , 'eps': cannot print &infin; contained in legend
        subprocess.check_call( args )

        # post-process the svg's: embed the images instead of referencing them as external files.
        with minidom.parse( fn + '.svg' ) as dom:
            images = dom.getElementsByTagName('image')
            for image in images:
                for idx in range(image.attributes.length):
                    attribute = image.attributes.item(idx)
                    if attribute.localName == 'href':
                        with open( os.path.join( self.sfm.outDir, attribute.value ),'rb' ) as fin:
                            encoded = urllib.parse.quote( base64.standard_b64encode( fin.read() ) )
                        attribute.value = 'data:image/png;base64,' + encoded
            
            # write the XML-tree in descending order of matches
            # "310197--310531: 6 matches"
            rex = re.compile(r'.*?:\s(?P<nmatches>\d+) (matches|common\sobjPts)')
            gs = dom.getElementsByTagName('g')
            ggraph = [ el for el in gs if el.getAttribute('class') == 'graph' ]
            assert(len(ggraph)==1)
            ggraph = ggraph[0]
            edgesNmatches = []
            for g in gs:
                if g.getAttribute('class') != 'edge':
                    continue
                gsNested = g.getElementsByTagName('g')
                assert( len(gsNested) == 1 )
                as_ = gsNested[0].getElementsByTagName('a')
                assert( len(as_)== 1 )
                title = as_[0].getAttribute('xlink:title')
                assert( len(title) )
                m = rex.match(title)
                nMatches = int(m.group('nmatches'))
                edgesNmatches.append( (g,nMatches) )

            edgesNmatchesSorted = sorted( edgesNmatches, key=lambda x: x[1] )
            old2newEdge = { old[0]:new[0] for old,new in zip_equal(edgesNmatches,edgesNmatchesSorted) }
        
            def writeElementxml( self, writer, indent="", addindent="", newl="" ):
                node = old2newEdge.get(self, self)
                node.oldwritexml( writer, indent, addindent, newl )

            minidom.Element.oldwritexml = minidom.Element.writexml
            minidom.Element.writexml = writeElementxml
            with open( fn + '.svg', 'w', encoding='utf-8' ) as fout:
                dom.writexml( fout, newl="\n", encoding='UTF-8')
            
            # prevent side-effects in other code that uses minidom
            minidom.Element.writexml = minidom.Element.oldwritexml
            del minidom.Element.oldwritexml
            
            # dom.unlink() # done by context manager

            
        self.cleanUpFns.add(fn)

        if minCut:
            # vertex ids in graph.ImageConnectivity must start at 0 and be consecutive!!
            iImg2iGraphImg = {}
            for iGraphImg,iImg in enumerate(iImg2nObs):
                iImg2iGraphImg[iImg] = iGraphImg
            conn = graph.ImageConnectivity( len(iImg2iGraphImg) )
            for edge,nMatches in edge2nMatches.items():
                graphEdge = graph.ImageConnectivity.Edge( graph.ImageConnectivity.Image(iImg2iGraphImg[edge[0]]),
                                                          graph.ImageConnectivity.Image(iImg2iGraphImg[edge[1]]),
                                                          nMatches )
                conn.addEdgeQuality( graphEdge )
            minCut = conn.minCut( returnIndicesSmallerSet=True )
            iGraphImg2iImg = { val : el for el,val in iImg2iGraphImg.items() }
            minCut.idxsImagesSmallerSet = np.array( [ iGraphImg2iImg[int(iGraphImg)] for iGraphImg in minCut.idxsImagesSmallerSet ], minCut.idxsImagesSmallerSet.dtype )
            return minCut

    def residuals( self, residuals, maxRes, fn ):
        nShown = np.sum( np.logical_and( residuals >=-maxRes, residuals <= maxRes ) )
        if nShown < len(residuals):
            cutoff = '{} of '.format( nShown )
        else:
            cutoff = ''
        bins = np.linspace( -maxRes, maxRes, 50 )
        hist,_ = np.histogram( residuals, bins=bins )
        plt.figure('residuals'); plt.clf()
        plt.bar( x=bins[:-1], height=hist, width=bins[1]-bins[0], color='b' )
        plt.ylabel('residual count', color='b')
        plt.xlabel('residuals')
        plt.xlim([-maxRes,maxRes])
        plt.title( cutoff + "{} img residuals; max:{:.1f}".format(
                            len(residuals), residuals.max() ))
        plt.savefig( os.path.join( self.sfm.outDir,  fn ), bbox_inches='tight', dpi=150 )
        plt.close('residuals')

    @contract
    def residualHistAndLoss( self,
                             robLoss,
                             residuals : 'array[N](float)',
                             maxResNorm : 'float,>0|None' = None,
                             fn : str = ''
                           ) -> None:
        # overlay a histogram of residuals
        # we better not compute the sqrt of all squared residuals, but get the histogram of the squared residuals, for the squared bin edges. Then plot the histogram with the un-squared bin edges.
        resNormSqr = residuals[0::2]**2 + residuals[1::2]**2
        # don't choke on extreme values!
        maxResNorm = maxResNorm or np.percentile(resNormSqr,99)**.5

        bins = np.linspace( 0, maxResNorm, 50 )
        binsSqr = bins**2
        hist,_ = np.histogram( resNormSqr, bins=binsSqr )
        plt.figure('residuals'); plt.clf()
        #plt.hist( resNormSqr, bins=50, range=(0,maxResNorm), hold=True, color='b' )
        plt.bar( x=bins[:-1], height=hist, width=bins[1]-bins[0], color='b' )
        plt.ylabel('residual count', color='b')
        plt.xlabel('residual norm')
        plt.xlim( right=maxResNorm )

        #squaredLoss = adjust_loss.Trivial()
        #resNorm = np.linspace( 0, maxResNorm, 100 )
        #residualLoss = np.empty_like( resNorm )
        #residualLossTrivial = np.empty_like( resNorm )
        #for idx in range(len(resNorm)): # would be more efficient with a ufunc
        #    resSqr = resNorm[idx]**2
        #    residualLoss[idx]        = robLoss.Evaluate( resSqr )[0]
        #    residualLossTrivial[idx] = squaredLoss.Evaluate( resSqr )[0]
        #plt.twinx()
        #plt.plot( resNorm, residualLoss, 'r-', label=str(robLoss) )
        #plt.plot( resNorm, residualLossTrivial, 'r--', label=str(squaredLoss) )
        #plt.legend(loc='upper right')
        #plt.ylim( top=residualLoss[-1] )
        #plt.ylabel('loss(residual norm squared)', color='r')
        nShown = hist.sum()
        if nShown < len(resNormSqr):
            cutoff = '{} of '.format( nShown )
        else:
            cutoff = ''

        plt.title( cutoff + "{} img residual norms; max:{:.1f}".format(
                            len(resNormSqr), resNormSqr.max()**.5 ))
        if fn:
            plt.savefig( os.path.join( self.sfm.outDir,  fn ), bbox_inches='tight', dpi=150 )
            plt.close('residuals') #  for some reason, plt.close(fig) doesn't work 


    @contract
    def epipolar( self,
                  iImg1 : int,
                  iImg2 : int,
                  pt1 : 'array[Nx2](float)',
                  pt2 : 'array[Nx2](float)',
                  essentialMatrix : 'array[3x3](float)' ):
        K1 = ori.cameraMatrix( self.sfm.imgs[iImg1].ior )
        K2 = ori.cameraMatrix( self.sfm.imgs[iImg2].ior )
        
        K1Inv = np.linalg.inv( K1 )
        K2Inv = np.linalg.inv( K2 )
        active = np.ones( (pt1.shape[0],), dtype=np.bool )
        fundamentalMatrix = K2Inv.T.dot(essentialMatrix).dot(K1Inv)
        
        plt.figure(11); plt.clf()
        plt_utils.plotEpipolar( self.sfm.imgs[iImg1].fullPath, pt1, pt2, active, 2, fundamentalMatrix )

        plt.figure(21); plt.clf()
        plt_utils.plotEpipolar( self.sfm.imgs[iImg2].fullPath, pt2, pt1, active, 1, fundamentalMatrix )

        img_1 = self.imageBGR(iImg1)
        img_2 = self.imageBGR(iImg2)
        hstacked = np.hstack( (img_1,img_2) ).copy(); # copy() returns a contiguous array, as required by cv2.line
        for iMatch in range(pt1.shape[0]):
            cv2.line( hstacked,
                      ( int(pt1[iMatch,0]), int(pt1[iMatch,1]) ),
                      ( img_1.shape[1]+int(pt2[iMatch,0]), int(pt2[iMatch,1]) ),
                      color = cv2.cv.RGB( 255,random.randint(0,127), 0 ) if ~active[iMatch] else cv2.cv.RGB( 0,random.randint(0,127),255 ),
                      thickness=0,
                      lineType=cv2.LINE_AA
                    )
            
        # schaut grob in Ordnung aus, ist im Detail bei den vielen Korrespondenzen aber schwierig nachzuprüfen
        plt.figure(3); plt.clf();
        plt.imshow( cv2.cvtColor( hstacked, cv2.COLOR_BGR2RGB ), interpolation='none' )
        cv2.imwrite( "{}/img_match_inlier_{:03d}_{:03d}.png".format( self.sfm.outDir, iImg1, iImg2 ),
                     hstacked )                   
    
        
    def candidates( self, candidates ):
        "candidates: [ namedtuple( 'Img', ( 'img', 'iKeyPts', 'imgPts', 'objPts', 'rvec', 'tvec', 'inliers', 'representatives' ) ) ]"
        figs = []
        for candidate in candidates:
            if candidate.inliers.shape[0] <= 5:
                continue # save memory, don't plot orientations without redundancy
            Rcv,_ = cv2.Rodrigues( candidate.rvec )

            outdir= os.path.join( self.sfm.outDir, "intermed" )
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            baseName_ = os.path.join( outdir, os.path.splitext(self.sfm.imgs[candidate.img.idx].fullPath)[0] )
            np.savetxt( baseName_ + "_imgObs.txt" , candidate.imgPts )
            np.savetxt( baseName_ + "_objPts.txt" , candidate.objPts )
            P = np.hstack((Rcv,candidate.tvec))
            np.savetxt( baseName_ + "_P.txt", P )
            np.savetxt( baseName_ + "_K.txt", ori.cameraMatrix( self.sfm.imgs[candidate.img.idx].ior ) )
            np.savetxt( baseName_ + "_inliers.txt", candidate.inliers )
            # TODO: exportiere objPts inkl. RGB-Farbwerte! -> ParaView
            
            R,t = ori.projectionMat2oriRotTrans( P ) 
            objPts = candidate.objPts.copy()
            objPts[:,1:] *= -1.
            xProj = ori.projection( objPts, t, ori.omfika(R), self.sfm.imgs[candidate.img.idx].ior, self.sfm.imgs[candidate.img.idx].adp )
            isInlier = np.zeros( candidate.imgPts.shape[0], dtype=np.bool )
            isInlier[candidate.inliers]=True
            
            figs.append( plt.figure("c"+self.sfm.imgs[candidate.img.idx].shortName) )
            plt.clf()
            for isInlier_,col in zip_equal( ( np.logical_not(isInlier), isInlier  ),
                                      ( (1,0,0),                  (0,1,0)   ) ):
                plt.scatter( x=candidate.imgPts[isInlier_,0],y=candidate.imgPts[isInlier_,1], color=col, marker="o", s=8, edgecolors='m', linewidths = (2.,) )
                
                plt.plot( np.vstack( ( xProj[isInlier_,0],candidate.imgPts[isInlier_,0]) ),
                          np.vstack( (-xProj[isInlier_,1],candidate.imgPts[isInlier_,1]) ), color=col, linewidth=2., marker=None )
        
            plt.imshow( self.imageRGB(candidate.img.idx) )
            plt.title("result of solvePnPRansac. img{} {} inliers of {} points".format( self.sfm.imgs[candidate.img.idx].shortName, isInlier.sum(), isInlier.shape[0] ))
            if     float(len(candidate.inliers)) / candidate.imgPts.shape[0] > 0.75 \
               and len(candidate.inliers) > 25:
                break # if there are more than 75 % inliers, then save memory and skip subsequent plots
            
        raw_input( "{} image(s) out of {} candidate(s) plotted, img{} selected: {} ({:.0%}) inliers out of {} imgObs. enter key to proceed".format(
            len(figs),
            len(candidates),
            self.sfm.imgs[candidates[0].img.idx].shortName,
            candidates[0].inliers.shape[0],
            float(candidates[0].inliers.shape[0]) / candidates[0].imgPts.shape[0],
            candidates[0].imgPts.shape[0] ) )
        
        # close figures to save memory. Don't close the selected one
        for fig in figs[1:]:
            plt.close( fig )   
                 