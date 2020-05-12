# -*- coding: cp1252 -*-
from .. import config as _config
from . import mlab
import numpy as np
import random
import hashlib
import colorsys

"""Objekt-Punktwolke plotten"""
def _camera( ior, R, t, name, sz, pos ):
    # plot the axes arrows with length sz.
    # plot the tetrahedrons-side in x-direction with length 2/3*sz
    # adjust the tetrahedron-side in y-direction and its height according to ior
    x = sz * 2/3
    y = ior[1] * x / ior [0]
    z = ior[2] * x / ior [0]
    if pos:
        z *= -1
    vtx = np.array([ [  0,  0,  0 ],
                        [  x,  y,  z ],
                        [ -x,  y,  z ],
                        [ -x, -y,  z ],
                        [  x, -y,  z ] ])
    # Cam-CS -> Obj-CS
    vtx = np.dot( R, vtx.T ).T + t
    triplets = [ (0,1,2),
                 (0,2,3),
                 (0,3,4),
                 (0,4,1),
                 (1,2,3),
                 (3,4,1) ]
                 
    if name is not None:
        name=str(name)
                         
    if not name or not len(name):
        surfaceColor=(random.random(),random.random(),random.random())
    else:
        # assuming that cameras in the same plot windows have different names, try to assign non-random, but different colors, based on the cam name.
        # digest_size is the length of the string returned by digest()
        # hexdigest() returns a string where each character returned by digest() has been converted to 2 hexadecimal digits
        # hexdigest() has thus twice the length of digest()
        surfaceHue = float( int(hashlib.md5(name).hexdigest(), 16) ) \
                     / 16.**(hashlib.md5().digest_size*2)
        surfaceColor = colorsys.hsv_to_rgb( surfaceHue, 1, .5 )
        
    # there seems to be no way to plot a surface with edges. Thus, issue 2 plot commands
    for representation,color in zip( ['surface','wireframe'],
                                     [surfaceColor, (0,0,0) ] ):
        mlab.triangular_mesh( vtx[:,0], vtx[:,1], vtx[:,2],
                              triplets, reset_zoom=False, color=color,
                              opacity=1., representation=representation, line_width=1. )
                              # opacity<1 would allow for seeing points inside the tetrahedron.
                              # However, any opacity<1 makes the tetrahedron's rear edges visible! 
    
    vtx = np.array([ [   0,   0,   0 ],
                        [  sz,   0,   0 ],
                        [   0,  sz,   0 ],
                        [   0,   0,  sz ] ] )
    vtx = np.dot( R, vtx.T ).T + t
    # plot the camera coordinate axes
    if 0: # check
        mlab.points3d( vtx[:,0], vtx[:,1], vtx[:,2], color=(1,1,0), reset_zoom=False, mode='sphere', scale_factor='.1')
    # quiver3d( x,y,z, u,v,w )
    # x,y,z: origin of glyph (e.g. tail of arrow)
    # u,v,w: direction of glyph (e.g. direction of tip of arrow, seen from tail)
    # scale_mode==1 -> length(glyph)==1 ?  
    # reduce to origin of Cam CS   
    vtx[1:,:] -= vtx[0,:]   
    mlab.quiver3d( vtx[0,0],vtx[0,1],vtx[0,2], vtx[1,0],vtx[1,1],vtx[1,2], color=(1,0,0), scale_factor=sz, scale_mode='none', mode='arrow', reset_zoom=False )
    mlab.quiver3d( vtx[0,0],vtx[0,1],vtx[0,2], vtx[2,0],vtx[2,1],vtx[2,2], color=(0,1,0), scale_factor=sz, scale_mode='none', mode='arrow', reset_zoom=False )
    mlab.quiver3d( vtx[0,0],vtx[0,1],vtx[0,2], vtx[3,0],vtx[3,1],vtx[3,2], color=(0,0,1), scale_factor=sz, scale_mode='none', mode='arrow', reset_zoom=False )
    if name is not None:
        mlab.text3d(vtx[0,0], vtx[0,1], vtx[0,2], name, scale=sz/2 )
       

# remote version of _camera
def camera( ior, R, t, name=None, sz=0.07, pos=True ):
    if not _config.redirectedPlotting:
        return _camera( ior, R, t, name, sz, pos )
    else:
        # remote version of _camera for speedup
        # doesn't work if IPython is running
        from .BlockingKernelManager import apply as _apply
        _apply( _camera, ior, R, t, name, sz, pos )   
   
"""Legende"""
def legend( lines, colours=None, charWidth=0.01 ):
    assert charWidth>0
    
    #assert colours is None or colours.shape[1]==3 and colours.shape[0]==len(lines)
    
    # genaugenommen müsste man hier den Renderer abfragen, wie groß er den Text plottet
    y = .9
    for line,colour in zip(lines,colours):
        txt=mlab.text(.1,y,line,color=colour,width=charWidth*len(line))
        y-=2*charWidth
             
def plotPtcl( pt, active ):   
    mlab.points3d( pt[active,0], pt[active,1], pt[active,2], color=(0,0,1),mode='sphere',scale_factor='.1')
    #mlab.text(0.1,0.9,"inlier", color=(0,0,1), width=.006*len("inlier"))
    mlab.points3d( pt[~active,0], pt[~active,1], pt[~active,2], color=(1,0,0), reset_zoom=False, mode='sphere',scale_factor='.1')
    # np.linalg.norm(t2-t1) == 1 
    mlab.view( azimuth=90, elevation=-45, distance=10, focalpoint=(0,0,0) )
