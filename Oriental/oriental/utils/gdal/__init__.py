# -*- coding: cp1252 -*-
"""not-so-buggy Python binding to GDAL"""

from ... import config as _config

if _config.debug:
    from ._gdal_d import *
    from ._gdal_d import __date__
else:
    from ._gdal import *
    from ._gdal import __date__

# -*- coding: cp1252 -*-
"""helper functions using GDAL"""

from contextlib import contextmanager
from oriental import utils
from oriental.ori.transform import AffineTransform2D
from oriental.utils.crs import fixCrsWkt
from contracts import contract
import numpy as np
#from osgeo import gdal
from osgeo import osr
osr.UseExceptions()

@contextmanager
def suppressGdalMesssages():
    oldVal = logGdalMessages(False)
    yield
    logGdalMessages( oldVal )

depth2dtype = { Depth.u8  : np.uint8,
                Depth.u16 : np.uint16,
                Depth.s32 : np.int32,
                Depth.f32 : np.float32  }

dtype2depth = { dtype : depth for depth,dtype in depth2dtype.items()  }

assert set(Depth.names.values()) == set(depth2dtype) | {Depth.unchanged}

dtype2gdalType = {
    np.dtype(np.uint8) : 'Byte',
    np.dtype(np.uint16) : 'UInt16',
    np.dtype(np.int16) : 'Int16',
    np.dtype(np.uint32) : 'UInt32',
    np.dtype(np.int32) : 'Int32',
    np.dtype(np.float32) : 'Float32',
    np.dtype(np.float64) : 'Float64'
}

@contract
def memDataset( img : 'array[RxC]|array[RxCxB]' ) -> str:
    return 'MEM:::DATAPOINTER={},PIXELS={},LINES={},BANDS={},DATATYPE={},PIXELOFFSET={},LINEOFFSET={},BANDOFFSET={}'.format(
        img.ctypes.data,
        img.shape[1],
        img.shape[0],
        1 if img.ndim==2 else img.shape[2],
        dtype2gdalType[img.dtype],
        img.strides[1],
        img.strides[0],
        1 if img.ndim==2 else img.strides[2] )

@contract
def interpolateRasterHeights( fn : str,
                              objPts : 'array[NxM](float) | array[M](float), M>1',
                              objPtsCS : 'int|str', # epsg code or WKT string
                              interpolation : Interpolation = Interpolation.bilinear
                            ) -> 'array[N](float) | float':
    objPts = np.atleast_2d( objPts )
    
    test = False

    imread = utils.gdal.imread( imageFilePath=fn,
                                bands=utils.gdal.Bands.unchanged,
                                maxWidth=0,
                                depth=utils.gdal.Depth.unchanged,
                                mask=test,
                                info=True,
                                skipData=not test )
    if test:
        dsm,mask,info = imread
    else:
        info, = imread

    rasterCs = osr.SpatialReference()
    rasterCs.ImportFromWkt( fixCrsWkt( info.projection ) )
    objPtCs = osr.SpatialReference()
    if type(objPtsCS)==int:
        objPtCs.ImportFromEPSG( objPtsCS )
    else:
        objPtCs.ImportFromWkt( objPtsCS )
    if not rasterCs.IsSame( objPtCs ): # transform object points into the CS of the surface model
        csTrafo = osr.CoordinateTransformation( objPtCs, rasterCs )
        objPts = np.array( csTrafo.TransformPoints( objPts.tolist() ) )[:,:2] # osr seems to always add the z-coordinate as 3rd column


    gdal2wrld = AffineTransform2D( A=info.geotransform[:,1:], t=info.geotransform[:,0] ) # Affine trafo pixel->CRS
    # The affine transform consists of six coefficients returned by GDALDataset::GetGeoTransform() which map pixel/line coordinates into georeferenced space using the following relationship:
    #     Xgeo = GT(0) + Xpixel*GT(1) + Yline*GT(2)
    #     Ygeo = GT(3) + Xpixel*GT(4) + Yline*GT(5)
    # 
    # In case of north up images, the GT(2) and GT(4) coefficients are zero, and the GT(1) is pixel width, and GT(5) is pixel height. The (GT(0),GT(3)) position is the top left corner of the top left pixel of the raster.
    # 
    # Note that the pixel/line coordinates in the above are from (0.0,0.0) at the top left corner of the top left pixel to (width_in_pixels,height_in_pixels) at the bottom right corner of the bottom right pixel. The pixel/line location of the center of the top left pixel would therefore be (0.5,0.5).
    if 0:
        luCorner_luPixel_wrld = gdal.ApplyGeoTransform( tuple(gdal2wrld.flat), 0, 0 )

        success,wrld2gdal = gdal.InvGeoTransform( tuple(gdal2wrld.flat) )
        assert success, "inversion of transform failed"

        Ainv = linalg.inv( gdal2wrld[:,1:] )
        wrld2gdal = np.column_stack(( -Ainv.dot(gdal2wrld[:,0]), Ainv ))
    
    assert info.bands == utils.gdal.Bands.grey, "Single channel image expected"

    
    #nNoData = ( mask == 0 ).sum() # undefinierte Höhen v.a. auf Gewässern.
    
    # Pixel is area oder point?
    # wird nur von TIFF unterstützt. GDAL interpretiert das nunmehr und 
    # amtl. Orthophotos Land NÖ. 2008/2009:
    # Auflösung: 10000 x 8000 px,
    #   mit GSD (=Pixelseitenlänge am Boden) = 0.125m
    # -> Gesamtausdehnung: 1250m (Y) x 1000m (X)
    
    imgPts = np.empty_like( objPts, dtype=np.float )
    for idx,(Y,X) in enumerate(objPts):
        if 0:
            c2,r2 = gdal.ApplyGeoTransform(  tuple(wrld2gdal.flat), Y, X )
        #c,r = wrld2gdal[:,0] + wrld2gdal[:,1:].dot( [Y,X] )
        c,r = gdal2wrld.inverse( np.array([Y,X]) )

        # Achtung: GDAL's Bildkoordinaten werden in der Reihenfolge 'Spalte', 'Zeile' ('Pixel','Line') angegeben,
        #   und der URSPRUNG liegt hier: top left corner of the top left pixel
        # PixelIsArea! -> beziehe die Bildkoordinaten auf den Mittelpunkt des linken/oberen Pixels, statt auf dessen linke/obere Ecke!
        # Deshalb wird auch hier verschoben: http://gis.stackexchange.com/questions/7611/bilinear-interpolation-of-point-data-on-a-raster-in-python
        c -= .5
        r -= .5
        imgPts[idx] = c,r

    heights = utils.gdal.interpolatePoints( fn, imgPts, interpolation )

    if test:
        heights_test = np.empty( objPts.shape[0], np.float )
        for idx,(c,r) in enumerate(imgPts):
            # bilinear interpolation

            c1 = np.floor(c)
            r1 = np.floor(r)
 
            if c1 < 0. or c1+1 > dsm.shape[1] or \
               r1 < 0. or r1+1 > dsm.shape[0]:
                heights_test[idx] = np.nan
                continue

            s = np.s_[r1:r1+2,c1:c1+2]
            if not mask[s].all():
                heights_test[idx] = np.nan
                continue

            neighbors = dsm[s]
            cr = c - c1
            rr = r - r1 
            heights_test[idx] = np.array([ 1-rr, rr ]).dot( neighbors ).dot( np.array([ 1-cr, cr ]) )

        isNan = np.isnan(heights)
        assert np.all( isNan == np.isnan(heights_test) )
        isValid = np.logical_not(isNan)
        assert np.all( np.abs( heights[isValid] - heights_test[isValid] ) < 1.e-8 )

    # transform into physically meaningful values
    for height in np.nditer( heights, op_flags=['readwrite'] ):
        height[...] = (height * info.scale) + info.offset

    if not rasterCs.IsSame( objPtCs ):
        csTrafo = osr.CoordinateTransformation( rasterCs, objPtCs ) # transform object points back into their original CS
        isValid = np.logical_not(np.isnan(heights))
        if isValid.any(): # otherwise, TransformPoints returns an empty list, and numpy chokes on the index into the 3rd column
            objPts3d = np.column_stack(( objPts, heights ))
            heights[isValid] = np.array( csTrafo.TransformPoints( objPts3d[isValid].tolist() ) )[:,2]


    if heights.size==1:
        return heights[0]
    return heights



