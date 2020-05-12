# -*- coding: cp1252 -*-
import math

import numpy as np
from osgeo import gdal
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()


def getOrtho( fnDSM, fnOrtho ):
    from oriental.utils import pyplot as plt

    # DSM und Orthophoto sind im gleichen System: EPSG:31256 - MGI / Austria GK East
    # siehe: D:\arap\data\120302_Carnuntum_TestdatenVonDoneus\Terrain\dsm.prj
    #        D:\arap\data\laserv\Geodaten\geodaten\oesterreich\niederoesterreich\orthofoto\793479.txt
            
    dsDSM = gdal.Open( fnDSM, gdal.GA_ReadOnly )
    # Dateien, die GDAL eingelesen hat:
    dsDSMFiles = dsDSM.GetFileList()
    
    # Arc/Info ASCII Grid
    # x nimmt nach rechts zu (east),
    # y nimmt nach oben zu (north)
    # 'xllcorner' und 'yllcorner' definieren die linke/untere Ecke der linksten/untersten Zelle im übergeordneten Koord.Sys. Das übergeordnete Koord.Sys. ist in der Datei nicht definiert/definierbar.
    # Zellen sind immer quadratisch. Ihre Seitenlänge ist definiert durch 'cellsize' 
    # Die einzelnen Werte repräsentieren eine Fläche, keinen Punkt a.k.a. PixelIsArea

    
    # The affine transform consists of six coefficients returned by GDALDataset::GetGeoTransform() which map pixel/line coordinates into georeferenced space using the following relationship:
    #   Xgeo = GT(0) + Xpixel*GT(1) + Yline*GT(2)
    #   Ygeo = GT(3) + Xpixel*GT(4) + Yline*GT(5)
    # In case of north up images, the GT(2) and GT(4) coefficients are zero, and the GT(1) is pixel width, and GT(5) is pixel height. The (GT(0),GT(3)) position is the top left corner of the top left pixel of the raster.
    # Note that the pixel/line coordinates in the above are from (0.0,0.0) at the top left corner of the top left pixel to (width_in_pixels,height_in_pixels) at the bottom right corner of the bottom right pixel.
    #   The pixel/line location of the center of the top left pixel would therefore be (0.5,0.5).
    # GDAL's Rasterkoord.sys. stimmt überein mit jenem von TIFF, für PixelIsArea: http://www.remotesensing.org/geotiff/spec/geotiff2.5.html
    gdal2wrld = dsDSM.GetGeoTransform() # Affine trafo pixel->CRS aus dsm.asc
    luCorner_luPixel_wrld = gdal.ApplyGeoTransform( gdal2wrld, 0, 0 )
    #gdal2wrld = np.array([ gdal2wrld[:3],
    #                       gdal2wrld[3:] ])
    wrld2gdal = gdal.InvGeoTransform( gdal2wrld )
    assert wrld2gdal[0]==1, "inversion of transform failed"
    wrld2gdal = wrld2gdal[1]
    #wrld2gdal = np.array([ wrld2gdal[:3],
    #                       wrld2gdal[3:] ]) 
    proj  = dsDSM.GetProjection() # unvollständige (Spheroid fehlt) CRS-Definition von EPSG:31256 als OpenGIS WKT
    meta  = dsDSM.GetMetadata() # nothing helpful: 'PyramidResamplingType' aus dsm.asc.aux.xml
    
    assert dsDSM.RasterCount==1
    band = dsDSM.GetRasterBand(1)
    noData = band.GetNoDataValue()
    
    # The GetMaskFlags() method returns a bitwise OR-ed set of status flags with the following available definitions that may be extended in the future:
    #   GMF_ALL_VALID(0x01): There are no invalid pixels, all mask values will be 255. When used this will normally be the only flag set.
    #   GMF_PER_DATASET(0x02): The mask band is shared between all bands on the dataset.
    #   GMF_ALPHA(0x04): The mask band is actually an alpha band and may have values other than 0 and 255.
    #   GMF_NODATA(0x08): Indicates the mask is actually being generated from nodata values. (mutually exclusive of GMF_ALPHA)
    maskFlags = band.GetMaskFlags()
    assert maskFlags==gdal.GMF_NODATA, "Data read from Arc/Info ASCII Grid files are expected to be masked according to their NODATA_value."
    mask = band.GetMaskBand()
    
    # raster values must be transformed with (offset,scale) to compute their exterior values (e.g. height[band.GetUnitType()] )
    # GetOffset() in combination with GetScale() is used to transform raw pixel values into the units returned by GetUnits(). For example this might be used to store elevations in GUInt16 bands with a precision of 0.1, and starting from -100.
    # Units value = (raw value * scale) + offset
    # Anscheinend nur von sehr wenigen Formaten unterstützt.
    offset = band.GetOffset()
    scale  = band.GetScale()
    unit   = band.GetUnitType()
    
    bandMeta = band.GetMetadata() # -> band statistics
    
    # ReadAsArray() by default reads the whole extents of all bands, using the internal data type as dtype
    # ReadAsArray() doesn't interpret <Scale> or <Offset>. They are interpretation hints for the user, but GDAL won't alter the values read from the dataset. http://lists.osgeo.org/pipermail/gdal-dev/2012-February/031755.html
    # Okay: GDAL liest die .asc-Datei korrekt, obwohl darin ',' als Dezimaltrenner verwendet wird.
    dsm = dsDSM.ReadAsArray()
    
    nNoData = (dsm==band.GetNoDataValue()).sum() # undefinierte Höhen v.a. auf Gewässern.
    
    # Pixel is area oder point?
    # wird nur von TIFF unterstützt. GDAL interpretiert das nunmehr und 
    # amtl. Orthophotos Land NÖ. 2008/2009:
    # Auflösung: 10000 x 8000 px,
    #   mit GSD (=Pixelseitenlänge am Boden) = 0.125m
    # -> Gesamtausdehnung: 1250m (Y) x 1000m (X)
    
    # im Orthophoto klar erkennbare Mauerecke des Amphitheaters. Wie Interpoliert QGIS die Geländehöhen? Vermutlich auch bilinear.
    Y = 38621.4444041
    X = 330266.945808
    Z =    184.701
    
    # bilinear interpolation
    # Achtung: GDAL's Bildkoordinaten werden in der Reihenfolge 'Spalte', 'Zeile' ('Pixel','Line') angegeben,
    #   und der URSPRUNG liegt hier: top left corner of the top left pixel
    c,r = gdal.ApplyGeoTransform( wrld2gdal, Y, X )
    if 1: 
        # PixelIsArea! -> beziehe die Bildkoordinaten auf den Mittelpunkt des linken/oberen Pixels, statt auf dessen linke/obere Ecke!
        # Deshalb wird auch hier verschoben: http://gis.stackexchange.com/questions/7611/bilinear-interpolation-of-point-data-on-a-raster-in-python
        c -= .5
        r -= .5
    c1 = math.floor(c)
    r1 = math.floor(r)
 
    if c1 >= 0. and c1 < dsDSM.RasterXSize-1 and \
       r1 >= 0. and r1 < dsDSM.RasterYSize-1:
        cr = c - c1
        rr = r - r1 
        neighbors = dsm[r1:r1+2,c1:c1+2]
        height = np.array([ 1-rr, rr ]).dot( neighbors ).dot( np.array([ 1-cr, cr ]) )

    
    # --------------------
    
    dsOrtho = gdal.Open( fnOrtho, gdal.GA_ReadOnly )
    dsOrthoFiles = dsOrtho.GetFileList()
    
    # Unlike GDAL's raster coord.sys., world-files define the origin in the center of the upper,left pixel: http://en.wikipedia.org/wiki/World_file#Definition
    # Line 5: C: x-coordinate of the center of the upper left pixel
    # Line 6: F: y-coordinate of the center of the upper left pixel
    # Thus, gdal2wrld[0] and gdal2wrld[3] are offset by half a pixel width/height from the respective values in 793479.jgw 
    gdal2wrld = dsOrtho.GetGeoTransform() # Affine trafo pixel->CRS aus 793479.jgw
    # The following agrees with the area covered by the orthophoto, as given in 793479.txt 
    luCorner_luPixel_wrld = gdal.ApplyGeoTransform( gdal2wrld, 0, 0 )
    rlCorner_rlPixel_wrld = gdal.ApplyGeoTransform( gdal2wrld, dsOrtho.RasterXSize, dsOrtho.RasterYSize )
    wrld2gdal = gdal.InvGeoTransform( gdal2wrld )
    assert wrld2gdal[0]==1, "inversion of transform failed"
    wrld2gdal = wrld2gdal[1]
    proj  = dsOrtho.GetProjection() # <leer>
    meta  = dsOrtho.GetMetadata()   # <leer>
    
    assert dsOrtho.RasterCount==3
    rgb = ( gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand )
    for idx in range(3):
        band = dsOrtho.GetRasterBand(idx+1)
        assert band.GetColorTable() is None
        assert band.GetColorInterpretation() == rgb[idx]

    ortho = dsOrtho.ReadAsArray()
    # ReadAsArray() returns an array with shape (depth,nRows,nCols)
    ortho = np.rollaxis( ortho, 0, 3 )
    # imshow sets the axes to make x increase to the right (as default), and y increase downwards
    plt.imshow( ortho, interpolation='nearest' )
    # matplotlib's raster coo.sys. has it's origin in the center of the top/left pixel!
    plt.scatter( 0, 0, marker="o", color='r' )
    
    c,r = gdal.ApplyGeoTransform( wrld2gdal, Y, X )
    if 1: # PixelIsArea! -> beziehe die Bildkoordinaten auf den Mittelpunkt des linken/oberen Pixels, statt auf dessen linke/obere Ecke!
        c -= .5
        r -= .5
    
    plt.scatter( x=c, y=r, marker="o", color='r' )
    