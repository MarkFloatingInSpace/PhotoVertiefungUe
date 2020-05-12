"""Access data using osgeo.ogr
"""
from contracts import contract
import numpy as np
from osgeo import ogr
ogr.UseExceptions()

from pathlib import Path

suffix2DriverName = {
    '.shp' : 'ESRI Shapefile'
}

# http://osdir.com/ml/gdal-development-gis-osgeo/2008-06/msg00093.html
"""OGR uses the simple features geometry model which places no requirements
on the winding direction of polygon rings. So generally speaking the
winding direction is not defined - often it will be whatever was most
convenient when reading from the source files.

The Shapefile *writer* does have to rewind polygon rings to put them into
the required winding order for shapefiles."""

@contract
def exportPolygons( fn : Path,
                    polygons : 'list[A]( $(array[Bx2](float)) )',
                    attributes : 'dict( str : array[A] )' = {} ) -> None:

    driver = ogr.GetDriverByName( suffix2DriverName[fn.suffix] )
    data_source = driver.CreateDataSource( str(fn) )
    layer = data_source.CreateLayer( fn.stem, geom_type=ogr.wkbPolygon )

    np2ogrTypes = { np.int32   : ogr.OFTInteger,
                    np.int64   : ogr.OFTInteger64,
                    np.float64 : ogr.OFTReal }

    for name, values in attributes.items():
        ogrType = np2ogrTypes.get(values.dtype, None)
        if ogrType is not None:
            fieldDef = ogr.FieldDefn( name, ogrType )
        else:
            fieldDef = ogr.FieldDefn( name, ogr.OFTString )
            fieldDef.SetWidth( max(len(value) for value in values) )
        layer.CreateField(fieldDef)

    for iPolygon,polygon in enumerate(polygons):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for pt in polygon:
            ring.AddPoint( *pt )
        ring.CloseRings()
        ring.FlattenTo2D() # even though we've constructed a 2D object, no matter if we call AddPoint(.) with 2 or 3 arguments, AddPoint(.) makes ring a 2.5D object!
        polyg = ogr.Geometry(ogr.wkbPolygon)
        polyg.AddGeometryDirectly(ring)
        assert polyg.IsValid()
        #polyg = polyg.Intersection( bbox )
        #if polyg is None:
        #    logger.warning('Image footprint completely outside of sparse point cloud')
        #    continue
        feature = ogr.Feature(layer.GetLayerDefn())
        for name, values in attributes.items():
            feature.SetField(name, values[iPolygon] )
        feature.SetGeometry(polyg)
        layer.CreateFeature(feature)

    # make sure that the file is okay, even if an exception is raised later on.
    data_source.SyncToDisk()