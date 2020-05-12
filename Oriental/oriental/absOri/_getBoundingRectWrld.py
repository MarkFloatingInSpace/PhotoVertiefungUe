# -*- coding: cp1252 -*-
from oriental import config
import sqlite3.dbapi2 as db

def getBoundingRectWrld():
    """Compute the approximate bounding window of a set of LBA-images, in EPSG:31256
       -> based on the result, appropriate orthophotos may be selected"""
    
    # n.b.: die geometries aus luftbild können auch in QGIS angezeigt werden,
    # und sogar der Filter funktioniert:
    # dbname='D:/swdvlp64/oriental/lba.sqlite' table="luftbild" (Geometry) sql=CAST(bild AS REAL) BETWEEN 2110503.047 AND 2110503.081
    # -> beim 'Stil' die Füllung entfernen, denn Bild 77 hat einen viel größeren Radius als die anderen Bilder, und dessen Kreis würde sonst alle anderen Kreise überdecken.
    with db.connect( config.dbLBA ) as conn:
        srids = conn.execute("""
            SELECT Distinct(Srid(geometry))
            FROM luftbild
            WHERE CAST(bild AS REAL) BETWEEN 2110503.047 AND 2110503.081 -- "x BETWEEN y AND z" is equivalent to "x>=y AND x<=z" except that with BETWEEN, the x expression is only evaluated once.
            """).fetchall()
        assert len(srids)==1, "Geometries do not share a common SRID"
        assert srids[0][0]==4312 # MGI(Greenwich)
        
        geometryTypes = conn.execute("""
            SELECT Distinct(GeometryType(geometry))
            FROM luftbild
            WHERE CAST(bild AS REAL) BETWEEN 2110503.047 AND 2110503.081
            """).fetchall()
        assert len(geometryTypes)==1, "Geometries do not share a common type"
        assert geometryTypes[0][0]=='POLYGON'     
        
        # Transform to EPSG:31256 "MGI / Austria GK East",
        # aggregate their bounding boxes to an overall bounding box,
        # and extract its extents
        minx,maxx,miny,maxy = conn.execute("""
            SELECT MbrMinX(ext),
                   MbrMaxX(ext),
                   MbrMinY(ext),
                   MbrMaxY(ext)
            FROM (
                SELECT Extent(Transform(geometry,31256)) as ext
                FROM luftbild
                WHERE CAST(bild AS REAL) BETWEEN 2110503.047 AND 2110503.081
            )
            """).fetchone()
        
        print( "Project extents Y/X [m]: {} {}".format( maxx-minx, maxy-miny ) )
        
        # Alternatively, we might consider bild.gkx, bild.gky, bild.radius: take the projection of the image center onto the ground(gkx,gky) and create a buffer zone of radius around it.
        # However, bild.bild is differently formatted, with no leading zeros after the '.':
        # e.g. '02110503.1' instead of '02110503.001' (as in luftbild.bild)
        # -> we cannot simply cast bild.bild to a REAL and select an interval of REALs, as above.
        #    SQL-function 'instr' would come in handy for that problem, but it is supported only starting from SQLite-version 3.7.15: http://www.sqlite.org/changes.html
        # However, we can filter by bild.film AND bild.bildnr:
        # WHERE film='02110503' AND CAST(bildnr AS INT) BETWEEN 47 AND 81

        # Anyway, we need to buffer the result, because the information on the footprint locations is fuzzy. 
        
        # Histogram of radii:
        rows = conn.execute("""
            SELECT COUNT(*) as occurrences,
                   CAST(radius AS REAL) as radius_
            FROM bild
            WHERE     film='02110503'
                  AND CAST(bildnr AS INT) BETWEEN 47 AND 81
            GROUP BY radius_
            ORDER BY radius_
            """)
        print( "#   radius[m]" )
        for row in rows:
            print("{:3d} {:4.1f}".format(*row))        
        
        dummy = 1