# -*- coding: cp1252 -*-
import environment
from oriental import config
import sqlite3.dbapi2 as db
from oriental.utils.db import initDataBase
import struct
import numpy as np
import unittest

class TestSpatialite(unittest.TestCase):
    
    def test_wkb(self):
        with db.connect( ':memory:' ) as conn:
            initDataBase( conn )
            conn.execute( """CREATE TABLE objpts( id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT )""" )
            conn.execute( """SELECT AddGeometryColumn(
                'objpts', -- table
                'pt',     -- column
                -1,       -- srid -1: undefined/local cartesian cooSys
                'POINT',  -- geom_type
                'XYZ',    -- dimension
                1         -- NOT NULL
                )""" )
            pointZWkb = struct.Struct('<bIddd')
            def packPointZ(pt):
                # 1001 is the code for points with z-coordinate
                # Must wrap the str returned by struct.pack(.) into a buffer object, or otherwise, SQLite complains about receiving 8-bit-strings instead of unicode. However, we want the string to be interpreted byte-wise.
                # Even though buffer is deprecated since Python 2.7, there is no other way to pass the data: http://bugs.python.org/issue7723
                return pointZWkb.pack( 1, 1001, *pt )
            
            pts = { 0 : np.array([0.,1.,2.]),
                    1 : np.array([3.,4.,5.]) }
            conn.executemany( """INSERT INTO objpts(id,pt) VALUES( ?, GeomFromWKB(?, -1) )""",
                              ( ( iPt, packPointZ(pt) ) for iPt,pt in enumerate(pts.values()) ) )
        
        
            rows = conn.execute("""
               SELECT asBinary(pt)
               FROM objpts
            """)
            for idx,row in enumerate(rows):
                val = row[0]
                res = pointZWkb.unpack( val )
                np.testing.assert_array_equal( pts[idx], np.array( res[2:] ), "Point coordinates of double precision are expected to be precisely equal when passed as Well-known-binary" )
                
if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestSpatialite.test_wkb',
                       exit=False )