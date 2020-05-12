# -*- coding: cp1252 -*-
"""util functions for data base access"""

from ... import config as _config, _setup_summary_docstring

if _config.debug:
    from ._db_d import *
    from ._db_d import __date__
else:
    from ._db import *
    from ._db import __date__

from oriental import log
from oriental.utils.crs import fixCrsWkt

import os
import struct

from sqlite3 import dbapi2
from contracts import contract
from pathlib import Path

logger = log.Logger(__name__)

pointZWkb = struct.Struct('<bIddd')
def packPointZ(pt):
    "represent a point as binary WKT of a 3-D point without m-value"
    # 1001 is the code for points with z-coordinate
    return pointZWkb.pack( 1, 1001, *pt )

@contract
def uri4sqlite( p : Path ):
    """SQLite supports file URIs only with authority being empty or 'localhost',
    except if compiled with SQLITE_ALLOW_URI_AUTHORITY - which is not the case by default,
    and not the case for Python's sqlite.dll
    However, one can trick SQLite to accept URIs of network files
    by passing URIs that start with 4 slashes!
    """
    s = p.resolve().as_uri()
    if len(s) > 7 and s[7] != '/':
        # a network path. Insert 2 more slashes after the scheme.
        s = '{}//{}'.format(s[:7], s[7:])
    return s

@contract
def tableExists( connection : dbapi2.Connection,
                 tableName : str 
               ) -> bool:
  res = connection.execute( """
      SELECT count(*)
      FROM sqlite_master
      WHERE     type='table'
            AND name=:tableName""",
      { 'tableName' : tableName } )
  for row in res:
      return row[0] > 0
  return False

@contract
def tableHasColumn( connection : dbapi2.Connection,
                    tableName : str,
                    columnName : str ) -> bool:
    return columnName in frozenset( columnNamesForTable( connection, tableName ) )

@contract
def columnNamesForTable( connection : dbapi2.Connection,
                         tableName : str ) -> list:
    res = connection.execute(
        "PRAGMA table_info('{}')".format( tableName ) )
    return [ row['name'] for row in res ]

@contract
def initDataBase( connection : dbapi2.Connection, readOnly : bool = False ) -> None:
    """Spatially enable the connection's db, and correct Tranformation Parameters"""
  
    connection.row_factory = dbapi2.Row
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    supportsForeignKeys = cursor.execute("PRAGMA foreign_keys").fetchone()[0]
    assert supportsForeignKeys

    nCpus = os.cpu_count()
    if nCpus:
        cursor.execute("PRAGMA threads = {}".format(nCpus))

    # since we now use the Python-builtin-module sqlite3, we need to load the spatialite-extension
    connection.enable_load_extension(True)
    # mod_spatialite.dll must be on PATH
    # as mod_spatialite.dll uses a non-standard combination of file name and initialization function name,
    # we need to supply the name of the initialization function. This cannot be done with conn.load_extension(dllName),
    # but only with a SELECT statement.
    connection.execute("SELECT load_extension('mod_spatialite','sqlite3_modspatialite_init')")

    # From the docs: the scope of 'InitSpatialMetadata' is to create (and populate) any metadata table internally required by SpatiaLite.
    # if any metadata table already exists, this function doesn't apply any action.
    # so, calling more times InitSpatialMetaData() is useless but completely harmless.
    # please note: spatialite_gui will automatically perform any required initialization task every time a new database is created:
    # so (using this tool) there is no need at all to explicitly call this function.

    # Okay, re-initializing an already initialized SpatiaLite-DB does not result in an error.
    # However, doing so, prints 'error:"table spatial_ref_sys already exists"' to the screen,
    # which is distracting for users.
    # Thus, check beforehand.
    #if not tableExists( connection, 'spatial_ref_sys' ): 
    #    connection.execute( 'SELECT InitSpatialMetadata()' )
    meta, = cursor.execute("SELECT CheckSpatialMetaData()").fetchone()

    if readOnly:
        if meta != 3:
            logger.warning( "DB connection's geometry tables do not exist or do not comply with the current layout" )
        return

    if meta == 0: # geometry_columns and spatial_ref_sys tables do not exist
        # if the optional argument 'transaction' is set to TRUE the whole operation will be handled as a single Transaction (faster).
        # the default setting  is 'transaction=FALSE' (slower, but safer)
        # Setting transaction=True is crucial on SpatiaLite 4 - or otherwise, the call may take minutes: https://groups.google.com/forum/#!msg/spatialite-users/La8BUrVKX_g/lGJKxnQzp1sJ
        cursor.execute( 'SELECT InitSpatialMetadata(1)' )
        assert cursor.execute("SELECT CheckSpatialMetaData()").fetchone()[0], 'Spatial metadata could not be inserted. Probably the SQLite db-connection was opened as read-only'
    elif meta == 1:
        # https://www.gaia-gis.it/fossil/libspatialite/wiki?name=switching-to-4.0
        raise Exception('Spatialite metadata tables follow a legacy layout. Please update the layout, e.g. using spatialite_convert')
    elif meta == 2: # both tables exist, and their layout is the one used by FDO/OGR
        cursor.execute( 'SELECT AutoFDOStart()' )
    elif meta == 3: # both tables exist, and their layout is the one currently used by SpatiaLite (4.0.0 or any subsequent version)
        pass
    elif meta == 4:
        cursor.execute( 'SELECT AutoGPKGStart()' )
    else:
        logger.warning( 'CheckSpatialMetaData() returned an unknown value: {}'.format( meta ) )

    sqlite_version = dbapi2.sqlite_version_info
    pysqlite_version = dbapi2.version_info
    # spatialite_version() may have appended a character, e.g. 4.3.0a
    #spatialite_version = tuple((int(el) for el in cursor.execute("SELECT spatialite_version()").fetchone()[0].split('.') ))
    spatialite_version = cursor.execute('SELECT spatialite_version()').fetchone()[0]
    proj4Version = cursor.execute('SELECT proj4_version()').fetchone()[0]
    hasEpsg = cursor.execute('SELECT HasEpsg()').fetchone()[0]

    # make sure that local coordinate systems form part of the table spatial_ref_sys.
    # Otherwise, AddGeometryColumn() will output the error "foreign key constraint failed" to std::cerr
    #   when defining a geometry column of a local CRS.
    # SpatiaLite v.4 already defines those 2 local CRS automatically: https://www.gaia-gis.it/fossil/libspatialite/wiki?name=switching-to-4.0 
    #if cursor.execute("SELECT COUNT(*) FROM spatial_ref_sys WHERE srid=-1").fetchone()[0] == 0:
    #    assert spatialite_version[0] < 4, "SpatiaLite v.4 is supposed to define Local coordinate systems automatically in its SRS table"    
    #    cursor.execute("""INSERT INTO spatial_ref_sys( srid, auth_name, auth_srid,            ref_sys_name, proj4text )
    #                                           VALUES(   -1,    'NONE',        -1, 'Undefined - Cartesian',        '' )""" )
    #if cursor.execute("SELECT COUNT(*) FROM spatial_ref_sys WHERE srid=0").fetchone()[0] == 0:
    #    assert dbapi2.spatialite_version[0] < 4, "SpatiaLite v.4 is supposed to define Local coordinate systems automatically in its SRS table"    
    #    cursor.execute("""INSERT INTO spatial_ref_sys( srid, auth_name, auth_srid,                      ref_sys_name, proj4text )
    #                                           VALUES(    0,    'NONE',         0, 'Undefined - Geographic Long/Lat',        '' )""" )

    # For some reason, some entries of PROJ.4 do not use the recent datum transformation parameters defined by BEV/EPSG:1618 from/to WGS84 to/from Hermannskogel/MGI,
    # (while PostGIS does), see thread at https://groups.google.com/forum/#!topic/spatialite-users/UuJWBfY4K70
    # Let's correct that for all SRSs that refer to this datum. SpatiaLite 4.3 comes with PROJ.4 version 4.9.1, and proj4text does not include +datum=hermannskogel any more.
    # Hence, the MGI-datum cannot be identified based on the proj4text any more, either. Thus, use the WKT
    #for row in cursor.execute("""
    #    SELECT srid, proj4text -- , ref_sys_name
    #   FROM   spatial_ref_sys
    #    WHERE     srid IN ({})
    #          AND proj4text LIKE "%+datum=hermannskogel%"
    #""".format( ','.join(('{}'.format(el) for el in srsMGI)) ) ):
    for row in cursor.execute("""
       SELECT srid, proj4text, srtext
       FROM spatial_ref_sys
       WHERE srtext LIKE '%DATUM["Militar_Geographische_Institut%'
    """ ).fetchall():
        # http://www.bev.gv.at/pls/portal/docs/PAGE/BEV_PORTAL_CONTENT_ALLGEMEIN/0100_NEWS/0150_ARCHIV/ARCHIV_2007/NEUE%20EPSG%20-%20CODES%20FUER%20OESTERREICH/PROJEKTIONEN_TRANSF.PDF
        # 577,326 90,129 463,919 5,137 1,474 5,297 2,4232
        # identisch mit EPSG-Code 1618, siehe http://www.epsg-registry.org/
        # identisch mit der im WKT-String in SpatiaLite selbst beschriebenen Trafo:
        # 31257	MGI / Austria GK M28	... TOWGS84[577.326,90.129,463.919,5.137,1.474,5.297,2.4232]
        #repl = re.sub( r"\+datum=hermannskogel", "+ellps=bessel +towgs84=577.326,90.129,463.919,5.137,1.474,5.297,2.4232", hermannskogel["proj4text"] )
        srtext, proj4text = fixCrsWkt( row["srtext"], True )
        if srtext.strip() == row["srtext"].strip() and proj4text.strip() == row["proj4text"].strip():
            continue
            
        cursor.execute("""
            UPDATE spatial_ref_sys
            SET srtext=:srtext,
                proj4text=:proj4text
            WHERE srid=:srid
        """, { 'srtext' : srtext, "proj4text" : proj4text, "srid" : row['srid'] })

def _summary():
    pass

_setup_summary_docstring( _summary, __name__ )