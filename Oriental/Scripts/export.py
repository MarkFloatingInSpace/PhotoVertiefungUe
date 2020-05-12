# -*- coding: cp1252 -*-
"""Export parts of an SQLite data base created by OrientAL to a text file.
Data-bases are expected to conform to the layout used by OrientAL.
"""

import _prolog
from sqlite3 import dbapi2
import argparse, os, glob

from oriental.utils.db import initDataBase
from oriental.utils.argparse import Formatter

def main( dbFn, what, outFn, floatFmt ):
    dbFn = os.path.abspath(dbFn)
    # SQLite creates a new DB if the file does not exist already.
    # SpatiaLite raises an error if opening the db-file does not succeed. However, that message does not include the file name.
    if not os.path.isfile(dbFn):
        raise Exception('Data base does not exist: "{}"'.format(dbFn))
    with dbapi2.connect( dbFn )  as conn, \
         open( outFn, 'wt' ) as fout:
        initDataBase( conn )
        if what=='objPts':
            fout.write("# X\tY\tZ\tR\tG\tB\n")
            tmpl = ( '{:' + floatFmt + '}\t' )*3 + '{}\t'*3 + '\n'
            objPts = conn.execute( """
                SELECT X(pt) as X, Y(pt) as Y, Z(pt) as Z,
                       CAST( round( avg(red  ) * 255 ) as INT ) as red, 
                       CAST( round( avg(green) * 255 ) as INT ) as green,
                       CAST( round( avg(blue ) * 255 ) as INT ) as blue
                FROM objpts
                JOIN imgobs ON objpts.id == imgobs.objPtID
                WHERE type ISNULL
                GROUP BY objpts.id
                -- HAVING count(*) > 2
                """ )
            for pt in objPts:
                fout.write( tmpl.format( *pt ) )
        
        elif what=='images':
            rows = conn.execute("""
                SELECT camId, X0, Y0, Z0, r1, r2, r3, parameterization, path
                FROM images
                """ )
            fout.write("# cam\tX0\tY0\tZ0\tangle1\tangle2\tangle3\tparameterization\tpath\n")
            tmpl = '{}\t' + ( '{:' + floatFmt + '}\t' )*6 + '{}\t"{}"\n'
            for row in rows:
                fout.write( tmpl.format( *row ) )
        
        elif what=='cameras':
            rows = conn.execute("""
                SELECT id, x0, y0, z0
                FROM cameras
                """ )
            fout.write("# id\tx0\ty0\tz0\n")
            tmpl = '{}\t' + ( '{:' + floatFmt + '}\t' )*3 + '\n'
            for row in rows:
                fout.write( tmpl.format( *row ) )
        
        else:
            raise Exception( "Export type '{}' not supported".format( what ) )

if __name__ == '__main__':
    # TODO: once this script supports the export of DBs from more than one OrientAL script, we might introduce parser sub-commands for each script
    # Ah no, better not. If every script needs custom handling, then it's better to embed the export functionalities (as subcommands) into those scripts.
    docList = __doc__.splitlines()
    parser = argparse.ArgumentParser( description=docList[0],
                                      epilog='\n'.join(docList[1:]),
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    parser.add_argument( 'what', nargs='?', default="objPts", choices=['objPts','images','cameras'],
                         help="which data to export")
    parser.add_argument( '--dbFn', default='oriental.sqlite',
                         help="file path of SpatiaLite data base")
    parser.add_argument( '--outFn',
                         help="output file to create. If not specified, a file path in the cwd is chosen whose name depends on the kind of data to be exported - e.g. 'objPts.txt'")
    parser.add_argument( '--floatFmt', default="+f",
                         help="print format for floating-point numbers")
    args = parser.parse_args()
    if args.outFn is None:
        args.outFn = args.what + '.txt'
    main( args.dbFn, args.what, args.outFn, args.floatFmt )
