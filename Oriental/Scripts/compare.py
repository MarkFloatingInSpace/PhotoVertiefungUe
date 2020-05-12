# -*- coding: utf-8 -*-
"""Compare image orientations estimated by multiple calls to relOri on the same image (sub-) set.

Use two or more DBs produced by `relOri`. Transform the blocks of subsequent calls onto the first block,
using a spatial similarity transform that minimizes the sum of squared distances between
the projection centers of the (sub-) set of images present in each pair of DBs.
"""

import _prolog
from oriental import adjust, log, ori, utils
import oriental.utils.argparse
import oriental.utils.db
import oriental.utils.filePaths

import argparse, itertools, sys
from collections import namedtuple
from pathlib import Path
from sqlite3 import dbapi2
from contextlib import suppress

import numpy as np
from scipy import linalg
from contracts import contract

Image = namedtuple( 'Image', 'prc rot' )

logger = log.Logger("compare")

class AtLeast2(argparse.Action):
    def __call__( self, parser, namespace, values, option_string=None ):
        if len(values) < 2:
            raise argparse.ArgumentTypeError( 'argument "{f}" requires at least 2 arguments'.format( f=self.dest ) )
        setattr( namespace, self.dest, values )

@contract
def parseArgs( args : 'list(str)|None' = None ):
    docList = __doc__.splitlines()
    parser = argparse.ArgumentParser( description=docList[0],
                                      epilog='\n'.join( docList[1:] ),
                                      formatter_class=utils.argparse.Formatter )

    parser.add_argument( 'dbs', type=Path, nargs='+', action=AtLeast2,
                         help='Use these DBs, at least 2.' )
    parser.add_argument( '--outDir', default=Path.cwd() / "compare", type=Path,
                         help='Store results in directory OUTDIR.' )
    utils.argparse.addLoggingGroup( parser, "compareLog.xml" )

    generalGroup = parser.add_argument_group('General', 'Other general settings')
    generalGroup.add_argument( '--no-progress', dest='progress', action='store_false',
                               help="Don't show progress in the console." )

    cmdLine = sys.argv[:]
    args = parser.parse_args( args=args )
    main( args, cmdLine, parser )

@contract
def main( args : argparse.Namespace, cmdLine : 'list(str) | None' = None, parser : 'ArgumentParser|None' = None ):
    with suppress(FileExistsError):
        args.outDir.mkdir(parents=True)
    args.outDir = args.outDir.resolve()
    utils.argparse.applyLoggingGroup( args, args.outDir, logger, cmdLine )

    if not args.progress:
        Progress.deactivate()

    args.dbs = [ db.resolve() for db in args.dbs ]

    logger.infoScreen( 'Output directory: {}', args.outDir )
    utils.argparse.logScriptParameters( args, logger, parser )

    firstBlock = None
    for dbFn in args.dbs:
        # Must be opened read-only, so file modification time stays the same, which we compare to the modification times of temporary files.
        with dbapi2.connect( '{}'.format( utils.db.uri4sqlite(dbFn) ), uri=True ) as db:
            utils.db.initDataBase( db )
            block = {}
            for row in db.execute('''
                SELECT path, X0, Y0, Z0, r1, r2, r3, parameterization
                FROM images ''' ):
                path = Path(row['path']).stem
                prc = np.array([ row[el] for el in 'X0 Y0 Z0'.split() ], float )
                rot = adjust.parameters.EulerAngles( adjust.EulerAngles.names[row['parameterization']],
                                                     np.array([ row[el] for el in 'r1 r2 r3'.split() ], float ) )
                block[path] = Image( prc=prc, rot=rot )

            if len(set(block)) < len(block):
                raise Exception('Block contains images with identical stems: {}'.format( ' '.join( set(block).symmetric_difference(block) ) ) )
            if len(block) < 3:
                raise Exception('Each block needs to consist of at least 3 images, but {} contains only {}'.format( dbFn, len(block) ) )

            prcs = np.array([ img.prc for img in block.values() ])
            red = prcs.mean(axis=0)
            for img in block.values():
                img.prc[:] -= red

            if firstBlock is None:
                firstBlock = block
                continue

            commonPaths = set(firstBlock).intersection( block )
            if len(commonPaths) < 3:
                raise Exception('Each pair of blocks needs to have at least 3 images in common, but {} and {} have only {}'.format( args.dbs[0], dbFn, len(commonPaths) ) )
            x = np.array([ block     [path].prc for path in commonPaths ])
            y = np.array([ firstBlock[path].prc for path in commonPaths ])
            # y=s*R.dot(x-x0)
            s, R, x0, sigma0 = ori.similarityTrafo( x=x, y=y )

            prcDiffs = y - s * R.dot( ( x - x0 ).T ).T
            thetas = []
            shortPaths = utils.filePaths.ShortFileNames(list(commonPaths))
            for path in commonPaths:
               R1 = ori.euler2matrix( block     [path].rot )
               R2 = ori.euler2matrix( firstBlock[path].rot )
               R0 = R2.T @ R @ R1
               # tr(R) = 1 + 2 cos(theta)
               theta = np.arccos( ( R0.trace() - 1 ) / 2 )
               thetas.append( theta / np.pi * 200. )
            msgs = [ 'Block pair {} - {}'.format( args.dbs[0], dbFn ),
                    'image\t\N{GREEK CAPITAL LETTER DELTA}X0\t\N{GREEK CAPITAL LETTER DELTA}Y0\t\N{GREEK CAPITAL LETTER DELTA}Z0\tnorm\t\N{GREEK SMALL LETTER THETA}[gon]' ]
            msgs += ( '\t'.join( str(el) for el in itertools.chain( [shortPaths(path)], prcDiff.tolist(), [linalg.norm(prcDiff), theta]) )
                                                for path, prcDiff, theta in utils.zip_equal( commonPaths, prcDiffs, thetas ) )
            for name, func in utils.zip_equal( ( 'min median mean max'.split() ),
                                               (np.min, np.median, np.mean, np.max ) ):
                vals = [ func(el) for el in ( prcDiffs[:,0], prcDiffs[:,1], prcDiffs[:,2], linalg.norm( prcDiffs, axis=0 ), thetas ) ]
                msgs.append( '\t'.join( str(el) for el in itertools.chain( [name], vals ) ) )
            logger.info( '\n'.join( msgs ) )



if __name__ == '__main__':
    utils.argparse.exitCode( parseArgs )