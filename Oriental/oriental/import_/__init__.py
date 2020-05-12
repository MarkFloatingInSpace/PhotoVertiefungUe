"""
Import a block from foreign software.
"""

import argparse, sqlite3, sys, typing
from contextlib import suppress
from pathlib import Path

from oriental import log, ObservationMode, utils
from oriental.import_.metashape import importMetaShape
from oriental.import_.matchAT import importMatchAT
import oriental.utils.argparse

Software = utils.argparse.ArgParseEnum('Software', 'auto metaShape matchAT')

logger = log.Logger('import')


def parseArgs(args: typing.Union[typing.List[str], None] = None):
    docList = __doc__.splitlines()
    parser = argparse.ArgumentParser(description=docList[0],
                                     epilog='\n'.join(docList[1:]),
                                     formatter_class=utils.argparse.Formatter)

    parser.add_argument('--project', type=Path, required=True,
                        help='File path of the project to be imported.')
    parser.add_argument('--chunk', type=int,
                        help='Id of chunk to be imported. Specify if multiple are available.')
    parser.add_argument('--software', default=Software.auto, choices=Software, type=Software,
                        help='Software that generated the project to be imported.')
    parser.add_argument('--outDir', default=Path.cwd() / "import", type=Path,
                        help='Store results in directory OUTDIR.')
    parser.add_argument('--outDb', type=Path,
                        help='Store the imported project in OUTDB. Default: OUTDIR/<PROJECT>.sqlite')

    utils.argparse.addLoggingGroup(parser, "importLog.xml")

    args = parser.parse_args(args=args)
    main(args)
    return args


def main(args: argparse.Namespace):
    with suppress(FileExistsError):
        args.outDir.mkdir(parents=True)
    args.outDir = args.outDir.resolve()
    args.project = args.project.resolve()
    utils.argparse.applyLoggingGroup(args, args.outDir, logger, sys.argv[:])
    if not args.outDb:
        args.outDb = args.outDir / args.project.with_suffix('.sqlite').name
    logger.info(f'Save data to {args.outDb}')
    with suppress(FileNotFoundError):
        args.outDb.unlink()

    if args.software == Software.auto:
        if args.project.suffix.lower() == '.psx':
            args.software = Software.metaShape
        elif args.project.suffix.lower() == '.prj':
            args.software = Software.matchAT
        else:
            raise Exception(f'Unable to guess software that produced {args.project}')

    utils.db.createUpdateSchema(str(args.outDb))

    with sqlite3.dbapi2.connect(utils.db.uri4sqlite(args.outDb), uri=True) as db:
        utils.db.initDataBase(db)
        db.execute("""
            SELECT AddGeometryColumn(
                'objpts', -- table
                'pt',     -- column
                -1,       -- srid -1: undefined/local cartesian cooSys
                'POINT',  -- geom_type
                'XYZ',    -- dimension
                1         -- NOT NULL
            )""")
        db.execute(
            'PRAGMA journal_mode = OFF')  # We create a new DB from scratch. Hence, no need for rollbacks. Journalling costs time!

        # Images may be inactive (enabled="false"),
        # And there may be image points without a corresponding object point.
        # Unfortunately, SQLite does not support INSERT OR IGNORE for foreign keys (generally, it does not support conflict resolution for foreign keys).
        # Otherwise, we could simply insert all image points and rely on the conflict resolution algorithm to skip those
        # whose objPtId references an inexistent object point id.
        # Since this is not possible, let's turn off foreign key checks temporarily, insert all image points,
        # remove all image points with inexistent object points, and finally check foreign keys.
        db.execute("PRAGMA foreign_keys = OFF")

        indices = db.execute("""
            SELECT name, sql
            FROM sqlite_master
            WHERE type='index'
            AND tbl_name IN ('cameras','images','imgobs','objpts','homographies')
            AND name LIKE 'idx_%' """).fetchall()
        for name, sql in indices:
            db.execute(f"DROP INDEX {name}")

        if args.software == Software.metaShape:
            importMetaShape(args.project, args.chunk, db, args.outDb.parent)
        elif args.software == Software.matchAT:
            importMatchAT(args.project, db, args.outDb.parent)
        else:
            raise Exception(f'Software unsupported: {args.software}')

        logger.info('Re-create indices, analyze tables, optimize file structure.')
        for name, sql in indices:
            db.execute(sql)
        db.execute("ANALYZE")
        db.commit()
        db.execute("VACUUM")

        db.execute("PRAGMA foreign_keys = ON")
        for row in db.execute("PRAGMA foreign_key_check"):
            tableName, rowid, referredTableName, iForeignKey = row
            logger.warning(
                f"row {rowid} of table {tableName} violates a foreign key constraint referring to table {referredTableName}.")

    with sqlite3.dbapi2.connect(utils.db.uri4sqlite(args.outDb) + '?mode=ro', uri=True) as db:
        utils.db.initDataBase(db, True)
        nCameras = db.execute('SELECT count(*) FROM cameras').fetchone()[0]
        nImages = db.execute('SELECT count(*) FROM images').fetchone()[0]
        nImgPts = db.execute('SELECT count(*) FROM imgObs').fetchone()[0]
        nAutoImgPts = db.execute(
            f'SELECT count(*) FROM imgObs WHERE type ISNULL OR type == {int(ObservationMode.automatic)}').fetchone()[0]
        nObjPts = db.execute('SELECT count(*) FROM objPts').fetchone()[0]
        if utils.db.tableHasColumn(db, 'objPts', 'refPt'):
            nRefImgPts = db.execute(
                'SELECT count(*) FROM imgObs JOIN objPts ON imgObs.objPtId == objPts.id WHERE objPts.refPt NOTNULL').fetchone()[
                0]
            nObjPtsRef = db.execute('SELECT count(*) FROM objPts WHERE refPt NOTNULL').fetchone()[0]
        else:
            nRefImgPts = 0
            nObjPtsRef = 0
    logger.info(f'''Number of imported objects
        #cameras\t{nCameras}
        #images\t{nImages}
        #image points total\t{nImgPts}
        #image points auto\t{nAutoImgPts}
        #image points reference\t{nRefImgPts}
        #object points total\t{nObjPts}
        #object points reference\t{nObjPtsRef}
    ''')
