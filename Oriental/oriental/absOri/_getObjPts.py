# -*- coding: cp1252 -*-
import numpy as np
import sqlite3.dbapi2 as db

def getObjPts(dbFn):
    """return all object points of the relatively oriented block"""
    
    with db.connect( dbFn ) as relOri:
        # using an SQL WITH-clause might be handy here,
        # but it is supported only starting from SQLite version 3.8.3 (2014-02-03), see: http://www.sqlite.org/changes.html
        nObjPts = relOri.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT *
                FROM objpts
                    JOIN    imgobs
                         ON imgobs.objPtID=objpts.id
                WHERE     X(objpts.pt) > -100. AND X(objpts.pt) < 100.
                      AND Y(objpts.pt) > -100. AND Y(objpts.pt) < 100.
                      AND Z(objpts.pt) > -100. AND Z(objpts.pt) < 100.
                GROUP BY objpts.id
                HAVING COUNT(*)>2	
            	)
            """).fetchone()[0]
            
        objPts = np.empty( (nObjPts,3) )
        rows = relOri.execute( """
            SELECT X(objpts.pt),
                   Y(objpts.pt),
                   Z(objpts.pt)
            FROM objpts
            	JOIN    imgobs
            	     ON imgobs.objPtID=objpts.id
            WHERE     X(objpts.pt) > -100. AND X(objpts.pt) < 100.
                  AND Y(objpts.pt) > -100. AND Y(objpts.pt) < 100.
                  AND Z(objpts.pt) > -100. AND Z(objpts.pt) < 100.
            GROUP BY objpts.id
            HAVING COUNT(*)>2
            """ )
       
        for curRow,arrRow in zip(rows,objPts):
            arrRow[:] = curRow
            
    return objPts