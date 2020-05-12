# -*- coding: cp1252 -*-

from oriental import log

import numpy as np
from contracts import contract
from osgeo import osr
osr.UseExceptions()

import math
import re

logger = log.Logger("oriental.utils.crs")

# EPSG Codes international:
# 4326 WGS 84 geographic
# 4978 WGS 84 cartesian
#  Die EPSG Codes für Österreich sind
# Datum (Geografische Länge und Breite):
#   4312 MGI, Nullmeridian=Greenwich
#   4805 MGI, Nullmeridian=Ferro
#   3906 MGI 1901 veraltet!?
# 31251-31286 und 31288-31290 (transversale Merkatorprojektion) und
# 31287 (Lambert-Projektion).

srsMGI = (
   3416, # ETRS89 / Austria Lambert  ETRS89  Austria Lambert  13d 20m  47d 30m  Greenwich  400000  400000 Lambert Conic Conformal (2SP)  49d  46d
  31251, # MGI (Ferro) / Austria GK West Zone    MGI (Ferro) Austria Gauss-Kruger West Zone    28d  0d Ferro 0 -5000000 Transverse Mercator
  31252, # MGI (Ferro) / Austria GK Central Zone MGI (Ferro) Austria Gauss-Kruger Central Zone 31d  0d Ferro 0 -5000000 Transverse Mercator
  31253, # MGI (Ferro) / Austria GK East Zone    MGI (Ferro) Austria Gauss-Kruger East Zone    34d  0d Ferro 0 -5000000 Transverse Mercator
  31254, # MGI / Austria GK West    MGI Austria Gauss-Kruger West    10d 20m 0d Greenwich 0      -5000000 Transverse Mercator
  31255, # MGI / Austria GK Central MGI Austria Gauss-Kruger Central 13d 20m 0d Greenwich 0      -5000000 Transverse Mercator
  31256, # MGI / Austria GK East    MGI Austria Gauss-Kruger East    16d 20m 0d Greenwich 0      -5000000 Transverse Mercator
  31257, # MGI / Austria GK M28     MGI Austria Gauss-Kruger M28     10d 20m 0d Greenwich 150000 -5000000 Transverse Mercator
  31258, # MGI / Austria GK M31     MGI Austria Gauss-Kruger M31     13d 20m 0d Greenwich 450000 -5000000 Transverse Mercator
  31259  # MGI / Austria GK M34     MGI Austria Gauss-Kruger M34     16d 20m 0d Greenwich 750000 -5000000 Transverse Mercator
)

srsMGI_deprecated = (
  31281, # MGI (Ferro) / Austria West Zone MGI (Ferro) Austria West Zone 28d 0d Ferro 0 0 Transverse Mercator
  31282, # MGI (Ferro) / Austria Central Zone MGI (Ferro) Austria Central Zone 31d 0d Ferro 0 0 Transverse Mercator
  31283, # MGI (Ferro) / Austria East Zone MGI (Ferro) Austria East Zone 34d 0d Ferro 0 0 Transverse Mercator
  31284, # MGI / Austria M28 MGI Austria M28 10d 20m 0d Greenwich 150000 0 Transverse Mercator
  31285, # MGI / Austria M31 MGI Austria M31 13d 20m 0d Greenwich 450000 0 Transverse Mercator
  31286, # MGI / Austria M34 MGI Austria M34 16d 20m 0d Greenwich 750000 0 Transverse Mercator
  31287, # MGI / Austria Lambert MGI Austria Lambert 13d 20m 47d 30m Greenwich 400000 400000 Lambert Conic Conformal (2SP) 49d 46d
  31288, # MGI (Ferro) / M28 MGI (Ferro) Austria zone M28 28d 0d Ferro 150000 0 Transverse Mercator
  31289, # MGI (Ferro) / M31 MGI (Ferro) Austria zone M31 31d 0d Ferro 450000 0 Transverse Mercator
  31290, # MGI (Ferro) / M34 MGI (Ferro) Austria zone M34 34d 0d Ferro 750000 0 Transverse Mercator Transformationen von MGI nach ETRS89 bzw. WGS84 Tfm Code Name X-axis translation Y-axis translation Z-axis translation X-axis rotation Y-axis rotation Z-axis rotation Scale difference EuroGeographics Identifier
  31297 # MGI / Austria Lambert (deprecated); identical to 31287, but with Y/X-axes swapped
)

# dhm_lamb_10m.tif does not contain -towgs84 ! Also, 1st and 2nd parallels of the Lambert projection are swapped!?
# There is an almost identical EPSG-code: 6843, which really has no -towgs84. However, the latitude of origin is wrong: 48. instead of 47.5
# Interestingly, tgtCs.ExportToProj4() prints +datum=hermannskogel, as with SpatiaLite. Thus, doing as with SpatiaLite, and re-importing the edited proj4-string should be a general fix. 
# -> use function that adds -towgs84 shift for datum definitions that miss it, as for SpatiaLite.

# Seemingly, all projection definitions with datum MGI in SpatiaLite, and the projection information of the Austrian-wide DTM dhm_lamb_10m.tif
# do not contain a shift to WGS84, and so the heights of transformed coordinates ARE affected, even when transforming to a CRS with the same horizontal and vertical datum and the same ellipsoid!
# Probably, PROJ.4 assumes a zero-shift for CRS that do not define a shift to WGS84!?
# According to https://trac.osgeo.org/gdal/ticket/3450
# osr.SpatialReference.ExportToProj4() even stripped any +towgs84 parameters for datums known to Proj.4. Proj.4 then used hard-coded +towgs84.
# Maybe that is the reason why there are data sets with +datum=hermannskogel that do not define +towgs84!
# Another reason might be the following: https://trac.osgeo.org/gdal/wiki/Release/1.9.0-News -> "Refresh with libgeotiff 1.4.0, to support for GeogTOWGS84GeoKey"
# We leave +datum=hermannskogel and just add +towgs84, but not bessel, as that seems to be implied by +datum=hermannskogel
@contract
def fixCrsProj4( proj4text : str ):
    #decimal = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
    #mgi2Wgs84 = '+towgs84=577.326,90.129,463.919,5.137,1.474,5.297,2.4232'
    #besselEllipsoid = '+ellps=bessel'
    #if '+datum=hermannskogel' in proj4text:
    #    if '+towgs84' not in proj4text:
    #        proj4text = re.sub( r"\+datum=hermannskogel", "+datum=hermannskogel {}".format( mgi2Wgs84 ), proj4text )
    #    else:
    #        proj4text = re.sub( r"\+towgs84={}".format( r'\s*,\s*'.join( (decimal,)*7 ) ), mgi2Wgs84, proj4text )
    #    if besselEllipsoid not in proj4text:
    #        proj4text = re.sub( r"\+datum=hermannskogel", "+datum=hermannskogel {}".format( besselEllipsoid ), proj4text )
    cs = osr.SpatialReference()
    cs.ImportFromProj4( proj4text )
    wkt = fixCrsWkt( cs.ExportToWkt() )
    cs = osr.SpatialReference()
    cs.ImportFromWkt( wkt )
    proj4text = cs.ExportToProj4()
    # Concerning vertical datums support by PROJ.4 and links to downloads of vertical datum shift grids, see:
    # https://trac.osgeo.org/proj/
    # for BEV's EVRS to Adria1875:
    # http://www.bev.gv.at/portal/page?_pageid=713,2204753&_dad=portal&_schema=PORTAL
    # "Das Höhen-Grid ermöglicht die Transformation orthometrischer Höhen im EVRS (European Vertical Reference System) mit Bezug Amsterdam in Höhen des MGI mit Bezug Adria 1875 (Gebrauchshöhen H.ü.A.)."

    # TODO: add +geoidgrids for transformations between ellisoidal and orthometric heights?
    return proj4text

@contract
def fixCrsWkt( wkt : str, proj4text : bool = False ):
    'converting WKT to PROJ.4-strings strips off the projection, datum, and ellipsoid names and EPSG codes. However, we want them to be preserved. Thus, do not convert to PROJ.4-string and back, but edit the WKT directly.'
    if not wkt.strip():
        return wkt
    cs = osr.SpatialReference()
    cs.ImportFromWkt( wkt )
    if cs.GetAttrValue('DATUM').lower().startswith( 'militar_geographische_institut' ):
        cs.SetTOWGS84( 577.326, 90.129, 463.919, 5.137, 1.474, 5.297, 2.4232 )
    #proj4text = fixCrsProj4( cs.ExportToProj4() )
    #cs.ImportFromProj4( proj4text )
    if not proj4text:
        return cs.ExportToWkt()
    return cs.ExportToWkt(), cs.ExportToProj4()

@contract
def utmZone( lon : float ) -> int:
    return math.ceil( ( lon + 180 ) / 6 )

def BMNToTB2( bmn ):
    """Returns the Triangulationsblatt 1:2000 (TB-2) number that contains the given point.
    bmn must be given in Bundesmeldenetz-coords.
    
    TB (10km x 10km): http://doris.ooe.gv.at/geoinformation/metadata/pdf/blattschn.pdf
    -> Bezugspunkt für die Blattbezeichnung ist die rechte obere
    Ecke. z. B.:  5638:
    56: gekürzter Rechtswert y = 560.000 + 450.000 (M31)
    38: gekürzter Hochwert x = 5.380.000, wobei der konstante Wert 5.000.000 weggelassen wird
    
    TB2 (TB unterteilt in Raster mit 8 Spalten und 10 Zeilen -> 1250m x 1000m): http://ispacevm01.researchstudio.at/ina/rest/document?f=html&id={28A18368-1694-460E-844F-8B2CC84C05A5}
    Bundesmeldenetz: http://de.wikipedia.org/wiki/Bundesmeldenetz
    """
    if bmn[1] > 5000000:
        raise Exception("y-coord is expected to be reduced by 5000000")
    #additionConstantsX = { 'M28' : 150000,
    #                       'M31' : 450000,
    #                       'M34' : 750000 }
    #additionConstantX = additionConstantsX[meridian]
    #if additionConstantX is None:
    #  raise Exception("Could not determine meridian")
    #gk = list(gk)
    # EPSG 31257 - 31259 sind das Bundesmeldenetz!
    #gk[0] += additionConstantX
    tb = ( int(bmn[0] // 10000 + 1),
           int(bmn[1] // 10000 + 1) )
    colRow = (   ( bmn[0] - (tb[0]-1) * 10000 ) // 1250,
               9-( bmn[1] - (tb[1]-1) * 10000 ) // 1000 )
    tb2 = int(colRow[0]+1 + colRow[1]*8)
    return "{:02}{:02}-{:02}".format(tb[0],tb[1],tb2)

# Die Kartenblätter im Bundesmeldenetz sind wie folgt codiert:
# ABCC-DD
# wobei:
# A .. Länge östl. von Ferro minus 27°
# B .. Breite nördl. minus 40°
# C .. fortlaufende Nummer (beginnend in NW-Ecke mit 1) eines 4x4-Rasters mit 15'x15'-Zellen, zentriert über AB
# D .. fortlaufende Nummer (beginnend in NW-Ecke mit 1) eines 8x8-Rasters mit 15'/8x15'/8-Zellen, zentriert über C
#nicht exakt: GPS bezieht sich auf WGS84, BMN auf MGI!
def geog2BMNBlatt(lonLat):
    """lonLat refers to MGI == EPSG 4805"""
    if lonLat[0] < 0:
        raise Exception("BMN is undefined in the West of Ferro")
    if lonLat[1] < 0:
        raise Exception("BMN is undefined in the South of the equator")
    geogBMN = np.array([ lonLat[0] - 27, lonLat[1] - 40 ])
    geogBMNRound = geogBMN.round()
    geogBMNSub = geogBMN - ( geogBMNRound + (-.5,+.5) )
    geogBMNSub[1] = -geogBMNSub[1]
    colRow = (geogBMNSub*4) // 1
    bmn = colRow[1]*4 + colRow[0] + 1
    return int( round( geogBMNRound[0]*1000 + geogBMNRound[1]*100 + bmn ) )

def lonGreenwichToMeridian( lon ):
    meridian = int( 28 + 3 * round( ( lon + 17 + 40/60 - 28 ) / 3 ) )
    if meridian not in ( 28, 31, 34 ):
        logger.warning("Undefined MGI meridian encountered: {}", meridian)
    return meridian

@contract
def prettyWkt( cs : osr.SpatialReference ) -> str:
    ''' return cs.ExportToPrettyWkt(), but with all but the root-authority removed
    Keep the root-authority, because it is usually the EPSG-code of the whole CRS
    '''
    pattern = r',?AUTHORITY\[".+?","\d+?"\]'
    wkt = cs.ExportToWkt()
    wkt2 = ''.join(el.strip() for el in re.split(pattern, wkt))
    typ = wkt2[:wkt2.find('[')].strip()
    name = cs.GetAuthorityName(None)
    code = cs.GetAuthorityCode(None)
    cs2 = osr.SpatialReference(wkt2)
    if name is not None and code is not None:
        cs2.SetAuthority( typ, name, int(code) )
    return cs2.ExportToPrettyWkt()
