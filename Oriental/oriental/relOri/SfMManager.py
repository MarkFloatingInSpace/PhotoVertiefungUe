# -*- coding: cp1252 -*-
import os, re, datetime, struct, math, traceback, enum
from itertools import chain
from collections import OrderedDict, namedtuple, Counter
from pathlib import Path
from contextlib import ExitStack

import numpy as np
from scipy import linalg, ndimage
from sqlite3 import dbapi2
import cv2
from osgeo import osr, ogr
osr.UseExceptions()

#import traits.api as _traits
#import traitlets as _traitlets
from oriental.utils import traitlets as _traitlets
from contracts import contract, new_contract

from oriental import config, ori, graph, adjust, log, utils
import oriental.adjust.cost
import oriental.adjust.local_parameterization
import oriental.adjust.loss
import oriental.adjust.parameters
import oriental.utils.dsm
import oriental.utils.crs

from oriental.utils import exif, crs, stats, zip_equal
from oriental.utils import db
import oriental.utils.filePaths
from oriental.utils.argparse import ArgParseEnum
from oriental.adjust.parameters import ADP
from oriental.ori.transform import ITransform2D, IdentityTransform2D
from oriental.relOri import fiducials

from oriental.utils import ( pyplot as plt, 
                             pyplot_utils as plt_utils )

class ObjPtState(enum.IntEnum):
    new = 0
    common = 1

new_contract('ImageConnectivityImage', graph.ImageConnectivity.Image )
new_contract('PhotoDistortion', adjust.PhotoDistortion )

logger = log.Logger("relOri")

class PairWiseOri(_traitlets.HasTraits): 
    E = _traitlets.NDArray( dtype=np.float, shape=(3,3) )
    R = _traitlets.NDArray( dtype=np.float, shape=(3,3) )
    t = _traitlets.NDArray( dtype=np.float, shape=(3,) )
    
    def __init__( self, E, R, t ):
        self.E = E
        self.R = R
        self.t = t

class LbaPos:
    def __init__(self, longitude, latitude, hoehe ):
        self.longitude = longitude
        self.latitude = latitude
        self.hoehe = hoehe

class CamParams(_traitlets.HasTraits):

    ior       = _traitlets.NDArray( dtype=np.float, shape=(3,) )
    iorID     = _traitlets.CInt
    adp       = _traitlets.NDArray( dtype=np.float, shape=(9,) )
    adpID     = _traitlets.CInt

    t         = _traitlets.NDArray( dtype=np.float, shape=(3,) )
    omfika    = _traitlets.NDArray( dtype=np.float, shape=(3,) )
    
    @property
    @contract
    def R( self ) -> 'array[3x3](float)':
        return ori.omfika( self.omfika )

    @R.setter
    @contract
    def R( self, r : 'array[3](float) | array[3x3](float)' ):
        self.omfika = ori.omfika(r)
        
    # Fiducial mark transformation matrix
    pix2cam      = _traitlets.Instance( ITransform2D )
    isCalibrated = _traitlets.CBool(False)

class ImgAttr(CamParams):
    fullPath  = _traitlets.CUnicode
    shortName = _traitlets.CUnicode
    make      = _traitlets.CUnicode # EXIF_Make or LBA:film:KAMERA
    model     = _traitlets.CUnicode # EXIF_Model or LBA:film:KAMERA
    camSerial = _traitlets.CUnicode # camera serial numbers are usually integers, of course. Still, LBA contains 'kamnr' like 'T-24'. Thus, store text instead of an integer.
    timestamp = _traitlets.Instance(datetime.date, allow_none=True)
    width     = _traitlets.CInt # the width of the digital image file. For scanned images, the actual image content will be smaller.
    height    = _traitlets.CInt # >= 1
    isDigital = _traitlets.CBool(True)
    format_mm = _traitlets.NDArray( dtype=np.float, shape=(2,) )
    focalLen  = _traitlets.CFloat # float>0. unit: either px (for phos from digital cams) or mm (for scanned phos from analogue cams)
    prcWgs84APriori = _traitlets.NDArray( dtype=np.float, shape=(3,) ) # PRC a priori in WGS84 geographic coordinates: lat, lon, ellips. height
    prcWgsTangentOmFiKaAPriori = _traitlets.NDArray( dtype=np.float, shape=(3,) ) # image a priori rotation w.r.t. local tangential system at PRC pos on WGS84/GRS80
    lbaPos = _traitlets.Instance(LbaPos, allow_none=True ) # LBA's bild.ELLL, bild.ELLB, bild.HOCH, if available. Interpreted as: geographic longitude & latitude of the image's center point projected onto the MGI (Greenwich) (EPSG:4312) ellipsoid. HOCH is the flying height [m] above the DSM.

    """(n,2) array that on each row holds a vertex (ORIENT image coordinates) of a polygon (possibly non-convex, but non-self-intersecting, arbitrary orientation) that defines the area within the img where actual image content can be expected.
    return a (0,2) array if the whole image area is meaningful
    For digital images, the area is most probably the whole image.
    For scanned images, it will be smaller. """    
    mask_px = _traitlets.NDArray( dtype=np.float, shape=(-1,2) )

    keypoints = _traitlets.NDArray( dtype=np.float32, shape=(-1,7) )

    nMatchedObjPts = _traitlets.CInt # >= 0  initialised, then constant
    nReconstObjPts = _traitlets.CInt # >= 0  grows during reconstruction

    @contract
    def lbaBild( self ) -> str:
        """convert file name to tuple(film,bildNr)
           filmnr is a 6 digit number with possibly leading zero
             the leading 2 digits indicate the century of data capture: "01" -> 1900, "02" -> 2000
             the next 2 digits indicate the year of that century: "0169" <=> 1969 
             the trailing 2 digits indicate the month of data capture
             verify this by querying film.aufdat:
                 select date(aufdat) from film where filmnr="016909"
           lfdnr is a 1- or 2-digit number, sequential (ever-increasing) number within a month. 
           film = filmnr + lfdnr, where a leading zero is prepended to lfdnr, if lfdnr has only 1 digit
           bildnr is a 1- to 3-digit number
           bild = film + '.' + bildnr, where no leading zeros are prepended to bildnr
           i.e. film must be returned as str, including leading zeros  
        """
        name = os.path.splitext( os.path.basename(self.fullPath) )[0]
        #ret = re.split( "_0*", name, maxsplit=1 )
        ret = name.rsplit( sep='_', maxsplit=1 )
        if len(ret)==2:
            return ret[0] + "." + ret[1]
        return name

IORgrouping = ArgParseEnum('IORgrouping', 'sequence none all')
AnalogCameras = utils.argparse.ArgParseEnum('AnalogCameras', 'WildRc8 ZeissRmkA Soviet1953 Hasselblad')

# LBA uses custom / abbreviated camera names, e.g.: 'FujiFPix S2'
lbaCamNames = {
  "fujifpix s2"   : ("Fujifilm","FinePix S2 Pro"),
  "fujifpix"      : ("Fujifilm","FinePix S2 Pro"),
  "nikon d90 nir" : ("Nikon"   , "D90")
}

class SfMManager(_traitlets.HasTraits):
    imageShortName2idx = _traitlets.Dict # str:int
    #commonName         = _traitlets.Unicode
    shortFileNames     = _traitlets.Instance( utils.filePaths.ShortFileNames )
    imgs               = _traitlets.List( _traitlets.Instance(ImgAttr) )

    # note: np.uintp (C++: size_t) would be nice as dtype, but cv::Mat does not support unsigned 64bit integers! e.g. ori.filterMatchesByEMatrix expects np.int32    
    edge2matches = _traitlets.Dict # (int,int):array(np.int, shape=(-1,2))
    
    edge2PairWiseOri = _traitlets.Dict # (int,int):PairWiseOri
    
    imageConnectivity = _traitlets.Instance(graph.ImageConnectivity, args=(), allow_none=False )
    
    featureTracks     = _traitlets.Instance(graph.ImageFeatureTracks, args=(), allow_none=False )
    
    # map the featureTrack of each track to the corresp. object point in the project coordinate system.
    # The number of entries should finally be <= nTracks (depending on outliers).
    # Specifying this as a trait causes a performance penalty, because featureTrack2ObjPt is altered during reconstruction
    featureTrack2ObjPt = _traitlets.Dict # int:array( np.float, shape=(3,) )
    
    featureTrack2ImgFeatures = _traitlets.Dict # int:(0,0)
    
    # map image index -> camera parameters (IOR,ADP,EOR); EOR: project coordinate system
    # Unfortunately, traits does not provide a class that works for OrderedDict, as _traits.Dict works for dict, i.e. checks the types of keys and values! 
    #imgIdx2CamParams = _traits.Instance( OrderedDict, args=(), allow_none=False )
    # the set of oriented images can also be retrieved using graph.ImageConnectivity.unorientedImages().
    # orientedImgs, however, provides the list of images, in the order in which they have been oriented
    # the first image index in the list defines the datum.
    orientedImgs = _traitlets.List( _traitlets.Int )
    
    imgFeature2costAndResidualBlockID = _traitlets.Instance( OrderedDict, args=(), allow_none=False )
    
    outDir = _traitlets.Unicode
    minIntersectAngleGon = _traitlets.Float
    
    # Cython doesn't like this, because a traitlets-notifier must be a Python-function, not a C(ython)-function
    #def _edge2matches_changed( self, name, oldItems, newItems ):
    #    for value in newItems.values():
    #        value.setflags(write=False)
            
    @contract       
    def __init__( self,
                  imageFns : 'list(str)',
                  iorGrouping : IORgrouping ,
                  outDir : str,
                  minIntersectAngleGon : float,
                  analogCameraMakeModelFocalFilmformat,
                  plotFiducials : bool = False ):
        if len(imageFns) < 2:
            raise Exception( "At least 2 images needed, while only {} have been passed".format(len(imageFns)) )
        
        #imageFns.sort()
        super().__init__()
        if os.path.exists( outDir ):
            if not os.path.isdir( outDir ):
                raise Exception("outDir refers to an existing path, but it is not an existing directory")
        else:
            os.mkdir( outDir )
        self.outDir = outDir
        self.minIntersectAngleGon = minIntersectAngleGon
         
        self.shortFileNames = utils.filePaths.ShortFileNames( imageFns )

        # frequently, the images to be processed share the same contradictions between Exif- and LBA-informations. Thus, collect them and output them in a compressed way.
        exifLbaContradictions = {}
        focalLengthDigUnknown = []
        focalLengthAnalUnknown = []
        sensorExtentsExifDbDiffer = {}

        with ExitStack() as stack:
            cameras = stack.enter_context( dbapi2.connect( '{}?mode=ro'.format( db.uri4sqlite(Path(config.dbCameras))), uri=True ) )
            if config.dbAPIS:
                lba = stack.enter_context( dbapi2.connect( '{}?mode=ro'.format( db.uri4sqlite(Path(config.dbAPIS   ))), uri=True ) )
                logger.info( "Using APIS DB at: '{}'", config.dbAPIS )
                db.initDataBase( lba, readOnly=True )
            else:
                lba = None

            cameras.row_factory = dbapi2.Row
            # Instead of loading and executing ExifTool for each image file separately, do it once for all. This results in an enormous speedup, e.g. 5s instead of 51s for 276 image files.
            phoInfos = exif.phoInfo( imageFns )
            for idx,(imgFn,phoInfo) in enumerate(zip_equal(imageFns,phoInfos)):
                img = ImgAttr()
                img.fullPath = imgFn
                img.shortName = self.shortFileNames( imgFn )
                self.imageShortName2idx[img.shortName] = idx

                if phoInfo.prcWgs84 is not None:
                    img.prcWgs84APriori = phoInfo.prcWgs84
                if phoInfo.prcWgsTangentOmFiKa is not None:
                    img.prcWgsTangentOmFiKaAPriori = phoInfo.prcWgsTangentOmFiKa
                
                # concerning the time stamp, rely on Exif, because it resolves to the date + time-of-day for digital images, instead of only the date from LBA
                # That's why we open the file even if LBA contains all IOR-relevant information 
                # for all other attributes, rely on LBA.
                img.width, img.height = phoInfo.ccdWidthHeight_px[:]
                img.timestamp = phoInfo.timestamp
                img.make = phoInfo.make or ''
                img.model = phoInfo.model or ''

                userCameraMakeModel, userCameraFocal, userCameraFilmformat = analogCameraMakeModelFocalFilmformat
                if userCameraMakeModel is not None:
                    if userCameraMakeModel == AnalogCameras.WildRc8:
                        img.make, img.model  = 'Wild', 'RC8'
                    elif userCameraMakeModel == AnalogCameras.ZeissRmkA:
                        img.make, img.model  = 'Zeiss', 'RMK A'
                    elif userCameraMakeModel == AnalogCameras.Soviet1953:
                        img.make, img.model = 'Soviet', '1953'
                    else:
                        raise Exception('{} not supported'.format(userCameraMakeModel) )
                    img.isDigital = False
                    if userCameraFocal is not None:
                        phoInfo.focalLength_mm = userCameraFocal
                    if userCameraFilmformat is not None:
                        img.format_mm = np.array( [userCameraFilmformat]*2, float )
                else:
                    if userCameraFocal is not None:
                        phoInfo.focalLength_px = userCameraFocal
                # temp
                # Eichstätt, Valle Lunga
                #img.isDigital = False
                #img.make = 'Wild'
                #img.model = 'RC8'
                #phoInfo.focalLength_mm = 152.27
                #img.format_mm = np.array([230.,230.])

                # AdV Benchmark, Wuppertal
                #img.isDigital = False
                #img.make = 'Zeiss'
                #img.model = 'RMK A'
                #phoInfo.focalLength_mm = 152.50
                #img.format_mm = np.array([230.,230.])

                # AdV Benchmark, Bonn
                #img.isDigital = False
                #img.make = 'Wild'
                #img.model = 'RC8'
                #phoInfo.focalLength_mm = 153.34
                #img.format_mm = np.array([230.,230.])

                # AdV Benchmark, Potsdam
                #img.isDigital = False
                #img.make = 'Soviet'
                #img.model = '1953'
                #phoInfo.focalLength_mm = 200.52
                #img.format_mm = np.array([300., 300.])

                # There exist data sets in LBA (e.g. film 02110507), for which there
                # - is an entry in table 'film' (make,model,IOR),
                # - are entries for each image in table 'luftbild' (->footprint-geometries)
                # - are NO entries in table 'bild'
                # For now, we only need to know if images are scanned or digital + IOR, but no location. Thus, rely on an entry in table 'film'.
                film = img.lbaBild().rsplit('.',maxsplit=1)[0]
                # if LBA contains pho, then rely on LBA concerning IOR
                        
                # md: Die einzige Möglichkeit der Interpretation, ob es sich um ein digitales Bild handelt, ist in den Feldern Form1 und Form2 (Format des Bildes):
                # wenn die auf "0" sind, dann ist das Bild digital.
                if lba:
                    rows = lba.execute("""
                        SELECT (form1 != 0 AND form2 != 0) AS isAnalog,
                                form1,
                                form2,
                                kamera,
                                kammerkonstante,
                                kalibrierungsnummer,
                                flugdatum
                        FROM film
                        WHERE filmnummer = ?
                    """, [film] ).fetchall()
                    if rows:
                        if len(rows) != 1:
                            raise Exception( "More than 1 entry found in table 'film' with filmnummer = '{}'".format( film ) )
                        row = rows[0]
                        if not img.timestamp and row["flugdatum"]:
                            img.timestamp = datetime.datetime.strptime( row["flugdatum"], "%Y-%m-%d" )
                        img.isDigital = not bool(row["isAnalog"])
                        if all((row[el] for el in ('form1','form2'))): # not None and != 0.0
                            img.format_mm = np.array( [row['form1'],row['form2']], dtype=np.float )
                        if row["kamera"] and row["kamera"].strip():
                            default = row["kamera"].strip().partition(' ') # LBA naming convention seems to be: MAKE <space> MODEL, e.g. 'RMK TOP 15', or 'Nikon D700'
                            maker,kamera = lbaCamNames.get( row["kamera"].lower().strip(), (default[0].strip(),default[2].strip()) )
                            if img.make  and not img.make .lower().startswith( maker .lower() ) or \
                               img.model and not img.model.lower().endswith( kamera.lower() ):
                                key = ( img.make, img.model,
                                        maker,    kamera )
                                exifLbaContradictions.setdefault( key, [] ).append(idx)
                            img.make  = maker
                            img.model = kamera
                        # n.b.: LBA stores KKON in [mm], also for digital cameras, see e.g. D:\arap\data\140217_relOri_Test_md\test2\2 -> KKON==35.0"
                        if row["kammerkonstante"]: # not None and != 0.0
                            phoInfo.focalLength_mm = row["kammerkonstante"]

                        if row['kalibrierungsnummer'] and row['kalibrierungsnummer'].strip():
                            img.camSerial = row['kalibrierungsnummer'].strip()
                
                if img.isDigital and phoInfo.focalLength_mm:
                    if img.make and img.model:
                        # lookup cameras.sqlite for sensor size [mm], with possible re-computation of phoInfo.focalLength_px
                        # Most camera manufacturers seem to prepend their brand to the Exif-tag 'model',  e.g. Exif-'make'=='NIKON CORPORATION',     Exif-'model'=='NIKON D700'
                        # Olympus does not seem to do that: Carnuntum_UAS_Geert: phos from Olympus PEN E-P2 -> Exif-'make'=='OLYMPUS IMAGING CORP.', Exif-'model': 'E-P2'
                        # Thus, let's use the rule that cameras.make must be a prefix of the Exif-'make',
                        #                          and  cameras.model must be a suffix of the Exif-'model',
                        # and compare case-insensitively
                        rows = cameras.execute("""
                            SELECT make,
                                   model,
                                   sensor_width_mm,
                                   sensor_height_mm
                            FROM cameras
                            WHERE     :exifMake  LIKE make || '%'
                                  AND :exifModel LIKE '%'  || model
                        """, 
                        { 'exifMake':img.make, 'exifModel':img.model } ).fetchall()
                        if rows:
                            if len(rows) > 1:
                                raise Exception( "Camera make/model '{}'/'{}' is ambiguous. Possible matches: {}".format(
                                                        img.make,
                                                        img.model,
                                                        ",".join( [ "'{}'/'{}'".format( row["make"], row["model"] ) for row in rows ] ) ) )
                            row = rows[0]
                            img.make  = row["make"].strip()
                            img.model = row["model"].strip()
                            ccdWidthHeight_mm = np.array([ row["sensor_width_mm"], row["sensor_height_mm"] ])
                            if phoInfo.ccdWidthHeight_mm is not None and \
                               np.abs( phoInfo.ccdWidthHeight_mm / ccdWidthHeight_mm - 1 ).max() > 0.1:
                                sensorExtentsExifDbDiffer.setdefault( tuple(chain( phoInfo.ccdWidthHeight_mm, ccdWidthHeight_mm )), [] ).append( idx )
                            phoInfo.ccdWidthHeight_mm = ccdWidthHeight_mm
                                
                    if phoInfo.ccdWidthHeight_mm is not None:
                        img.format_mm = phoInfo.ccdWidthHeight_mm
                        phoInfo.focalLength_px = phoInfo.focalLength_mm * np.mean( phoInfo.ccdWidthHeight_px / phoInfo.ccdWidthHeight_mm )
                    
                focalLen = phoInfo.focalLength_px if img.isDigital else phoInfo.focalLength_mm
                if not focalLen:
                    if img.isDigital:
                        focalLen = (img.width**2 + img.height**2)**.5
                        focalLengthDigUnknown.append( idx )
                    else:
                        focalLengthAnalUnknown.append( idx  )
                        continue

                img.focalLen = focalLen
                
                if lba: 
                    # In case not enough PRC positions / rotations are available from Exif, then LBA may still have information for coarse geo-referencing.
                    bildSplit = img.lbaBild().rsplit('.',maxsplit=1)
                    if len(bildSplit) > 1:
                        bildnummer = film + '.' + bildSplit[1]
                        rows = lba.execute("""
                                SELECT longitude, latitude, hoehe
                                FROM luftbild_senk_cp
                                WHERE bildnummer = :bildnummer
                            UNION ALL
                                SELECT longitude, latitude, hoehe
                                FROM luftbild_schraeg_cp
                                WHERE bildnummer = :bildnummer
                        """, { 'bildnummer' : bildnummer } )
                        for iRow,row in enumerate(rows):
                            if iRow > 0:
                                raise Exception( "More than 1 entry found in tables 'luftbild_senk_cp' and 'luftbild_schraeg_cp' with bildnummer = '{}'".format( bildnummer ) )
                            img.lbaPos = LbaPos( longitude = float(row['longitude']),
                                                 latitude  = float(row['latitude' ]),
                                                 hoehe     = float(row['hoehe'    ]) )
                                  
                self.imgs.append( img )
        
        if len({ img.isDigital for img in self.imgs }) > 1:
            raise Exception( 'Either all or none of the images must have been scanned' )

        if exifLbaContradictions:
            logger.warning( 'contradicting information on camera make/model. Using camera according to LBA.\n{}{}',
                    'Exif Make\tExif Model\tLBA Make\tLBA Model\tAffected photos\n',
                    '\n'.join( '\t'.join(key) + '\t' +
                               (', '.join(self.imgs[iImg].shortName for iImg in iImgs) if len(iImgs) < len(imageFns) else '<all>' )
                               for key,iImgs in exifLbaContradictions.items() ) )

        if sensorExtentsExifDbDiffer:
            logger.warning( 'Sensor extents [mm] according to Exif and sensor data base differ considerable. Have you resized those images? Using sensor extents according to DB.\n{}{}',
                            'Exif width\tExif height\tDB width\tDBheight\tAffected photos\n',
                            '\n'.join( '\t'.join('{:.4f}'.format(el) for el in key ) + '\t' +
                                       ( ', '.join(self.imgs[iImg].shortName for iImg in iImgs) if len(iImgs) < len(imageFns) else '<all>' )
                                       for key, iImgs in sensorExtentsExifDbDiffer.items() ) )

        for iUnknown,unknowns in enumerate(( el for el in (focalLengthDigUnknown, focalLengthAnalUnknown) if el )):
            makeModel2shortNames = {}
            for iImg in unknowns:
                makeModel2shortNames.setdefault( (self.imgs[iImg].make,self.imgs[iImg].model), [] ).append( self.imgs[iImg].shortName )
            logger.warning( '{} image: {} file lacks the Exif tags necessary to estimate the focal length. {}:\n'
                            'Make\tModel\tAffected photos\n{}',
                            'Digital' if iUnknown==0 else 'Scanned',
                            'LBA contains no focal length, and ' if config.dbAPIS else '',
                            'Assuming normal angle lens' if iUnknown==0 else 'Image(s) will not be oriented',
                            '\n'.join( '\t'.join(key) + '\t' +
                                       ( ', '.join(values) if len(values) < len(imageFns) else '<all>' )
                                       for key,values in makeModel2shortNames.items() ) )

        if not len(self.imgs):
            raise Exception("No images left that could be oriented!")

        self._setIORsADPs( iorGrouping, plotFiducials )

        # avoid cluttering the screen with too many digits after the comma,
        # but log to file with full precision
        def initIorsAdps(fmt):
            storedIorIDs = set()
            storedAdpIDs = set()
            return ( 'Initial IORs\n' +
                     '\t'.join(('ID','x_0','y_0','z_0')) + '\n' +
                     '\n'.join((
                         str(img.iorID) + '\t' +
                         '\t'.join(( fmt.format(val) for val in img.ior ))
                         for img in self.imgs if not ( img.iorID in storedIorIDs or storedIorIDs.add(img.iorID) ) )) +
                    '\v'
                    'Initial ADPs\n' +
                    '\t'.join(('ID','skew','scaleY','r3','r5','t1','t2','r7','r9','r11')) + '\n' +
                    '\n'.join((
                        str(img.adpID) + '\t' +
                        '\t'.join(( fmt.format(val) for val in img.adp ))
                        for img in self.imgs if not ( img.adpID in storedAdpIDs or storedAdpIDs.add(img.adpID) ) ))
                   )

        logger.infoFile(
            'Initial photo properties\n' +
            'file\tmake\tmodel\tdigital\tIOR/ADP ID\tcalibrated\n' +
            '\n'.join((
                '\t'.join((
                    img.shortName,
                    img.make  if len(img.make)  else "<unknown>",
                    img.model if len(img.model) else "<unknown>",
                    str(img.isDigital),
                    str(img.iorID),
                    str(img.isCalibrated) ))
                for img in self.imgs )) + '\v' +
            initIorsAdps('{:f}') )

        counter = Counter( ( img.make  if len(img.make)  else "<unknown>",
                             img.model if len(img.model) else "<unknown>",
                             img.isDigital,
                             img.iorID,
                             img.isCalibrated ) for img in self.imgs )
        logger.infoScreen(
            'Initial photo properties\n' +
            'nPhos\tmake\tmodel\tdigital\tIOR/ADP ID\tcalibrated\n' +
            '\n'.join(
                '\t'.join( chain( [str(count)], (str(el) for el in els ) ) )
                for els,count in counter.items() ) +
            '\v' +
            initIorsAdps('{:.2f}') )

    @contract
    def _getIorAdp( self,
                    iImg : int,
                    plotFiducials : bool ):
        img = self.imgs[ iImg ]
        if not img.isDigital:
            makeModel = ( img.make + ' ' + img.model ).strip().lower()
            camSerial = img.camSerial or None
            filmFormatFocal = ( tuple(img.format_mm) if img.format_mm is not None else (None,None) ) + (img.focalLen,)
            plotDir = os.path.join( self.outDir, 'fiducials' ) if plotFiducials else ''
            if plotDir:
                os.makedirs( plotDir, exist_ok=True )
            if makeModel == "Zeiss RMK A".lower():
                return fiducials.zeissRmkA( img.fullPath, camSerial, filmFormatFocal, plotDir )
            elif makeModel in [ "Zeiss RMK TOP".lower(), "RMK TOP 15".lower() ]:
                return fiducials.zeissRmkTop( img.fullPath, camSerial, filmFormatFocal, plotDir )
            elif makeModel == "Zeiss RMK".lower():
                excs = []
                for rmkFunc in [ fiducials.zeissRmkTop, fiducials.zeissRmkA ]:
                    try:
                        return rmkFunc( img.fullPath, camSerial, filmFormatFocal, plotDir )
                    except Exception as ex:
                        excs.append( ex )
                if len(excs) == 1:
                    raise excs[0]
                raise Exception( 'Fiducial mark detection has failed for both Zeiss RMK Top and RMK A:\n' + '\n'.join((str(el) for el in excs )) )
            elif makeModel in ( 'hasselblad', 'hasselbl', 'h553elx', 'h205fcc' ):
                return fiducials.hasselblad( img.fullPath, None, filmFormatFocal, plotDir )
            elif makeModel == "Wild RC8".lower():
                return fiducials.wildRc8(img.fullPath, camSerial, filmFormatFocal, plotDir)
            elif makeModel == "Soviet 1953".lower():
                return fiducials.potsdam(img.fullPath, camSerial, filmFormatFocal, plotDir)
            else:
                # for cameras from Bundesheer/Langenlebarn:
                # Vinten 500; c=45/98/101mm and
                # Hasselblad; c=50/100mm
                raise Exception("Fiducial mark transformation not implemented for make/model '{}/{}'".format(img.make,img.model))
        else:
            pix2cam = IdentityTransform2D()
            mask_px = None
            rmse_microns = 0.
            if os.path.dirname(img.fullPath) == r'D:\_data\car':
                isCalibrated = True # Hasselblad H3DII-39,5412x7216,"1/800",f12.0, 35
                ior = np.array([ 3575.708, -2690.199, 5175.363 ])
                adp = ADP( normalizationRadius=3007. )
                adp[3-1] = -1.013834 
                adp[4-1] =  1.103173
                adp[5-1] =   .120043
                adp[6-1] =   .241401 
            elif os.path.dirname(img.fullPath).startswith( r'D:\MilutinPielach' ):
                isCalibrated = True # IPF NIKON D800 IPF Nikon-Objektiv 28mm
                ior = np.array([ 3719.50854,  -2459.58936,  5669.88281 ])
                adp = ADP( normalizationRadius=2950. )
                adp[3-1] = -95.9791
                adp[4-1] =  18.2746
                adp[5-1] =   .959034
                adp[6-1] =   -1.68611
            elif os.path.dirname(img.fullPath).startswith(r'D:\MVI_3122_5fps'):
                isCalibrated = True # IPF Canon EOS 60D Nr.3 IPF Sigma-lens 20mm
                ior = np.array([ 2605.80371, -1730.37866, 4715.50635 ])
                adp = ADP( normalizationRadius=2077. )
                adp[3-1] = -38.8805847
                adp[4-1] =   6.1733217
            else:
                isCalibrated = False
                ior = np.array([ (img.width-1.)/2., -(img.height-1.)/2., img.focalLen ])
                twoThirdsOfHalfImgDiag = 2. / 3. * ( img.width **2 + img.height **2 )**0.5 / 2.
                adp = ADP( normalizationRadius=twoThirdsOfHalfImgDiag )
                if img.make == 'Sony' and img.model == 'Alpha NEX-5':
                    adp[adjust.PhotoDistortion.optPolynomRadial3] = -64.
                elif os.path.dirname(img.fullPath) == r'D:\BA Raskovic\flight4_singleFocal_singleFlight':
                    #ior[:] = 3001.708216, -2013.051508,	13261.385020
                    adp[adjust.PhotoDistortion.optPolynomRadial3] = 20.1
                elif os.path.dirname(img.fullPath) == r'D:\Livia\ISPRS_Prague\Sequence1_188Images_PanoFrame\Images_188':
                    ior[:] = 2851.822, -1917.141, 4900.630
                    adp[adjust.PhotoDistortion.affinityScaleOfYAxis] = 0.508
                    adp[adjust.PhotoDistortion.optPolynomRadial3] = -76.152
                    adp[adjust.PhotoDistortion.optPolynomRadial5] = 18.335
        
        return ior,adp,pix2cam,mask_px,isCalibrated,rmse_microns
    ##    # Milutin's data set: Gravel_bed_2 - camera is calibrated!
    ##    adp = ADP( normalizationRadius=1200 )
    ##    adp[ adjust.PhotoDistortion.optPolynomRadial3 ] = -16.3964
    ##    adp[ adjust.PhotoDistortion.optPolynomRadial5 ] =  1.29049
    ##    
    ##
    ##    # Forkert data set: a compound of 5 cameras that are fixed to each other.
    ##    # Camera index is indicated by last digit of file name.
    ##    iors = {}
    ##    iors_ = {}
    ##    adps = {}
    ##    adps_ = {}
    ##    for iImg in range(len(imgFns)):
    ##        camIdx = int( os.path.splitext( os.path.basename( imgFns[iImg] ) )[0][-1] )
    ##        if camIdx not in iors_:
    ##            cameraMatrix = ori.cameraMatrix( os.path.join( fns_luftbild_dir, imgFns[iImg] ) )    
    ##            iors_[camIdx] = np.array([ cameraMatrix[0,2], -cameraMatrix[1,2], cameraMatrix[0,0] ])
    ##            
    ##        if camIdx not in adps_:
    ##            adps_[camIdx] = ADP( normalizationRadius=2950 ) # 2/3 of half the image diagonal
    ##
    ##        iors[iImg] = iors_[camIdx]
    ##        adps[iImg] = adps_[camIdx]
        
    @contract
    def _setIORsADPs( self,
                      iorGrouping : IORgrouping,
                      plotFiducials : bool
                    ) -> None:
        if iorGrouping == IORgrouping.none:
            # assign different ior,adp-params to each image 
            items = []
            for iImg in range(len(self.imgs)):
                items.append( self._getIorAdp( iImg, plotFiducials ) )
            iors,adps,pix2cams,mask_pxs,areCalibrated,rmses_microns = zip_equal(*items)
    
        elif iorGrouping == IORgrouping.sequence:
            # assign the same ior,adp-params to all images
            # that were taken in sequence with the same camera, the same resolution and the same focal length -
            # i.e. only if there has not been captured another photo with different settings in between.
            
            # LBA defines the day of the data capture, but not the time of day
            # Exif gives the time of day for digital images, but not for scanned images. 
            # For scanned images, we assume that the focal length and focus are fixed by construction (cannot be altered by users),
            # and so we adjust only 1 pair of adp,ior
            # If a calibration protocol is available, then better not adjust IOR,ADP at all (by default).
            
            # sort by time stamp only if all phos have a time stamp, and if there are at least 2 different time stamps (e.g. LBA contains only the date, but no time -> better sort by file name)
            if all( (x.timestamp is not None for x in self.imgs ) ) and \
               len( { x.timestamp for x in self.imgs } ) > 1:
                imgs = sorted( self.imgs, key = lambda x: x.timestamp )
            else:
                imgs = sorted( self.imgs, key = lambda x: x.fullPath )
            
            items = []
            for iImg in range(len(imgs)):
                origIdx = self.imgs.index( imgs[iImg] )
                if iImg > 0 and \
                   imgs[iImg-1].isDigital and imgs[iImg].isDigital and \
                   all(( getattr(imgs[iImg-1],attr) == getattr(imgs[iImg],attr) for attr in ('make','model','camSerial','width','height','focalLen') )):
                    ior,adp,pix2cam,mask_px,isCalibrated,rmse_microns = items[-1][1:]
                else:            
                    ior,adp,pix2cam,mask_px,isCalibrated,rmse_microns = self._getIorAdp( origIdx, plotFiducials )
                    if iImg > 0 and \
                       not imgs[iImg-1].isDigital and not imgs[iImg].isDigital and \
                       all(( getattr(imgs[iImg-1],attr) == getattr(imgs[iImg],attr) for attr in ('make','model','camSerial','focalLen') )) and \
                       np.all(imgs[iImg-1].format_mm == imgs[iImg].format_mm ):
                        ior,adp = items[-1][1:3]
                        assert items[-1][5] == isCalibrated
                    
                items.append( ( origIdx, ior, adp, pix2cam, mask_px, isCalibrated, rmse_microns ) )

            items.sort( key=lambda x: x[0] )
            _,iors,adps,pix2cams,mask_pxs,areCalibrated,rmses_microns = tuple(zip_equal(*items))
        
        elif iorGrouping == IORgrouping.all:
            pix2cams=[]
            mask_pxs=[]
            rmses_microns=[]
            for iImg in range(len(self.imgs)):
                ior,adp,pix2cam,mask_px,isCalibrated,rmse_microns = self._getIorAdp( iImg, plotFiducials )
                pix2cams.append(pix2cam)
                mask_pxs.append(mask_px)
                rmses_microns.append(rmse_microns)
            iors = (ior,) * len(self.imgs)
            adps = (adp,) * len(self.imgs)
            areCalibrated = (isCalibrated,) * len(self.imgs)
        
        else:
            raise Exception( "IORgrouping '{}' not supported. Supported ones are: {}".format(iorGrouping,', '.join(IORgrouping.names)))
        
        if any(( not img.isDigital for img in self.imgs )):
            logger.info( 'Transformations (pix->cam):\v' +
                         'Photo\tRMSE [\xB5m]\n' +
                         '\n'.join(( '{}\t{:.2f}'.format(img.shortName,rmse) for img,rmse in zip_equal(self.imgs,rmses_microns) )) )

        iorIDs = dict()
        nextIORID = 0
        adpIDs = dict()
        nextADPID = 0
        for img,ior,adp,pix2cam,mask_px,isCalibrated in zip_equal(self.imgs,iors,adps,pix2cams,mask_pxs,areCalibrated):
            img.ior = ior
            img.adp = adp
            if pix2cam is not None: # assigning None to this trait is illegal.
                img.pix2cam = pix2cam
            if mask_px is not None:
                img.mask_px = mask_px
            img.isCalibrated = isCalibrated
            iorID = id(ior)
            if iorID not in iorIDs:
                iorIDs[iorID] = nextIORID
                nextIORID += 1
            img.iorID = iorIDs[iorID]
            adpID = id(adp)
            if adpID not in adpIDs:
                adpIDs[adpID] = nextADPID
                nextADPID += 1
            img.adpID = adpIDs[adpID]
        
    def buildFeatureTracks( self ):
        self.featureTracks = graph.ImageFeatureTracks()
        for edge,matches in self.edge2matches.items():
            for iMatch in range( matches.shape[0] ):
                self.featureTracks.join( graph.ImageFeatureID( edge[0], matches[iMatch,0].item() ),
                                         graph.ImageFeatureID( edge[1], matches[iMatch,1].item() ) )
        self.featureTracks.compute()

    def removeMultipleProjections( self ) -> 'tuple(list,list)':
        # Jeder track darf pro pho nur einmal beobachtet sein. Ansonsten beide (alle) matches entfernen, aber nur in diesem einen Bild! 
        # for each image pair, collect a set of match indices that are deemed inconsistent, and thus shall be removed
        Msg = namedtuple( 'Msg', ( 'iImg1', 'iImg2', 'nRemoved', 'nRemain' ) )  
        edge2matchesToRemove = {}
        for iImg in range(len(self.imgs)):
            # map: featureTrack -> first encountered corresponding (edge,iMatch) for current image
            featureTrack2FirstEdgeMatch = {}
            for thisEdge,matches in self.edge2matches.items():
                if iImg not in thisEdge:
                    continue
                iImgInThisEdge = thisEdge.index(iImg)
                assert thisEdge[iImgInThisEdge]==iImg
                for iThisMatch in range(matches.shape[0]):
                    iThisKeypt = matches[iThisMatch,iImgInThisEdge].item()
                    featureTrack = self.featureTracks.component( graph.ImageFeatureID( iImg, iThisKeypt ) )[1]
                    thatEdge, iThatMatch = featureTrack2FirstEdgeMatch.setdefault( featureTrack, (thisEdge, iThisMatch) )
                    if (thatEdge, iThatMatch) == (thisEdge, iThisMatch):
                        continue
                    iImgInThatEdge = thatEdge.index(iImg)
                    iThatKeypt = self.edge2matches[thatEdge][iThatMatch,iImgInThatEdge]
                    if iThatKeypt == iThisKeypt:
                        # the current and the first match correspond to the same track.
                        continue
                    # if they refer to different feature points, then remove both matches!
                    for edge,iMatch in [ (thisEdge,iThisMatch),
                                         (thatEdge,iThatMatch) ]:
                        assert iImg in edge
                        edge2matchesToRemove.setdefault( edge, set() ).add( iMatch )
                        
        del featureTrack2FirstEdgeMatch        
        
        logger.info('Remove multiple projections')
        msgs = []
        removedEdges = []
        for edge,matches in edge2matchesToRemove.items():
            matchesOrig = self.edge2matches[edge]
            nRemaining = matchesOrig.shape[0] - len(matches)
            if nRemaining >= 5:
                keep = np.ones( matchesOrig.shape[0], dtype=np.bool )
                keep[ np.array( list(matches) ) ] = False
                #self.edge2matches[edge].resize( (self.edge2matches[edge].shape[0] - len(matches), 2 ) ) # resize in-place; error: array does not own its data!
                #self.edge2matches[edge][:] = self.edge2matches[edge][keep,:] # copy in-place!
                self.edge2matches[edge] = matchesOrig[keep,:]
            else:
                # 2014-05-21: avoid edges that do not contain any matches (important), avoid edges that contain less matches than the minimum needed for rel. ori. (for performance)
                del self.edge2matches[edge]
                removedEdges.append( edge )
                nRemaining = 0

            msgs.append( Msg( edge[0], edge[1], len(matches), nRemaining ) )

        msgs.sort()
        return msgs, removedEdges
        
    @contract
    def thinOutTracks( self, nCols : int, nRows : int, minFeaturesPerCell : int ):
        # Divide the area of each image into a grid
        # Remove as many feature tracks as possible, such that a each cell of each image projects a certain minimum number of the remaining tracks.
        # Try to keep tracks with large multiplicities, discard tracks with low ones.
        logger.info( 'Thin out feature tracks on {}x{} (cols x rows) grids, keeping at least {} features per cell with largest multiplicity', nCols, nRows, minFeaturesPerCell )

        def tracks2Keep():
            def getRowCol( feature ):
                img = self.imgs[feature.iImage]
                pt = img.keypoints[feature.iFeature]
                col = int(  pt[0] / ( img.width  / nCols ) )
                row = int( -pt[1] / ( img.height / nRows ) )
                return row, col

            keptTracks = []
            featureCounts = { iImg : np.zeros( (nRows, nCols), int ) for iImg in range(len(self.imgs)) }
            nCellsLeft = len(featureCounts) * nRows * nCols
            components = list( self.featureTracks.components().items() )
            components.sort( key=lambda x: len(x[1]), reverse=True )
            for component,features in components:
                for feature in features:
                    row, col = getRowCol(feature)
                    if featureCounts[feature.iImage][row,col] < minFeaturesPerCell:
                        break
                else:
                    continue
                for feature in features:
                    row, col = getRowCol(feature)
                    featureCount = featureCounts[feature.iImage]
                    featureCount[row,col] += 1
                    if featureCount[row,col] == minFeaturesPerCell:
                        nCellsLeft -= 1
                keptTracks.append(component)
                if not nCellsLeft: # it is quite unlikely that all cells of all images get filled. However, this test doesn't cost much, so let's test and break early in that case.
                    break
            # TODO: the output of #features is misleading: while it suggests that it is the total number of features per image, it is actually only a lower bound.
            logger.log( log.Severity.info,
                        log.Sink.all if len(featureCounts) < 500 else log.Sink.file,
                        'Thinned out features per image\n'
                        'img\t#features\t#fullCells\n'
                        '{}',
                        '\n'.join( '{}\t{}\t{}'.format( self.imgs[iImg].shortName, counts.sum(), np.sum( counts >= minFeaturesPerCell ) )  for iImg,counts in featureCounts.items() ) )
            return keptTracks

        keptTracks = set(tracks2Keep())
        removedEdges = []
        edge2matches = {}
        for edge,oldMatches in self.edge2matches.items():
            matches = []
            for oldMatch in oldMatches:
                found,component = self.featureTracks.component( graph.ImageFeatureID( edge[0], oldMatch[0].item() ) )
                assert found
                if component in keptTracks:
                    matches.append( oldMatch )
            if len(matches):
                edge2matches[edge] = np.array(matches, oldMatches.dtype)
            else:
                removedEdges.append(edge)

        nTracksBefore = self.featureTracks.nComponents()
        self.edge2matches = edge2matches
        logger.info('Re-compute feature tracks')
        self.buildFeatureTracks()
        nTracksAfter = self.featureTracks.nComponents()
        logger.info('#objPts reduced to {}({:2.0%}) from {}', nTracksAfter, nTracksAfter/nTracksBefore, nTracksBefore )
        return removedEdges

    def countFeatures( self ):
        "return the number of features over all images that are matched at least once"
        featureCounts = [ np.zeros( len(img.keypoints), dtype=bool ) for img in self.imgs ]
        for edge,matches in self.edge2matches.items():
            featureCounts[edge[0]][matches[:,0]] = True
            featureCounts[edge[1]][matches[:,1]] = True
        nFeaturesTotal = 0
        for img,featureCount in zip_equal(self.imgs,featureCounts):
            img.nMatchedObjPts = featureCount.sum()
            nFeaturesTotal += img.nMatchedObjPts
        return nFeaturesTotal


    @contract
    def logCamParams( self,
                      block : adjust.Problem,
                      severity = log.Severity.info,
                      withIorAdp : bool = False ) -> None:
        # avoid cluttering the screen with too many digits after the comma.
        # Still, log to file with full precision
        for idx,prec in enumerate([1,None]):
            msgs = [ 'EORs',
                     '\t'.join(['pho','X0','Y0','Z0','\N{GREEK SMALL LETTER OMEGA}','\N{GREEK SMALL LETTER PHI}','\N{GREEK SMALL LETTER KAPPA}','IOR/ADP ID']) ]
            precFmt = '{:' +  ( '.{}'.format(prec) if prec else '' ) + 'f}'
            fmt = '{}\t' + '\t'.join( [precFmt]*6 ) + '\t{}'
            for iImg in self.orientedImgs:
                img = self.imgs[iImg]
                args = chain( [img.shortName],
                              img.t,
                              img.omfika,
                              [img.iorID] )
                msgs.append( fmt.format( *args ) )
    
            msgs = '\n'.join( msgs )

            if withIorAdp:
                iorMsgs = []
                storedIorIDs = set()
                fmt = '{}\t' + '\t'.join([precFmt+'{}']*3) + '\t' + '\t'.join(['{:.1f}{}']*2)
                for iImg in self.orientedImgs:
                    img = self.imgs[iImg]
                    if img.iorID not in storedIorIDs:
                        storedIorIDs.add( img.iorID )
                        areConst = []
                        for parBlock in ( img.ior, img.adp ):
                            isParBlockConst = block.IsParameterBlockConstant( parBlock )
                            locParam = block.GetParameterization( parBlock )
                            areConst.append( [ isParBlockConst or locParam is not None and locParam.isConstant(idx) for idx in range(len(parBlock)) ] )
                        args = [ img.iorID ] + \
                               [ el for idx in range(len(img.ior))                                                                  for el in ( img.ior[idx], 'c' if areConst[0][idx] else 'v' ) ] + \
                               [ el for idx in [adjust.PhotoDistortion.optPolynomRadial3, adjust.PhotoDistortion.optPolynomRadial5] for el in ( img.adp[idx], 'c' if areConst[1][idx] else 'v' ) ]
                        iorMsgs.append( fmt.format( *args ) )
                iorMsgs = [ 'IORs, ADPs (c=constant, v=variable)',
                            '\t'.join(['IOR/ADP ID','x0','y0','z0','r3','r5']) ] + sorted( iorMsgs, key=lambda x: x[0] )
                msgs += '\v' + '\n'.join( iorMsgs )

            if idx==0:
                logger.log( severity, log.Sink.screen, msgs )
            else:
                logger.log( severity, log.Sink.file, msgs )
        
    @contract
    def logActiveFeaturesPerImage( self ) -> None:
        imgs = [ (self.imgs[iImg].shortName, self.imgs[iImg].nReconstObjPts, iImg) for iImg in self.orientedImgs ]
        imgs.sort( key=lambda x: x[1] )
        logger.info( 'Active features per image\n'
                     'img\t#features\torder\n'
                     '{}',
                     '\n'.join( '\t'.join( str(el) for el in img ) for img in imgs ) )


    @contract
    def logStatistics( self, resultsFn : str ) -> None:
        idx2imageShortName = { val:key for key,val in self.imageShortName2idx.items() }
        with dbapi2.connect( resultsFn ) as conn:
            #rows = conn.execute("""
            #    SELECT imgid, COUNT(*)
            #    FROM imgobs
            #    GROUP BY imgid""")
            #nPts = sorted( ( "{}\t{}".format( idx2imageShortName[row[0]], row[1] ) for row in rows ),
            #               key=lambda x: x[0] )
            #logger.info( 'pho\t#image points\n{}',
            #             '\n'.join( nPts ),
            #             tag='#image points per image' )
            rows = conn.execute("""
                SELECT imgobs.imgID,
                       COUNT(*) as total,
                       sum(n2obs_) as n2obs,
                       sum(n3obs_) as n3obs,
                       sum(n4obs_) as n4obs,
                       sum(nMore_) as nMore
                FROM imgobs
                JOIN (
	                SELECT objPtID as oPtID,
	                CASE WHEN COUNT(*)==2 THEN 1 ELSE 0 END as n2obs_,
	                CASE WHEN COUNT(*)==3 THEN 1 ELSE 0 END as n3obs_,
	                CASE WHEN COUNT(*)==4 THEN 1 ELSE 0 END as n4obs_,
	                CASE WHEN COUNT(*)>4  THEN 1 ELSE 0 END as nMore_
	                FROM imgobs
	                GROUP BY objPtID
                ) ON imgobs.objPtID==oPtID
                GROUP BY imgobs.imgID
                ORDER BY imgobs.imgID""")
            nPts = sorted( ( '\t'.join(['{}']*6).format( *( (idx2imageShortName[row[0]],) + row[1:] ) ) for row in rows ),
                           key=lambda x: x[0] )
            logger.info( 'pho\t#total\t#2folds\t#3folds\t#4folds\t#morefolds\n{}',
                         '\n'.join( nPts ),
                         tag='#image points per image: total and grouped by multiplicity' )
            ###
            rows = conn.execute("""
                SELECT CASE WHEN cnt>640 THEN 7
                            WHEN cnt>320 THEN 6
                            WHEN cnt>160 THEN 5
                            WHEN cnt>80  THEN 4
                            WHEN cnt>40  THEN 3
                            WHEN cnt>20  THEN 2
                            WHEN cnt>10  THEN 1
                            ELSE              0
                       END AS bin,
                       COUNT(*)
                FROM (
                    SELECT COUNT(*) AS cnt
                    FROM imgobs
                    GROUP BY imgID
                )
                GROUP BY bin
                ORDER BY bin""")
            lowerBoundsExclusive = [0] + [10*2**exp for exp in range(7)]
            bin2nImgs = { bin : nImgs for bin,nImgs in rows }
            binNImgs = [ (bin,bin2nImgs.get(bin,0)) for bin in range(8) ]
            logger.info( '#image points\t#images\n{}',
                         '\n'.join((
                             '{}\t{}'.format(
                                 '>{}'.format( lowerBoundsExclusive[bin] ) if lowerBoundsExclusive[bin]>0 else '<=10',
                                 count
                             )
                             for bin,count in reversed(binNImgs) )),
                         tag='Histogram of #image points per image' )

            ###
            rows = conn.execute("""
                SELECT cnt, COUNT(cnt)
                FROM (
	                SELECT COUNT(*) AS cnt
	                FROM imgobs
	                GROUP BY objPtID
                )
                GROUP BY cnt
                ORDER BY cnt""")
            nObs2nPts = { el[0]:el[1] for el in rows }
            nObsnPts = [ (idx,nObs2nPts.get(idx,0)) for idx in range(2,1+max(nObs2nPts)) ]
            logger.info( '#image points\t#object points\n{}',
                         '\n'.join(( '{}\t{}'.format(*el) for el in nObsnPts )),
                         tag='Histogram of #image points per object point' )
            # TODO: create graphics from that using plt.bar(.)

            ### stdDevs, optionally non-NULL
            # Note: within the scope of relOri, the model scale is arbitrary, and so are the stdDevs.
            # Still, let's use robust measures of the distribution of stdDevs to normalize the stdDevs and highlight outliers.
            # Even though the scale is arbitrary, it is homogeneous and isotropic - thus, use the same for all coordinates.
            # Also, the rotation of the model CS is arbitrary, but there is no unknown scale of the angles :-D
            
            # 'median' is available in one of the SQLite extensions, but not in SQLite that comes with Python 3.3
            # The following computes the median within SQLite. Too verbose!
            # To come by that verbosity, we may register a function defined in Python to be called by SQLite. But that is probably slow.
            #medX0 = conn.execute("""
            #    SELECT AVG(s_X0)
            #    FROM (
            #        SELECT s_X0
            #        FROM images
            #        ORDER BY s_X0
            #        LIMIT 2 - (SELECT COUNT(s_X0) FROM images) % 2    -- odd 1, even 2
            #        OFFSET (
            #            SELECT (COUNT(s_X0) - 1) / 2
            #            FROM images
            #        )
            #    )
            #""").fetchone()



            stdDevs = conn.execute("""
                SELECT s_X0, s_Y0, s_Z0
                FROM images
                WHERE s_X0!=0 AND s_Y0!=0 AND s_Z0!=0""").fetchall()
            if len(stdDevs): # otherwise, stdDevs have not been computed and stored.
                stdDevs = np.array( stdDevs )
                median = np.median( stdDevs )
                stdDevs = np.abs( stdDevs - median )
                sigmaMAD = 1.4826 * np.median( stdDevs )
                if sigmaMAD > 0:
                    stdDevs /= sigmaMAD
                    # Use numpy to generate a histogram,
                    # with ranges defined by the normalized standard deviations,
                    # e.g. ranging from 0 to 7
                    # separately for each coordinate? That makes sense only if the coordinate axes directions are meaningful
                    # E.g. the Z-coordinate is meaningful for an aerial data set, having fitted a robust plane through the sparse point cloud,
                    # and having transformed the whole block accordingly.
                    plt_utils.logHisto( logger,
                                        stdDevs,
                                        'Histogram of normalized PRC coordinate standard deviations',
                                        'std.dev.',
                                        ('X0','Y0','Z0'),
                                        [0] + [el/3 for el in range(1,10)],
                                        ('$X_0$','$Y_0$','$Z_0$') )

            # Do similar for rotation angles.
            # Rotation angles have a meaningful scale of course.
            # Still, robustly estimate their distribution in order to have meaningful bin sizes.
            stdDevs = conn.execute("""
                SELECT s_r1, s_r2, s_r3
                FROM images
                WHERE s_r1!=0 AND s_r2!=0 AND s_r3!=0""").fetchall()
            if len(stdDevs):
                stdDevs = np.array( stdDevs )
                median = np.median( stdDevs )
                sigmaMAD = 1.4826 * np.median( np.abs( stdDevs - median ) )
                if sigmaMAD > 0:
                    # separately for each coordinate? That makes sense only if the coordinate axes directions are meaningful
                    # E.g. the kappa-coordinate is meaningful for an aerial data set, having fitted a robust plane through the sparse point cloud,
                    # and having transformed the whole block accordingly.
                    bins = np.linspace(start=median-3*sigmaMAD, stop=median+3*sigmaMAD, num=9)
                    bins = bins[bins>0]
                    plt_utils.logHisto( logger,
                                        stdDevs,
                                        'Histogram of ROT angle standard deviations',
                                        'std.dev.',
                                        ('om','phi','ka'),
                                        np.concatenate(( [0], bins )),
                                        (r'$\omega$',r'$\phi$',r'$\kappa$') )
            # object points.
            # As with PRCs, their scale is meaningless
            stdDevs = conn.execute("""
                SELECT s_X, s_Y, s_Z
                FROM objpts
                WHERE s_X!=0 AND s_Y!=0 AND s_Z!=0""").fetchall()
            if len(stdDevs): # otherwise, stdDevs have not been computed and stored.
                stdDevs = np.array( stdDevs )
                median = np.median( stdDevs )
                stdDevs = np.abs( stdDevs - median )
                sigmaMAD = 1.4826 * np.median( stdDevs )
                if sigmaMAD > 0:
                    stdDevs /= sigmaMAD
                    # TODO: scale separately for each coordinate! That makes sense only if the coordinate axes directions are meaningful
                    # E.g. the Z-coordinate is meaningful for an aerial data set, having fitted a robust plane through the sparse point cloud,
                    # and having transformed the whole block accordingly.
                    plt_utils.logHisto( logger,
                                        stdDevs,
                                        'Histogram of normalized object point coordinate standard deviations',
                                        'std.dev.',
                                        ('X0','Y0','Z0'),
                                        [0] + [el/3 for el in range(1,10)],
                                        ('$X_0$','$Y_0$','$Z_0$') )

    @contract
    def makeIorAdpVariable( self,
                            block : adjust.Problem,
                            principalPoint : bool = False,
                            maxAbsCorr : float = 0.7,
                            affectedImgs : 'set(int)|None' = None
                          ) -> str:
        """check if focal length and/or radial distortion should be estimated.
        For uncalibrated photos only.
        For scenes with high depth variation, both focal length and radial distortion may be estimable.
        For aerial photos, it is quite probable that radial distortion is estimable. Concerning focal length, this is questionable.
        """
        # We use an ordering among potential free parameters: in each list, a parameter may only be set variable if all its predecessors have already been set variable.
        # Thus, we test for r5 only if r3 has already been set variable.
        # This approach is not very flexible, however:
        # - parameters may only make sense if used in a certain combination, e.g. adjust either both of x0,y0, or none.
        # - parameters that compensate for different defects may be set variable independently of each other: e.g. radial distortion parameters vs. affine distortion parameters.
        
        # Estimate focal length only
        iorCoeffs = ([2],)
        if principalPoint:
           iorCoeffs = iorCoeffs + ([0,1],) # either adjust both or none of the PP coordinates
        # Estimate r3 first, then r5.
        adpCoeffs = [adjust.PhotoDistortion.optPolynomRadial3],
        if principalPoint: # speedup: r5 usually cannot be estimated anyway. If it can be estimated, then it usually does not reduce image residuals much enough to affect the set of reconstructed images / object points. So estimate r5 only after reconstruction.
            adpCoeffs = adpCoeffs + ([adjust.PhotoDistortion.optPolynomRadial5],)
        constIorsAdps = {}
        for iImg in affectedImgs or self.orientedImgs:
            #speedup: do not iterate over all oriented images, but only over those that observe new structure since the last call of this function!
            img = self.imgs[iImg]
            if img.isCalibrated:
                continue
            # do not estimate IOR/ADP for images with less active observations. They may have IORs/ADPs with low correlations, but they are probably unreliable.
            # For a thorough check, we may compute the ratio of their estimated value to their std.dev. However, we would need to do a full adjustment for that. Too costly!
            #nImgPtsMin = 15
            # TODO: nReconstObjPts is an invalid measure here: many of those pts may just have been introduced. A better measure would be nReconstObjPts, counting only pts that have been observed at least 3 times.
            #if img.nReconstObjPts < nImgPtsMin:
            #    continue
            entry = constIorsAdps.get( img.ior.ctypes.data )
            if entry:
                entry[-1].append(iImg)
            else:
                iorBlockIsConstant, iorLocPar = block.IsParameterBlockConstant( img.ior ), block.GetParameterization( img.ior )
                adpBlockIsConstant, adpLocPar = block.IsParameterBlockConstant( img.adp ), block.GetParameterization( img.adp )
                if iorBlockIsConstant or iorLocPar and any(( iorLocPar.isConstant(coeff) for coeffGroup in iorCoeffs for coeff in coeffGroup )) or \
                   adpBlockIsConstant or adpLocPar and any(( adpLocPar.isConstant(coeff) for coeffGroup in adpCoeffs for coeff in coeffGroup )):
                    # np.ndarray is not hashable!
                    constIorsAdps[img.ior.ctypes.data] = ( img.ior, iorBlockIsConstant, iorLocPar,
                                                           img.adp, adpBlockIsConstant, adpLocPar, [iImg] )

        for elems in constIorsAdps.values():
            iImgs = elems[-1]
            eors = [ eor for iImg in iImgs for eor in ( self.imgs[iImg].omfika, self.imgs[iImg].t ) if not block.IsParameterBlockConstant(eor) ]
            for checkIor in (True,False):
                par , blockIsConstant , locPar  = elems[ 3*(not checkIor) : 3*(not checkIor)+3 ]
                aPar, aBlockIsConstant, aLocPar = elems[ 3*checkIor : 3*checkIor+3 ]
                coeffs  = iorCoeffs if checkIor else adpCoeffs
                aCoeffs = adpCoeffs if checkIor else iorCoeffs
                iPars = coeffs[0]
                if blockIsConstant:
                    block.SetParameterBlockVariable( par )
                if not locPar:
                    # set all parameters constant except for the first in coeffs
                    locPar = adjust.local_parameterization.Subset( par.size, [ el for el in range(par.size) if el not in coeffs[0] ] )
                    block.SetParameterization( par, locPar )
                elif not blockIsConstant:
                    # set the next parameter(s) variable
                    for coeffGroup in coeffs:
                        if any(( locPar.isConstant( coeff ) for coeff in coeffGroup )):
                            iPars = coeffGroup
                            for iPar in iPars:
                                locPar.setVariable( iPar )
                            block.ParameterizationLocalSizeChanged( par )
                            break
                    else:
                        continue

                # Default value for Covariance.Options.algorithm_type depends on ORIENTAL_GPL: SPARSE_CHOLESKY or SPARSE_QR
                # DENSE_SVD uses Eigen's JacobiSVD, which is incredibly slow for matrices with the max. of (nrows,ncols) being larger than a few hundred! https://forum.kde.org/viewtopic.php?f=74&t=102088
                cov = adjust.Covariance()
                els = eors
                if not aBlockIsConstant:
                    els = eors + [aPar]
                covBlockPairs = [ ( el,el) for el in [par]+els ] + \
                                [ (par,el) for el in els ]
                success = False
                try:
                    #log.setScreenMinSeverity( log.Severity.debug )
                    #adjust.setCeresMinLogLevel( log.Severity.debug )
                    cov.Compute( covBlockPairs, block )
                except Exception as ex:
                    pass
                else:
                    # we check the maximum correlation value only.
                    # Thus, let's avoid the computation of square roots!
                    maxAbsCorrSqr_ = -1.
                    maxAbsCorrSqr = maxAbsCorr**2.
                    cofacPar = cov.GetCovarianceBlock( par, par )
                    # Compute the correlations among iPar and the other non-constant parameters of par
                    # i.e. do not introduce r5 if it is highly correlated with r3

                    assert all(( not locPar.isConstant(iPar) for iPar in iPars ))
                    for oPar in ( oPar for oPars in coeffs for oPar in oPars if maxAbsCorrSqr_ < maxAbsCorrSqr and not locPar.isConstant( oPar ) ):
                        for iPar in ( iPar for iPar in iPars if maxAbsCorrSqr_ < maxAbsCorrSqr and iPar != oPar ):
                            maxAbsCorrSqr_ = cofacPar[iPar,oPar]**2 / ( cofacPar[iPar,iPar] * cofacPar[oPar,oPar] )
                    # check correlations between ior and adp or vice versa
                    if not aBlockIsConstant and maxAbsCorrSqr_ < maxAbsCorrSqr:
                        cofacAPar  = cov.GetCovarianceBlock( aPar, aPar )
                        cofacMixed = cov.GetCovarianceBlock( par, aPar )
                        for aCoeff in ( aCoeff for aCoeffGroup in aCoeffs for aCoeff in aCoeffGroup if maxAbsCorrSqr_ < maxAbsCorrSqr and not aLocPar or not aLocPar.isConstant( aCoeff ) ):
                            for iPar in ( iPar for iPar in iPars if maxAbsCorrSqr_ < maxAbsCorrSqr ):
                                maxAbsCorrSqr_ = cofacMixed[iPar,aCoeff]**2 / ( cofacPar[iPar,iPar] * cofacAPar[aCoeff,aCoeff] )
                    for eor in ( eor for eor in eors if maxAbsCorrSqr_ < maxAbsCorrSqr ):
                        cofacEor   = cov.GetCovarianceBlock( eor, eor )
                        cofacMixed = cov.GetCovarianceBlock( par, eor )
                        for iPar in ( iPar for iPar in iPars if maxAbsCorrSqr_ < maxAbsCorrSqr ):
                            maxAbsCorrSqr_ = ( cofacMixed[iPar,:]**2 / ( cofacPar[iPar,iPar] * cofacEor.diagonal() ) ).max()
                    success = maxAbsCorrSqr_ < maxAbsCorrSqr
                finally:
                    if success:
                        # do a full adjustment. After that, call this function again to check if further parameters shall be set variable.
                        return 'Free parameter {} for IOR/ADP ID {}'.format( str(iPars[0]) if not checkIor else 'focal length' if iPars == [2] else 'principal point',
                                                                             self.imgs[iImgs[0]].iorID )
                    elif iPars is coeffs[0]:
                        # Setting all parameters of a block to constant is illegal. Instead, set the whole block constant!
                        block.SetParameterBlockConstant( par )
                    else:
                        for iPar in iPars:
                            locPar.setConstant( iPar )
                        block.ParameterizationLocalSizeChanged( par )
        return ''

    @contract
    def makeIorsAdpsVariableAtOnce( self,
                                    block : adjust.Problem,
                                    blockSolveOptions : adjust.Solver.Options,
                                    params : 'seq' = ( adjust.PhotoDistortion.optPolynomRadial3, 2, 0, adjust.PhotoDistortion.optPolynomRadial5 ),
                                    iorIds : 'seq(int)|None' = None,
                                    maxAbsCorr : float = 0.7  ) -> None:
        atOnce = True
        if iorIds is None:
            iorIds = tuple(OrderedDict.fromkeys(self.imgs[iImg].iorID for iImg in self.orientedImgs if not self.imgs[iImg].isCalibrated)) # remove duplicates while preserving order
        if not params or not iorIds:
            return # all calibrated images
        iIorId = 0
        iLastSuccessParam=0
        iLastSuccessIorId=0
        iParam = 0
        while 1:
            if atOnce:
                # Introduce r3, z0, (x0,y0), r5 one after another, but for all cameras at the same time.
                # This may not yield the maximum set of variable ior/adp parameters. Thus, call makeIorAdpVariable later on for single iorIDs.
                msgs = self.makeIorAdpVariableAtOnce( block, params[iParam], maxAbsCorr=maxAbsCorr )
                if msgs:
                    iLastSuccessParam = iParam
                else:
                    iParam += 1
                    if iParam == len(params):
                        iParam=0
                    if iLastSuccessParam == iParam:
                        if len(iorIds) == 1: # Since there is only one camera, !atOnce doesn't make a difference
                            break
                        logger.verbose('Switching makeIorAdpVariableAtOnce to single-ior/adp mode')
                        atOnce = False
                        iParam = 0
                        iLastSuccessParam = 0
                    continue
            else:
                msgs = self.makeIorAdpVariableAtOnce( block, params[iParam], iorId=iorIds[iIorId], maxAbsCorr=maxAbsCorr )
                if msgs:
                    iLastSuccessParam=iParam
                    iLastSuccessIorId=iIorId
                iIorId += 1
                if iIorId == len(iorIds):
                    iIorId=0
                    iParam += 1
                    if iParam == len(params):
                        iParam=0
                if not msgs:
                    if iLastSuccessParam==iParam and iLastSuccessIorId==iIorId:
                        break
                    continue
            logger.info( '\n'.join(msgs) )
            logger.verbose("Full adjustment ...")
            summary = adjust.Solver.Summary()
            adjust.Solve(blockSolveOptions, block, summary)
            logger.info("Full adjustment done.")
        
            if not adjust.isSuccess( summary.termination_type ):
                # this state could be handled by removing the added observations from the block again.
                logger.info( summary.FullReport() )
                self.logActiveFeaturesPerImage()
                raise Exception("adjustment failed after additional parameters have been introduced into the block")

            self.logCamParams( block, withIorAdp=True, severity=log.Severity.verbose )

    @contract
    def makeIorAdpVariableAtOnce( self,
                                  block : adjust.Problem,
                                  param : '(int,>=0,<=2) | PhotoDistortion', # ior param index or adjust.PhotoDistortion; for ior, index 0 or 1 are both treated as 'principal point'
                                  iorId : int = -1,
                                  maxAbsCorr : float = 0.7
                                ) -> 'list(str)':
        # Try to set variable at once as many ior/adp parameters as possible and reasonable.
        # Need to pass all and only non-constant parameter blocks to adjust.sparseQxx

        msgs = []

        def isIor(par):
            return type(par) == int

        paramName = str(param) if not isIor(param) else 'focal length' if param==2 else 'principal point'
        params = [0,1] if isIor(param) and param in (0,1) else [param]

        Intrinsic = namedtuple( 'Intrinsic', [ 'parBlock', 'wasConst', 'subset' ] )
        IntrinsicPair = namedtuple( 'IntrinsicPair', [ 'parBlock', 'wasConst', 'subset', 'anyParsSetFree', 'oParBlock', 'oIsConst', 'oSubset', 'iImgs' ] )

        def getIntrinsicPairs():
            intrinsicPairs = OrderedDict()
            for iImg in self.orientedImgs:
                img = self.imgs[iImg]
                if img.isCalibrated:
                    continue # avoid zero-columns in jacobian!
                intrinsicPair = intrinsicPairs.get(img.iorID)
                if intrinsicPair:
                    intrinsicPair[-1].append( iImg )
                    continue
                intrinsic = Intrinsic( img.ior, block.IsParameterBlockConstant( img.ior ), block.GetParameterization( img.ior ) )
                oIntrinsic = Intrinsic( img.adp, block.IsParameterBlockConstant( img.adp ), block.GetParameterization( img.adp ) )
                if not isIor(param):
                    intrinsic, oIntrinsic = oIntrinsic, intrinsic

                anyParsSetFree = False
                if iorId in ( -1, img.iorID ):
                    if intrinsic.wasConst:
                        block.SetParameterBlockVariable(intrinsic.parBlock)
                        anyParsSetFree = True

                    if not intrinsic.subset:
                        if intrinsic.wasConst: # for parameters there have already been free and did not have a subset parameterization, don't introduce a new subset.
                            # set all parameters constant except for the first in coeffs
                            locPar = adjust.local_parameterization.Subset( intrinsic.parBlock.size, [ el for el in range(intrinsic.parBlock.size) if el not in params ] )
                            intrinsic = intrinsic._replace(subset=locPar)
                            block.SetParameterization( intrinsic.parBlock, locPar )
                            anyParsSetFree = True
                    else:
                        if intrinsic.wasConst:
                            # We must wipe all other free parameters from subset: e.g. adjustment of ior's focal length has been tried before, but failed, and was set constant, again. Now, we may want to estimate the principal point
                            wantedConstancyMask = np.ones_like( intrinsic.subset.constancyMask )
                            wantedConstancyMask[params] = 0
                        else:
                            wantedConstancyMask = intrinsic.subset.constancyMask.copy()
                            wantedConstancyMask[params] = 0
                        iDiffs = np.flatnonzero( intrinsic.subset.constancyMask != wantedConstancyMask )
                        if iDiffs.size:
                            for iDiff in iDiffs:
                                if wantedConstancyMask[iDiff]:
                                    intrinsic.subset.setConstant( int(iDiff) )
                                else:
                                    intrinsic.subset.setVariable( int(iDiff) )
                            block.ParameterizationLocalSizeChanged( intrinsic.parBlock )
                            anyParsSetFree = True

                intrinsicPairs[img.iorID] = IntrinsicPair( *chain( intrinsic, [anyParsSetFree], oIntrinsic, [[iImg]] ) )

            return intrinsicPairs

        def logOrRevert( intrinsicPairs, maxAbsCorrsSqr ):
            # for all parameters/parameter blocks with correlations above maxAbsCorr, revert the changes from above. For the others, produce log messages.
            # Once a parameter has been set variable, it shall never be set constant, again.
            maxAbsCorrSqr = maxAbsCorr**2
            iPar = 0
            for currIorID,intrinsicPair in intrinsicPairs.items():
                # do not re-set parameters to constant that were set free in preceding function calls
                if iorId in ( -1, currIorID ):
                    nVariable = intrinsicPair.subset.constancyMask.size - intrinsicPair.subset.constancyMask.sum()
                    if intrinsicPair.anyParsSetFree:
                        # check only the parameter(s) under question, but not all of non-const subset
                        offsets = np.cumsum( np.logical_not(intrinsicPair.subset.constancyMask) ) - 1
                        idxs = offsets[ params ]
                        if maxAbsCorrsSqr is not None and maxAbsCorrsSqr[iPar + idxs].max() <= maxAbsCorrSqr:
                            msgs.append( 'Free parameter {} for IOR/ADP ID {}'.format( paramName, self.imgs[intrinsicPair.iImgs[0]].iorID ) )
                        elif nVariable - len(params) == 0:
                            # Setting all parameters of a block to constant is illegal. Instead, set the whole block constant!
                            block.SetParameterBlockConstant( intrinsicPair.parBlock )
                        else:
                            for coeff in params:
                                intrinsicPair.subset.setConstant( coeff )
                            block.ParameterizationLocalSizeChanged( intrinsicPair.parBlock )
                else:
                    if intrinsicPair.wasConst: # in this case, the par block has not been freed!
                        nVariable = 0
                    elif intrinsicPair.subset:
                        nVariable = intrinsicPair.subset.constancyMask.size - intrinsicPair.subset.constancyMask.sum()
                    else:
                        nVariable = intrinsicPair.parBlock.size

                iPar += nVariable

                if not intrinsicPair.oIsConst:
                    if intrinsicPair.oSubset:
                        iPar += intrinsicPair.oSubset.constancyMask.size - intrinsicPair.oSubset.constancyMask.sum()
                    else:
                        iPar += intrinsicPair.oParBlock.size

        intrinsicPairs = getIntrinsicPairs()
        if not any( el.anyParsSetFree for el in intrinsicPairs.values() ):
            return msgs # all cameras had the current parameters already set free, nothing to do.

        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.apply_loss_function = True
        paramBlocks = list( chain( ( par for el in intrinsicPairs.values() for par in (el.parBlock,el.oParBlock) if not block.IsParameterBlockConstant(par) ),
                                   ( self.imgs[iImg].t      for iImg in self.orientedImgs[1:] ), #leave out the first oriented image, as its PRC and ROT are const
                                   ( self.imgs[iImg].omfika for iImg in self.orientedImgs[1:] ),
                                   self.featureTrack2ObjPt.values() ) )
        evalOpts.set_parameter_blocks( paramBlocks )
        # jacobian contains columns only for paramBlocks
        # jacobian contains no columns for parameters that are set constant by way of a Subset-parameterization
        jacobian, = block.Evaluate( evalOpts, residuals=False, jacobian=True ) # we might ask for the cost, and compute sigmas!
        maxAbsCorrsSqr = None
        if 0:
            import pickle
            with open( os.path.join(self.outDir,'jacobian.pickle'), 'wb' ) as fout:
                pickle.dump( jacobian, fout, protocol=pickle.HIGHEST_PROTOCOL )
        try:
            # A.T.dot(A) may not be invertible!
            # returns an upper triangular crs matrix
            # TODO: simplicial factorization (which is selected automatically for small problems) crashes with a segmentation fault or throws 'out of memory'
            if jacobian.shape[1] != np.unique( jacobian.nonzero()[1] ).size:
               logger.warning("zero columns in jacobian")
               self.logActiveFeaturesPerImage()
               raise Exception()
            QxxAll = adjust.sparseQxx( jacobian, adjust.Factorization.supernodal )
        except Exception as ex:
            pass
        else:
            nObjPts = len(self.featureTrack2ObjPt)
            # column slicing is very inefficient on csr matrices. Thus, convert to csc
            Qxx = QxxAll[:-nObjPts*3,:].tocsc()[:,:-nObjPts*3]
            # TODO: Not even the rows and columns concerning ior/adp seem to be dense, but only the whole diagonal.
            # Is that only the case for supernodal factorization?
            #if Qxx.nnz != Qxx.shape[0]*(Qxx.shape[0]+1)/2:
            #    import pickle
            #    with open('qxx.pickle','wb') as fout:
            #        pickle.dump( Qxx, fout, protocol=pickle.HIGHEST_PROTOCOL )
            #    raise Exception( 'sub-matrix for cameras is not dense! Qxx dumped to file: {}'.format('qxx.pickle') )
            Qxx = Qxx.toarray()
            diag = Qxx.diagonal().copy()
            Rxx = Qxx + Qxx.T
            np.fill_diagonal(Rxx,0)
            # we check the maximum correlation value only.
            # Thus, let's avoid the computation of square roots!
            #sqrtDiag = diag ** .5
            #Rxx = ( ( Rxx / sqrtDiag ).T / sqrtDiag ).T
            #maxAbsCorrs = np.abs(Rxx).max( axis=1 )
            RxxSqr = ( ( Rxx ** 2 / diag ).T / diag ).T
            # It's enough if at least 1 photo's EOR/ROT has low correlations with a certain IOR/ADP. Hence, don't take the maximum correlation coefficient over all photos!
            #maxAbsCorrsSqr = RxxSqr.max( axis=1 )
            nEorPars = (len(self.orientedImgs)-1) * 6 - 1
            nIorPars = RxxSqr.shape[1] - nEorPars
            maxAbsCorrsSqr = np.empty( nIorPars )
            for iIorPar in range(nIorPars):
                minOfMaxAbsCorrSqrPerPho = RxxSqr[ iIorPar, nIorPars : nIorPars + 5 ].max()
                for iCam in range(len(self.orientedImgs)-2):
                    offset = nIorPars + 5 + iCam*6
                    minOfMaxAbsCorrSqrPerPho = min( minOfMaxAbsCorrSqrPerPho, RxxSqr[ iIorPar, offset : offset + 6 ].max() )
                maxAbsCorrsSqr[iIorPar] = max( minOfMaxAbsCorrSqrPerPho, RxxSqr[iIorPar,:nIorPars].max() )
        finally:
            logOrRevert( intrinsicPairs, maxAbsCorrsSqr )
        return msgs

    @contract
    def maxAbsCorrsIorAdp( self, block : adjust.Problem ) -> dict:
        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.apply_loss_function = False

        def getIorAdps():
            visitedIorIds = set()
            for iImg in self.orientedImgs:
                img = self.imgs[iImg]
                if img.iorID in visitedIorIds:
                    continue
                visitedIorIds.add(img.iorID)
                for par in (img.ior,img.adp):
                    if not block.IsParameterBlockConstant(par):
                        yield par

        paramBlocks = list( chain( getIorAdps(),
                                   ( self.imgs[iImg].t      for iImg in self.orientedImgs[1:] ), #leave out the first oriented image, as its PRC and ROT are const
                                   ( self.imgs[iImg].omfika for iImg in self.orientedImgs[1:] ),
                                   self.featureTrack2ObjPt.values() ) )
        evalOpts.set_parameter_blocks( paramBlocks )
        jacobian, = block.Evaluate( evalOpts, residuals=False, jacobian=True ) # we might ask for the cost, and compute sigmas!
        maxAbsCorrs = {}
        assert jacobian.shape[1] == np.unique( jacobian.nonzero()[1] ).size, "zero columns in jacobian"
        # A.T.dot(A) may not be invertible!
        # returns an upper triangular crs matrix
        # TODO: simplicial factorization (which is selected automatically for small problems) crashes with a segmentation fault
        #import pickle
        #with open( os.path.join(self.outDir,'jacobian.pickle'), 'wb' ) as fout:
        #    pickle.dump( jacobian, fout, protocol=pickle.HIGHEST_PROTOCOL )
        #QxxAll = adjust.sparseQxx( jacobian, adjust.Factorization.automatic )
        QxxAll = adjust.sparseQxx( jacobian, adjust.Factorization.supernodal )
        nObjPts = len(self.featureTrack2ObjPt)
        # column slicing is very inefficient on csr matrices. Thus, convert to csc
        Qxx = QxxAll[:-nObjPts*3,:].tocsc()[:,:-nObjPts*3]
        # TODO: Not even the rows and columns concerning ior/adp seem to be dense, but only the whole diagonal.
        # Is that only the case for supernodal factorization?
        #if Qxx.nnz != Qxx.shape[0]*(Qxx.shape[0]+1)/2:
        #    import pickle
        #    with open('qxx.pickle','wb') as fout:
        #        pickle.dump( Qxx, fout, protocol=pickle.HIGHEST_PROTOCOL )
        #    raise Exception( 'sub-matrix for cameras is not dense! Qxx dumped to file: {}'.format('qxx.pickle') )
        Qxx = Qxx.toarray()
        diag = Qxx.diagonal().copy()
        Rxx = Qxx + Qxx.T
        np.fill_diagonal(Rxx,0) # fill the diagonal, so we can easily find the maximum correlation coefficients off the diagonal
        # we check the maximum correlation value only.
        # Thus, let's avoid the computation of all square roots, but compute only the needed ones.
        #sqrtDiag = diag ** .5
        #Rxx = ( Rxx / sqrtDiag ).T / sqrtDiag # don't need to transpose in the end, as matrix is symmetric
        #maxAbsCorrs = np.abs(Rxx).max( axis=1 )
        RxxSqr = ( Rxx ** 2 / diag ).T / diag # don't need to transpose in the end, as matrix is symmetric
        maxAbsCorrsSqr = RxxSqr.max( axis=1 )

        iPar = 0
        for iImg in self.orientedImgs:
            img = self.imgs[iImg]
            if img.iorID in maxAbsCorrs:
                continue
            if img.isCalibrated:
                maxAbsCorrs[img.iorID] = np.full( img.ior.size, np.nan ), np.full( img.adp.size, np.nan )
                continue
            theMaxAbsCorrs = []
            for par in (img.ior,img.adp):
                if block.IsParameterBlockConstant(par):
                    theMaxAbsCorrs.append( np.full( par.size, np.nan ) )
                    continue
                subset = block.GetParameterization( par )
                if subset:
                    nVariable = subset.constancyMask.size - subset.constancyMask.sum()
                else:
                    nVariable = par.size
                rxx = maxAbsCorrsSqr[iPar : iPar + nVariable]**.5
                if subset:
                    theArr = np.full( par.size, np.nan )
                    theArr[ np.logical_not( subset.constancyMask ) ] = rxx
                else:
                    theArr = rxx
                theMaxAbsCorrs.append( theArr )
                iPar += nVariable

            maxAbsCorrs[img.iorID] = theMaxAbsCorrs

        return maxAbsCorrs

    @contract
    def computePrecision( self,
                          block : adjust.Problem,
                          sigma0 : float ) -> dict:
        covOpts = adjust.Covariance.Options()
        covOpts.apply_loss_function = False
        covariance = adjust.Covariance( covOpts )
        paramBlockPairs = [ (objPt,objPt) for objPt in self.featureTrack2ObjPt.values() ]
        storedIorIDs = set()
        for img in ( self.imgs[iImg] for iImg in self.orientedImgs ):
            paramBlocks = [ img.t, img.omfika ]
            if img.iorID not in storedIorIDs:
                if img.adpID != img.iorID:
                    raise oriental.Exception("Not implemented: ior and adp IDs different")
                storedIorIDs.add( img.iorID )
                paramBlocks += [ img.ior, img.adp ]
            
            for paramBlock in paramBlocks:
                paramBlockPairs.append( (paramBlock,paramBlock) )

        covariance.Compute( paramBlockPairs, block )
        stdDevs = {}
        for paramBlockPair in paramBlockPairs:
            cofactorBlock = covariance.GetCovarianceBlock( *paramBlockPair )
            stdDevs[ id(paramBlockPair[0]) ] = sigma0 * np.diag(cofactorBlock)**0.5
        
        return stdDevs
    
    @contract       
    def getConstancyAsPrecision( self,
                                 block : adjust.Problem ) -> dict:
        """return a dict that contains for parameter blocks with constant parameters their ids as keys.
           if the whole parameter block is constant then the value is np.zero(par.size)
           if only a subset is constant, then the value is a np.ndarray with 0.0 for constants and NaN for non-constant parameters."""
        stdDevs = dict()
        # there are no constant objPts.

        img = self.imgs[self.orientedImgs[0]]
        stdDevs[id(img.t)] = np.zeros(3)
        stdDevs[id(img.omfika)] = np.zeros(3)

        # How to mark the local_parameterization.UnitSphere for the second image?
        img = self.imgs[self.orientedImgs[1]]
        stdDevs[id(img.t)] = None, None, 0.

        storedIorIDs = set()
        for iImg in self.orientedImgs:
            img = self.imgs[iImg]
            if img.iorID not in storedIorIDs:
                storedIorIDs.add( img.iorID )
                for parBlock in ( img.ior, img.adp ):
                    if block.IsParameterBlockConstant( parBlock ):
                        stdDevs[id(parBlock)] = np.zeros(parBlock.size)
                    else:
                        locParam = block.GetParameterization( parBlock )
                        if locParam is not None:
                            # ndarray of dtype=np.float holds NaN for non-constant parameters. SQLite will consider those as NULL
                            # dtype=np.float must be passed, otherwise: ndarray.dtype==object
                            stdDevs[id(parBlock)] = np.array([ (0.0 if locParam.isConstant(idx) else None) for idx in range(parBlock.size) ], dtype=np.float )

        return stdDevs
    
    new_contract('SpatialReference', osr.SpatialReference )

    @contract
    def saveSQLite( self,
                    resultsFn : str,
                    targetCS : 'SpatialReference | None',
                    stdDevs : 'dict' = {} ) -> None:
        # saving WKT geometries has the advantage that e.g. QGIS can directly visualize them.
        # otherwise, save coordinates in separate columns
        saveAsGeometry = True

        db.createUpdateSchema( resultsFn, db.Access.readWriteCreate )

        conn = dbapi2.connect( resultsFn )
        # removing the sqlite-file beforehand is the only way to clear the database.
        # However, if the sqlite-file is opened in e.g. spatialite_gui.exe, then we cannot remove it.
        # Still, we have write access to the db content. Thus, drop our tables!
        #for name in ( "imgobs", "cameras", "images", "objpts" ):
        #    conn.execute("DROP TABLE IF EXISTS {}".format(name) )

        db.initDataBase( conn )
        conn.execute('PRAGMA journal_mode = OFF')  # We create a new DB from scratch. Hence, no need for rollbacks. Journalling costs time!

        #conn.executescript( db.createTables() )
        # the SQL-String used to create a table can be queried with:
        # SELECT name, sql FROM sqlite_master WHERE type='table' AND name='cameras' ORDER BY name;
        # which will output that String, including any comments in it!
        # This seems to be the standard way of commenting tables and columns.

        conn.execute( """
            SELECT AddGeometryColumn(
                'objpts', -- table
                'pt',     -- column
                -1,       -- srid -1: undefined/local cartesian cooSys
                'POINT',  -- geom_type
                'XYZ',    -- dimension
                1         -- NOT NULL
            )""" )
            
        # Connection objects can be used as context managers that automatically commit or rollback transactions. In the event of an exception, the transaction is rolled back; otherwise, the transaction is committed:
        # However, non-DML-statements must be called outside the context manager (-> create tables beforehand, above)
        # calling BEGIN TRANSACTION and COMMIT TRANSACTION seems to be illegal unless the connection has been opened with autocommit=0
        with conn:  
            
            # Let's store the Wkt string, and not the PROJ.4-string, because the Wkt-string preserves EPSG-codes (Parameter 'authority' of root node).
            if targetCS:
                conn.execute( f"""INSERT INTO config ( name, value )
                                  VALUES ( '{db.ConfigNames.CoordinateSystemWkt}', ? ) """,
                              ( targetCS.ExportToWkt(), ) )
                                                               
            # Must quote 
            # POINTZ(x y z)
            # , because otherwise, the SQL parser complains about the unknown function 'POINTZ'
            # It seems impossible to feed arguments inside a quoted string using additional arguments of executemany, because executemany "just quotes everything"
            # The most efficient solution seems to be to use a generator-expression, as below: 
            # The second argument to executemany must be a tuple/list of tuples/lists, not of scalars! i.e. the return value of the generator expression must be a tuple
            # Note if not explicitly given, then SQLite will set the lowest primary key to 1, not zero! Thus, set it explicitly.
            
            # Pass values as well-known binary, because well-known text (or only the SpatiaLite-WKT-Parser?) does not support scientific notation. Note that SQLite itself supports it: "SELECT 1.1E-3 > 3."
            # Passing values always in fixed-point notation would require formatting with {:.15f}, to account for both very small and very large coordinates, which is probably inefficient. 
            # If a number is passed in scientific notation, then GeomFromText returns NULL, without error.
            # Within the same call, SQLite then tries to insert NULL into the geometry column.
            # This results in the confusing error message: 'spatialite.dbapi2.IntegrityError: objpts.pt may not be NULL'
            # http://en.wikipedia.org/wiki/Well_Known_Text#Well-known_binary
            # pack with little-endian byte order ('<'). This must correspond to the value of the first byte in the packed data: 1 for little-endian
            # We may use e.g. osgeo/ogr for that conversion, but that is probably slower?
            pointZWkb = struct.Struct('<bIddd')
            def packPointZ(pt):
                # 1001 is the code for points with z-coordinate
                # Must wrap the str returned by struct.pack(.) into a memoryview object, or otherwise, SQLite complains about receiving 8-bit-strings instead of unicode. However, we want the string to be interpreted byte-wise.
                # Even though buffer is deprecated since Python 2.7, there is no other way to pass the data: http://bugs.python.org/issue7723
                # For Python 3, SQLite accepts bytes/memoryview and returns bytes for BLOBs!
                return pointZWkb.pack( 1, 1001, *pt )

            conn.executemany( """INSERT INTO objpts(id,pt,s_X,s_Y,s_Z) VALUES( ?, GeomFromWKB(?, -1), ?, ?, ? )""",
                              ( ( iPt, packPointZ(pt) ) + tuple( stdDevs.get( id(pt), (None,)*3 ) )
                                for iPt,pt in enumerate(self.featureTrack2ObjPt.values()) ) )
            
            photoDistortionSorted = sorted( adjust.PhotoDistortion.values.items(), key=lambda x: x[0] )
            insertCameras =   "INSERT INTO cameras(id, make, model, sensorWidth_mm, sensorHeight_mm, isDigital, x0, y0, z0, s_x0, s_y0, s_z0, reference, normalizationRadius," \
                            + ','.join(   [        str(val[1]) for val in photoDistortionSorted ]
                                        + [ 's_' + str(val[1]) for val in photoDistortionSorted ] ) \
                            + ')\n' \
                            + 'VALUES( ' + ','.join( '?' * (8 + 3*2 + 9*2) ) +  ' )'

            storedIorIDs = set()
            for iImg in self.orientedImgs:
                img = self.imgs[iImg]
                if img.iorID not in storedIorIDs:
                    if img.adpID != img.iorID:
                        raise Exception("Not implemented: ior and adp IDs different")
                    storedIorIDs.add( img.iorID )
                    conn.execute( insertCameras,
                                  tuple( chain(
                                     ( img.iorID, img.make, img.model, img.format_mm[0] if img.format_mm is not None else None, img.format_mm[1] if img.format_mm is not None else None, img.isDigital ),
                                     img.ior,
                                     stdDevs.get( id(img.ior), (None,)*3 ),
                                     ( str(adjust.AdpReferencePoint.principalPoint),
                                       img.adp.normalizationRadius ),
                                     img.adp,
                                     stdDevs.get( id(img.adp), (None,)*9 ) ) )
                                )
        

               
            def genImages():
                for iImg in self.orientedImgs:
                    img = self.imgs[iImg]
                    mask = None
                    if img.mask_px is not None and len(img.mask_px):
                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        for pt in img.mask_px:
                            ring.AddPoint( *pt )
                        ring.CloseRings()
                        ring.FlattenTo2D()
                        polyg = ogr.Geometry(ogr.wkbPolygon)
                        polyg.AddGeometryDirectly(ring)
                        assert polyg.IsValid()
                        mask = polyg.ExportToWkb()

                    yield tuple( chain(
                        ( iImg,
                          img.iorID,
                          utils.filePaths.relPathIfExists( img.fullPath, os.path.dirname(resultsFn) ),
                          img.width,
                          img.height ),
                          img.t,
                          stdDevs.get( id(img.t), (None,)*3 ),
                          img.omfika,
                          stdDevs.get( id(img.omfika), (None,)*3 ),
                         (None,)*6 if type(img.pix2cam) is IdentityTransform2D else chain( img.pix2cam.A.flat, img.pix2cam.t ),
                         (mask,)
                    ) )

            conn.executemany( """INSERT INTO images( id, camID, path, nCols, nRows, X0, Y0, Z0, s_X0, s_Y0, s_Z0, r1, r2, r3, s_r1, s_r2, s_r3, parameterization, fiducial_A00, fiducial_A01, fiducial_A10, fiducial_A11, fiducial_t0, fiducial_t1,               mask )
                                             VALUES(  ?,     ?,    ?,     ?,     ?,  ?,  ?,  ?,    ?,    ?,    ?,  ?,  ?,  ?,    ?,    ?,    ?,         'omfika',            ?,            ?,            ?,            ?,           ?,           ?, GeomFromWKB(?, -1) )""",
                              genImages() )


            #featureTracks = list( self.featureTrack2ObjPt.keys() ) # we use features.index(.), thus we need to create an actual list.
            # Better create a temporary dict for this purpose
            featureTrack2Idx = { featureTrack:idx for idx,featureTrack in enumerate(self.featureTrack2ObjPt) }

            # use a generator instead of a generator expression. As efficient, but easier to read
            # pack with little-endian byte order. This must correspond to the value of the first byte in the packed data: 1 for little-endian
            #pointWkb = struct.Struct('<bIdd')
            #def packPoint(pt):
            #    # 1 is the code for 2D points
            #    return pointWkb.pack( 1, 1, *pt )
            
            def getImgPts():
                for iImg in self.orientedImgs:
                    for iPt in range(self.imgs[iImg].keypoints.shape[0]):
                        if (iImg,iPt) in self.imgFeature2costAndResidualBlockID:
                            found,featureTrack = self.featureTracks.component( graph.ImageFeatureID( iImg, iPt ) )
                            #iObjPt = featureTracks.index( featureTrack )
                            iObjPt = featureTrack2Idx[featureTrack]
                            row = self.imgs[iImg].keypoints[iPt].astype(np.float)
                            #yield ( iImg, iPt, iObjPt, packPoint(row[:2]) ) + tuple(row[2:])
                            # Reserve the first 1000 point names for non-automatic points, to be suggested for use in MonoScope
                            yield ( iImg, iPt+1000, iObjPt ) + tuple(row)
        
            #conn.executemany( """INSERT INTO imgobs( imgID, serial, objPtID, pt                , R, G, B, angle, diameter )
            #                                 VALUES(     ?,      ?,       ?, GeomFromWKB(?, -1), ?, ?, ?,     ?, ?        )""",
            conn.executemany( """INSERT INTO imgobs( imgID, name, objPtID, x, y, red, green, blue, angle, diameter )
                                             VALUES(     ?,    ?,       ?, ?, ?,   ?,     ?,    ?,     ?, ?        )""",
                              getImgPts() )

        
        conn.execute("ANALYZE")
                                           
        logger.info("Results stored in SpatiaLite DB '{}'".format(resultsFn))   
            
    @contract
    def orientByEMatrix( self,
                         iImgOld : int,
                         iImgNew : int,
                         maxEpipolarDist : 'float,>0.',
                         inlierRatio     : 'float,>0.',
                         confidenceLevel : 'float,>0.,<1.' ) -> 'tuple( array[3x3](float), array[3](float), array[N](uint64), array[N](uint64) )':
        """returns (R,t,iKeyPts,representatives)
        R,t is the EOR of the new image in the project CS, according to the E-matrix of old,new images, and the scale estimated by the commonly observed, already reconstructed objPts
        representatives represent the commonly observed, already reconstructed object points i.e. only a subset of the already reconstr. objpts observed by the new image 
        iKeyPts are the idxs into the keypts of new image of those reps
        Since IOR,ADP may have changed since the initial estimation of E-matrices, let's compute the E-matrix from scratch and not re-use self.edge2PairWiseOri"""
        ret = np.eye(3), np.zeros((3,)), np.empty( (0,), dtype=np.uint64 ), np.empty( (0,), dtype=np.uint64 )
        
        if (iImgNew, iImgOld) in self.edge2matches:
            matches = np.fliplr( self.edge2matches[ (iImgNew, iImgOld) ] )
        else:
            matches =            self.edge2matches[ (iImgOld, iImgNew) ]
        
        assert iImgOld     in self.orientedImgs
        assert iImgNew not in self.orientedImgs

        pt1 = self.imgs[iImgOld].keypoints[matches[:,0],:2]   
        pt2 = self.imgs[iImgNew].keypoints[matches[:,1],:2]
        
        pt1 = ori.distortion( pt1*(1.,-1.),
                              self.imgs[iImgOld].ior,
                              ori.adpParam2Struct(self.imgs[iImgOld].adp),
                              ori.DistortionCorrection.undist )*(1.,-1.)

        pt2 = ori.distortion( pt2*(1.,-1.),
                              self.imgs[iImgNew].ior,
                              ori.adpParam2Struct(self.imgs[iImgNew].adp),
                              ori.DistortionCorrection.undist )*(1.,-1.)
                              
        cameraMatrix1 = ori.cameraMatrix( self.imgs[iImgOld].ior )
        cameraMatrix2 = ori.cameraMatrix( self.imgs[iImgNew].ior )
    
        cameraMatrixInv1 = linalg.inv( cameraMatrix1 )
        cameraMatrixInv2 = linalg.inv( cameraMatrix2 )
    
        pth1 = np.hstack((pt1,np.ones((pt1.shape[0],1)))).T
        pth2 = np.hstack((pt2,np.ones((pt2.shape[0],1)))).T
        pt_normalized1 = cameraMatrixInv1.dot( pth1 ) 
        pt_normalized2 = cameraMatrixInv2.dot( pth2 )
        pt_normalized1 = pt_normalized1[:2,:] / pt_normalized1[2,:]
        pt_normalized2 = pt_normalized2[:2,:] / pt_normalized2[2,:]                    
        
        # 10% -> 690773 maxNumIters @99.9% confidence
        # 20% ->  21584 maxNumIters @99.9% confidence
        # 25% ->   7071 maxNumIters @99.9% confidence
        # 37% ->    993 maxNumIters @99.9% confidence
        # OpenCV-default: maxNumIters=1000
        maxNumIters = ori.maxNumItersRANSAC( nModelPoints=5, inlierRatio=inlierRatio, confidence=confidenceLevel, nDataPoints=matches.shape[0] )
    
        # directly use image calibrations, essential matrix
        # cv2.findEssentialMat expects normalized image coordinates, and accepts only 1 reprojection error threshold.
        # If the focal lengths of the 2 images are different, then it's hard to find a meaningful threshold!
        threshold = .5*( maxEpipolarDist/cameraMatrix1[0,0] + maxEpipolarDist/cameraMatrix2[0,0] )
        essentialMatrix,mask = cv2.findEssentialMat( points1=pt_normalized1.T,
                                                     points2=pt_normalized2.T,
                                                     method=cv2.FM_RANSAC,
                                                     threshold=threshold, # default: 1.
                                                     prob=confidenceLevel, # default: 0.999
                                                     maxNumIters=maxNumIters
                                                   )
        # mask ... Output array of N elements, every element of which is set to 0 for outliers and to 1 for the other points.
        assert mask.min() >= 0
        assert mask.max() <= 1
        assert mask.shape == (pt1.shape[0],1)
        # -> convert to 1-dim boolean array
        mask = mask.squeeze() > 0
        
        if float(mask.sum()) / matches.shape[0] < inlierRatio:
            return ret
    
        # Achtung: cv2.recoverPose verwendet einen hartcodierten Schwellwert für die Objektpunktdistanz==50.
        # Punkte, die weiter weg liegen, werden als instabil betrachtet (liegen nahe 'unendlich' weit weg - daher ist keine Entscheidung möglich, ob sie vor oder hinter der Kamera liegen),
        # und nicht für die Entscheidung verwendet, ob die Kombination von R und t eine gute Wahl ist
        nGood,R2,t2,mask2 = cv2.recoverPose( essentialMatrix, pt_normalized1[:,mask].T, pt_normalized2[:,mask].T )
        mask2 = mask2.squeeze() > 0                        
        mask[mask] = mask2
               
        P2 = np.hstack((R2,t2))
        P1 = np.hstack( (np.eye((3)), np.zeros((3,1)) ) )
        # P2 is the EOR of iImgNew in the model CS
        # P1 is the EOR of iImgOld in the model CS.
        X = cv2.triangulatePoints( P1, P2, pt_normalized1[:,mask], pt_normalized2[:,mask] )
        X /= X[3,:]

        # transform object coordinates to ORIENT system: invert both y- and z-coordinates
        Xori = np.empty( (mask.shape[0],3) )
        Xori[mask,:] = X[:3,:].T
        Xori[:,1:] *= -1.                    
        
        # coords of the second PRC in the Orient-CS of the first camera
        R2,t2 = ori.projectionMat2oriRotTrans( P2 )
        
        # the model coordinate system is identical to camera CS # 1!
        # Xori is in the model CS
        # Beginning with the second pair, adjust model scale to project scale
        #   based on common ObjPts present in modelCS and in projCS
        #newlyOriented = [False,False]
        
        # iImgOld is already in the project CS
        
        camOld = self.imgs[iImgOld]
        
        # prepare for after the loop
        representatives = []
        iKeyPts = []
        
        # estimate scale: modelCS->projCS based on common points
        distSqrRatios = []
        for iMatch in mask.nonzero()[0]:
            featureTrack = self.featureTracks.component( graph.ImageFeatureID( iImgOld, matches[iMatch,0].item() ) )
            if not featureTrack[0]:
                continue
            featureTrack = featureTrack[1]
            if featureTrack in self.featureTrack2ObjPt:
                distObjSqr = np.sum( ( camOld.t - self.featureTrack2ObjPt[featureTrack] )**2 )
                distModelSqr = np.sum( Xori[iMatch,:]**2 )
                distSqrRatios.append( distObjSqr / distModelSqr )
                
                iKeyPts.append(matches[iMatch,1])
                representatives.append( featureTrack )
                

        if len(distSqrRatios) == 0:
            return ret

        iKeyPts = np.array( iKeyPts, dtype=np.uint64 )
        representatives = np.array( representatives, dtype=np.uint64 )

        distSqrRatio = np.median( np.array( distSqrRatios ) )
        scale = distSqrRatio**.5

        t2 *= scale

        # Xp ...... objPts in projCS
        # Xm ...... objPts in modelCS
        # X1 ...... objPts in camCS1
        # X2 ...... objPts in camCS2
        # R2m,t2m . trafo camCS1 -> camCS2 (known)
        # R1p,t1p . trafo projCS -> camCS1 (known)
        # R2p,t2p . trafo projCS -> camCS2 (wanted)
        #
        # X1=Xm 
        # X1=R1p'•(Xp-t1p)
        
        # transform new objPts from modelCS==camCS1 -> projCS
        # Xp = R1p•X1+t1p
        #    = R1p•(X1+R1p'•t1p)
        #    = (R1p')' • ( X1 - (-R1p'•t1p) )
        R = camOld.R.T
        t = - camOld.R.T.dot(camOld.t)
        
        # transform cam2 to project coordinate system
        # X2 = R2m'•(X1-t2m)
        #    = R2m'•(R1p'•(Xp-t1p)-t2m)
        #    = R2m'•(R1p'•(Xp-t1p-R1p•t2m))
        #    = (R1p•R2m)' • ( Xp - (t1p+R1p•t2m) )
        camNew_R = camOld.R.dot( R2 )
        camNew_t = camOld.t + camOld.R.dot(t2)
        
        return camNew_R, camNew_t, iKeyPts, representatives 
        
    @contract
    def orientCandidatesByEMatrix( self,
                                   candidates : list,
                                   maxEpipolarDist : float,
                                   minPairInlierRatio : float,
                                   confidenceLevel : float ) -> tuple:
        # candidates ist absteigend sortiert nach der Anzahl der gemeinsamen, bereits rekonstruierten obPts -> einfach den ersten nehmen, oder denjenigen mit den meisten matches mit einem bereits orientierten pho wählen?
        #bereits berechnete E-Matrizen wiederverwenden!! self.edge2PairWiseOri
        for candidate in candidates:
        
            neighborEdges = self.imageConnectivity.adjacentUnusedOriented( candidate.iImg )
            if len(neighborEdges)==0:
                continue
                            
            selection = None
            for neighborImg in [ edge.img2 for edge in neighborEdges ]:
                matches = self.edge2matches.get( (neighborImg.idx, candidate.iImg.idx) )
                if matches is None:
                    matches = np.fliplr( self.edge2matches[ (candidate.iImg.idx, neighborImg.idx) ] )
                if selection is None or \
                    selection[1] < matches.shape[0]:
                    selection = neighborImg,matches.shape[0]
                        
            if selection is None or \
                selection[1] < 50:
                continue
                                        
            R, t, iKeyPts, representatives = self.orientByEMatrix(
                iImgOld=selection[0].idx,
                iImgNew=candidate.iImg.idx, 
                maxEpipolarDist=maxEpipolarDist, 
                inlierRatio=minPairInlierRatio,
                confidenceLevel=confidenceLevel )
                                        
            if iKeyPts.shape[0] == 0:
                continue
                            
            return candidate.iImg, R, t, iKeyPts, representatives, np.arange( len(representatives) )
        
        return ()
        

    @contract
    def imgPairQuality( self, iImg1 : int, iImg2 : int, matches : 'array[Nx2]', pwo = None ) -> float:
        # Compute Sxx of the relative, exterior orientation parameters.
        # Use the reciprocal of the maximum of standard deviations of the position and rotation parameters, with the rotation parameters in gon/400.
        # Unfortunately, this does not seem to be a good quality measure, either: very similar photos with many matches get a high quality,
        # even if the intersection angles at object points later in addClosures turn out to be too small.

        edge = iImg1, iImg2

        ptsPix = [ self.imgs[iImg].keypoints[matches[:,idx],:2] for idx,iImg in enumerate(edge) ]
            
        allPtsCam = [ self.imgs[iImg].pix2cam.forward(ptsPix[idx]) for idx,iImg in enumerate(edge) ]

        if pwo is None:
            pwo = self.edge2PairWiseOri[edge]

        cams = [ CamParams(), CamParams() ]
        cams[0].ior = self.imgs[iImg1].ior
        cams[0].adp = self.imgs[iImg1].adp
        cams[0].t = np.zeros(3)
        cams[0].omfika = np.zeros(3)
        cams[1].ior = self.imgs[iImg2].ior
        cams[1].adp = self.imgs[iImg2].adp
        cams[1].t = pwo.t
        cams[1].omfika = ori.omfika(pwo.R)

        objPts = ori.triangulatePoints( allPtsCam[0],
                                        allPtsCam[1],
                                        cams[0],
                                        cams[1] ).copy()
        block = adjust.Problem()
        loss = adjust.loss.Trivial()
        for cam,ptsCam in zip_equal( cams, allPtsCam ):
            for ptCam,objPt in zip_equal( ptsCam, objPts ):
                cost = adjust.cost.PhotoTorlegard( *ptCam.astype(float) )
                block.AddResidualBlock( cost,
                                        loss,
                                        cam.t,
                                        cam.omfika,
                                        cam.ior,
                                        cam.adp,
                                        objPt )

            block.SetParameterBlockConstant(cam.ior)
            block.SetParameterBlockConstant(cam.adp)

        block.SetParameterBlockConstant( cams[0].t )
        block.SetParameterBlockConstant( cams[0].omfika )

        assert( abs( linalg.norm( cams[1].t ) - 1 ) < 1.e-7 )
        parameterization = adjust.local_parameterization.UnitSphere()
        block.SetParameterization( cams[1].t, parameterization )

        if 0: # skip this adjustment, as it may cost lots of time, and relative image orientations have already been computed with the same data in ori.filterMatchesByEMatrix
            options = adjust.Solver.Options()
            options.max_num_iterations = 50
            summary = adjust.Solver.Summary()
            adjust.Solve(options, block, summary)
            if not adjust.isSuccess( summary.termination_type ):
                return 0.

        covOpts = adjust.Covariance.Options()
        cov = adjust.Covariance(covOpts)
        try:
            cov.Compute([(cams[1].t,)*2,
                         (cams[1].omfika,)*2], block )
        except:
            return 0.
        qtt = cov.GetCovarianceBlockInTangentSpace( cams[1].t     , cams[1].t )
        qrr = cov.GetCovarianceBlockInTangentSpace( cams[1].omfika, cams[1].omfika )
        cost, = block.Evaluate(cost=True,residuals=False)
        sigma0 = ( cost*2 / ( len(matches)*4 - (len(matches)*3+5) ) )**.5

        badness = np.max( np.r_[ np.diag(qtt), np.diag(qrr) / 400.**2 ] )**.5 * sigma0
        return 1. / badness

    @contract
    def adjustSingle( self,
                      iImg : int,
                      bKeyPts         : 'array[>0](bool)',
                      representatives : 'array[>0](uint64)', # note: np.uintp would be appropriate here, but unsupported by pycontracts
                      bInliers        : 'array[>0](bool)', 
                      printer ) \
                    -> adjust.Solver.Summary:
        single = adjust.Problem()
        loss = adjust.loss.Trivial()
        img = self.imgs[iImg]
        ptsPix = img.keypoints[bKeyPts,:2]
        ptsCam = img.pix2cam.forward(ptsPix)

        for ptCam, objPt in zip_equal( ( img.pix2cam.forward(ptPix) for ptPix in ptsPix[bInliers] ),
                                       ( self.featureTrack2ObjPt[featureTrack] for featureTrack in representatives[bInliers] ) ):
            cost = adjust.cost.PhotoTorlegard( *ptCam.astype(float) )
            single.AddResidualBlock( cost,
                                     loss,
                                     img.t,
                                     img.omfika,
                                     img.ior,
                                     img.adp,
                                     objPt )
            single.SetParameterBlockConstant( objPt )
        
        if printer:
            X = np.array([ self.featureTrack2ObjPt[featureTrack] for featureTrack in representatives ], float )
            printer.single(70,"block before closure before adj", iImg, ptsPix, X, bInliers, img )
         
        single.SetParameterBlockConstant( img.ior )
        single.SetParameterBlockConstant( img.adp )
        options = adjust.Solver.Options()   
        options.max_num_iterations = 500
        options.max_num_consecutive_invalid_steps = 15
        options.linear_solver_type = adjust.LinearSolverType.DENSE_SCHUR
        summary = adjust.Solver.Summary()
        adjust.Solve(options, single, summary)
        
        if printer:
            # objPts have been set constant, so no need to update X
            printer.single(71,"block before closure after adj", iImg, ptsPix, X, bInliers, img )
                
        return summary


    @contract
    def initBlock( self,
                   maxResidualNorm : float,
                   #globalLoss : adjust.loss.Wrapper,
                   #block : adjust.Problem,
                   blockSolveOptions : adjust.Solver.Options,
                   robustLossFunc, # callable
                   initPair : 'list[2](number)|None',
                   printer  = None ) -> 'tuple(*,*,dict(int:int))':
        """initialize the incremental reconstruction with an appropriate image pair
        
        before that, no image has been oriented yet."""

        def resetPair( imgPair ):
            for img in ( imgPair.img1, imgPair.img2 ):
                img.state = graph.ImageConnectivity.Image.State.unoriented
                self.imageConnectivity.setImageState( img )
                self.imgs[img.idx].nReconstObjPts = 0
            imgPair.state = graph.ImageConnectivity.Edge.State.failed
            self.imageConnectivity.setEdgeState( imgPair )
            self.featureTrack2ObjPt.clear()
            blockSolveOptions.linear_solver_ordering.Clear()
            self.imgFeature2costAndResidualBlockID.clear()
            self.featureTrack2ImgFeatures.clear()
            self.orientedImgs = []

        def orientPair( imgPair ):
            pairWiseOri = self.edge2PairWiseOri.get( (imgPair.img1.idx, imgPair.img2.idx) )
            if pairWiseOri is None:
                pairWiseOri = self.edge2PairWiseOri[ (imgPair.img2.idx, imgPair.img1.idx) ]
                pairWiseOri.E = pairWiseOri.E.T
                pairWiseOri.t = - pairWiseOri.R.T.dot( pairWiseOri.t )
                pairWiseOri.R = pairWiseOri.R.T
                
            img1 = self.imgs[imgPair.img1.idx]
            img2 = self.imgs[imgPair.img2.idx]

            if False and printer:
                pt1pix = img1.keypoints[matches[:,0],:2]   
                pt2pix = img2.keypoints[matches[:,1],:2]
                printer.epipolar( imgPair.img1.idx, imgPair.img2.idx, pt1pix, pt2pix, pairWiseOri.E )
        
            img1.R,img1.t = np.eye(3), np.zeros(3)
            # coords of the second PRC in the Orient-CS of the first camera
            img2.R,img2.t = pairWiseOri.R.copy(), pairWiseOri.t.copy()
                    
            block = adjust.Problem()
            globalLoss = adjust.loss.Wrapper( robustLossFunc() )
            for img in ( imgPair.img1, imgPair.img2 ):
                img.state = graph.ImageConnectivity.Image.State.oriented
                self.imageConnectivity.setImageState( img )
            addFeatureTracks = {}
            msgs,affectedImgs = self.addClosures(
                imgPair.img2,
                maxResidualNorm,
                block,
                blockSolveOptions,
                globalLoss,
                addFeatureTracks )
            assert len(msgs)==1
            if msgs[0].nValid < 5:
                logger.info('Not enough valid matches. Trying next best edge')
                return False

            for cam in ( img1, img2 ):                
                blockSolveOptions.linear_solver_ordering.AddElementToGroup( cam.t     , 1 )
                blockSolveOptions.linear_solver_ordering.AddElementToGroup( cam.omfika, 1 )
                blockSolveOptions.linear_solver_ordering.AddElementToGroup( cam.ior   , 1 )
                blockSolveOptions.linear_solver_ordering.AddElementToGroup( cam.adp   , 1 )
                        
                block.SetParameterBlockConstant(cam.ior)
                block.SetParameterBlockConstant(cam.adp)

            block.SetParameterBlockConstant(img1.t)
            block.SetParameterBlockConstant(img1.omfika)
        
            assert( abs( linalg.norm( img2.t ) - 1 ) < 1.e-7 )
            parameterization = adjust.local_parameterization.UnitSphere()
            block.SetParameterization( img2.t, parameterization )
                
            options = adjust.Solver.Options()
            options.max_num_iterations = 500
            summary = adjust.Solver.Summary()
            adjust.Solve(options, block, summary)

            assert not len(self.orientedImgs)
            self.orientedImgs.append( imgPair.img1.idx )
            self.orientedImgs.append( imgPair.img2.idx )

            return block,globalLoss,addFeatureTracks,summary

        # early success:
        # - min. #features added to the block (considering visibility, residuals, and INTERSECTION ANGLE, according to addClosures
        # - image area covered by those features.
        #     This checks not only the overlap region, but also, how many matches were rejected by addClosures
        #     due to small intersection angles in the objPts.
        #     The area of the convex hull of features is not robust enough.
        #     #pixels/#features/medianDistanceToNextFeature
        #     i.e. pixels per feature, weighted by the median distance from each pixel to the nearest feature
        #                              distances must be normalized w.r.t. image resolution!
        # simple solution:
        # divide the image plane into a raster with a fixed resolution.
        # Count the number of cells that have at least 3 features in them.

        subOptimalPair = None

        def getImgPairs():
            if initPair:
                yield self.imageConnectivity.getEdge( *(graph.ImageConnectivity.Image(iImg) for iImg in initPair) )
            else:
                imgPair = graph.ImageConnectivity.Edge()
                while self.imageConnectivity.nextBestEdge(imgPair):
                    yield imgPair

        for imgPair in getImgPairs():
            logger.info( "{} - {}",
                self.imgs[imgPair.img1.idx].shortName,
                self.imgs[imgPair.img2.idx].shortName,
                tag='initial image pair' )

            res = orientPair( imgPair )
            if not res:
                resetPair( imgPair )
                continue
            block,globalLoss,addFeatureTracks,summary = res

            if not adjust.isSuccess( summary.termination_type ):
                logger.info('Adjustment failed. Trying next best edge')
                resetPair( imgPair )
                continue
                
            if printer:
                evalOpts = adjust.Problem.EvaluateOptions()
                evalOpts.apply_loss_function = False
                residuals, = block.Evaluate(evalOpts)
                printer.block( {}, title='init' )
                printer.allImageResiduals( 'init', addFeatureTracks, save=False )
                printer.residualHistAndLoss( robustLossFunc(), residuals  )
            
            nRowsCols = 5
            areas = []
            for iImg in ( imgPair.img1.idx, imgPair.img2.idx ):
                iKeyPts = np.fromiter( ( feature[1] for feature in self.imgFeature2costAndResidualBlockID if feature[0] == iImg ), dtype=int )
                img = self.imgs[iImg]
                keypoints = img.keypoints[iKeyPts,:2]
                hist, xedges, yedges = np.histogram2d( x=keypoints[:,0], y=keypoints[:,1], bins=nRowsCols, range=( (0,img.width), (-img.height+1,1) ) )
                #low = max( 1., len(keypoints) / np.count_nonzero(hist) / 2. )
                # Be more robust here: most features may be scattered along a linear feature (e.g. a measuring tape, see Milutin).
                # Assuming that at least 90% of matches are inliers.
                low = np.percentile( hist[hist>0], 10 )
                hist = hist >= low
                # corner points of grid cells
                points = np.array( [ (iX,iY) for iCol in range(nRowsCols) for iRow in range(nRowsCols) for iX in (iCol,iCol+1) for iY in (iRow,iRow+1) if hist[iRow,iCol] ] )
                hull = cv2.convexHull(points)
                areas.append( cv2.contourArea(hull) )
            
            area = min( areas )
            if area < nRowsCols**2 / 3:
                logger.info('Features cover less than a third of the image area. Trying next best edge')
                resetPair( imgPair )
                if not subOptimalPair or subOptimalPair[1] < area:
                    subOptimalPair = ( imgPair, area )
                continue

            # measure the spatial dispersion of the feature points:
            # median of absolute deviations from median.
            #disps = []
            #for iImg in ( imgPair.img1.idx, imgPair.img2.idx ):
            #    iKeyPts = np.fromiter( ( feature[1] for feature in self.imgFeature2costAndResidualBlockID if feature[0] == iImg ), dtype=int )
            #    img = self.imgs[iImg]
            #    keypoints = img.keypoints[iKeyPts,:2].copy() # copy needed?
            #    keypoints /= ( img.width, img.height )
            #    center = np.median( keypoints, axis=0 )
            #    distSqr = np.sum( (keypoints - center)**2, axis=1 )
            #    disp = np.median( distSqr )
            #    disp = disp**.5
            #    disps.append( disp )
            #
            #disp = min( disps )
            #if disp < .25 # less than
            return block,globalLoss,addFeatureTracks

        if subOptimalPair:
            imgPair = subOptimalPair[0]
            logger.info( "once more, initial relative orientation of phos {} and {}, without checking overlap and intersection angle",
                self.imgs[imgPair.img1.idx].shortName,
                self.imgs[imgPair.img2.idx].shortName )

            res = orientPair( imgPair )
            if res:
                block,globalLoss,addFeatureTracks,summary = res
                if adjust.isSuccess( summary.termination_type ):
                    return block,globalLoss,addFeatureTracks
            resetPair( imgPair )
        
        raise Exception("All edges have been tried, but all relative orientations have failed")

    @contract
    def candidatesPnP( self, exclude = None ) -> 'list(ImageConnectivityImage)':
        exclude = exclude or set()
        unorientedImages = self.imageConnectivity.unorientedImagesAdjacent2Oriented()
        unorientedImagesNReconst = []
        for unorientedImage in unorientedImages:
            if unorientedImage in exclude:
                continue
            nReconst = 0
            for orientedImage in self.imageConnectivity.adjacentOriented( unorientedImage ):
                #nReconst += self.imgs[orientedImage.idx].nReconstObjPts
                # try to estimate the number of common object points that have already been reconstructed:
                # the number of matches * ratio of reconstructed object points for the other image.
                matches = self.edge2matches.get((unorientedImage.idx,orientedImage.idx))
                if matches is None:
                    matches = self.edge2matches[orientedImage.idx,unorientedImage.idx]
                nReconst += matches.shape[0] * self.imgs[orientedImage.idx].nReconstObjPts / self.imgs[orientedImage.idx].nMatchedObjPts
            unorientedImagesNReconst.append( ( unorientedImage, nReconst ) )
        unorientedImagesNReconst.sort( key = lambda x: x[1], reverse=True )
        return [ unorientedImage for (unorientedImage,nReconstObjPts) in unorientedImagesNReconst ]

        #Candidate = namedtuple( 'Candidate', ( 'iImg', 'iKeyPts', 'objPts', 'representatives' ) )  
        #candidates = []
        #unorientedImages = self.imageConnectivity.unorientedImages()
        ## These nested loops take really long. Maybe it would be faster to determine beforehand for each oriented image, which keyPts have already been constructed, and store their index together with the objPt.
        #for imgLoc in unorientedImages:
        #    keyPts = self.imgs[imgLoc.idx].keypoints
        #    iKeyPts = []
        #    objPts  = []
        #    representatives = []                        
        #    for iKeyPt in range(keyPts.shape[0]):
        #        found,featureTrack = self.featureTracks.component( graph.ImageFeatureID( int(imgLoc.idx), int(iKeyPt) ) )
        #        if not found:
        #            continue
        #        objPts_ = self.featureTrack2ObjPt.get(featureTrack)
        #        if objPts_ is not None:
        #            if len( self.featureTrack2ImgFeatures.get(featureTrack,[]) ) < nObsMin:
        #                continue
        #            iKeyPts.append( iKeyPt )
        #            objPts .append( objPts_ )
        #            representatives.append( featureTrack )
        #                   
        #    if len(iKeyPts) < 4:
        #        continue    
        #    iKeyPts  = np.array( iKeyPts , dtype=np.uint64 )#.reshape( (-1,1) )
        #    objPts = np.array( objPts, dtype=np.float32 )#.reshape( (-1,3) )
        #    representatives = np.array( representatives, dtype=np.uintp )
        #    candidates.append( Candidate( imgLoc, iKeyPts, objPts, representatives ) )
        #    if iKeyPts.shape[0] >= nPtsBreak:
        #        return candidates
        #
        #return candidates

    def getCandidate( self,
                      img : graph.ImageConnectivity.Image ) :
        # These nested loops take really long. Maybe it would be faster to determine beforehand for each oriented image, which keyPts have already been constructed, and store their index together with the objPt.
        keyPts = self.imgs[img.idx].keypoints
        bKeyPts = np.zeros( keyPts.shape[0], bool )
        objPts  = []
        representatives = []                        
        for iKeyPt in range(keyPts.shape[0]):
            found,featureTrack = self.featureTracks.component( graph.ImageFeatureID( int(img.idx), int(iKeyPt) ) )
            if not found:
                continue
            objPts_ = self.featureTrack2ObjPt.get(featureTrack)
            if objPts_ is None:
                continue
            bKeyPts[iKeyPt] = True
            objPts .append( objPts_ )
            representatives.append( featureTrack )
                           
        if len(objPts) < 4:
            return     
        
        objPts = np.array( objPts, dtype=np.float32 )#.reshape( (-1,3) )
        representatives = np.array( representatives, dtype=np.uintp )
        return bKeyPts, objPts, representatives

    @contract
    def PnP( self,
             candidates : 'list(ImageConnectivityImage)',
             minInlierRatio   : float,
             confidenceLevel  : float,
             maxResidualNorm  : float,
             nInliersBreak    : int,
             inlierRatioBreak : float ) -> 'tuple(bool,list)':
        Img = namedtuple( 'Img', ( 'img', 'bKeyPts', 'R', 't', 'bInliers', 'representatives', 'density' ) )
        imgs = []
        for candidate in candidates:
            imgLoc = candidate
            res = self.getCandidate( candidate )
            if res is None:
                continue
            bKeyPts, objPts, representatives = res
            img = self.imgs[imgLoc.idx]
            imgPts = img.keypoints[bKeyPts,:2] #.astype( dtype=np.float64, copy=False )
                            
            # Generally, there is no lossless way to transform ORIENT ADP into openCV distortion parameters
            # Thus, pass zero-valued distortion parameters,
            # and undistorted images observations!
            # Mind that ori.distortion expects imgPts in the ORIENT image coordinate system, so we need to invert the y-coordinate twice!
            # As we return imgPts from here, we must not change them in-place.
            imgPts_undist = ( ori.distortion( img.pix2cam.forward(imgPts),
                                              img.ior,
                                              ori.adpParam2Struct(img.adp),
                                              ori.DistortionCorrection.undist )*(1.,-1.) # ORIENT -> cv
                            ).astype( img.keypoints.dtype )
                        
            objPts[:,1:] *= -1.
            # Achtung: auch EPNP ergibt manchmal grob fehlerhafte Orientierungen, u.z. bei planarem RefSys und geneigter Kamera (d.h. nicht-Normalfall).
            # siehe die EPNP-Originalpublikation:
            # F.Moreno-Noguer and V.Lepetit and P.Fua: "Accurate Non-Iterative O(n) Solution to the PnP Problem", ICCV 2007
            # http://infoscience.epfl.ch/record/179767/files/top.pdf Kap.4.1.2
            # Moren-Noguer et al. vergleichen die Ergebnisse von EPNP in diesem Fall (planares RefSys, geneigte Kamera) mit
            # G. Schweighofer and A. Pinz. Robust pose estimation from a planar target. PAMI, 28(12):2024–2030, 2006. 
            # -> fast immer richtig, aber viel langsamer.
                        
            # cv2.solvePnPRansac ist mit der TBB parallelisiert: die vorab definierte Anzahl an maximalen Iterationen wird auf die threads aufgeteilt.
            # Vermutlich deshalb unterstützt es keinen Parameter param2 wie z.B. cv2.findFundamentalMat "desirable level of confidence (probability) that the estimated matrix is correct.",
            #   denn cv2.findFundamentalMat passt die maximale Anzahl an Iterationen nach jeder Modellschätzung an, auf Basis des Anteils der inlier des Modells im Verhältnis zur Gesamtanzahl Punkte,
            #   siehe modelest.cpp: cvRANSACUpdateNumIters
            #   aus dem Anteil der outlier der aktuellen Modellschätzung wird nach untenstehender Formel die maximale Anzahl an Iterationen geschätzt.
            #   Ist diese Anzahl kleiner als die bisherige max. Anzahl an Iterationen,
            #   dann wird diese auf den neuen Wert reduziert.
            #   Gestartet wird mit einer max. Anzahl an Iterationen von 2000.
                        
            # Mit dem derzeitigen Ansatz fällt der Anteil der inlier grundsätzlich mit jedem zusätzlich orientierten Bild:
            # 90% beim dritten Bild (dem ersten nach dem initialen Bildpaar), <10% beim x-ten Bild 
            # Damit der gesamte Block sicher zum globalen Optimum konvergiert (d.h. auch das Bild mit dem kleinsten Anteil an inliern richtig orientiert wird),
            #   müssen wir w entsprechend niedrig ansetzen, womit sich eine sehr hohe Anzahl an maximalen Iterationen ergibt.
            #   Stösst cv2.solvePnPRansac auf eine Lösung mit mehr als minInliersCount inliern, dann bricht es vorzeitig ab. Andernfalls werden wirklich so viele Iterationen / Modelle berechnet,
            #   was trotz Parallelisierung recht lange dauert.
            #   Beschleunigen ließe sich das, in dem wie bei den nicht parallelisierten RANSAC-Funktionen in OpenCV aus dem Anteil der outlier des aktuellen Modells
            #   die max. Anzahl an Iterationen bestimmt und die bisherige max. Anzahl an Iterationen ggf. reduziert wird! (siehe cv::RANSACUpdateNumIters)
            #w = 0.08 # 10% inliers -> 69074       @ 99.9% probability of a correct model
            #         #  8% inliers -> 168643      @ 99.9% probability of a correct model
            #         #  5% inliers -> 1.1 million @ 99.9% probability of a correct model
            # SOLVEPNP_EPNP needs min. 5 points, SOLVEPNP_P3P needs exactly 4
            nModelPoints = 5 if len(imgPts_undist) >= 5 else 4
            maxNumIters = ori.maxNumItersRANSAC( nModelPoints=nModelPoints, inlierRatio=minInlierRatio, confidence=confidenceLevel, nDataPoints=len(imgPts_undist) )
            # it would help to introduce a heuristic for selecting the sample points:
            # prefer objPts that are observed in at least 3 images to those that are only observed twice!
                        
            # Even though confidenceLevel has been introduced, iterationsCount must not be set to an unreasonably high value,
            #  because iterationsCount iterations are executed for the many images that do not even have our low ratio of 8% inliers - for example, images that do not overlap at all the current point cloud, and thus contain only erroneous correspondences!
                        
            # Note: there could be a hard upper bound on the number of iterations: the number of possibilities to choose 4 different correspondences from the submitted correspondences, irrespective of their order:
            #  that is, the binomial coefficient (n,4)
            #  (n,4) is really large for the usual numbers of submitted correspondences, e.g. (441,4) == 1.6e9
            # Also, cv2.solvePnPRansac chooses each sample randomly without consideration of already chosen samples,
            #  and thus, the same samples are chosen multiple times, generally - hence, in turn, that upper bound of (n,4) is not applicable.
                        
            #minInliersCount = max( 50, objPts.shape[0]//2 ) # 50% if many points, else set it to the number of correspondences (-1), because e.g. 5 inliers out of only 10 points still seems unreliable
                        
            cameraMatrix = ori.cameraMatrix( img.ior )

            # As reprojectionError, let's use the same threshold as is used below (loop over closures) for insertion of observations of existing points!
            # Thus, the total count of inserted observations for distinct, non-newly-triangulated points should match the count of inliers here!
            # NOTE: For OpenCV 3.0, we need to make sure that objectPoints has 3 channels and 1 column and that
            #                                                 imagePoints  has 2 channels and 1 column,
            # or otherwise, the internals of solvePnPRansac, RANSACPointSetRegistrator won't work: too few bytes are copied from local models to the best model!!
            success, rvec, tvec, iInliers = cv2.solvePnPRansac( objectPoints=objPts.reshape((-1,1,3)),
                                                                imagePoints=imgPts_undist.reshape((-1,1,2)),
                                                                cameraMatrix=cameraMatrix,
                                                                distCoeffs=np.empty(0),
                                                                useExtrinsicGuess=False,
                                                                iterationsCount=maxNumIters, # exit at latest after trying iterationsCount random samples; return the result with most inliers 
                                                                reprojectionError=maxResidualNorm,
                                                                confidence=confidenceLevel, # estimate the ratio of inliers after each model estimation, and reduce the max. number of iters accordingly!
                                                                # solvePnPRansac runs RANSAC either with SOLVEPNP_EPNP (>=5 pts) or SOLVEPNP_P3P (4 pts), and only the final solution is computed with the method passed in flags
                                                                #flags=cv2.SOLVEPNP_ITERATIVE # For initialization, CV_ITERATIVE uses DLT for non-planar RefSys (disregarding passed ior, or clamped?) and Homography for planar RefSys
                                                                #flags=cv2.SOLVEPNP_P3P # doesn't support redundancy.
                                                                flags=cv2.SOLVEPNP_EPNP # is said to handle planar and non-planar RefSys!
                                                                #flags=cv2.SOLVEPNP_DLS # would be the new hot stuff. More precise results.
                                                              )
            if not success or iInliers is None:
                continue 
            # inliers.shape == (n,1) -> make it (n,)
            iInliers = iInliers.squeeze()

            if 1:
                bInliers = np.zeros( imgPts.shape[0], bool )
                bInliers[iInliers] = True
                nInliers = len(iInliers)
            else:
                # solvePnPRansac determines the final inliers based on one of the tried models.
                # Using those inliers, solvePnPRansac refines the unknowns (rvec,tvec) using Levenberg-Marquardt (but leaves inliers unchanged afterwards).
                # Considering those refined parameters, the set of inliers is generally changed!
            
                # How to suppress the computation and output of the Jacobian in Python?
                projected,_ = cv2.projectPoints( objectPoints=objPts,
                                                 rvec=rvec,
                                                 tvec=tvec,
                                                 cameraMatrix=cameraMatrix,
                                                 distCoeffs=np.empty(0) )
                # imagePoints.shape == (n,1,2) -> make it (n,2)
                projected = projected.squeeze()
                residualsSq = np.sum( ( imgPts_undist - projected )**2, axis=1 )
                bInliers = residualsSq < maxResidualNorm**2
                nInliers = np.count_nonzero(bInliers)
                        
            if nInliers < 3:
                continue
            #if inliers.shape[0] >= minInliersCount and \
            if nInliers / imgPts.shape[0] < minInlierRatio:
                logger.warning("solvePnPRansac has tried fewer samples than would be appropriate for this low ratio of inliers. Result is unreliable, thus skipping.")
                continue

            # Check if OpenCV applies the threshold correctly. There was a bug in OpenCV-3.0:
            # the L2-norm of the image residual vectors were not compared to reprojectionError, but to reprojectionError**2 !! see RANSACPointSetRegistrator::findInliers (expects squared norms) vs. PnPRansacCallback::computeError (computes unsquared L2-norms)
            #bInliers = np.zeros( len(imgPts_undist), np.bool )
            #bInliers[inliers] = True
            #assert residualsSq[ bInliers].max()**.5 <= maxResidualNorm
            #assert residualsSq[~bInliers].min()**.5 >= maxResidualNorm

            # Compute a simple feature density that approximates the expected orientation quality.
            H, *_ = np.histogram2d( x = -imgPts[bInliers][:,1], # rows
                                    y =  imgPts[bInliers][:,0], # cols
                                    bins = 10,
                                    range = [[ -.5, img.height - .5 ], [ -.5, img.width - .5 ]],
                                    normed=False )
            thresh = 10
            H[H>thresh] = thresh
            Hblurred = ndimage.filters.gaussian_filter( H, sigma=1, mode='nearest' )
            density = np.maximum( H, Hblurred ).sum() / thresh / H.size

            Rcv,_ = cv2.Rodrigues( rvec )
            R,t = ori.projectionMat2oriRotTrans( np.column_stack((Rcv,tvec)) ) 

            imgs.append( Img(imgLoc, bKeyPts, R, t, bInliers, representatives, density ) )
                        
            if nInliers > nInliersBreak and \
               nInliers / imgPts.shape[0] > inlierRatioBreak:
                return True, imgs # result seems very reliable, don't try any further
                    
        return False,imgs

    @contract
    def addClosures( self,
                     img : graph.ImageConnectivity.Image,
                     maxResidualNorm : float,
                     block : adjust.Problem,
                     blockSolveOptions : adjust.Solver.Options,
                     globalLoss : adjust.loss.Wrapper,
                     addFeatureTracks : dict ) -> 'tuple(list,set)':
        # query imageConnectivity for edges between already oriented images and the newly oriented one,
        #   and insert the resp. correspondences as observations into the global block
        #   -> stabilize the so far oriented images in the block.
        # How to take care of outliers? Before adding these observations into the block?

        # these 2 functions get called frequently. Better define them outside the loops. Even though their code object is created only once anyway, a function object is created each time the interpreter encounters their definition, which costs a little.
        def checkVisibility( camPars_, R_, objPt_ ):
            return 0. > R_.T.dot( objPt_ - camPars_.t )[2]
                     
        def checkResidual( imgPtPinHole, camPars_, R_, objPt_, thresh_ ):
            # distorting projected image points due to ADP is iterative and thus slow.
            # instead, undistort the observed image point!
            #return thresh_**2 > np.sum( ( ori.projection( objPt_, camPars_.t, camPars_.omfika, camPars_.ior, camPars_.adp ) - imgPt_ )**2. )
            pinHoleProj = ori.projection( objPt_, camPars_.t, R_, camPars_.ior )
            return thresh_**2 > np.sum( ( pinHoleProj - imgPtPinHole )**2. )

        
        minIntersectAngleRad = self.minIntersectAngleGon / 200. * math.pi
        maxIntersectAngleRad = math.pi - minIntersectAngleRad

        def checkIntersectionAngle( newP0, oldP0, objPt ):
            ray1 = newP0 - objPt
            ray2 = oldP0 - objPt
            ray1n = linalg.norm( ray1 )
            ray2n = linalg.norm( ray2 )
            if ray1n < 1.e-15 or ray2n < 1.e-15:
                return False
            ray1 /= ray1n
            ray2 /= ray2n
            # avoid np.arccos' domain error, resulting in inf and setting the floating point error flag
            angle = np.arccos( np.clip( ray1.dot(ray2), -1., 1. ) ) # -> [0,pi]
            return angle > minIntersectAngleRad and \
                   angle < maxIntersectAngleRad

        camNew = self.imgs[img.idx]
        Msg = namedtuple( 'Msg', ( 'iOther', 'nValid', 'nTotal', 'nObsAddedSelf', 'nObsAddedOther', 'nObjNew' ) )
        msgs = []
        affectedImgs = set()
        for closure in self.imageConnectivity.adjacentUnusedOriented( img ):
            logger.debug('Adding closure {} -> {}', closure.img1.idx, closure.img2.idx )
            assert closure.img1.idx   == img.idx, "self.imageConnectivity.adjacentUnusedOriented(.) should return an edge whose first vertex is the function argument"
            assert closure.img1.state == graph.ImageConnectivity.Image.State.oriented
            assert closure.img2.state == graph.ImageConnectivity.Image.State.oriented
            assert closure.state      != graph.ImageConnectivity.Edge.State.used
            # TODO make edge hashable, so we can skip the following swap?
            matches = self.edge2matches.get( (closure.img1.idx, closure.img2.idx) )
            if matches is None:
                matches = np.fliplr( self.edge2matches[ (closure.img2.idx, closure.img1.idx) ] )
            
            closure.state = graph.ImageConnectivity.Edge.State.used
            self.imageConnectivity.setEdgeState( closure )
            
            pt1pix = self.imgs[closure.img1.idx].keypoints[matches[:,0],:2]  # the newly oriented image 
            pt2pix = self.imgs[closure.img2.idx].keypoints[matches[:,1],:2]
            
            pt1cam = self.imgs[closure.img1.idx].pix2cam.forward(pt1pix)
            pt2cam = self.imgs[closure.img2.idx].pix2cam.forward(pt2pix)

            camOld = self.imgs[closure.img2.idx]
            # need to triangulate
            # compute X[:,:] once for all.
            Xori = ori.triangulatePoints( pt1cam,
                                          pt2cam,
                                          camNew,
                                          camOld )
            # pre-compute the rotation matrices, so we don't need to compute sin and cos for each call of checkVisibility, checkResidual
            Rnew = ori.omfika( camNew.omfika )
            Rold = ori.omfika( camOld.omfika )
            
            pt1camPinhole = ori.distortion( pt1cam.reshape(-1,2), camNew.ior , ori.adpParam2Struct( camNew.adp ), ori.DistortionCorrection.undist )
            pt2camPinhole = ori.distortion( pt2cam.reshape(-1,2), camOld.ior , ori.adpParam2Struct( camOld.adp ), ori.DistortionCorrection.undist )
                        
            nNew = 0
            nNewRec = 0
            nCommon = 0
            nObsAdded = [ 0, 0 ]
            for iMatch in range( matches.shape[0] ):
                # Wenn a mit b gematcht wurde
                # und  b mit c, nicht
                # aber a mit c,
                # und in b manche Beobachtungen nicht eingefügt werden wegen zu großer Residuen / schlechtem Schnittwinkel,
                # dann fügt der folgende Ansatz keine neuen Objektpunkte für a/c ein!!!
                # Abhilfe würde schaffen:
                # iteriere über alle keypoints des neu orientierten phos
                #   bestimme den featureTrack
                #   frage alle features dieses representatives ab
                #   iteriere über diese features
                #   falls das pho des features schon orientiert wurde, dann füge eine Beobachtung ein, falls diese noch nicht vorhanden ist.  
                            
                # Nach der Orientierung eines phos prüfen wir immer auf outlier, deaktivieren diese im Block, und entfernen Einträge in:
                # del self.imgFeature2costAndResidualBlockID[imgPt]
                #   self.featureTrack2ImgFeatures[featureTrack].remove( imgPt )
                # ODER
                #   del self.featureTrack2ImgFeatures[featureTrack]  
                # del self.featureTrack2ObjPt[featureTrack]           falls der ganze track entfernt wird, weil nur eine imgObs übrigbleiben würde
                #
                # Den Eintrag in featureTracks können wir nicht löschen, weil featureTracks zwar inkrementell, aber nicht dynamisch ist (es erlaubt also das Einfügen von Vertizes und Kanten, nicht aber das Entfernen derselben).          
                # Es fehlen daher dann diese Einträge für einen imgPt in self.imgFeature2costAndResidualBlockID, etc., obwohl imgPt noch immer in featureTracks ist. 
                # Auch wenn eine imgObs zuvor entfernt wurde, macht es Sinn, diese später wieder einzufügen, wenn:
                #   sich die erste von 2 imgObs als Ausreißer herausstellt: die erste imgObs muss also entfernt werden. Aber auch die zweite, weil der zugehörige objPt ja sonst nicht bestimmbar wäre.
                #   Stößt das Skript später auf eine dritte imgObs, die mit der zweiten verknüpft ist, dann wird die zweite wieder eingefügt, gemeinsam mit der dritten.         
                            
                # Verschiedene mögliche Situationen, kategorisiert nach der Anzahl der beiden imgPts, die bereits in featureTracks sind:
                # 0: neuen objPt triangulieren, und dessen Residuen in beiden phos prüfen. 
                # 1: Für jenen imgPt, der nicht in featureTracks ist, gibt es sicher noch keine imgObs.
                #    Für den anderen imgPt gab es sicher einmal eine imgObs, diese kann aber schon wieder entfernt worden sein!
                #      Falls es schon eine gibt, dann bräuchte dessen Residuum nicht überprüft zu werden, weil das schon beim Einfügen dieser imgObs durchgeführt wurde und nach jeder Iteration geschieht.
                # 2: 
                #  a falls featureTracks die beiden imgPts mit demselben featureTrack verknüpft,
                #      dann wurden beide Beobachtungen schon in den Block eingefügt, können aber auch schon wieder entfernt worden sein (siehe oben).
                #      Falls schon beide imgObs im Block und aktiv sind, dann bräuchten wir gar keine Residuen prüfen, und auch keine neuen imgObs einfügen.
                #  b andernfalls stellt sich die Frage, ob sie vereinigt werden sollen, oder die aktuelle Korrespondenz verworfen werden soll.
                #     Für diese Entscheidung muss die Anzahl der beiden representatives bestimmt werden, für die featureTrack2ObjPt objPts enthält:
                #     0: immer vereinigen; trianguliere einen Neupunkt
                #     1: immer vereinigen; übernehme den einzigen bestehenden objPt
                #     2: Es wird nur einer der beiden representatives 'überleben', wir wissen vorab aber nicht, welcher (ließe sich aber herausfinden: internal von boost::disjoint_sets).
                #        Die Residuen des überlebenden representatives müssen auf jeden Fall in beiden phos okay sein. Aber das prüft nur auf Konsistenz entlang der Beobachtungsstrahlen.
                #        Der sterbende objPt wird auch in mind. 1 anderen pho beobachtet, in anderer Perspektive - denn sonst gäbe es ihn nicht (mehr).
                #        Genaugenommen müssten wir also die Residuen des überlebenden representatives in allen phos prüfen, die den sterbenden representatives beobachten!
                #        Daher besser auch die Residuen des sterbenden representatives in beiden phos prüfen. 
                #        Sind die Residuen okay, dann vereinige die beiden objPts, d.h. ersetze die vorhandenen imgObs im Block für ALLE imgObs, die den sterbenden objPt verwenden!.
        
                # Wenn es einen objPt gibt, dann muss es einen rep geben.
                # Falls es einen rep gibt, aber keinen objPt, dann kann es keine anderen phos geben, die diesen rep beobachten (abgesehen von deaktivierten obs)
                            
                featureTrack = self.featureTracks.component( graph.ImageFeatureID( int(closure.img1.idx),  int(matches[iMatch,0].item()) ) )[1]
                #assert featureTrack == self.featureTracks.component( graph.ImageFeatureID( int(closure.img2.idx),  int(matches[iMatch,1].item()) ) )[1]
        
                            
                objPt = self.featureTrack2ObjPt.get( featureTrack )
                if objPt is not None:
                    # TODO:
                    # If objPt is observed in only 2 images so far, then we should deem its position as unreliable.
                    #   In that case, the 1 out of the 2 image observations might very well be erroneous, even though they happen to satisfy the epipolar geometry.
                    # Thus, check the residuals of the triangulated point in that case.
                    #   This, way, the eventual erroneous match may be replaced by a good one, and objPt be pulled into the right position.
##                            if not checkVisibility( camNew, objPt_ ) or \
##                               not checkVisibility( camOld, objPt_ ):
##                                continue
##
##                            if not checkResidual( pt1[iMatch,:], camNew, objPt_, maxResidualNorm*10 ) or \
##                               not checkResidual( pt2[iMatch,:], camOld, objPt_, maxResidualNorm*10 ):
##                                continue                                
                    if featureTrack in addFeatureTracks:
                        nCommon += 1
                    else:
                        nNewRec += 1
                                
                else:
                    objPt = adjust.parameters.ObjectPoint( Xori[iMatch,:].copy() ) # don't reference a slice of the large array!?
                                
                    if not checkVisibility( camNew, Rnew, objPt ) or \
                       not checkVisibility( camOld, Rold, objPt ):
                        continue
                                
                    if not checkResidual( pt1camPinhole[iMatch,:], camNew, Rnew, objPt, maxResidualNorm ) or \
                       not checkResidual( pt2camPinhole[iMatch,:], camOld, Rold, objPt, maxResidualNorm ):
                        continue
        
                    # check the intersection angle of (reverse) observation rays in the newly triangulated objPt
                    if not checkIntersectionAngle( camNew.t, camOld.t, objPt ):
                        continue
                                
                    self.featureTrack2ObjPt[featureTrack] = objPt
                    addFeatureTracks[ featureTrack ] = ObjPtState.new
                    blockSolveOptions.linear_solver_ordering.AddElementToGroup( objPt, 0 )
                    nNew += 1
                                
                            
                # Concerning the newly oriented image (camNew):
                # If this is a newly triangulated objPt, then we know for sure that the observation in closure.img1 needs to be introduced.
                # Otherwise, the observation in the first image has already been inserted, if there is a correspondence with another image with lower index in the current list of closures
                #   (inserted in a previous execution of this loop body for the currently being oriented img1)
        
                # Concerning the image that was already oriented:                            
                # If this is a newly triangulated objPt, then we know for sure that the observation in closure.img2 needs to be introduced.
                # Otherwise we don't know, because
                #   that observation may or may not already have been introduced because of a correspondence of a previously oriented image with closure.img2
                for( iImg, iMatchCol, ptcam, cam ) in ( ( closure.img1.idx, 0, pt1cam, camNew ),
                                                        ( closure.img2.idx, 1, pt2cam, camOld ) ):
                    feature = (iImg, matches[iMatch,iMatchCol])        
                    affectedImgs.add(iImg)            
                    if feature not in self.imgFeature2costAndResidualBlockID:
                        cost = adjust.cost.PhotoTorlegard( *ptcam[iMatch,:2].astype(float) )
                        resId=block.AddResidualBlock( cost,
                                                      globalLoss,
                                                      cam.t,
                                                      cam.omfika,
                                                      cam.ior,
                                                      cam.adp,
                                                      objPt )
                        self.imgFeature2costAndResidualBlockID[feature]=(cost,resId)
                        self.featureTrack2ImgFeatures.setdefault( featureTrack, [] ).append( feature )
                        self.imgs[iImg].nReconstObjPts += 1
                        nObsAdded[iMatchCol] += 1
                   
            
            msgs.append( Msg( closure.img2.idx,
                              nCommon+nNew+nNewRec,
                              matches.shape[0],
                              nObsAdded[0],
                              nObsAdded[1],
                              nNew ) )
                            
            #validMatchesRatio = float(nCommon+nNew+nNewRec)/matches.shape[0]
            #if closure.quality > 0. and \
            #    matches.shape[0] > 50 and \
            #    validMatchesRatio < 0.8:
            #    logger.info( "Warning: Matches of {}<->{} have been filtered based on their relative orientation, and still, now only {:3.0%} of matches have been found valid.\nReconstruction may be invalid!",
            #        self.imgs[closure.img1.idx].shortName,
            #        self.imgs[closure.img2.idx].shortName,
            #        validMatchesRatio
            #        )
            
        return msgs,affectedImgs

    @contract
    def deactivateOutliers( self,
                            cutoff : float,
                            block : adjust.Problem,
                            robustLossFunc, # callable
                          ) -> 'tuple(int,int,int)':
        evalOpts = adjust.Problem.EvaluateOptions()
        evalOpts.apply_loss_function = False
        logger.debug( "Compute residuals" )
        res, = block.Evaluate( evalOpts )
        logger.debug( "Residuals computed" )
        resNormSqr = res[0::2]**2 + res[1::2]**2
        deactivate = resNormSqr > cutoff**2

        def checkMinIntersectionAngles():
            deacFeatures = []
            nObjPtsUnstable = 0
            minIntersectAngleRad = self.minIntersectAngleGon / 200. * math.pi
            maxIntersectAngleRad = math.pi - minIntersectAngleRad
            def checkMinIntersectionAngle(features,objPt):
                # we expect the great majority of objPts to have valid intersection angles,
                # and we expect the first or first few observation rays to meet the criterion.
                # Thus, we do not pre-compute the unit length observation rays!
                for iF1,f1 in enumerate(features):
                    ray1 = self.imgs[f1[0]].t - objPt
                    ray1n = linalg.norm( ray1 )
                    if ray1n < 1.e-15:
                        continue
                    ray1 /= ray1n
                    for f2 in features[iF1+1:]:
                        ray2 = self.imgs[f2[0]].t - objPt
                        ray2n = linalg.norm( ray2 )
                        if ray2n < 1.e-15:
                            continue
                        ray2 /= ray2n
                        # avoid np.arccos' domain error, resulting in inf and setting the floating point error flag
                        angle = np.arccos( np.clip( ray1.dot(ray2), -1., 1. ) ) # -> [0,pi]
                        if angle > minIntersectAngleRad and angle < maxIntersectAngleRad:
                            return True
                return False
            for featureTrack,objPt in self.featureTrack2ObjPt.items():
                features = self.featureTrack2ImgFeatures[featureTrack]
                if not checkMinIntersectionAngle(features,objPt):
                    nObjPtsUnstable += 1
                    deacFeatures.extend( features )
            return set(deacFeatures), nObjPtsUnstable
        deacFeatures, nObjPtsUnstable = checkMinIntersectionAngles()

        test=False
        if test:
            assert len(deactivate) == len(self.imgFeature2costAndResidualBlockID)
            nResidualBlocks = block.NumResidualBlocks()
            nResiduals      = block.NumResiduals()
        # only deactivate the bad observations, not the whole tracks.
        # If after deactivation, only 1 active observation of an objPt is left, then deactivate the whole track.
        # But don't insert the track into deactivatedFeatureTracks,
        # as that means that the objPt will never be restored, even if more correspondences are found subsequently
        logger.verbose( "Removing outliers" )
        del_imgFeature2costAndResidualBlockID = []
        nRemovedObjPts = 0
        
        def outlierFeatures():
            assert len(deactivate)==len(self.imgFeature2costAndResidualBlockID)
            # that's why it must be an OrderedDict:
            for deac,feature in zip_equal(deactivate,self.imgFeature2costAndResidualBlockID):
                if deac or feature in deacFeatures:
                    yield feature[0], int(feature[1])
        
        for feature in outlierFeatures():
            found,featureTrack = self.featureTracks.component( graph.ImageFeatureID( *feature ) )
            assert found
            # use a list comprehension instead of a for-loop to gain performance. Additionally, local variables in list comprehensions do not leak into our namespace!
            features = self.featureTrack2ImgFeatures.get(featureTrack)
            if features is None:
                # while removing a preceding outlier feature, only 1 active feature would have remained that observed their common objPt., and was removed.
                assert not self.imgFeature2costAndResidualBlockID[feature][0].isAct()
                assert feature in del_imgFeature2costAndResidualBlockID
                assert featureTrack not in self.featureTrack2ObjPt
                continue
            if test:
                assert feature in features
                for feature2 in features:
                    assert self.imgFeature2costAndResidualBlockID[feature2][0].isAct()

            if len(features) > 2: # more than 2 observations will still observe the objPt
                        
                # re-triangulate! Otherwise, the block adjustment may not converge any more.
                # TODO: select pair with best intersection angle
                iImg1, iFeat1 = features[0]
                iImg2, iFeat2 = features[1]
                img1 = self.imgs[iImg1]
                img2 = self.imgs[iImg2]
                Xori = ori.triangulatePoints( img1.pix2cam.forward( img1.keypoints[iFeat1,:2] ).reshape((-1,2)),
                                              img2.pix2cam.forward( img2.keypoints[iFeat2,:2] ).reshape((-1,2)),
                                              img1,
                                              img2 ).squeeze()
                # Note: the newly triangulated point may still have too large residuals.
                # Note: Xori is the solution of SVD-triangulation. Least-squares solution may deviate from that.
                #       Otherwise, if len(features) == 3, we could avoid usage of adjust.Problem() and simply project Xori into the images and check the residuals.
                problem = adjust.Problem()
                loss = robustLossFunc()
                for (iImg,iFeat) in features:
                    if (iImg,iFeat) == feature:
                        continue
                    img = self.imgs[iImg]
                    cost = adjust.cost.PhotoTorlegard( *img.pix2cam.forward( img.keypoints[iFeat,:2] ).astype(float) )
                    problem.AddResidualBlock( cost,
                                              loss,
                                              img.t,
                                              img.omfika,
                                              img.ior,
                                              img.adp,
                                              Xori )
                    for par in ( img.t, img.omfika, img.ior, img.adp ):
                        problem.SetParameterBlockConstant( par )
                options = adjust.Solver.Options()   
                options.linear_solver_type = adjust.LinearSolverType.DENSE_QR
                summary = adjust.Solver.Summary()
                adjust.Solve(options, problem, summary)
                if adjust.isSuccess( summary.termination_type ):
                    resNew, = problem.Evaluate( evalOpts )
                    resNewNormSqr = resNew[0::2]**2 + resNew[1::2]**2
                    if resNewNormSqr.max() <= cutoff**2:
                        self.imgFeature2costAndResidualBlockID[feature][0].deAct()
                        del_imgFeature2costAndResidualBlockID.append( feature )
                        features.remove( feature )
                        self.featureTrack2ObjPt[featureTrack][:] = Xori # [:] copy values, not the object!
                        continue
                        # otherwise proceed and remove all observations of objPt

            for feature2 in features:
                self.imgFeature2costAndResidualBlockID[feature2][0].deAct()
            del_imgFeature2costAndResidualBlockID.extend( features )
            del self.featureTrack2ImgFeatures[featureTrack]
            # oriental.adjust.Problem holds a reference to self.featureTrack2ObjPt[featureTrack], so we can safely delete the reference here, without deleting the object
            objPt = self.featureTrack2ObjPt.pop( featureTrack )
            # otherwise, the Jacobian gets rank deficient!
            # Alternatively, we may call block.RemoveParameterBlock(.).
            # However, the docs say: "will destroy the implicit ordering, rendering the jacobian or residuals returned from the solver uninterpretable"
            # Does that also make Covariance unusable?
            # Anyway, we depend on the matching order of the returned residuals and imgFeature2costAndResidualBlockID.
            block.SetParameterBlockConstant( objPt )
            # If we removed the objPt from the problem instead of setting it constant, then we'd need to call this:
            #blockSolveOptions.linear_solver_ordering.Remove( objPt )

            nRemovedObjPts += 1

        nRemovedImgObs = len(del_imgFeature2costAndResidualBlockID)
        if test:
            assert len(set(del_imgFeature2costAndResidualBlockID))==nRemovedImgObs, 'expected unique values in del_imgFeature2costAndResidualBlockID'
            assert block.NumResiduals() == nResiduals - 2*nRemovedImgObs
            resNew, = block.Evaluate( evalOpts )
            assert len(resNew) == len(res) - 2*nRemovedImgObs
            resNewNormSqr = resNew[0::2]**2 + resNew[1::2]**2
            assert resNewNormSqr.max() <= cutoff**2

        for feature in del_imgFeature2costAndResidualBlockID:
            del self.imgFeature2costAndResidualBlockID[feature]
            self.imgs[feature[0]].nReconstObjPts -= 1
        
        return nRemovedObjPts, nRemovedImgObs, nObjPtsUnstable

    @contract
    def atLeast3obs( self, block : adjust.Problem ) -> 'tuple(int,int)':
        del_representatives = []
        nRemovedObjPts = 0
        nRemovedImgObs = 0
        for featureTrack,features in self.featureTrack2ImgFeatures.items():
            if len(features) < 3:
                for feature in features:
                    self.imgFeature2costAndResidualBlockID[feature][0].deAct()
                    del self.imgFeature2costAndResidualBlockID[feature]
                    self.imgs[feature[0]].nReconstObjPts -= 1
                    nRemovedImgObs += 1
                del_representatives.append( featureTrack )
        for featureTrack in del_representatives: 
            del self.featureTrack2ImgFeatures[featureTrack]
            #del self.featureTrack2ObjPt[featureTrack] # oriental.adjust.Problem holds a reference to self.featureTrack2ObjPt[featureTrack], so we can safely delete the reference here, without deleting the object
            objPt = self.featureTrack2ObjPt.pop( featureTrack )
            block.SetParameterBlockConstant( objPt ) # otherwise, the Jacobian gets rank deficient!

        return len(del_representatives), nRemovedImgObs

    def estimateObjPlane( self ):
        assert len(self.featureTrack2ImgFeatures) == len(self.featureTrack2ObjPt), "featureTrack2ImgFeatures and featureTrack2ObjPt do not share the same number of featureTracks"
        assert all(( track1==track2 for track1,track2 in zip_equal( self.featureTrack2ImgFeatures,
                                                              self.featureTrack2ObjPt        ) )), "featureTrack2ImgFeatures and featureTrack2ObjPt do not share the same set of featureTracks"
        def getObjPts( nAtLeastObs ):
            return [ objPt for features,objPt in zip_equal( self.featureTrack2ImgFeatures.values(),
                                                      self.featureTrack2ObjPt      .values() )
                     if len(features) >= nAtLeastObs ]

        objPts = getObjPts(3)
        if len(objPts) < 3:
            objPts = getObjPts(2)
            if len(objPts) < 3:
                logger.warning("Not enough objPts available to estimate a plane through them.")
            return None,None

        objPts = np.array( objPts )
        # plane gives us the new z-direction - however, with not yet defined orientation
        #plane = ori.fitPlaneSVD( objPts ) # not robust!
        plane = ori.adjustPlane( objPts ) # robust!
        normal = plane.normal
        offset = plane.offset
        assert abs( 1. - linalg.norm(normal) ) < 1.e-7, "expected the plane normal to be close to unit length"

        # define the orientation of the new z-axis: cameras shall be on the positive semi-z-axis
        prcs = np.array([ self.imgs[iImg].t for iImg in self.orientedImgs ])
        try:
            medCams = stats.geometricMedian( prcs )
        except linalg.LinAlgError as ex: # if there are more than 3 points and all of them are (nearly) collinear, then geometricMedian will not converge.
            medCams = np.median( prcs, axis=0 )
        if medCams.dot( normal ) < offset:
            normal *= -1.
            offset *= -1.

        return normal,offset

    @contract
    def targetCs( self, targetCS, ptsWgs84 : 'array[Nx2]' ):
        tgtCs = osr.SpatialReference()
        if targetCS == "UTM":
            # The optimal target coordinate system for relOri would be a local tangential cartesian CS:
            # the origin is some point on the ellipsoid,
            # and the X,Y-plane is the local tangent plane of the ellipsoid at that point.
            # And the Y is local geographic north, Z is the local ellipsoid normal.
            # I.e. relOri would output cartesian coordinates whose X,Y-coordinates are meaningful, and unaffected by projection distortions.
            # GDAL uses PROJ.4 for transformations.
            # While there seem to be ways how to define a local tangential cartesian coordinate system as WKT-string,
            # GDAL is unable to translate them to PROJ.4-strings.
            # I cannot find a way either how to directly define a PROJ.4-string for a local tangential cartesian CS.

            # see: http://portal.opengeospatial.org/files/?artifact_id=999 "OpenGIS Coordinate Transformation Service Implementation Specification" 9.5 "Transform steps from NAD27 to NAD83 in California Zone 1"
            # PARAM_MT has been deprecated by "Geographic information — Well known text representation of coordinate reference systems"

            # Try with a topographic engineering CRS, see
            # http://docs.opengeospatial.org/is/12-063r5/12-063r5.html "Geographic information — Well known text representation of coordinate reference systems" 15.5.2: "Examples of WKT describing a derived engineering CRS"
                
            meanLonLat = stats.geometricMedian( ptsWgs84 )
            utmZone = crs.utmZone( meanLonLat[0] )
            tgtCs.SetWellKnownGeogCS( "WGS84" )
            tgtCs.SetUTM( utmZone, int(meanLonLat[1] >= 0.) )
        else:
            tgtCs.SetFromUserInput( targetCS )
        return tgtCs

    @contract
    def transformToPointCloud( self, targetCS, dsmFn : Path ):
        """estimate a plane through the point cloud. Transform PRC, ROT, OBJ such that
              - 'most' cameras point 'down' i.e. omega,phi ~= 0
              - the CS origin is in the center of gravity
              
           If there are enough of them, then let's use only object points that are observed at least 3 times."""
        try:
            normal,offset = self.estimateObjPlane()
            if normal is None:
                return logger.warning('Failed to estimate object plane.')

            ## use the geometric median of the point cloud as the new origin, ...
            #origin = stats.geometricMedian( objPts )
            ## ... projected onto the plane
            #diff = normal.dot(origin) - offset
            #origin -= normal*diff
            #
            ## IMPORTANT: the second oriented image has a local parameterization attached,
            ## which keeps its norm to 1.!
            ## Thus, we cannot shift the origin.
            origin = np.zeros(3)

            # compute the minimum angle rotation from the positive z-axis of the camera that defines the datum to the new positive z-axis
            z = np.array([0.,0.,1.])
            angle = np.arccos( z.dot(normal) )
            if angle * 200./np.pi < 5.:
                # The rotation axis would be unstably defined. The norm of the axis would be small, and we would divide by that small number. The angle-axis representation would be numerically unstable.
                logger.verbose( "Current z-axis and object plane normal draw an angle of only {}gon. Omitting rotation in datum transformation", angle * 200./np.pi )
                R = np.eye(3)
            else:
                axis = np.cross( z, normal )
                axis /= linalg.norm( axis )
                R = cv2.Rodrigues( axis*angle )[0]

            # Transform all object points, including those that are observed only twice.
            for el in chain( self.featureTrack2ObjPt.values(),
                            ( self.imgs[iImg].t for iImg in self.orientedImgs ) ):
                el[:] = R.T.dot( el - origin )
            for iImg in self.orientedImgs:
                img = self.imgs[iImg]
                img.omfika[:] = ori.omfika( R.T.dot(img.R) )

            # LBA may provide information that let's us additionally estimate the 3D-shift, scale, and azimuth.
            # The goal is to have a meaningful geo-reference, even if imprecise - at least the location should roughly be correct, so absOri can provide an appropriate initial view for digitizing GCPs.
            # However, LBA information is not straight-forward to interpret: longitude and latitude specify the position of the image center projected onto the ground, while hoehe is the flying height above ground.
            # Actually, longitude and latitude may not refer to the image center, but just to the object of interest (which is probably close to the center), and therefore, multiple images may have identical longitude and latitude.
            # longitude and latitude may be very coarse and thus may be identical for many or all images of a data set!
            # longitude and latitude should be present for every image in the LBA, while hoehe may be missing.
            # It may be impossible to estimate azimuth and/or scale. In that case, we still apply the shift
            # and we need to communicate to absOri that it should estimate azimuth and/or scale based on the digitized GCPs, even though relOri produced a (complete) target CS.
            lbaPoss = []
            for iImg in self.orientedImgs:
                img = self.imgs[iImg]
                if img.lbaPos and \
                   img.lbaPos.longitude is not None and \
                   img.lbaPos.latitude is not None:
                    lbaPoss.append((iImg, img.lbaPos ))
            if not lbaPoss:
                return logger.info('Object CS rotated w.r.t. object plane normal.')

            logger.info('Transform object CS according to back-projected image centers and flying heights from LBA.')
            # model CS: compute 3D coordinates of image centers projected onto the ground: intersect with ground plane.
            midPtsLocal = []
            for idx,(iImg,_) in enumerate(lbaPoss[:]):
                img = self.imgs[iImg]
                if isinstance( img.pix2cam, IdentityTransform2D ):
                    midPt = np.array([ (img.width-1)/2, -(img.height-1)/2 ])
                else:
                    midPt = np.zeros(2)
                midPt = ori.distortion( np.atleast_2d(midPt),
                                        img.ior,
                                        ori.adpParam2Struct(img.adp),
                                        ori.DistortionCorrection.undist )[0]
                midPt  = np.r_[ midPt, 0 ] - img.ior
                R = ori.euler2matrix( img.omfika )
                midPt = R.dot( midPt )
                # images and object points have already been transformed, so normal is now [0,0,1]
                factor = ( img.t[2] - offset ) /  -midPt[2]
                if factor <= 0.: # Back-projection of image center is behind camera ... image center above the horizon
                    del lbaPoss[idx]
                    continue
                midPtsLocal.append( midPt * factor + img.t ) # projected onto the plane
            if not len(midPtsLocal):
                return logger.warning('All image centers point above the horizon. Cannot compute back-projections.')
            midPtsLocal = np.array(midPtsLocal)

            # target CS: compute 3D coordinates of image centers projected onto the ground.
            mgi = osr.SpatialReference()
            mgi.ImportFromEPSG(4312)
            wgs84 = osr.SpatialReference()
            wgs84.SetWellKnownGeogCS( 'WGS84' )
            mgi2wgs84 = osr.CoordinateTransformation( mgi, wgs84 )
            midPtsMgi = np.atleast_2d([ (lbaPos.longitude,lbaPos.latitude) for _,lbaPos in lbaPoss ])
            midPtsWgs84 = np.array( mgi2wgs84.TransformPoints( midPtsMgi.tolist() ) )[:,:2]
            tgtCs = self.targetCs( targetCS, midPtsWgs84 )
            mgi2tgt = osr.CoordinateTransformation( mgi, tgtCs )
            midPtsGlobal = np.array( mgi2tgt.TransformPoints( midPtsMgi.tolist() ) ) # osr always adds the z-coordinate as 3rd column. Here, we are happy with it.
            infoDsm = utils.dsm.info( dsmFn )
            if not infoDsm.projection:
                raise Exception("Surface model is not geo-referenced.")
            infoDsm.projection = utils.crs.fixCrsWkt( infoDsm.projection )
            dsmCs = osr.SpatialReference( wkt = infoDsm.projection )
            if not dsmCs.IsProjected():
                raise Exception( "Surface model lacks a projected coordinate system: '{}'.", dsmCs.ExportToWkt() )
            midPtsGlobal[:,2] = utils.gdal.interpolateRasterHeights( infoDsm.path, midPtsGlobal[:,:2], objPtsCS=tgtCs.ExportToWkt(), interpolation=utils.gdal.Interpolation.bilinear )
            
            assert len(midPtsLocal) == len(midPtsGlobal)
            ctrGlobal = stats.geometricMedian(midPtsGlobal)
            ctrLocal  = stats.geometricMedian(midPtsLocal)
            azimuth = 0.
            scale = 1.
            if len(midPtsGlobal) > 1:
                midPtsGlobal -= ctrGlobal
                midPtsLocal  -= ctrLocal
                azimuths = []
                for midPtGlobal,midPtLocal in zip_equal( midPtsGlobal, midPtsLocal ):
                    if midPtGlobal.any() and midPtLocal.any():
                        azimuthGlobal = np.arctan2( midPtGlobal[1], midPtGlobal[0] )
                        azimuthLocal  = np.arctan2( midPtLocal [1], midPtLocal [0] )
                        azimuths.append( azimuthGlobal - azimuthLocal )
                if azimuths:
                    azimuth = stats.circular_median( azimuths )
                else:
                    logger.warning('All image centers have identical planar positions. Cannot estimate azimuth.')
                    tgtCs.SetAttrValue('PROJCS|ROTATION_IS_VALID','0') # communicate to absOri that the rotation is invalid. osgeo.crs.ExportToWkt() prints this, so we use this for embedding our information.

                scales = []
                for midPtGlobal,midPtLocal in zip_equal( midPtsGlobal, midPtsLocal ):
                    lenGlobal = linalg.norm(midPtGlobal)
                    lenLocal  = linalg.norm(midPtLocal)
                    if lenGlobal and lenLocal: # exclude points at origin
                        scales.append( lenGlobal / lenLocal )
                # add scales according to LBA flying height
                for iImg,lbaPos in lbaPoss:
                    img = self.imgs[iImg]
                    heightLocal = img.t[2] - offset
                    if lbaPos.hoehe and heightLocal:
                        # LBA stores HOCH in meters. Here, we assume that the vertical axis of the target CS has the same unit. We might check that and scale accordingly.
                        scales.append( lbaPos.hoehe / heightLocal )
                if scales:
                    scale = np.median(scales)
                else:
                    logger.warning('All image centers have identical planar positions, and all flying heights are zero. Cannot estimate scale.')
                    tgtCs.SetAttrValue('PROJCS|SCALE_IS_VALID','0') # communicate to absOri that the scale is invalid.

            # reduce to ctrLocal, then apply azimuth & scale, add ctrGlobal
            R = ori.rz( azimuth * 200. / np.pi )
            for el in chain( self.featureTrack2ObjPt.values(),
                            ( self.imgs[iImg].t for iImg in self.orientedImgs ) ):
                el[:] = R.dot( el - ctrLocal ) * scale + ctrGlobal
            for iImg in self.orientedImgs:
                img = self.imgs[iImg]
                img.omfika[:] = ori.omfika( R.dot(img.R) )
            
            return tgtCs

        except Exception as ex:
            logger.warning( "Datum transformation failed:\n{}", str(ex) )

    @contract
    def transformToAPrioriEOR( self,
                               targetCS,
                               stdDevPos : 'array[3](float)',
                               stdDevRot : 'array[3](float)',
                               dsmFn : Path ) -> 'SpatialReference | None':
        """
        Aerials may feature a priori PRC positions either from LBA (flight plan), or from a GNSS receiver (with the antenna either attached to the camera, or placed close to it).
        In addition to PRC positions, aerials may feature image rotations w.r.t. to a local tangential system from an IMU.
        If rotations are unavailable for all aerials,
          - and if the aerials were taken along a straight line, then the estimates of omega,phi will be imprecise.
          - if there are less than 3 aerials with PRC positions, then the rotations will be undetermined.
        We neglect the case that aerials have rotations, but not any PRC position is available.
        """

        # the following fails if none of the images has prcWgs84APriori: need more than zero values to unpack
        #PRCsWGS84, PRCsLocal, shortNamesWGps = zip_equal(
        #    *[ (self.imgs[iImg].prcWgs84APriori, self.imgs[iImg].t, self.imgs[iImg].shortName)
        #        for iImg in self.orientedImgs if self.imgs[iImg].prcWgs84APriori is not None ])
        PRCsWGS84, PRCsLocal, shortNamesWGps = [], [], []
        omFiKasWGS84Tangent, omFiKasLocal, omFiKaShortNames = [], [], []
        for iImg in self.orientedImgs:
            img = self.imgs[iImg]
            if img.nReconstObjPts < 10:
                continue
            if img.prcWgs84APriori is not None:
                PRCsWGS84.append( img.prcWgs84APriori )
                PRCsLocal.append( img.t )
                shortNamesWGps.append( img.shortName )

            if img.prcWgsTangentOmFiKaAPriori is not None:
                omFiKasWGS84Tangent.append( img.prcWgsTangentOmFiKaAPriori )
                omFiKasLocal.append( img.omfika )
                omFiKaShortNames.append( img.shortName )

        nWithPos = len(PRCsWGS84)

        if nWithPos < 3: # 4 points would be needed for initialization with affine 3d trafo: 12 unknowns to be estimated
            # TODO: it's okay to either provide full georeferencing, or none (i.e.: don't compute a shift based on a single PRC, leaving the azimuth undetermined)
            # However, we may provide full georeferencing even if only 1 image has full a priori EOR (PRC+ROT)!
            logger.info( "Too few PRC positions available:  {}", nWithPos )
            return self.transformToPointCloud(targetCS, dsmFn)

        logger.info( "A priori EOR data\n"
                     "type\tnImages\n"
                     "position\t{}\n"
                     "rotation\t{}",
                     nWithPos,
                     len(omFiKasWGS84Tangent) )

        test = False
        if test:
            test_featureTrack, test_imgFeatures = next(iter(self.featureTrack2ImgFeatures.items()))
            test_objPt = self.featureTrack2ObjPt[test_featureTrack]
            test_cam = self.imgs[test_imgFeatures[0][0]]
            test_projection = ori.projection( test_objPt, test_cam.t, test_cam.omfika, test_cam.ior )
            test_residualBefore = test_cam.keypoints[ test_imgFeatures[0][1] ][:2] - test_projection
        try:
            srcCs = osr.SpatialReference()
            srcCs.SetWellKnownGeogCS( 'WGS84' ) # srcCs.ImportFromEPSG( 4326 )
            tgtCs = self.targetCs( targetCS, np.array(PRCsWGS84)[:,:2] )
            logger.info( "Transforming to: {}", "EPSG:{}".format(tgtCs.GetAuthorityCode("PROJCS")) if tgtCs.GetAuthorityCode("PROJCS") else tgtCs.ExportToProj4() )
            csTrafo = osr.CoordinateTransformation( srcCs, tgtCs )
            PRCsGlobal = [ np.array( csTrafo.TransformPoint( *prcWgs84 ) ) for prcWgs84 in PRCsWGS84 ]
            # Find initial values for sim-trafo using affine-trafo
            #designMatrix = np.zeros( (nWithPos*3,12) )
            #obsVector = np.zeros( nWithPos*3 )
            #for iPRCGlobal,PRCGlobal in enumerate(PRCsGlobal):
            #    for iCoo in range(3):
            #        designMatrix[ iPRCGlobal*3 + iCoo, iCoo] = 1
            #        designMatrix[ iPRCGlobal*3 + iCoo, iCoo*3+3 : iCoo*3+6 ] = PRCsLocal[iPRCGlobal]
            #    obsVector[iPRCGlobal*3:iPRCGlobal*3+3]=PRCGlobal
            #
            #spatialAffineTrafo = linalg.lstsq( designMatrix, obsVector )[0]
            #P0 = spatialAffineTrafo[:3]
            #mat = np.reshape( spatialAffineTrafo[3:], (3,3) )
            #scale = np.sum( mat**2 )**.5 / 3.**.5 # (4.2-2) or: np.trace( mat.T.dot(mat) )**.5 / 3**.5
            #mat /= scale
            #omfika = ori.omfika( mat )
            #R = ori.omfika( omfika )
            #if 0:
            #    for iPRCGlobal,PRCGlobal in enumerate(PRCsGlobal):
            #        res = PRCGlobal - ( P0 + scale * R.dot( PRCsLocal[iPRCGlobal] ) )
            #scale = np.array([1./scale]) # prepare for usage of R.T to transform from object to model space

            # for robustness, replace with geometric median of PRCs, L1 rotation averaging, and median of scales. Also, only 3 correspondences will be needed then (instead of 4).
            PRCsLocal = np.array(PRCsLocal)
            PRCsGlobal = np.array(PRCsGlobal)
            # y=s*R.dot(x-x0)
            scale, Rt, P0, res = ori.similarityTrafo( x=PRCsGlobal, y=PRCsLocal )
            omfika = ori.omfika( Rt.T )
            scale = np.array([scale])

            # TODO: images may be oriented completely wrongly, so make this robust! Either by using a robust loss, or by using only a subset of the oriented images (with reliable pts)
            trafoProbl = adjust.Problem()
            trafoLoss = adjust.loss.Trivial()
            for PRCLocal,PRCGlobal in zip_equal(PRCsLocal,PRCsGlobal):
                # P = R.dot( p / s ) + P0
                cost = adjust.cost.SimTrafo3d( p=PRCLocal, P=PRCGlobal, stdDevsP=stdDevPos )
                trafoProbl.AddResidualBlock( cost,
                                             trafoLoss,
                                             P0,
                                             omfika,
                                             scale )

            # R_tangential = R_sim * R_model
            # -> aus IMU-Winkel R_tangential aufbauen
            for omFiKaWGS84Tangent, omFiKaLocal in zip_equal(omFiKasWGS84Tangent, omFiKasLocal):
                cost = adjust.cost.ObservedOmFiKa( omFiKaLocal, omFiKaWGS84Tangent, stdDevRot )
                trafoProbl.AddResidualBlock( cost,
                                             trafoLoss,
                                             omfika )

            options = adjust.Solver.Options()   
            summary = adjust.Solver.Summary()
            P0_orig, omfika_orig, scale_orig = P0.copy(), omfika.copy(), scale.copy()
            adjust.Solve(options, trafoProbl, summary)
            if not adjust.isSuccess(summary.termination_type):
                raise Exception("Adjustment of similarity transformation has not converged")
            else:
                covOpts = adjust.Covariance.Options()
                covariance = adjust.Covariance( covOpts )
                paramBlockPairs = [ (omfika,omfika) ]
                covariance.Compute( paramBlockPairs, trafoProbl )
                cofactorBlock = covariance.GetCovarianceBlock( omfika, omfika )
                redundancy = summary.num_residuals_reduced - summary.num_effective_parameters_reduced
                sigma0 = ( summary.final_cost * 2 / redundancy ) **.5 if redundancy > 0 else np.nan
                stdDevs = sigma0 * np.diag(cofactorBlock)**0.5
                # check stdDevs. if sigma_om or sigma_phi are too large (because of no rotation observations available, and cameras along a straight line in space),
                # then incorporate the object plane.
                maxStdDevOmFi = 10
                localZDirRes = None
                if np.any( stdDevs[:2] > maxStdDevOmFi ):
                    logger.info( 'Precisions of \N{GREEK SMALL LETTER OMEGA} and / or \N{GREEK SMALL LETTER PHI} are worse than {}gon. Introduce object plane normal as observation.', maxStdDevOmFi )
                    normal,offset = self.estimateObjPlane()
                    if normal is not None:
                        # TODO: we should weight this observation, too. But how? As it is introduced only in case of weakly determined z-direction, weighting is less important.
                        cost = adjust.cost.LocalZDirection( normal )
                        localZDirRes = trafoProbl.AddResidualBlock( cost,
                                                                    trafoLoss,
                                                                    omfika )
                        P0[:] = P0_orig
                        omfika[:] = omfika_orig
                        scale[:] = scale_orig
                        adjust.Solve(options, trafoProbl, summary)
                        if not adjust.isSuccess(summary.termination_type):
                            raise Exception("Adjustment of similarity transformation has not converged")

        
                residuals, = trafoProbl.Evaluate()
                residualsPos = residuals[:nWithPos*3].reshape((-1,3)) * stdDevPos # un-weight the residuals
                residualsRot = residuals[nWithPos*3: -1 if localZDirRes is not None else None ].reshape((-1,3)) * stdDevRot
                if residualsPos.size:
                    # provide statistics for X,Y,Z separately
                    logger.info( 'PRC residual statistics (target system) [m]\n'
                                 'statistic\tX\tY\tZ\n'
                                 'min\t{:.3f}\t{:.3f}\t{:.3f}\n'
                                 'med\t{:.3f}\t{:.3f}\t{:.3f}\n'
                                 'max\t{:.3f}\t{:.3f}\t{:.3f}\n'
                                 'max@pho\t{}\t{}\t{}\n',
                                 #'\N{GREEK SMALL LETTER SIGMA}_0\t{sigma0:.3}\n',
                                 *chain( np.min   (residualsPos, axis=0),
                                         np.median(residualsPos, axis=0),
                                         np.max   (residualsPos, axis=0),
                                         [ shortNamesWGps[idx] for idx in np.argmax(residualsPos, axis=0) ] )
                                )
                if residualsRot.size:
                    # the same for om,phi,ka
                    logger.info( 'ROT residual statistics (target system) [gon]\n'
                                 'statistic\t\N{GREEK SMALL LETTER OMEGA}\t\N{GREEK SMALL LETTER PHI}\t\N{GREEK SMALL LETTER KAPPA}\n'
                                 'min\t{:.3f}\t{:.3f}\t{:.3f}\n'
                                 'med\t{:.3f}\t{:.3f}\t{:.3f}\n'
                                 'max\t{:.3f}\t{:.3f}\t{:.3f}\n'
                                 'max@pho\t{}\t{}\t{}\n',
                                 *chain( np.min   (residualsRot, axis=0),
                                         np.median(residualsRot, axis=0),
                                         np.max   (residualsRot, axis=0),
                                         [ omFiKaShortNames[idx] for idx in np.argmax(residualsRot, axis=0) ] )
                                )

            #apply the similarity trafo
            R = ori.omfika( omfika )
            for img in [ self.imgs[iImg] for iImg in self.orientedImgs ]: #omfika
                img.t[:] = P0 + 1./scale * R.dot( img.t )
                img.omfika[:] = ori.omfika( R.dot( img.R ) )
            for objPt in self.featureTrack2ObjPt.values():
                objPt[:] = P0 + 1./scale * R.dot( objPt )
        except:
            logger.warning( "Datum transformation failed:\n{}", ''.join( traceback.format_exc() ) or 'Exception raised, but traceback unavailable' )

        if test:
            test_projection = ori.projection( test_objPt, test_cam.t, test_cam.omfika, test_cam.ior )
            test_residualAfter = test_cam.keypoints[ test_imgFeatures[0][1] ][:2] - test_projection
            assert np.abs( test_residualBefore - test_residualAfter ).max() < 1

        return tgtCs

    @contract
    def objSpaceBbox( self ) -> 'tuple( array[3](float), array[3](float) )':
        minObj = np.array([np.inf,np.inf,np.inf])
        maxObj = -minObj.copy()
        for obj in self.featureTrack2ObjPt.values():  
            for idx in range(3):
                minObj[idx] = min( minObj[idx], obj[idx] )
                maxObj[idx] = max( maxObj[idx], obj[idx] )
        return minObj,maxObj

    def cleanUp( self ):
        logger.verbose('Delete temporary files')
        fn = os.path.join( self.outDir, 'features.h5' )
        try:
            os.remove( fn )
        except:
            logger.warning( 'File could not be deleted: {}', fn )
