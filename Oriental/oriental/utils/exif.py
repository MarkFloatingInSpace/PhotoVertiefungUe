# -*- coding: cp1252 -*-
import contextlib, datetime, json, os, re, subprocess, tempfile

import numpy as np
from contracts import contract, new_contract

from oriental import log, ori, utils
import oriental.utils.filePaths

logger = log.Logger("relOri")

def exifGPSStringToGeogKoord(s):
    m = re.match( r"\((?P<deg>\d+\.?\d*)\)\s*\((?P<min>\d+\.?\d*)\)\s*\((?P<sec>\d+\.?\d*)\)\s*", s) # is string!
    if not m:
        raise Exception("Couldn't parse EXIF string to GPS coordinate: {}".format(s) )
    d = m.groupdict()
    coord = float(d["deg"]) + float(d["min"])/60 + float(d["sec"])/3600
    return coord

class PhoInfo(object):
    make = None # str
    model = None # str
    focalLength_px = None # float
    focalLength_mm = None # float
    ccdWidthHeight_px = None # np.array( dtype=np.int, shape=(2,) )
    ccdWidthHeight_mm = None # np.array( dtype=np.float, shape=(2,) )
    timestamp = None # datetime.date
    prcWgs84 = None
    prcWgsTangentOmFiKa = None

new_contract('PhoInfo', lambda x: isinstance(x, PhoInfo) )

@contract
def phoInfo( imgFns         : 'seq[N](str)',
             addAttribs     : 'seq(str)' = [] # query additional Exif attributes, to avoid the need to call Exiftool again for that purpose
           ) -> 'list[N](PhoInfo)':

    """compute approximate focal length [px] from EXIF Tags
     mind: XResolution, YResolution, ResolutionUnit (2==Inch), PixelXDimension, PixelYDimension, FocalLength [mm]
    
    EXIF 2.3:
        
    XResolution 
        The number of pixels per ResolutionUnit in the ImageWidth direction. When the image resolution is unknown, 72 [dpi] is designated.
        note wk: The meaning of XResolution seems unclear according to Exif 2.3 itself. However, according to Exif 2.3, XResolution shall be converted to Flashpix's 'Default display width' - meaning that XResolution must not be used for estimating the metric sensor size!
    YResolution 
        The number of pixels per ResolutionUnit in the ImageLength direction. The same value as XResolution is designated.
    ResolutionUnit 
        The unit for measuring XResolution and YResolution. The same unit is used for both XResolution and YResolution. 
        If the image resolution in unknown, 2 (inches) is designated. 
        3 = centimeters               
    FocalLength 
        The actual focal length of the lens, in mm. Conversion is not made to the focal length of a 35 mm film camera.     
    FocalPlaneXResolution 
        Indicates the number of pixels in the image width (X) direction per FocalPlaneResolutionUnit on the camera focal plane.
    FocalPlaneYResolution 
        Indicates the number of pixels in the image height (Y) direction per FocalPlaneResolutionUnit on the camera focal plane.   
    FocalPlaneResolutionUnit 
        Indicates the unit for measuring FocalPlaneXResolution and FocalPlaneYResolution. This value is the same as the ResolutionUnit.
        Default=2 (inch)  
    FocalLengthIn35mmFilm
        This tag indicates the equivalent focal length assuming a 35mm film camera, in mm. A value of 0 means the focal length is unknown.
    PixelXDimension 
        Information specific to compressed data. When a compressed file is recorded, the valid width of the meaningful 
        image shall be recorded in this tag, whether or not there is padding data or a restart marker. This tag should not 
        exist in an uncompressed file.  
    PixelYDimension 
        Information specific to compressed data. When a compressed file is recorded, the valid height of the meaningful 
        image shall be recorded in this tag, whether or not there is padding data or a restart marker. This tag should not 
        exist in an uncompressed file.  
        
        1 inch = 2.54 cm           
        
    Possible alternative: ExifTool by Phil Harvey
    http://www.sno.phy.queensu.ca/~phil/exiftool/
    e.g.:
    exiftool -json -short -forcePrint -ScaleFactor35efl 02090601_011.JPG
    -short -> Prints tag names instead of descriptions.
    -forcePrint -> print '-' for nonexistent tags' values
    -json -> print output in JSON-format, which can easily be parsed in Python
    """

    
    # The Exif-'standard' seems to not specify any obligatory attributes - camera makers are free to supply any subset of the attributes defined by Exif.
    # Thus, rely on as few Exif attributes as possible - e.g. do not use PixelXDimension & PixelYDimension, but rely on GDAL to extract the correct values.
    
    # Note: ExifTool offers 'composite tags', which are not stored in files, but which it derives from actual meta data:
    # ImageWidth, ImageHeight, ScaleFactor35efl, ... http://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/Composite.html
    # http://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/Extra.html

    # ExifTool cannot read all formats that we need. Most importantly, MrSID, which is output by the RMK-scanner.
    # If ExifTool cannot read a file, then try extracting at least the image resolution in pixels using gdal.
    # The other meta data of scanned MrSID files doesn't matter for our purpose.
    try:
        args = ([
            # FILE:ImageWidth is the number of columns, irrespective of eventual meta information as by Exif, etc.
            # However, ExifTool does not report that value for some files, but instead only EXIF:ImageWidth (in addition to EXIF:ExifImageWidth) - e.g. Carnuntum_UAS_Geert\3240438_DxO_no_dist_corr.tif
            # Thus, simply query ImageWidth without a specific group.
            # With -json, duplicates are suppressed (if found in more than 1 group), unless -groupHeadings is specified. Which duplicate wins, seems unspecified.
            '-ImageWidth',
            '-ImageHeight',
            '-Make',
            '-Model',
            '-FocalLength',
            '-FocalPlaneXResolution',
            '-FocalPlaneYResolution',
            '-FocalPlaneResolutionUnit',
            '-FocalLengthIn35mmFormat', # as noted in the docs of ExifTool, this tag is actually called 'FocalLengthIn35mmFilm' by the EXIF spec.
            '-DateTime',
            '-DateTimeOriginal',
            '-DateTimeDigitized',
            '-EXIF:GPSLongitude',
            '-EXIF:GPSLatitude',
            '-EXIF:GPSAltitude',
            '-EXIF:GPSImgDirection', # yaw
            '-EXIF:GPS_0xd000', # pitch
            '-EXIF:GPS_0xd001', # roll
            '-json',
            '--printConv', # this makes ExifTool output numerical values as such, instead of numbers as text with their unit appended.
            #  E.g. FocalLength -> 35.0 instead of '35.0 mm'
            #       FocalPlaneResolutionUnit -> 3 instead of 'cm'
            '-quiet' # don't clutter the screen of our main application with messages from ExifTool like '276 image files read' 
          ]
          + [ '-' + el for el in addAttribs ]
          + ['-unknown']
          + imgFns )
        # Don't supply the arguments on the commandline, but in a temporary file,
        # because the concatenation of all image file paths may exceed the maximum length of a commandline string, which may be e.g. only 8191 characters.
        with tempfile.NamedTemporaryFile(mode='wt', encoding='utf8', delete=False) as fout:
            fout.write( '\n'.join(args) )
        try:
            all_attributes = subprocess.run(
                ['exiftool', '-charset', 'filename=utf8', '-@', fout.name], # read contents of file arguments with UTF8-encoding. So we can read exif info from files with non-ascii paths. However, if OrientAL is install in a non-ASCII-path, then exiftool will still simply not start up!
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, # otherwise, ExifTool clutters the screen e.g. when it encounters an unknown file format
                check=True, 
                universal_newlines=True ).stdout
        finally:
            with contextlib.suppress(OSError):
                os.remove( fout.name )
    # If subprocess was unable to open exiftool.exe, it throws OSError.
    # CalledProcessError is raised if exiftool itself returned anything but zero, probably because it could not read the file - maybe because it does not support the format, e.g. MrSID.
    #   In any case, it reports valid JSON, just that the JSON won't have the attributes we need. If attributes are missing, fall back to GDAL.
    except subprocess.CalledProcessError as ex:
        all_attributes = ex.output
    all_attributes = json.loads( all_attributes )
    results = []
    focalPlaneResolutionsDiffer = {}
    for iAttrib,attributes in enumerate(all_attributes):
        res = PhoInfo()
        results.append(res)
        if addAttribs:
            res.addAttribs = {}
            for addAttrib in addAttribs:
                addAttribName = addAttrib.split(':')[-1]
                res.addAttribs[addAttribName] = attributes.get( addAttribName )
        
        prcWgs84 = [ attributes.get('GPSLongitude'), attributes.get('GPSLatitude'), attributes.get('GPSAltitude') ]
        if all(( el is not None for el in prcWgs84 )):
            res.prcWgs84 = np.array(prcWgs84,float)

        if all( el in attributes for el in 'GPS_0xd001 GPS_0xd000 GPSImgDirection'.split() ):
            # angles in Exif are in degrees!
            om = attributes['GPS_0xd001']     / 180. * np.pi
            fi = attributes['GPS_0xd000']     / 180. * np.pi
            ka = attributes['GPSImgDirection'] / 180. * np.pi

            som = np.sin(om)
            com = np.cos(om)
            sfi = np.sin(fi)
            cfi = np.cos(fi)
            ska = np.sin(ka)
            cka = np.cos(ka)

            R_om = np.array([ [   1.,  0.,   0. ],
                              [   0., com, -som ],
                              [   0., som,  com ] ] )

            R_fi  = np.array([ [  cfi,  0.,  sfi ],
                               [   0.,  1.,   0. ],
                               [ -sfi,  0.,  cfi ] ] )

            R_ka = np.array([ [ cka, -ska,  0. ],
                              [ ska,  cka,  0. ],
                              [  0.,   0.,  1. ] ] )

            R = R_ka @ R_fi @ R_om

            R = R.dot( np.array([[  0., +1., 0. ],
                                 [ -1.,  0., 0. ],
                                 [  0.,  0., 1. ]])  # rot about z by +90°
                ).dot( np.array([[  1.,  0.,  0. ],
                                 [  0.,  0.,  -1. ],
                                 [  0., +1.,  0. ]]) ) # rot about x by -90°

            res.prcWgsTangentOmFiKa = ori.omfika( R )

        def parseTimeStr( timeStr ):
            fmts = ( "%Y:%m:%d %H:%M:%S", # format according to Exif 2.3 standard
                     "%Y-%m-%dT%H:%M:%S" ) # format used by ISPRS image orientation benchmark dataset
            for fmt in fmts:
                try:
                    return datetime.datetime.strptime( timeStr.strip(), fmt )
                except ValueError:
                    pass
            raise Exception( "Exif datetime '{}' does not adhere to any of the supported formats: {}".format( timeStr.strip(), ', '.join(fmts) ) )

        imageWidth, imageHeight = attributes.get('ImageWidth'), attributes.get('ImageHeight')
        if all(( el is not None for el in (imageWidth, imageHeight) )):
            res.ccdWidthHeight_px = np.array([ imageWidth, imageHeight ])
            res.make                 = attributes.get( 'Make' )
            res.model                = attributes.get( 'Model' )
            res.focalLength_mm       = attributes.get( 'FocalLength' )
            focalPlaneXResolution    = attributes.get( 'FocalPlaneXResolution' )
            focalPlaneYResolution    = attributes.get( 'FocalPlaneYResolution' )
            focalPlaneResolutionUnit = attributes.get( 'FocalPlaneResolutionUnit', 2 )
            focalLengthIn35mmFilm_mm = attributes.get( 'FocalLengthIn35mmFormat' )
            res.timestamp = attributes.get( 'DateTimeOriginal',
                                            attributes.get( 'DateTimeDigitized',
                                                            attributes.get( 'DateTime' ) ) )
        
            if res.timestamp is not None:
                res.timestamp = parseTimeStr( res.timestamp )
        
            if focalPlaneXResolution is not None and \
               focalPlaneYResolution is not None:
                
                INCH_IN_MM = 25.4
                if focalPlaneXResolution != focalPlaneYResolution: # otherwise, we'd need to either use 2 focal lengths, or resample the images
                    focalPlaneResolutionsDiffer.setdefault( (focalPlaneXResolution, focalPlaneYResolution, focalPlaneResolutionUnit), [] ).append( iAttrib )
    
                # 0 actually means 'unknown'. However, the Exif standard says 'If the image resolution is unknown, 2 (inches) shall be designated'. So let's default to inches in that case.
                if focalPlaneResolutionUnit in (0,2): # inch
                    fac = INCH_IN_MM
                elif focalPlaneResolutionUnit==3: # cm
                    fac = 10.
                elif focalPlaneResolutionUnit==4: # mm, non-standard, but e.g. also used by ExifTool
                    fac = 1.
                elif focalPlaneResolutionUnit==5: # µm, non-standard, but e.g. also used by ExifTool
                    fac = 1./1000
                else:
                    raise Exception( "FocalPlaneResolutionUnit not implemented: '{}'".format(focalPlaneResolutionUnit) )
    
                res.ccdWidthHeight_mm = np.array([ res.ccdWidthHeight_px[0] * fac / focalPlaneXResolution, 
                                                   res.ccdWidthHeight_px[1] * fac / focalPlaneYResolution ])
                                  
            elif focalLengthIn35mmFilm_mm is not None and \
                 res.focalLength_mm is not None:
                # 35mm is the width of a standard 35mm film resulting in classical still images of size 36mm x 24mm
                # TODO: what about portrait format images? Does GDAL consider the according flag when reading image data?
                # Let's assume landscape format for now.
                ccdHeight_mm =  24. / focalLengthIn35mmFilm_mm * res.focalLength_mm
                res.ccdWidthHeight_mm = np.array([ ccdHeight_mm / res.ccdWidthHeight_px[1] * res.ccdWidthHeight_px[0], 
                                                   ccdHeight_mm ])
        else:
            # GDAL treats EXIF-tags differently for JPEG and TIFF files:
            # while for JPEG files, EXIF-tags are stored in the 'default domain' (undocumented; to be accessed with ds.GetMetadata_Dict()),
            # for TIFF-files, it stores them in the 'EXIF'-domain (as documented)
            # -> query both domains, and merge the returned dicts.
            from osgeo import gdal
            ds = gdal.Open( imgFns[iAttrib], gdal.GA_ReadOnly )
            meta = ds.GetMetadata_Dict().copy()
            meta.update( ds.GetMetadata_Dict('EXIF') )
            if "EXIF_PixelXDimension" in meta:
                assert ds.RasterXSize==int(meta["EXIF_PixelXDimension"]) # otherwise, the image may have been resized
            if "EXIF_PixelYDimension" in meta:
                assert ds.RasterYSize==int(meta["EXIF_PixelYDimension"])
       
            res.ccdWidthHeight_px = np.array([ ds.RasterXSize, ds.RasterYSize ])
            for idx in range(1): # Pillow may be able to extract more information. However, let's avoid that additional dependency.
                # GDAL prepends 'EXIF_' to the EXIF tag names
                infix = ("EXIF_","")
    
                # EXIF_FocalPlaneXResolution, EXIF_FocalPlaneYResolution, EXIF_FocalPlaneResolutionUnit
                # seem to be the right attributes to inspect. These values are stored e.g. by Canon DIGITAL IXUS 400.
                # However, e.g. Nikon D90 seems to not store them.
                # In that case, let's inspect 'FocalLengthIn35mmFilm'
                Make                     = "{}Make"                    .format( infix[idx] )
                Model                    = "{}Model"                   .format( infix[idx] )
                DateTime                 = "{}DateTime"                .format( infix[idx] ) # TODO: consider Exif-tag 'SubsecTime', eventually
                FocalLength              = "{}FocalLength"             .format( infix[idx] )
                FocalPlaneXResolution    = "{}FocalPlaneXResolution"   .format( infix[idx] )
                FocalPlaneYResolution    = "{}FocalPlaneYResolution"   .format( infix[idx] )
                FocalPlaneResolutionUnit = "{}FocalPlaneResolutionUnit".format( infix[idx] )
                FocalLengthIn35mmFilm    = "{}FocalLengthIn35mmFilm"   .format( infix[idx] )
                DateTimeOriginal         = "{}DateTimeOriginal"        .format( infix[idx] )
                DateTimeDigitized        = "{}DateTimeDigitized"       .format( infix[idx] )
        
                if idx==0:
                    meta_ = meta
                else:
                    # even though GDAL (v.1.10.1) extracts Exif tags (in the default or in the 'Exif'-domain, depending on the file format),
                    # it extracts only TIFF tags that are known by GDAL;
                    # 'Model', 'Make' are TIFF tags unknown by GDAL, and thus, it is not possible to extract those tags from TIFF files using GDAL - e.g. for Olympus E-P2, see D:\arap\data\Carnuntum_UAS_Geert
                    # In that case, use PIL
                    from PIL import Image, TiffImagePlugin, ExifTags
                    img = Image.open( imgFns[iAttrib] )
                    if not hasattr(img,'tag'):
                        continue # no Exif tags present
                    meta_ = img.tag.named().copy()
                    ExifIFD = img.tag.get( TiffImagePlugin.EXIFIFD )
           
                    if ExifIFD is not None:
                        img.fp.seek( ExifIFD[0] )
                        img.tag.load( img.fp )
                        #img.tag.named() # -> transform with ExifTags.py
                        metaExif = { ExifTags.TAGS.get( k, k ) : v for k,v in img.tag.items()  }
                        meta_.update( metaExif )
                
                if not set(( Make, Model, FocalLength )).issubset( meta_ ):
                    continue
    
                def getASCII( value ):
                    if value is None:
                        return None
                    return value.strip()
    
                def getRational( value ):
                    if value is None:
                        return None
                    if idx==0:
                        # GDAL encloses EXIF-values of type 'RATIONAL' with round braces.
                        return float( value[1:-1] )
                    return float( value[0][0] ) / value[0][1]
            
                def getShort( value ):
                    if value is None:
                        return None
                    if idx==0:
                        return int(value)
                    return int(value[0])
        
                res.make                 = getASCII( meta_[Make] )
                res.model                = getASCII( meta_[Model] )
                res.focalLength_mm       = getRational( meta_.get(FocalLength) )
                focalPlaneXResolution    = getRational( meta_.get(FocalPlaneXResolution) )
                focalPlaneYResolution    = getRational( meta_.get(FocalPlaneYResolution) )
                focalPlaneResolutionUnit = getShort( meta.get( FocalPlaneResolutionUnit, 2 ) )
                focalLengthIn35mmFilm_mm = getShort( meta_.get( FocalLengthIn35mmFilm ) )
                res.timestamp = getASCII( meta_.get( DateTimeOriginal,
                                                     meta_.get( DateTimeDigitized,
                                                                meta_.get( DateTime ) ) ) )
        
                if res.timestamp is not None:
                    res.timestamp = parseTimeStr( res.timestamp )
        
                if focalPlaneXResolution is not None and \
                   focalPlaneYResolution is not None:
                
                    INCH_IN_MM = 25.4
                    if focalPlaneXResolution != focalPlaneYResolution: # otherwise, we'd need to either use 2 focal lengths, or resample the images
                        focalPlaneResolutionsDiffer.setdefault( (focalPlaneXResolution, focalPlaneYResolution, focalPlaneResolutionUnit), [] ).append( iAttrib )
    
                    # 0 actually means 'unknown'. However, the Exif standard says 'If the image resolution is unknown, 2 (inches) shall be designated'. So let's default to inches in that case.
                    if focalPlaneResolutionUnit in (0,2): # inch
                        fac = INCH_IN_MM
                    elif focalPlaneResolutionUnit==3: # cm
                        fac = 10.
                    elif focalPlaneResolutionUnit==4: # mm, non-standard, but e.g. also used by ExifTool
                        fac = 1.
                    elif focalPlaneResolutionUnit==5: # µm, non-standard, but e.g. also used by ExifTool
                        fac = 1./1000
                    else:
                        raise Exception( "FocalPlaneResolutionUnit not implemented: '{}'".format(focalPlaneResolutionUnit) )
    
                    res.ccdWidthHeight_mm = np.array([ res.ccdWidthHeight_px[0] * fac / focalPlaneXResolution, 
                                                       res.ccdWidthHeight_px[1] * fac / focalPlaneYResolution ])
                                  
                elif focalLengthIn35mmFilm_mm is not None:
                    # 35mm is the width of a standard 35mm film resulting in classical still images of size 36mm x 24mm
                    # TODO: what about portrait format images? Does GDAL consider the according flag when reading image data?
                    # Let's assume landscape format for now.
                    ccdHeight_mm =  24. / focalLengthIn35mmFilm_mm * res.focalLength_mm
                    res.ccdWidthHeight_mm = np.array([ ccdHeight_mm / res.ccdWidthHeight_px[1] * res.ccdWidthHeight_px[0], 
                                                       ccdHeight_mm ])
                else:
                    res.ccdWidthHeight_mm = None
        
                break                    

    if focalPlaneResolutionsDiffer:
        shortFileNames = utils.filePaths.ShortFileNames( imgFns )
        logger.warning( 'OrientAL assumes images with square pixels. However, Exif says that focal plane x- and y-resolutions differ:\n{}{}',
                        'x-resolution\ty-resolution\tUnit\tAffected photos\n',
                        '\n'.join( '\t'.join( '{}'.format(el) for el in (key[0],key[1],'[pix/' + ('inch' if key[2]==2 else 'cm') + ']') ) + '\t' +
                                   (', '.join(shortFileNames(imgFns[iImg]) for iImg in iImgs) if len(iImgs) < len(imgFns) else '<all>' )
                                   for key, iImgs in focalPlaneResolutionsDiffer.items() ) )


    return results
