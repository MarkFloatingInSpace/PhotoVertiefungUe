# -*- coding: cp1252 -*-
"""
General considerations:

Considering that all current cameras directly create digital images, 
for which the relation between the pixel CS and the camera frame is considered constant,
it seems convincing to compute the fiducial trafo, and then transform all parameters given in the camera CS by a calibration protocol
into respective parameters in the pixel CS (and hence forget about the camera CS).
While this is possible for the principal point's position in any case,
for the focal length, this is only possible for fiducial transformation models that have a constant scale!
For distortion parameters, this is generally impossible or at least cumbersome (thinking about polygonal radial distortions, etc.).

Anyway, if OrientAL shall be able to not only use, but adjust parameters in the camera CS of scanned photos,
then image observations must be transformed into the camera CS, before being introduced into the adjustment.
While the adjustment of IOR and/or ADP of scanned phos may not ever be necessary for professional aerial cameras
(as we may require a calibration protocol), this will certainly be necessary for processing scanned photos from classical 35mm film
(without fiducials / reseau ).

We need not only the forward-transform from px to mm (the camera CS), but also the inverse:
e.g. the area outside the actual photo area must be masked for feature detection. This mask is defined in the camera CS,
but needs to be applied in the pixel CS.
Also, the inverse transformation is needed for plotting: e.g. residuals in image space.

For photos with 8 fiducial marks, Kraus recommends to use the bilinear transformation model (see Kraus A 3.2.1.2: Tab. 3.2-4, 7th ed.).
However, the bilinear transformation model has the serious drawback that it's inverse depends on the image position;
for computing the inverse, a quadratic equation needs to be solved.
Note that simply computing the inverse bilinear transformation based on 4 points that have been transformed
by the formerly computed forward bilinear transformation will not result in a pair of forward/inverse transformations that round-trips any points but those 4!
see e.g. http://www.fmwconcepts.com/imagemagick/bilinearwarp/FourCornerImageWarp2.pdf

Some authors also mention usage of a homography / projective transformation / collinearity as the transformation model.
Like bilinear transformation, it has 8 unknowns.
However, a homography does not map the center of the source image onto the center of the target image
 - which seems inappropriate to me in the context of fiducial transformation.

Anyway, Orpheus (chapter 6.3.5) offers only 3 transformation models:
- congruent: 3 Parameters (2 shifts and 1 rotation)
- conformal = Helmer = similarity: 4 Parameters (2 shifts, 1 rotation and 1 scale)
- affine: 6 Parameters (2 shifts, 2 rotation and 2 scales; recommended for scanned images having their fiducial marks in their corners)

Orient (chapter 24.3) additionally lists bilinear and perspective=collinear transformations, but they are marked as not implemented.
This probably indicates that bilinear/projective transformations are not common / have not proven to be advantageous.

Thus, for scanned photos, let's introduce image observations defined in the camera CS into the bundle adjustment, and use an affine transformation for fiducials.
Following Kraus, let's use an affine transformation if fiducials are available in the image corners, and
                           a similarity transformation if fiducials are available only in the middle of the image edges.

Let's use Orient's definitions (chapter 24.3):
fiducials_mm = A.dot( fiducials_px ) + b
"""

from .hasselblad import hasselblad
from .wildRc8 import wildRc8
from .zeissRmk20 import zeissRmk20
from .zeissRmkA import zeissRmkA
from .zeissRmkTop import zeissRmkTop

if __name__ == '__main__':
    if 0:
        import os
        import glob, datetime
        import sqlite3.dbapi2 as db
        from oriental import config
        os.chdir(r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\fiducials')

        interestingImage = '01750901_034.sid'
        interestingImageHasPassed = True
        nProcessed = 0
        exceptions = []
        with db.connect( config.dbLBA ) as lba:
            lba.row_factory = db.Row
            for fn in glob.glob(r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy\*'):
                ext = path.splitext(fn)[1]
                if ext.lower() not in ('.jpg','.sid','.tif','.raf','.nef'):
                    continue
                pho = path.basename(fn)
                if not interestingImageHasPassed:
                    if pho == interestingImage:
                        interestingImageHasPassed = True
                    else:
                        continue
                film = pho.split('_')[0]
                rows = lba.execute("""
                    SELECT kamera, kamnr, form1,form2, kkon
                    FROM film
                    WHERE film=?
                """, (film,) ).fetchall()
                if not len(rows):
                    continue
                row = rows[0]
                try:
                    if row['kamera'] in ('Hasselblad','Hasselbl','H553ELX','H205FCC'):
                        hasselblad( fn, plotDir='.')
                        #continue
                    elif row['kamera'] == 'Zeiss RMK':
                        continue

                        filmFormatFocal=(row['form1'],row['form2'],row['kkon'])
                        kamnr = row['kamnr'] if len(row['kamnr'].strip()) else None
                        for func in (zeissRmkTop,zeissRmkA):
                            try:
                                func( fn, camSerial=kamnr, filmFormatFocal=filmFormatFocal, plotDir='.' )
                                #pass
                            except Exception as ex:
                                exc = ex
                            else:
                                break
                        else:
                            raise exc
                        nProcessed += 1

                except Exception as ex:
                    exceptions.append(ex)
                    print( 'Exception raised while processing {}: {}'.format( fn, ex ) )

            print( '#processed: ', nProcessed )
            print( '#exceptions: ', len(exceptions) )
            if len(exceptions):
                print( 'Exceptions:' )
                for ex in exceptions:
                    print(ex)
    else:
        difficultPhos = [
            #'01750901_021.sid', # detected rotation is wrong if the threshold for 1 vs. 3 data fields is too high (20 gray values) -> 3 data fields are detected instead of 1, in wrong orientation
            #'01750901_034.sid'  # 1 data field detected (correct), but in wrong orientation: median in forest area is lower than in image serial data field (1 vs. 0 gray value).
            #'01880503_013.sid'
            '02010604_052.sid'
        ]
        filmFormatFocal=(230.,230.,305)
        for fn in difficultPhos:
            for func in (zeissRmkTop,zeissRmkA):
                try:
                    func( path.join( r'D:\arap\data\laserv\Projekte\ARAP\Luftbilder CaseStudy', fn ), camSerial=None, filmFormatFocal=filmFormatFocal, plotDir='.', debugPlot=True )
                except Exception as ex:
                    exc = ex
                else:
                    break
            else:
                print( 'Exception raised while processing {}: {}'.format( fn, exc ) )
    