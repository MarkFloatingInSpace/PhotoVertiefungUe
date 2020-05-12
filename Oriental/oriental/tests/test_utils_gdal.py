# -*- coding: cp1252 -*-
import environment
from oriental import config
from oriental import utils
import oriental.utils.gdal
import contextlib, os, unittest, tempfile
import numpy as np
from osgeo import ogr, osr

ogr.UseExceptions()
osr.UseExceptions()

class TestGDAL(unittest.TestCase):    
    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as fout:
            cls.tmpFn = fout.name

    @classmethod
    def tearDownClass(cls):
        with contextlib.suppress(OSError):
            os.remove( cls.tmpFn )

    def test_writeMasked(self):
        img = np.empty((5000,4009,3), dtype=np.uint8 )
        mask = np.ones( img.shape[:2], dtype=np.uint8 )
        info = utils.gdal.ImageInfoWrite()
        info.geotransform = np.array([[ 100., .1,   0. ],
                                      [ 200.,  0., -.1 ]])
        #info.maskBoundary = np.array([ 125.12, 126.16 ])
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for corner in ((0.,0.),
                       (0.,-100.),
                       (100,-100),
                       (100,0)):
            ring.AddPoint( *corner )
        ring.FlattenTo2D() # even though we've constructed a 2D object, and call AddPoint(.) with only 2 arguments, AddPoint(.) makes ring a 2.5D object!
        ring.CloseRings()
        polyg = ogr.Geometry(ogr.wkbPolygon)
        polyg.AddGeometry(ring)
        # GDAL's GeoTIFF tags are limited in size. Better not store those thousands of vertices, or GDAL issues a warning about lost metadata during writing.
        info.vectorMask = polyg.ExportToWkt()

        utils.gdal.imwrite( self.tmpFn, img, mask, info )
        #img,mask,info = utils.gdal.imread( 'test.tif', mask=True, info=True )
        img,mask,info = utils.gdal.imread( self.tmpFn, mask=True, info=True )
        self.assertGreater( len(info.vectorMask), 0, 'Vector mask has either not been written to file, or it could not be read back.' )
        geom = ogr.Geometry(wkt=info.vectorMask)
        self.assertGreater( geom.Area(), 0 )

    def test_write3channel(self):
        img = np.empty((5000,4009,3), dtype=np.uint8 )
        info = utils.gdal.ImageInfoWrite()
        info.geotransform = np.array([[ 100., .1,   0. ],
                                        [ 200.,  0., -.1 ]])
        utils.gdal.imwrite( self.tmpFn, img, info=info )

    def test_writeJPEG2000(self):
        img = np.empty((5000,4009,3), dtype=np.uint16 )
        mask = np.empty((5000,4009,1), dtype=np.uint8 )
        with tempfile.NamedTemporaryFile(suffix='.jp2', delete=False) as fout:
            tmpFn = fout.name
        utils.gdal.imwrite( tmpFn, img, mask )
        with contextlib.suppress(OSError):
            os.remove( tmpFn )


    def test_writeInt32SingleChannelMasked(self):
        img = np.ones( (3,2), np.int32 ) * -1
        img[1,1]=0
        img[2,1]=1
        utils.gdal.imwrite( self.tmpFn, img, mask=(img>=0).astype(np.uint8)*255 )

    @unittest.skipUnless(config.isDvlp, "Needed data not available in distro")
    def test_readMasked(self):
        fnDSM = fnDSM = os.path.join( config.testDataDirNotInDistro, 'dsm.asc' )
        dsm,mask,info = utils.gdal.imread( imageFilePath=fnDSM,
                                     bands=utils.gdal.Bands.unchanged,
                                     maxWidth=0,
                                     depth=utils.gdal.Depth.unchanged,
                                     mask=True,
                                     info=True )
        self.assertEqual( np.bool, mask.dtype )

    @unittest.skipUnless(config.isDvlp, "Needed data not available in distro")
    def test_readImage(self):
        fns_nBandsDepth = [
            ( r'E:\arap\data\laserv\Projekte\ARAP\2017_Willi_Senkrecht\0220000602_046.tif', 3, utils.gdal.Depth.u8 ), # per-band NODATA value 255
            ( r'E:\PleiAlps\Austria_Groglockner_208_SO17013362-9-01_DS_PHR1A_201609251017273_FR1_PX_E012N47_0804_00974\TPP1600674671\IMG_PHR1A_PMS-N_001\IMG_PHR1A_PMS-N_201609251017273_SEN_2394355101-001_R1C1.JP2', 3, utils.gdal.Depth.u16 ),
            ( os.path.join( config.testDataDirNotInDistro, '02030613_039_u8_NIR_pixel-interleaved_no-mask_wrong-georef.sid'     ), 3, utils.gdal.Depth.u8  ), # 8bit 3-channel NIR Falschfarben; pixel-interleaved; unmasked; geo-referencing seems wrong
            #( os.path.join( config.testDataDirNotInDistro, 'DSC01438_u8_RGB_line-interleaved_no-mask.JPG'                       ), 3, utils.gdal.Depth.u8  ), # 8bit 3-channel RGB; line-interleaved; unmasked
            ( os.path.join( config.testDataDirNotInDistro, '3240438_DxO_no_dist_corr_grey-u8_gray_pixel-interleaved_no-mask.jpg'), 1, utils.gdal.Depth.u8  ), # 8bit 1-channel / grey; pixel-interleaved; unmasked
            ( os.path.join( config.testDataDirNotInDistro, '02020502_009_u16_grey_band-interleaved_no-mask.tif'                 ), 1, utils.gdal.Depth.u16 ), # 16bit 1-channel / grey; band-interleaved; unmasked
            ( os.path.join( config.testDataDirNotInDistro, 'dsm.asc'                                                            ), 1, utils.gdal.Depth.f32 ), # DSM: 1-channel float32; NO_DATA mask
            ( os.path.join( config.testDataDirNotInDistro, 'mosaicOrigin_s32_grey_band-interleaved_no-mask.tif'                 ), 1, utils.gdal.Depth.s32 )
            #( "E:\\arap\\data\\laserv\\Geodaten\\geodaten\\oesterreich\\niederoesterreich\\orthofoto\\793479.jpg", 3, utils.gdal.Depth.u8 ) # 8bit 3-channel RGB; correct geo-referencing
        ]

        for fn,nInBands,inDepth in fns_nBandsDepth:
            for outDepth in ( utils.gdal.Depth.unchanged, utils.gdal.Depth.u8, utils.gdal.Depth.u16 ):
                for outBands in ( el[1] for el in sorted( utils.gdal.Bands.values.items(), key=lambda x: x[0] ) ):
                    for maxWidth in ( 0, 200 ):
                        if 1:
                            # calling gdal in C++ seems okay ...
                            if ( outDepth != utils.gdal.Depth.unchanged and inDepth not in ( utils.gdal.Depth.u8, utils.gdal.Depth.u16 ) ) or \
                               ( # OpenCV's cvtColor supports conversion only for u8, u16, and f32
                                 inDepth not in (utils.gdal.Depth.u8,utils.gdal.Depth.u16,utils.gdal.Depth.f32) and outBands not in ( utils.gdal.Bands.unchanged, utils.gdal.Bands.grey ) ) :
                                try:
                                    utils.gdal.imread( fn, bands = outBands, maxWidth = maxWidth, depth=outDepth, info = True )
                                except Exception as ex:
                                    pass
                                self.assertRaises( Exception, lambda: utils.gdal.imread( fn, bands = outBands, maxWidth = maxWidth, depth=outDepth, info = True ) )
                                continue
                            img,info = utils.gdal.imread( fn, bands = outBands, maxWidth = maxWidth, depth=outDepth, info = True )
                        else:
                            # ... while calling the gdal-Python-bindings results in a crash upon interpreter shutdown!!!
                            from osgeo import gdal as _gdal
                            dataset = _gdal.Open( fn, _gdal.GA_ReadOnly )
                            img = dataset.ReadAsArray()
                            if img.ndim == 3:
                                img = np.rollaxis( img, 0, 3 )
                            # The following loads osgeo.gdalnumeric.pyd and also numpy.
                            # importing gdalnumeric is enough to make the interpreter crash upon shutdown. This can also be reproduced by:
                            # python -c "from osgeo import gdalnumeric"
                            # -> crash
                            # updating GDAL to v. 1.11.0 does not help.
                            # compiling GDAL without SCOP-driver does not help.
                            # compiling GDAL with the same compiler as numpy (i.e. MSVC 2010) does not help.
                            # AND: it's the same in the Opals shell (Python 2.7)
                            # AND: it's the same in an OSGeo4W shell (Python 2.7)!
                            # The problem cannot be traced using gdal's CPL_debug(): it doesn't output anything.
                            # bug filed: http://trac.osgeo.org/gdal/ticket/5527
        
                            #numpyArray = dataset.ReadAsArray()
                            #print(numpyArray.flags)
                            #del dataset

                        self.assertEqual( inDepth, info.depth )
                        if outBands == utils.gdal.Bands.unchanged:
                            self.assertEqual( np.atleast_3d(img).shape[2], nInBands )
                        else:
                            self.assertEqual( np.atleast_3d(img).shape[2], utils.gdal.nBands( outBands ) )
                        if maxWidth:
                            self.assertLessEqual( img.shape[1], maxWidth )
                        if outDepth != utils.gdal.Depth.unchanged:
                            self.assertEqual( img.dtype, utils.gdal.depth2dtype[outDepth] )

    def test_overviewPosition(self):
        import cv2
        import osgeo.gdal

        full = np.zeros((1000, 1000), np.uint8)
        wid = 8
        oddRow = np.zeros((wid, full.shape[1]), np.uint8)
        for iCol in range( 0, full.shape[1]//wid, 2 ):
            oddRow[:, iCol*8:(iCol+1)*8] = 255
    
        evenRow = np.zeros_like(oddRow)
        evenRow[:,wid:] = oddRow[:,:-wid]
        for iRow in range(full.shape[1]//wid):
            full[iRow*wid:(iRow+1)*wid,:] = oddRow if iRow % 2 else evenRow

        # OpenCV docs say:
        # "First, it convolves the source image with the kernel: [5x5 Gaussian]
        # Then, it downsamples the image by rejecting even rows and columns."
        halfCv = cv2.pyrDown(full)
        # However, it is much more sensible to drop the odd rows and columns.
        # The following shows that in contrast to its documentation, OpenCV indeed rejects the odd rows and columns:
        # Additionally, the size of the downsampled image is computed such that no image content ever gets lost at the right/bottom edge
        # i.e. for odd numbers of rows/columns, the size is int( (n+1)/2 )
        filtered = cv2.GaussianBlur( full, ksize=(5,5), sigmaX=0 )
        down1 = filtered[0::2,0::2] # drop odd rows and columns
        down2 = filtered[1::2,1::2] # drop even rows and columns
        self.assertTrue ( ( halfCv == down1 ).all() )
        self.assertFalse( ( halfCv == down2 ).all() )

        # oriental.utils.lsm assumes that reading at a higher image pyramid level only scales the image coordinates, but never shifts them.
        # In the original implementation of GDAL's overview.cpp, this is not the case, however.
        # With our small changes, it is.
        info = utils.gdal.ImageInfoWrite()
        info.compression = utils.gdal.Compression.none
        utils.gdal.imwrite( self.tmpFn, full, info=info, buildOverviews=True )

        ds = osgeo.gdal.OpenEx( self.tmpFn, nOpenFlags=osgeo.gdal.OF_RASTER | osgeo.gdal.OF_READONLY, open_options=['OVERVIEW_LEVEL=0only'] )
        halfGdal = ds.ReadAsArray()
        self.assertTrue(halfCv.dtype == halfGdal.dtype)
        self.assertTrue(halfCv.shape == halfGdal.shape)
        self.assertTrue( ( halfCv == halfGdal ).all() )

    @unittest.skipUnless(config.isDvlp, "Needed data not available in distro")
    def test_createOverviews(self):
        imgPath = os.path.join( config.testDataDirNotInDistro, '370_1_120.tif' )
        with contextlib.suppress(OSError):
            os.remove( imgPath + '.ovr' )
        res = utils.gdal.buildOverviews( imgPath, external=True, test=False )
        self.assertTrue( res )

    @unittest.skipUnless(config.isDvlp, "Needed data not available in distro")
    def test_interpolatePoints(self):
        fnDSM = os.path.join( config.testDataDirNotInDistro, 'dsm.asc' )
        img, info = utils.gdal.imread( fnDSM, depth=utils.gdal.Depth.unchanged, info=True )
        points = np.array([ [ -1, 0 ],
                            [ 0, 0 ],
                            [ 0.01, 0 ],
                            [ 0.01, 0.01 ],
                            [ 1, 1 ],
                            [ info.nCols, 0 ] ], dtype=np.float )
        for interpolation in utils.gdal.Interpolation.names.values():
            interpolated = utils.gdal.interpolatePoints( fnDSM, points, interpolation )
            self.assertTrue( np.isnan( interpolated[0] ) )
            for idx in range(1,5):
                self.assertFalse( np.isnan( interpolated[idx] ) )
            self.assertTrue( np.isnan( interpolated[-1] ) )
            np.testing.assert_array_almost_equal( img[0,0], interpolated[1] )
            np.testing.assert_array_almost_equal( img[1,1], interpolated[4] )

    @unittest.skipUnless(config.isDvlp, "Needed data not available in distro")
    def test_interpolatePoints2(self):
        # this is the Austrian-wide DTM
        fnDtm = os.path.join( config.dataDir, 'dhm_lamb_10m.tif' )
        info, = utils.gdal.imread( fnDtm, info=True, skipData=True )
        points = np.array([[ info.nCols*3./4., info.nRows/2. ], # somewhere in Austria
                           [ 0, 0 ]], dtype=np.float) # somewhere in Bavaria
        interpolated = utils.gdal.interpolatePoints( fnDtm, points )
        self.assertFalse( np.isnan( interpolated[0] ) )
        self.assertTrue( np.isnan( interpolated[1] ) )

    def test_wmts(self):
        map = utils.gdal.wms(utils.gdal.DataSet.ortho)

    def test_gdal_data(self):
        # this fails if GDAL_DATA is not set as an environment variable
        # that points to a directory containing the GDAL .csv-files that define GDAL's coordinate systems.
        cs = osr.SpatialReference()
        cs.ImportFromEPSG(31254)

if __name__ == '__main__':
    if not config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='TestGDAL.test_overviewPosition',
                       exit=False )
