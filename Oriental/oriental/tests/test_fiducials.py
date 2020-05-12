import environment

import glob, os, unittest

import numpy as np

import oriental
from oriental.relOri import fiducials
from oriental import utils
import oriental.utils.gdal

class Test(unittest.TestCase):

    @unittest.skipUnless( oriental.config.isDvlp, "Needed data only available on dvlp machines" )
    def test_wildRc8(self):
        plotDir = os.path.join(oriental.config.pkgRoot, 'tests', 'fiducials', 'wildRc8')
        phoFns = []
        phoFns.extend( glob.glob( r'E:\Livia\Historical_Images_ValleLunga\alte_IGM_Luftbilder\*.tif') )
        phoFns.extend( glob.glob( r'P:\Studies\17_AdV_HistOrthoBenchmark\07_Work_Data\Bonn_1962\Tif_8bit\*.tif' ) )

        rotPhoFns = []
        # Use np.rot90 to check if rotation gets recognized successfully
        for iRot in range(4):
            imgFn, imgExt = os.path.splitext( os.path.basename( phoFns[0] ) )
            rotImgFn = os.path.join( oriental.config.pkgRoot, 'tests', imgFn + '_rot{:03d}'.format(iRot*90) + imgExt )
            if not os.path.exists(rotImgFn):
                img = utils.gdal.imread( phoFns[0], depth=utils.gdal.Depth.unchanged )
                img = np.ascontiguousarray( np.rot90( img, k=iRot ) )
                utils.gdal.imwrite( rotImgFn, img )
            rotPhoFns.append( rotImgFn )
        phoFns.extend( rotPhoFns )

        #phoFns = [r'E:\Livia\Historical_Images_ValleLunga\alte_IGM_Luftbilder\6842.tif']
        #phoFns = [r'E:\AdV_Benchmark_Historische_Orthophotos\Bonn_1962\Tif_8bit\168_4_2044_b.tif']
        for phoFn in phoFns:
            try:
                fiducials.wildRc8( phoFn, filmFormatFocal=(230,230,152.27), plotDir=plotDir, debugPlot=False )
            except Exception as ex:
                print('{} failed: {}'.format(phoFn,ex))

    @unittest.skipUnless( oriental.config.isDvlp, "Needed data only available on dvlp machines" )
    def test_zeissRmk20(self):
        plotDir = os.path.join(oriental.config.pkgRoot, 'tests', 'fiducials', 'zeissRmk20')
        phoFns = glob.glob(r'P:\Projects\17_AdV_HistOrthoBenchmark\07_Work_Data\BB_Testdaten-1953\Luftbilder\*.tif')
        #phoFns=['E:\\AdV_Benchmark_Historische_Orthophotos\\BB_Testdaten-1953\\Luftbilder\\6163_b.tif']
        for phoFn in sorted(phoFns):
            try:
                print(os.path.basename(phoFn))
                fiducials.zeissRmk20( phoFn, plotDir=plotDir, debugPlot=False )
            except Exception as ex:
                print('{} failed: {}'.format(phoFn,ex))

if __name__ == '__main__':
    if not oriental.config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='Test.test_potsdam',
                       exit=False )
