import environment
import sys, unittest
from pathlib import Path

import oriental
from oriental import import_

def importMe(project: str, chunk = None):
    cmdline = f"--project {project}"
    if chunk is not None:
        cmdline += f" --chunk {chunk}"
    import_.parseArgs(cmdline.split())
    if sys.gettrace():  # We are in a debugging session (or a profiling session)
        outDb = Path('import') / (Path(project).stem + '.sqlite')
        load(outDb)

def load(outDb : Path):
    from collections import namedtuple
    import numpy as np
    from oriental import absOri, adjust, blocks, log
    import oriental.adjustScript
    import oriental.absOri.main
    import oriental.adjust.loss
    import oriental.blocks.export

    logger = log.Logger('import')
    loss = adjust.loss.Trivial()

    def getImgObsData(row):
        return oriental.adjustScript.AutoImgObsData(row['id'], row['imgId'], row['objPtId'],
                                                    (row['red'], row['green'], row['blue']))

    block, solveOpts, cameras, images, objPts = absOri.main.restoreRelOriBlock(
        outDb,
        getImgObsLoss=lambda row: loss,
        getImgObsData=getImgObsData)
    logger.info('Block restored')
    residuals, = block.Evaluate()
    sse = np.sum(residuals ** 2)
    mad = np.median(np.abs(np.median(residuals) - residuals))
    logger.info(f'residuals\nsse\t{sse}\nmad\t{mad}')
    import matplotlib.pyplot as plt

    plt.hist(residuals, bins=100, range=(-5 * mad, 5 * mad))

    # adapt to blocks.export.webGL and block.deactivate
    Image = namedtuple('Image', absOri.main.Image._fields + ('camId', 'rot'))
    for imgId, image in images.items():
        images[imgId] = Image(**image._asdict(), camId=image.camera.id, rot=image.omfika)
    objPts = {key: value.pt for key, value in objPts.items()}

    blocks.export.webGL(outDb.parent / 'reconstruction.html', block, cameras, images, objPts,
                        oriental.adjustScript.AutoImgObsData)
    blocks.export.ply(outDb.parent / 'reconstruction.ply', cameras, images, objPts, block)

    plt.show()

class Test(unittest.TestCase):
    @unittest.skipUnless(oriental.config.isDvlp, "Needed data only available on dvlp machines")
    def test_photoScanEinsicht(self):
        # Einsicht-PhotoScan-Version: <chunk version="1.2.0">
        importMe(r'P:\Projects\17_EINSICHT\07_Work_Data\20180618_Lavantal\Projekt\Lavanttal\Pfeiler1\Pfeiler1-Export.psx', 28)

    @unittest.skipUnless(oriental.config.isDvlp, "Needed data only available on dvlp machines")
    def test_photoScan124(self):
        # our PhotoScan-version 1.2.4: <chunk> without version-attribute
        importMe(r'P:\Projects\17_EINSICHT\07_Work_Data\20180625_FalkensteinII\b\Projekt_180623.psx')

    @unittest.skipUnless(oriental.config.isDvlp, "Needed data only available on dvlp machines")
    def test_metaShape(self):
        # MetaShape 1.5.1
        importMe(r'E:\_dataOrientAL\testDataMetaShape.psx')

    @unittest.skipUnless(oriental.config.isDvlp, "Needed data only available on dvlp machines")
    def test_matchAT(self):
        # Inpho Match-AT $PROJECT 7.1.0
        importMe(r'P:\Projects\16_VOLTA\07_Work_Data\ICGC_2018505_Cornella_6cm\control\cornella_6cm.prj')

if __name__ == '__main__':
    if not oriental.config.ide:
        unittest.main()
    else:
        import sys
        unittest.main( argv=sys.argv[:1], # we don't set anything useful in the debugging options.
                       defaultTest='Test.test_photoScanEinsicht',
                       exit=False )