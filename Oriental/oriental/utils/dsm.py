# -*- coding: cp1252 -*-

from oriental import config, Progress, log, utils
import oriental.utils.gdal

from pathlib import Path
import os.path

from contracts import contract

logger = log.Logger(__name__)

urlDhmLamb10mZip = 'http://gis.ktn.gv.at/OGD/Geographie_Planung/ogd-10m-at.zip'

@contract
def info( dsmFn : Path ):
    dsmFns = []
    # a file in the 'data' directory is probably the common use-case. GDAL logs an error when it cannot open a data set, which our logging system logs automatically. Thus, try opening in the 'data' folder first.
    if not dsmFn.is_absolute():
        dsmFns.append( os.path.abspath( str(Path( config.dataDir ) / dsmFn ) ) )
    dsmFns.append( os.path.abspath(str(dsmFn)) )
    with utils.gdal.suppressGdalMesssages():
        excpts = []
        for fn in dsmFns:
            try:
                infoDsm, = utils.gdal.imread( fn, info=True, skipData=True )
            except Exception as ex:
                excpts.append(ex)
            else:
                break
        else:
            if str(dsmFn) == 'dhm_lamb_10m.tif':
                zipFn = Path(urlDhmLamb10mZip).name
                zipPath = Path( config.dataDir ) / zipFn
                logger.warning('{} is missing. Either copy it to {}, or to the current working directory.', str(dsmFn), config.dataDir )
                res = input( 'Download data set from {} to {} (y/n)? '.format( urlDhmLamb10mZip, config.dataDir ) )
                if not res.lower().startswith( 'n' ):
                    logger.info( 'Downloading {}.', zipFn )
                    import urllib.request, zipfile, shutil, ssl
                    # Let's not validate SSL certificates. Usually, it works, but e.g. not on stu10.
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    # zipfile.ZipFile needs random file access. However, the HTTP server does not seem to provide that. Thus, we need to download the .zip first, and only then we can extract its contents.
                    with urllib.request.urlopen(urlDhmLamb10mZip, context=ctx) as response, \
                         zipPath.open('wb') as fout:
                        progress = None
                        contentLen = response.info().get('Content-Length')
                        if contentLen:
                            progress = Progress(int(contentLen))
                        #shutil.copyfileobj(response, fout) # implementation of shutil.copyfileobj follows below, with progress handling added.
                        length=16*1024
                        while 1:
                            buf = response.read(length)
                            if not buf:
                                break
                            fout.write(buf)
                            if progress:
                                progress += len(buf)
                        if progress:
                            progress.finish()
                    logger.info( '{} downloaded.', zipFn )
                    with zipfile.ZipFile(str(zipPath)) as myzip:
                        myzip.extractall(config.dataDir)
                        logger.info( '{} extracted', ', '.join(myzip.namelist()) )
                    zipPath.unlink()
                    logger.info( '{} deleted.', zipFn )
                    try:
                        infoDsm, = utils.gdal.imread( str(Path( config.dataDir ) / dsmFn ), info=True, skipData=True )
                    except Exception as ex:
                        excpts = [ex]
                    else:
                        return infoDsm
            raise Exception( "Unable to read surface model {} from any of the following locations:\n{}\n\n{}".format( str(dsmFn), '\n'.join(dsmFns), '\n'.join(str(el) for el in excpts) ) )
    return infoDsm