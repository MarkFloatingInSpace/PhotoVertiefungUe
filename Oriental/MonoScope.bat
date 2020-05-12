
@ECHO OFF
SET GDAL_DATA=%~dp0\bin\gdal_data
SET GDAL_DEFAULT_WMS_CACHE_PATH=%TEMP%\OrientalGdalWmsCache
START "MonoScope" "%~dp0bin\MonoScope.exe" %*
