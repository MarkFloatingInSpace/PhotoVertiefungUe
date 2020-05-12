@echo off
setlocal enableDelayedExpansion
REM This script sets up a shell environment for OrientAL. All necessary
REM environment variables are set automatically and OrientAL is also configured
REM to be used with the built-in Python version.
REM ===========================================================================
SET PYTHON_ASSOCIATED=
SET ASSOCIATED_WITH_INTERAL_PYTHON=
SET PY_IN_PATHEXT=
set ORIENTAL_MESSAGE=

goto run

:ANALYSE_ASSOCIATION
  set PYTHON_TEXT=
  set PYTHON_EXT=
  FOR /F "delims== tokens=2 USEBACKQ" %%F IN (`assoc .py`) DO SET PYTHON_TEXT=%%F
  if "%PYTHON_TEXT%"=="" goto :eof

  FOR /F "delims== tokens=2 USEBACKQ" %%F IN (`ftype %PYTHON_TEXT%`) DO SET PYTHON_EXT=%%F
  rem set PYTHON_EXT=
  :: remove appendix "%1" %* from PYTHON_EXT (and quotes)
  call :SELECT_FIRST %PYTHON_EXT%

  if "%PYTHON_EXT%"=="" goto :eof
  :: add quotes again since it is needed later on
  set PYTHON_EXT="%PYTHON_EXT%"
  rem echo -%PYTHON_EXT%-

  set PYTHON_INT="%~dp0\python\python.exe"
  
  SET PYTHON_ASSOCIATED=1
  
  if %PYTHON_INT% == %PYTHON_EXT% set ASSOCIATED_WITH_INTERAL_PYTHON=1
  goto :eof

:SELECT_FIRST
  :: store first parameter and remove quotes 
  SET PYTHON_EXT=%~1
  goto :eof
  
:ANALYSE_PATHEXT
  FOR %%G IN (%PATHEXT%) DO if /I "%%G"==".py" set PY_IN_PATHEXT=1
  goto :eof

:ADD_PY_TO_PATHEXT
  SET PATHEXT=%PATHEXT%;.PY
  goto :eof
  
:REMOVE_PY_FROM_PATHEXT
  SET PATHEXT=%PATHEXT:.PY;=%
  SET PATHEXT=%PATHEXT:;.PY=%
  goto :eof
  
:run
TITLE OrientAL Shell 2020-04-29
SET ORIENTAL_ROOT=%~dp0
REM we need sqlite3.dll on PATH, so include python/DLLs
SET PATH=%ORIENTAL_ROOT%python;%ORIENTAL_ROOT%bin;%ORIENTAL_ROOT%Scripts;%ORIENTAL_ROOT%python\DLLs;%ORIENTAL_ROOT%python\Scripts;%PATH%
SET GDAL_DATA=%ORIENTAL_ROOT%bin\gdal_data
SET PYTHONPATH=

:: Now do some analyses regarding .py in %PATHEXT% and corresponding file association
:: We have to ensure that .py is not in %PATHEXT% if the  .py file association points to an external Python interpreter
call :ANALYSE_ASSOCIATION
call :ANALYSE_PATHEXT

if "%PYTHON_ASSOCIATED%"=="1" if "%ASSOCIATED_WITH_INTERAL_PYTHON%"=="" if "%PY_IN_PATHEXT%"=="1" (
  :: ".py" file association does not match internal python. hence, remove ".py" from PATHEXT
  set ORIENTAL_MESSAGE=ATTENTION: '.py' was removed from PATHEXT
  call :REMOVE_PY_FROM_PATHEXT
)

if "%PYTHON_ASSOCIATED%"=="1" if "%ASSOCIATED_WITH_INTERAL_PYTHON%"=="1" if "%PY_IN_PATHEXT%"==""  (
  :: ".py" file association matches internal python. hence, ".py" is safe to be added to PATHEXT
  set ORIENTAL_MESSAGE=Note: '.py' was added to PATHEXT
  call :ADD_PY_TO_PATHEXT
)

REM If launched without arguments, we print the welcome screen and remain open until the user exits.
REM Otherwise, we change the color temporarily and exit after calling the shifted arguments.

REM There is no built-in functionality to query the number of arguments. Testing on "%*"=="" or [%*]==[] fails for double-quoted arguments.
REM Hence, test only the first argument on being empty. To handle both quoted and unquoted arguments, use %~1
IF "%~1"=="" (
  REM http://patorjk.com/software/taag/#p=display&f=Graffiti&t=OrientAL
  REM There are several special characters that generally must be escaped when used in Windows batch files. Here is a partial list: < > & | ^ %
  REM The escape character is ^
  REM Open the Shell
  CMD /S /T:17 /K "cls & echo.  & echo Welcome to & echo   ________        .__               __     _____  .____             & echo   \_____  \_______^|__^| ____   _____/  ^|_  /  _  \ ^|    ^|       & echo    /   ^|   \_  __ \  ^|/ __ \ /    \   __\/  /_\  \^|    ^|        & echo   /    ^|    \  ^| \/  \  ___/^|   ^|  \  ^| /    ^|    \    ^|___  & echo   \_______  /__^|  ^|__^|\___  ^>___^|  /__^| \____^|__  /_______ \ & echo           \/              \/     \/             \/        \/         & echo. & if not "%ORIENTAL_MESSAGE%"=="" echo %ORIENTAL_MESSAGE%"
) ELSE (
  REM Launched from a console, call program
  COLOR 17
  SHIFT
  CMD /S /C "%*"
  SET OPALS_ERRORLEVEL=!ERRORLEVEL!
  COLOR
  EXIT /B !OPALS_ERRORLEVEL!
)
