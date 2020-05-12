
@echo off
SET PATH=%~dp0bin;%PATH%
SET PYTHONHOME=
REM SET PYDEVD_USE_FRAME_EVAL=NO 
CALL "%~dp0python\python" "%~dp0bin\PyCharm\registerOrientALInterpreter.py" ".PyCharmCE2020.1"
START "PyCharm" "%~dp0bin\PyCharm\bin\pycharm64.exe" "%~dp0oriental"
