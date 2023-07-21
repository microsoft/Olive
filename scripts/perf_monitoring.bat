REM -------------------------------------------------------------------------
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
REM --------------------------------------------------------------------------
@echo off

set PIPELINE=%1
set ROOT_DIR=%2
set PERF_MONITORING_SCRIPT_NAME=%3
set PERF_MONITORING_SCRIPT_FUNCTION=%4

if "%PIPELINE%"=="True" (
    call olive-venv\\Scripts\\activate.bat || goto :error
)

rem install pytest
call python -m pip install pytest

rem performance monitoring
call echo "performance monitoring examples"
call python -m pip install -r %ROOT_DIR%\\perf_monitoring\\requirements.txt || goto :error

call python -m pytest -v -s --log-cli-level=WARNING --junitxml=%ROOT_DIR%\\logs\\performance-monitoring-TestOlive.xml^
 %ROOT_DIR%\\perf_monitoring\\test_%PERF_MONITORING_SCRIPT_NAME%.py::%PERF_MONITORING_SCRIPT_FUNCTION% || goto :error

goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
