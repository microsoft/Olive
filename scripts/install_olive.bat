REM -------------------------------------------------------------------------
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
REM --------------------------------------------------------------------------
@echo off

set PIPELINE=%1
set INSTALL_DEV_MODE=%2

rem Create virtual environment
if "%PIPELINE%"=="True" (
    call echo "Creating virtual environment for pipeline"
    call python -m venv olive-venv || goto :error
    call olive-venv\\Scripts\\activate.bat || goto :error
) else (
    call echo "Using active python environment"
)

rem Install olive
if "%INSTALL_DEV_MODE%"=="True" (
    call echo "Installing olive in dev mode"
    call python -m pip install -e . || goto :error
) else (
    call echo "Installing olive"
    call python -m pip install . || goto :error
)

goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
