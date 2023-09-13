REM -------------------------------------------------------------------------
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
REM --------------------------------------------------------------------------
@echo off

set INSTALL_DEV_MODE=%1

rem Upgrade pip
call echo "Upgrading pip"
call python -m pip install --upgrade pip || goto :error

rem Install olive
if "%INSTALL_DEV_MODE%"=="True" (
    call echo "Installing olive in dev mode"
    call python -m pip install -e ".%INSTALL_EXTRAS%" || goto :error
) else (
    call echo "Installing olive"
    call python -m pip install ".%INSTALL_EXTRAS%" || goto :error
)

goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
