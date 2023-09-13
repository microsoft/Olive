REM -------------------------------------------------------------------------
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
REM --------------------------------------------------------------------------
@echo off

set INSTALL_DEV_MODE=%1
set MODEL_NAME=%2
set DEVICE=%3

call python -m pip install -r %ROOT_DIR%\\.azure_pipelines\\performance_check\\requirements-%DEVICE%.txt || goto :error
call python %ROOT_DIR%\\.azure_pipelines\\performance_check\\run_performance_check.py --model_name %MODEL_NAME% --device %DEVICE% || goto :error

REM clean up
call rmdir /s /q %ROOT_DIR%\\.azure_pipelines\\performance_check\\run_cache || goto :error

goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
