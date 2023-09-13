REM -------------------------------------------------------------------------
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
REM --------------------------------------------------------------------------
@echo off

set PIPELINE=%1
set INSTALL_DEV_MODE=%2
set MODEL_NAME=%3
set DEVICE=%4

if "%PIPELINE%"=="True" (
    call olive-venv\\Scripts\\activate.bat || goto :error
)

call python -m pip install -r %ROOT_DIR%\\.azure_pipelines\\performance_check\\requirements-%DEVICE%.txt
call python %ROOT_DIR%\\.azure_pipelines\\performance_check\\run_performance_check.py --model_name %MODEL_NAME% --device %DEVICE%

REM clean up
call rmdir /s /q %ROOT_DIR%\\.azure_pipelines\\performance_check\\run_cache
