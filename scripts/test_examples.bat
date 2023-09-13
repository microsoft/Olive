REM -------------------------------------------------------------------------
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
REM --------------------------------------------------------------------------
@echo off

set ROOT_DIR=%1
set EXAMPLE_FOLDER=%2
set EXAMPLE_NAME=%3

rem install pytest
call python -m pip install pytest

rem test samples
call echo "Testing examples"
call python -m pip install -r %ROOT_DIR%\\examples\\%EXAMPLE_FOLDER%\\requirements.txt || goto :error

call python -m pytest -v -s --log-cli-level=WARNING --junitxml=%ROOT_DIR%\\logs\\test_examples-TestOlive.xml^
 %ROOT_DIR%\\examples\\test\\test_%EXAMPLE_NAME%.py || goto :error

goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
