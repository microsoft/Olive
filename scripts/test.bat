REM -------------------------------------------------------------------------
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
REM --------------------------------------------------------------------------
@echo off

set PIPELINE=%1
set ROOT_DIR=%2
set TEST_TYPE=%3

rem install pytest
call python -m pip install pytest
call python -m pip install -r %ROOT_DIR%\\test\\requirements-test.txt || goto :error

rem run tests
call coverage run --source=%ROOT_DIR%\\olive -m pytest -v -s --log-cli-level=WARNING --junitxml=%ROOT_DIR%\\logs\\test-TestOlive.xml^
 %ROOT_DIR%\\test\\%TEST_TYPE% || goto :error
call coverage xml

goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
