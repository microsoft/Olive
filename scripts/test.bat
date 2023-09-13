REM -------------------------------------------------------------------------
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
REM --------------------------------------------------------------------------
@echo off

set ROOT_DIR=%1
set TEST_TYPE=%2

rem install pytest
call python -m pip install pytest

if %TEST_TYPE% == multiple_ep (
    call curl --output openvino_toolkit.zip https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0.1/windows/w_openvino_toolkit_windows_2023.0.1.11005.fa1c41994f3_x86_64.zip
    call 7z x openvino_toolkit.zip
    call w_openvino_toolkit_windows_2023.0.1.11005.fa1c41994f3_x86_64\\setupvars.bat
    call python -m pip install numpy psutil coverage protobuf==3.20.3 || goto :error
) else (
    call python -m pip install -r %ROOT_DIR%\\test\\requirements-test.txt || goto :error
)

rem run tests
call coverage run --source=%ROOT_DIR%\\olive -m pytest -v -s --log-cli-level=WARNING --junitxml=%ROOT_DIR%\\logs\\test-TestOlive.xml^
 %ROOT_DIR%\\test\\%TEST_TYPE% || goto :error
call coverage xml

goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
