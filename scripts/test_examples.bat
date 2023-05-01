REM -------------------------------------------------------------------------
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
REM --------------------------------------------------------------------------
@echo off

set PIPELINE=%1
set ROOT_DIR=%2
set EXAMPLE=%3

if "%PIPELINE%"=="True" (
    call olive-venv\\Scripts\\activate.bat || goto :error
)

rem install pytest
call python -m pip install pytest

rem test samples
call echo "Testing examples"
call python -m pip install -r %ROOT_DIR%\\examples\\%EXAMPLE%\\requirements.txt || goto :error

rem TODO: need to remove later
if "%EXAMPLE%"=="whisper" (
    call :whisper-setup || goto :error
)

call python -m pytest -v -s --log-cli-level=WARNING --junitxml=%ROOT_DIR%\\logs\\test_examples-TestOlive.xml^
 %ROOT_DIR%\\examples\\test_%EXAMPLE%.py || goto :error

goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%

:whisper-setup
call echo "Installing custom packages for whisper"
call python -m pip uninstall -y onnxruntime onnxruntime-extensions || goto :error
call python -m pip install ort-nightly==1.15.0.dev20230429003 onnxruntime-extensions==0.8.0.303816 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/  || goto :error
