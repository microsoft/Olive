@echo off
echo Killing processes on ports 3000, 3001, 3002, 8000, 8001...
echo.

set PORTS=3000 3001 3002 8000 8001

for %%p in (%PORTS%) do (
    echo Checking port %%p...
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%%p ^| findstr LISTENING') do (
        if not "%%a"=="" (
            echo Killing process on port %%p (PID: %%a)
            taskkill /F /PID %%a 2>nul
            if !errorlevel!==0 (
                echo Successfully killed process on port %%p
            ) else (
                echo Failed to kill process on port %%p - may require admin rights
            )
        )
    )
)

echo.
echo Done! All specified ports have been cleared.