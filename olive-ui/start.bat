@echo off
setlocal enabledelayedexpansion

echo =================================================
echo              Olive UI Launcher
echo =================================================
echo.

:: Check if port argument is provided
set FRONTEND_PORT=%1
set BACKEND_PORT=%2

:: Set default ports if not provided
if "%FRONTEND_PORT%"=="" set FRONTEND_PORT=3000
if "%BACKEND_PORT%"=="" set BACKEND_PORT=8000

echo Frontend port: %FRONTEND_PORT%
echo Backend port: %BACKEND_PORT%
echo.

:: Kill existing processes on the specified ports
echo Cleaning up existing processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%FRONTEND_PORT%') do (
    taskkill /F /PID %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%BACKEND_PORT%') do (
    taskkill /F /PID %%a 2>nul
)

:: Wait a moment for ports to be freed
timeout /t 2 /nobreak > nul

:: Create frontend environment file
echo PORT=%FRONTEND_PORT% > client\.env.local
echo REACT_APP_API_URL=http://localhost:%BACKEND_PORT%/api >> client\.env.local

echo.
echo Starting Olive UI...
echo Frontend: http://localhost:%FRONTEND_PORT%
echo Backend:  http://localhost:%BACKEND_PORT%
echo.

:: Start backend in background
echo Starting backend server...
cd server
start /B python app.py --port %BACKEND_PORT%
cd ..

:: Wait for backend to start
timeout /t 3 /nobreak > nul

:: Start frontend (this will block and show output)
echo Starting frontend...
cd client
npm start
cd ..

echo.
echo Olive UI stopped.
echo Usage: start.bat [frontend_port] [backend_port]
echo Example: start.bat 3001 8001
pause