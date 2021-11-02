set framework=%1
set framework_version=%2

call copy requirements\%framework%_%framework_version%.txt .\requirements.txt
call docker build -t olive_conversion:%framework%_%framework_version% --no-cache .
