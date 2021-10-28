set onnxruntime_version=%1
set use_gpu=%2

if "%use_gpu%"=="True" (call docker build -t olive_optimization:%onnxruntime_version%_gpu --no-cache --build-arg ort_version=%onnxruntime_version% -f Dockerfile.gpu .) else (call docker build -t olive_optimization:%onnxruntime_version%_cpu --no-cache --build-arg ort_version=%onnxruntime_version% -f Dockerfile.cpu .)

