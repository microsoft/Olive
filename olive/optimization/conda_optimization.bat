set conda_env_name=%1
set python_version=%2
set use_gpu=%3
set opt_args_str=%4
set onnxruntime_version=%5
set opt_args_str=%opt_args_str:"=%

:: create conda env
call conda create -n %conda_env_name% python=%python_version% -y

:: activate conda env
call conda activate %conda_env_name%

:: install dependencies
call pip install numpy onnx psutil coloredlogs sympy onnxconverter_common docker==5.0.0 six

:: install olive
call pip install --extra-index-url https://olivewheels.azureedge.net/oaas onnxruntime-olive==0.5.0

:: optimization setup in conda env
if %use_gpu%=="True" (call olive setup --onnxruntime_version %onnxruntime_version% --use_gpu) else (call olive setup --onnxruntime_version %onnxruntime_version%)

:: run optimization in conda env
call olive optimize %opt_args_str%

:: deactivate conda env
call conda deactivate

:: call conda env remove -n %conda_env_name%
