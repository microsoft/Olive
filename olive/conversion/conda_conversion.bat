set conda_env_name=%1
set python_version=%2
set model_framework=%3
set framework_version=%4
set cpt_args_str=%5
set cpt_args_str=%cpt_args_str:"=%

:: create conda env
call conda create -n %conda_env_name% python=%python_version% -y

:: activate conda env
call conda activate %conda_env_name%

:: install dependencies
call pip install numpy onnx psutil coloredlogs sympy onnxconverter_common docker==5.0.0 six

:: install olive
call pip install --extra-index-url https://olivewheels.azureedge.net/test onnxruntime-olive==0.3.0

:: conversion setup in conda env
call olive setup --model_framework %model_framework% --framework_version %framework_version%

:: run optimization in conda env
call olive convert %cpt_args_str%

:: deactivate conda env
call conda deactivate

call conda env remove -n %conda_env_name%
