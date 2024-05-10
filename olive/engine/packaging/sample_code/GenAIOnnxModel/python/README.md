# Olive sample code instructions

## ONNXRuntime GenAI installation
Install onnxruntime-genai package:

### install by pip (CPU)
```
python -m pip install onnxruntime-genai
```

### install by pip (CUDA)
```
python -m pip install onnxruntime-genai-cuda --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

### install by pip (DirectML)
```
python -m pip install onnxruntime-genai-directml
```

For updated instructions and/or configuring a locally built package, refer to the instructions [here](https://github.com/microsoft/onnxruntime-genai).

## Running the same code
```
python code_sample.py <Model's directory path> --prompts [prompt1, [prompt2, ...]]
```

For full list of available options, run the script in help mode.
```
python code_sample.py -h
```
