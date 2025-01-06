# Prepare env

```
pip install virtualenv
virtualenv olive-venv
./olive-venv/scripts/activate
```

# Install needed modules

- pyinstaller will use current python's
- May also need to update spec's hiddenimports etc.
    * olive uses `inspect.getsourcelines`
    * olive uses `from optimum.exporters.tasks import TasksManager`
    * olive uses `from onnxruntime_genai.models.builder import create_model`
    * optimum uses `_onnx_available = _is_package_available("onnx")`
    * File "onnxruntime\transformers\onnx_utils.py", line 5, in <module> ModuleNotFoundError: No module named 'fusion_utils'

```
pip install olive-ai[ort-genai,auto-opt]
pip install transformers==4.44.2
```

# Install pyinstaller

```
pip install pyinstaller
```

# Generate exe


```
pyinstaller pyinstaller/olive.spec --workpath pyinstaller/build --distpath pyinstaller/dist --noconfirm
```

# Test

```
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct *.json *.safetensors *.txt

.\pyinstaller\dist\olive\olive.exe auto-opt `
    --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct `
    --output_path models/smolm2 `
    --device cpu `
    --provider CPUExecutionProvider `
    --use_ort_genai `
    --precision int4 `
    --log_level 1
```

# Build Msix

# Questions

- remove duplicate 