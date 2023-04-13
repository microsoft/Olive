# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
AML_BENCHMARK_PATH=$(pwd)
# activate python env
# & $PYTHON_PATH
# traverse the directory under AML_BENCHMARK_PATH
for filename in $AML_BENCHMARK_PATH/*; do
    # check if the file is a directory
    if [ -d "$filename" ]; then
        # check if the directory name is a valid model name
        cd $filename
        echo "Current directory: $(pwd)"
        pip install -r requirements.txt
        mkdir -p cache
        python -m olive.workflows.run --config cpu_config.json --output_dir ./engine_results --output_name cpu 2>&1 > ./cache/engine_logs.log
        # python -m olive.workflows.run --config gpu_config.json --clean_cache --output_dir ./engine_results --output_name gpu
        cd ..
    fi
done
