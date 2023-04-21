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
        # if filename in ["t5", "gpt2", "bart"] skip
        if [ "$filename" == "$AML_BENCHMARK_PATH/t5" ] || [ "$filename" == "$AML_BENCHMARK_PATH/gpt2" ] || [ "$filename" == "$AML_BENCHMARK_PATH/bart" ]; then
            echo "Skip $filename"
            continue
        fi

        # check if the directory name is a valid model name
        cd $filename
        echo "Current directory: $(pwd)"
        pip install -r requirements.txt
        mkdir -p cache
        python -m olive.workflows.run --config gpu_config.json --clean_cache --log_level 2 --output_dir ./cache --output_name gpu 2>&1 > ./cache/gpu_engine_logs
        cd ..
    fi
done
