# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
$AML_BENCHMARK_PATH = Get-Location
# traverse the directory under AML_BENCHMARK_PATH
# activate python env
# & $PYTHON_PATH
Get-ChildItem -Path $AML_BENCHMARK_PATH -Directory | ForEach-Object {
    # get the directory name
    $dir_name = $_.Name
    # get the full path of the directory
    $dir_path = $_.FullName
    # cd $dir_path
    Set-Location $dir_path
    Write-Host "Current directory: $dir_path"
    pip install -r requirements.txt
    # run olive engine to sumit the job
    # if dir_name in ["bert", "gpt2", "t5"], ignore accuracy
    if ($dir_name -in "bart", "gpt2", "t5") {
        Write-Host "Ignore accuracy for $dir_name"
        (python -m olive.workflows.run --config cpu_config.json --output_dir ./cache --output_name cpu --only_metrics latency --system aml_system --clean_cache) | Tee-Object -file ./cache/engine_run.log
    }
    else {
        (python -m olive.workflows.run --config aml_cpu_config.json --output_dir ./cache --output_name cpu --system aml_system --clean_cache) | Tee-Object -file ./cache/engine_run.log
    }
    # cd parent folder of $dir_path
    Set-Location ..
}
