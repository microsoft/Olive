# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

model_ep_map = {
    # Placeholder for recommending execution provider based on models
}

ep_graphOptimizer_map = {
    # Placeholder for assigning graph optimizer to executer provider
}

ep_envvar_map = {
    # Placeholder to tune environment variables based on execution provider
    "cpu_openmp": {
        "OMP_WAIT_POLICY": ["active", "passive"],
        },
    "mklml": {
        "OMP_WAIT_POLICY": ["active", "passive"],
        },
    "dnnl": {
        "OMP_WAIT_POLICY": ["active", "passive"],
        },
    "ngraph": {
        "OMP_WAIT_POLICY": ["active", "passive"],
        },
    "nuphar": {
        "OMP_WAIT_POLICY": ["active", "passive"],
        }
}