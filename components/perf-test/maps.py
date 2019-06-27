model_ep_map = {
    # Placeholder for recommending execution provider based on models
}

ep_graphOptimizer_map = {
    # Placeholder for assigning graph optimizer to executer provider
    "cpu": 2,
    "cpu_openmp": 2,
    "cuda": 2,
    "mkldnn": 2,
    "mkldnn_openmp": 2,
    "ngraph": 2,
    "tensorrt": 2
}

ep_envvar_map = {
    # Placeholder to tune environment variables based on execution provider
    "mkldnn_openmp": {
        "OMP_WAIT_POLICY": ["active", "passive"],
        },
    "cpu_openmp": {
        "OMP_WAIT_POLICY": ["active", "passive"],
        }
}