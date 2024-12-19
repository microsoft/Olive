# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
CONDA_CONFIG = {
    "name": "olive_finetune",
    "channels": ["defaults"],
    "dependencies": [
        "python=3.8.13",
        "pip=22.3.1",
        {
            "pip": [
                "accelerate",
                "bitsandbytes",
                "peft",
                "sentencepiece",
                "datasets",
                "evaluate",
                "psutil",
                "optimum",
                "scipy",
                "scikit-learn",
                "torch",
                "onnxruntime-genai",
                "--extra-index-url https://download.pytorch.org/whl/cu118",
                "transformers>=4.41.1",
                "git+https://github.com/microsoft/Olive#egg=olive-ai[gpu,azureml]",
            ]
        },
    ],
}
