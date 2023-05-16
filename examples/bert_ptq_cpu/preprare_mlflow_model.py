# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import transformers
from azureml.evaluate import mlflow as aml_mlflow


def prepare_model():
    model_path = "mlflow_bert"
    task = "text-classification"
    architecture = "Intel/bert-base-uncased-mrpc"
    original_model = transformers.AutoModelForSequenceClassification.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    hf_conf = {
        "task_type": task,
    }
    aml_mlflow.hftransformers.save_model(
        original_model,
        model_path,
        tokenizer=tokenizer,
        config=original_model.config,
        hf_conf=hf_conf,
    )


if __name__ == "__main__":
    prepare_model()
