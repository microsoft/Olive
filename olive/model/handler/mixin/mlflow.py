# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

logger = logging.getLogger(__name__)


class MLFlowHFConfigMixin:
    def load_mlflow_model(self):
        """Load the model from mlflow.

        Load the model from mlflow which return a tuple of task_type, model, tokenizer, config
        with this way, the olive's hf_config in input model should not work as the model is
        loaded from mlflow where the MLmodel file is the source of truth
        For more loading details like `trust_remote_code` in from_pretrained_args will be derived
        from the MLmodel file but not Olive's hf_config
        e.g.
        force_load_config:
          config_hf_load_kwargs:
           trust_remote_code: true
          tokenizer_hf_load_kwargs:
           trust_remote_code: true
          model_hf_load_kwargs:
           trust_remote_code: true

        Return: a tuple of task_type, model, tokenizer, config
        """
        from azureml.evaluate import mlflow as aml_mlflow

        return aml_mlflow.hftransformers.load_model(self.model_path)

    def get_task_type_from_mlflow_model(self):
        return self.load_mlflow_model()[0]

    def get_model_from_mlflow_model(self):
        model = self.load_mlflow_model()[1]
        model.eval()
        return model

    def get_config_from_mlflow_model(self):
        return self.load_mlflow_model()[3]

    def get_tokenizer_from_mlflow_model(self):
        return self.load_mlflow_model()[2]
