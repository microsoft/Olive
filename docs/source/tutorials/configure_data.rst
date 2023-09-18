.. _how_to_configure_data:

How To Configure Data
=====================

Olive Data config organizes the data loading, preprocessing, batching and post processing into a single json config and defines several **popular templates** to serve Olive optimization.

The configuration of Olive data config is positioned under Olive run config with the field of `data_configs`, and the data config is defined as a list of data config items. Here is an example of data config: `open_llama_sparsegpt_gpu <https://github.com/microsoft/Olive/blob/main/examples/open_llama/open_llama_sparsegpt_gpu.json#L11-L49>`_ .

.. code-block::

    "data_configs": {
        "dataset_1": {...},
        "dataset_2": {...},
    }

Then if there is any requirement to leverage the data config in Olive passes/evaluator, we can simply refer to the data config **key name**. For above `open_llama_sparsegpt_gpu` case, the passes/evaluator data config is:
`open_llama_sparsegpt_gpu data_config reference <https://github.com/microsoft/Olive/blob/main/examples/open_llama/open_llama_sparsegpt_gpu.json#L59>`_ .

.. code-block::

    "evaluators": {
        "common_evaluator": {
            ...,
            "data_config": "dataset_1"
        },
        ...
    },
    "passes": {
        "common_pass": {
            ...,
            "data_config": "dataset_2"
        },
        ...
    }


Before deep dive to the generic data config, let's first take a look at the data config template.

Supported Data Config Template
------------------------------

The data config template is defined in `olive.data.template <https://github.com/microsoft/Olive/blob/main/olive/data/template.py>`_ module which is used to create data_config easily.

Currently, we support the following data container which can be generated from `olive.data.template`:

1. `DummyDataContainer <https://github.com/microsoft/Olive/blob/main/olive/data/template.py#L9>`_ :
Convert the dummy data config to the data container.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "dummy_data_config_template",
                "type": "DummyDataContainer",
                "params_config": {
                    "input_shapes": [[1, 128], [1, 128], [1, 128]],
                    "input_names": ["input_ids", "attention_mask", "token_type_ids"],
                    "input_types": ["int64", "int64", "int64"],
                },
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.data.config import DataConfig
            data_config = DataConfig(
                name="dummy_data_config_template",
                type="DummyDataContainer",
                params_config={
                    "input_shapes": [[1, 128], [1, 128], [1, 128]],
                    "input_names": ["input_ids", "attention_mask", "token_type_ids"],
                    "input_types": ["int64", "int64", "int64"],
                },
            )

2. `HuggingfaceContainer <https://github.com/microsoft/Olive/blob/main/olive/data/template.py#L9>`_ :
Convert the huggingface data config to the data container.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "huggingface_data_config_template",
                "type": "HuggingfaceContainer",
                "params_config": {
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "task_type": "text-classification",
                    "batch_size": 1,
                    "data_name": "glue",
                    "input_cols": ["sentence1", "sentence2"],
                    "label_cols": ["label"],
                    "split": "validation",
                    "subset": "mrpc",
                },
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.data.config import DataConfig
            data_config = DataConfig(
                name="huggingface_data_config_template",
                type="HuggingfaceContainer",
                params_config={
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "task_type": "text-classification",
                    "batch_size": 1,
                    "data_name": "glue",
                    "input_cols": ["sentence1", "sentence2"],
                    "label_cols": ["label"],
                    "split": "validation",
                    "subset": "mrpc",
                },
            )

.. note::
    If the input model for Olive is huggingface model, we can update above config under `input_model`:

    .. code-block:: json

        {
            "input_model":{
                "type": "PyTorchModel",
                "config": {
                    "hf_config": {
                        "model_name": "Intel/bert-base-uncased-mrpc",
                        "task": "text-classification",
                        "dataset": {
                            "data_name":"glue",
                            "subset": "mrpc",
                            "split": "validation",
                            "input_cols": ["sentence1", "sentence2"],
                            "label_cols": ["label"],
                            "batch_size": 1
                        }
                    }
                }
            }
        }


3. `RawDataContainer <https://github.com/microsoft/Olive/blob/main/olive/data/template.py#L72>`_ :
Convert the raw data config to the data container.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "raw_data",
                "type": "RawDataContainer",
                "params_config": {
                    "data_dir": "data",
                    "input_names": ["data"],
                    "input_shapes": [[1, 3, 224, 224]],
                    "input_dirs": ["."],
                    "input_suffix": ".raw",
                    "input_order_file": "input_order.txt"
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.data.config import DataConfig
            data_config = DataConfig(
                name="raw_data",
                type="RawDataContainer",
                params_config={
                    "data_dir": "data",
                    "input_names": ["data"],
                    "input_shapes": [[1, 3, 224, 224]],
                    "input_dirs": ["."],
                    "input_suffix": ".raw",
                    "input_order_file": "input_order.txt"
                }
            )



Generic Data Config
-------------------

If no data config template can meet the requirement, we can also define the `data config <https://github.com/microsoft/Olive/blob/main/olive/data/config.py#L35>`_ directly. The data config is defined as a dictionary which includes the following fields:
    1. `name`: the name of the data config.
    2. `type`: the type name of the data config. Available `type`:
        - `DataContainer <https://github.com/microsoft/Olive/blob/main/olive/data/container/data_container.py#L17>`_ : the base class of all data config.
        - `DummyDataContainer <https://github.com/microsoft/Olive/blob/main/olive/data/template.py#L9>`_ in above section.
        - `HuggingfaceContainer <https://github.com/microsoft/Olive/blob/main/olive/data/template.py#L9>`_ in above section.
        - `RawDataContainer <https://github.com/microsoft/Olive/blob/main/olive/data/template.py#L72>`_ in above section.
    3. `components`: the dictionary of four `components <https://github.com/microsoft/Olive/blob/main/olive/data/constants.py#L12>`_ which contain:
        .. list-table:: Title
            :widths: 25 100
            :header-rows: 1

            * - Components
              - Available component type
            * - `load_dataset <https://github.com/microsoft/Olive/blob/main/olive/data/component/load_dataset.py>`_
              - local_dataset(default), simple_dataset, huggingface_dataset, raw_dataset
            * - `pre_process_data <https://github.com/microsoft/Olive/blob/main/olive/data/component/pre_process_data.py>`_
              - pre_process(default), huggingface_pre_process, ner_huggingface_preprocess, text_generation_huggingface_pre_process
            * - `post_process_data <https://github.com/microsoft/Olive/blob/main/olive/data/component/post_process_data.py>`_
              - post_process(default), text_classification_post_process, ner_post_process, text_generation_post_process
            * - `dataloader <https://github.com/microsoft/Olive/blob/main/olive/data/component/dataloader.py>`_
              - default_dataloader(default), skip_dataloader, no_auto_batch_dataloader

        each component can be customized by the following fields:
            - `name`: the name of the component.
            - `type`: the type name of the available component type. Besides the above available type in above table, user can also define their own component type in `user_script` with the way Olive does for `huggingface_dataset <https://github.com/microsoft/Olive/blob/main/olive/data/component/load_dataset.py#L26>`_. In this way, they need to provide `user_script` and `script_dir`. There is an `example <https://github.com/microsoft/Olive/blob/main/examples/snpe/inception_snpe_qualcomm_npu/user_script.py#L9>`_ with customized component type.
            - `params`: the dictionary of component function parameters. The key is the parameter name for given component type and the value is the parameter value.
    4. `user_script`: the user script path which contains the customized component type.
    5. `script_dir`: the user script directory path which contains the customized script.


Configs with built-in component:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then the complete config would be like:

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "data",
                "type": "DataContainer",
                "components": {
                    "load_dataset": {
                        "name": "_huggingface_dataset",
                        "type": "huggingface_dataset",
                        "params": {
                            "data_dir": null,
                            "data_name": "glue",
                            "subset": "mrpc",
                            "split": "validation",
                            "data_files": null
                        }
                    },
                    "pre_process_data": {
                        "name": "_huggingface_pre_process",
                        "type": "huggingface_pre_process",
                        "params": {
                            "model_name": "Intel/bert-base-uncased-mrpc",
                            "input_cols": [
                                "sentence1",
                                "sentence2"
                            ],
                            "label_cols": [
                                "label"
                            ],
                            "max_samples": null
                        }
                    },
                    "post_process_data": {
                        "name": "_text_classification_post_process",
                        "type": "text_classification_post_process",
                        "params": {}
                    },
                    "dataloader": {
                        "name": "_default_dataloader",
                        "type": "default_dataloader",
                        "params": {
                            "batch_size": 1
                        }
                    }
                },
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.data.config import DataConfig
            data_config = DataConfig(
                name="data",
                type="DataContainer",
                components={
                    "load_dataset": {
                        "name": "_huggingface_dataset",
                        "type": "huggingface_dataset",
                        "params": {
                            "data_dir": null,
                            "data_name": "glue",
                            "subset": "mrpc",
                            "split": "validation",
                            "data_files": null
                        }
                    },
                    "pre_process_data": {
                        "name": "_huggingface_pre_process",
                        "type": "huggingface_pre_process",
                        "params": {
                            "model_name": "Intel/bert-base-uncased-mrpc",
                            "input_cols": [
                                "sentence1",
                                "sentence2"
                            ],
                            "label_cols": [
                                "label"
                            ],
                            "max_samples": null
                        }
                    },
                    "post_process_data": {
                        "name": "_text_classification_post_process",
                        "type": "text_classification_post_process",
                        "params": {}
                    },
                    "dataloader": {
                        "name": "_default_dataloader",
                        "type": "default_dataloader",
                        "params": {
                            "batch_size": 1
                        }
                    }
                },
            )



Configs with customized component:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above case shows to rewrite all the components in data config. But sometime, there is no need to rewrite all the components. For example, if we only want to customize the `load_dataset` component for `DataContainer`, we can just rewrite the `load_dataset` component in the data config and ignore the other default components.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "data",
                "type": "DataContainer",
                "user_script": "user_script.py",
                "script_dir": "user_dir",
                "components": {
                    "load_dataset": {
                        "name": "_huggingface_dataset",
                        "type": "customized_huggingface_dataset",
                        "params": {
                            "data_dir": null,
                            "data_name": "glue",
                            "subset": "mrpc",
                        }
                    },
                },
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.data.registry import Registry

            @Registry.register_dataset()
            def customized_huggingface_dataset(output):
                ...

            from olive.data.config import DataConfig
            data_config = DataConfig(
                name="data",
                type="DataContainer",
                user_script="user_script.py",
                script_dir="user_dir",
                components={
                    "load_dataset": {
                        "name": "_huggingface_dataset",
                        "type": "customized_huggingface_dataset",
                        "params": {
                            "data_dir": null,
                            "data_name": "glue",
                            "subset": "mrpc",
                        }
                    },
                },
            )

.. note::
    User should provide the `user_script` and `script_dir` if they want to customize the component type. The `user_script` should be a python script which contains the customized component type. The `script_dir` should be the directory path which contains the `user_script`. Here is an example for `user_script`:

    .. code-block:: python

        from olive.data.registry import Registry

        @Registry.register_dataset()
        def customized_huggingface_dataset(dataset):
            ...

        @Registry.register_pre_process()
        def customized_huggingface_pre_process(dataset):
            ...

        @Registry.register_post_process()
        def customized_post_process(output):
            ...

        @Registry.register_dataloader()
        def customized_dataloader(dataset):
            ...

    More examples:
        1. inception_post_process:
            - user_script https://github.com/microsoft/Olive/blob/main/examples/snpe/inception_snpe_qualcomm_npu/user_script.py#L8-L10
            - json_config https://github.com/microsoft/Olive/blob/main/examples/snpe/inception_snpe_qualcomm_npu/inception_config.json#L14-L16
        2. dummy_dataset_dataroot:
            - user_script https://github.com/microsoft/Olive/blob/main/test/unit_test/test_data_root.py#L31
            - json_config https://github.com/microsoft/Olive/blob/main/test/unit_test/test_data_root.py#L107
