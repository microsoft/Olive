.. _how_to_configure_data:

How To Configure Data
=====================

Olive Data config organizes the data loading, preprocessing, batching and post processing into a single json config and defines several **popular templates** to serve Olive optimization.

The configuration of Olive data config is positioned under Olive run config with the field of ``data_configs``, and the data config is defined as a list of data config items. Here is an example of data config: `open_llama_sparsegpt_gpu <https://github.com/microsoft/Olive/blob/main/examples/open_llama/open_llama_sparsegpt_gpu.json#L11-L49>`_ .

.. code-block::

    "data_configs": [
        { "name": "dataset_1", ...},
        { "name": "dataset_2", ...},
    ]

Note: `name` for each dataset should be unique, and must be composed letters, numbers, and underscores.

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

Currently, we support the following data container which can be generated from ``olive.data.template``:

1. DummyDataContainer:
Convert the dummy data config to the data container.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "dummy_data_config_template",
                "type": "DummyDataContainer",
                "load_dataset_config": {
                    "input_shapes": [[1, 128], [1, 128], [1, 128]],
                    "input_names": ["input_ids", "attention_mask", "token_type_ids"],
                    "input_types": ["int64", "int64", "int64"],
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.data.config import DataConfig
            data_config = DataConfig(
                name="dummy_data_config_template",
                type="DummyDataContainer",
                load_dataset_config=DataComponentConfig(params={
                    "input_shapes": [[1, 128], [1, 128], [1, 128]],
                    "input_names": ["input_ids", "attention_mask", "token_type_ids"],
                    "input_types": ["int64", "int64", "int64"],
                })
            )

2. HuggingfaceContainer:
Convert the huggingface data config to the data container.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "huggingface_data_config",
                "type": "HuggingfaceContainer",
                "load_dataset_config": {
                    "data_name": "glue",
                    "split": "validation",
                    "subset": "mrpc"
                },
                "pre_process_data_config": {
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "task": "text-classification",
                    "input_cols": ["sentence1", "sentence2"],
                    "label_cols": ["label"]
                },
                "post_process_data_config": {
                    "task": "text-classification"
                },
                "dataloader_config": {
                    "batch_size": 1
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.data.config import DataConfig
            data_config = DataConfig(
                name="huggingface_data_config",
                type="HuggingfaceContainer",
                load_dataset_config=DataComponentConfig(params={
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "task": "text-classification",
                    "data_name": "glue",
                    "split": "validation",
                    "subset": "mrpc",
                }),
                pre_process_data_config=DataComponentConfig(params={
                    "input_cols": ["sentence1", "sentence2"],
                    "label_cols": ["label"],
                })
                dataloader_config=DataComponentConfig(params={
                    "batch_size": 1,
                })
            )


3. RawDataContainer:
Convert the raw data config to the data container.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "raw_data",
                "type": "RawDataContainer",
                "load_dataset_config": {
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
                load_dataset_config=DataComponentConfig(params={
                    "data_dir": "data",
                    "input_names": ["data"],
                    "input_shapes": [[1, 3, 224, 224]],
                    "input_dirs": ["."],
                    "input_suffix": ".raw",
                    "input_order_file": "input_order.txt"
                })
            )

4. TransformersDummyDataContainer:
Convert the transformer dummy data config to the data container.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "transformers_dummy_data_config",
                "type": "TransformersDummyDataContainer"
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.data.config import DataConfig
            data_config = DataConfig(
                name="transformers_dummy_data_config",
                type="TransformersDummyDataContainer",
                load_dataset_config=DataComponentConfig(params={
                    # model_name can be filled with the model name in input model's model_path
                    # if you start olive with olive run --config <config_path>
                    "model_name": "meta-llama/Llama-2-7b-hf"
                })
            )

    Also, based on ``TransformersDummyDataContainer``, Olive provides templates for transformer inference based on prompt(first prediction, no kv_cache now) and token(with kv_cache) inputs.
    - ``TransformersPromptDummyDataContainer`` where ``seq_len`` >= 1(default 8) and ``past_seq_len`` = 0.
    - ``TransformersTokenDummyDataContainer`` where ``seq_len`` == 1 and ``past_seq_len`` >= 1(default 8).


Generic Data Config
-------------------

If no data config template can meet the requirement, we can also define the `data config <https://github.com/microsoft/Olive/blob/main/olive/data/config.py#L35>`_ directly. The data config is defined as a dictionary which includes the following fields:
    1. ``name``: the name of the data config.
    2. ``type``: the type name of the data config. Available ``type``:
        - `DataContainer <https://github.com/microsoft/Olive/blob/main/olive/data/container/data_container.py#L17>`_ : the base class of all data config.
        - `DummyDataContainer <https://github.com/microsoft/Olive/blob/main/olive/data/template.py#L9>`_ in above section.
        - `HuggingfaceContainer <https://github.com/microsoft/Olive/blob/main/olive/data/template.py#L9>`_ in above section.
        - `RawDataContainer <https://github.com/microsoft/Olive/blob/main/olive/data/template.py#L72>`_ in above section.
    3. ``components``: the dictionary of four `components <https://github.com/microsoft/Olive/blob/main/olive/data/constants.py#L12>`_ which contain:
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
              - default_dataloader(default), no_auto_batch_dataloader

        each component can be customized by the following fields:
            - ``name``: the name of the component.
            - ``type``: the type name of the available component type. Besides the above available type in above table, user can also define their own component type in ``user_script`` with the way Olive does for `huggingface_dataset <https://github.com/microsoft/Olive/blob/main/olive/data/component/load_dataset.py#L26>`_. In this way, they need to provide ``user_script`` and ``script_dir``. There is an `example <https://github.com/microsoft/Olive/blob/main/examples/inception/user_script.py#L9>`_ with customized component type.
            - ``params``: the dictionary of component function parameters. The key is the parameter name for given component type and the value is the parameter value.
    4. ``user_script``: the user script path which contains the customized component type.
    5. ``script_dir``: the user script directory path which contains the customized script.


Configs with built-in component:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then the complete config would be like:

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "data",
                "type": "DataContainer",
                "load_dataset_config": {
                    "type": "huggingface_dataset",
                    "data_dir": null,
                    "data_name": "glue",
                    "subset": "mrpc",
                    "split": "validation",
                    "data_files": null
                },
                "pre_process_data_config": {
                    "type": "huggingface_pre_process",
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "input_cols": [
                        "sentence1",
                        "sentence2"
                    ],
                    "label_cols": [
                        "label"
                    ],
                    "max_samples": null
                },
                "post_process_data_config": {
                    "type": "text_classification_post_process"                },
                "dataloader_config": {
                    "type": "default_dataloader",
                    "batch_size": 1
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.data.config import DataConfig
            data_config = DataConfig(
                name="data",
                type="DataContainer",
                load_dataset_config=DataComponentConfig(
                    type="huggingface_dataset",
                    params={
                        "data_dir": null,
                        "data_name": "glue",
                        "subset": "mrpc",
                        "split": "validation",
                        "data_files": null
                    }
                ),
                pre_process_data_config=DataComponentConfig(
                    type="huggingface_pre_process",
                    params={
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
                ),
                post_process_data_config=DataComponentConfig(
                    type="text_classification_post_process",
                    params={}
                ),
                dataloader_config=DataComponentConfig(
                    type="default_dataloader",
                    params={
                        "batch_size": 1
                    }
                )
            )



Configs with customized component:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above case shows to rewrite all the components in data config. But sometime, there is no need to rewrite all the components. For example, if we only want to customize the ``load_dataset`` component for ``DataContainer``, we can just rewrite the ``load_dataset`` component in the data config and ignore the other default components.

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "name": "data",
                "type": "DataContainer",
                "user_script": "user_script.py",
                "script_dir": "user_dir",
                "load_dataset_config": {
                    "type": "customized_huggingface_dataset",
                    "data_dir": null,
                    "data_name": "glue",
                    "subset": "mrpc"
                }
            }

    .. tab:: Python Class

        .. code-block:: python

            from olive.data.registry import Registry

            @Registry.register_dataset()
            def customized_huggingface_dataset(_output, trust_remote_code=None, **kwargs):
                ...

            from olive.data.config import DataConfig
            data_config = DataConfig(
                name="data",
                type="DataContainer",
                user_script="user_script.py",
                script_dir="user_dir",
                load_dataset_config=DataComponentConfig(
                    type="customized_huggingface_dataset",
                    params={
                        "data_dir": null,
                        "data_name": "glue",
                        "subset": "mrpc",
                    }
                )
            )

.. note::
    User should provide the ``user_script`` and ``script_dir`` if they want to customize the component type. The ``user_script`` should be a python script which contains the customized component type. The ``script_dir`` should be the directory path which contains the ``user_script``. Another thing you might need to notice is that when your customized dataset is from huggingface, you should at least allow the ``trust_remote_code`` in your function's arguments list to indicate whether you trust the remote code or not. ``kwargs`` is the additional keyword arguments provided in the config, it can cover the case of ``trust_remote_code`` as well.
    Here is an example for ``user_script``:

    .. code-block:: python

        from olive.data.registry import Registry

        @Registry.register_dataset()
        def customized_huggingface_dataset(data_dir, **kwargs):
            # kwargs can cover the case of trust_remote_code or user can add trust_remote_code in the function's
            # arguments list, like, customized_huggingface_dataset(data_dir, trust_remote_code=None, **kwargs):
            ...

        @Registry.register_pre_process()
        def customized_huggingface_pre_process(dataset, **kwargs):
            ...

        @Registry.register_post_process()
        def customized_post_process(output):
            ...

        @Registry.register_dataloader()
        def customized_dataloader(dataset):
            ...

    More examples:
        1. inception_post_process:
            - user_script https://github.com/microsoft/Olive/blob/main/examples/inception/user_script.py#L8-L10
            - json_config https://github.com/microsoft/Olive/blob/main/examples/inception/inception_config.json#L14-L16

.. note::
    The components will be called with the following arguments along with any additional keyword arguments provided in the config:
        - load_dataset: ``data_dir`` (required, but the type can be `Optional[str]`)
        - pre_process_data: ``dataset`` (required, must be the first argument)
        - post_process_data: ``output`` (required, must be the first argument)
        - dataloader: ``dataset`` (required, must be the first argument)

    the required arguments for ``pre_process_data/post_process_data/dataloader`` must start with ``_`` to avoid the conflict with the additional keyword arguments provided in the config.
