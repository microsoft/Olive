.. _how_to_configure_pass:

How To Configure Pass
=====================

This document describes how to configure a Pass.

When configuring a Pass, the user can chose to set the values of parameters to their default value (no search), pre-defined search space
(search for the best value from the possible options) or a combination of the two (fix some parameters to a certain value, default or
user provided, and/or search for other parameters).

To fully configure a Pass, we require three things: :code:`type`, :code:`disable_search`, and :code:`config`.

* :code:`type`: This is the type of the Pass. Check out :ref:`passes` for the full list of supported Passes.
* :code:`disable_search`: This decides whether to use the default value (:code:`disable_search=True`) or the default searchable values,
  if any, (:code:`disable_search=False`) for the optional parameters. This is :code:`False` by default.
* :code:`config`: This is a dictionary of the config parameters and values. It must contain all required parameters. For optional parameters
  the default value or default searchable values (depending on whether :code:`disable_search` is :code:`True` or :code:`False`) can be
  overridden by providing user defined values. You can also assign the value for a specific parameter as :code:`"DEFAULT_VALUE"` to use the default
  value or :code:`"SEARCHABLE_VALUES"` to use the default searchable values (if available).

Let's take the example of the :ref:`onnx_quantization` Pass:

.. tabs::
    .. tab:: Config JSON

        .. code-block:: json

            {
                "type": "OnnxQuantization",
                "disable_search": false,
                "user_script": "./user_script.py",
                "dataloader_func": "glue_calibration_reader",
                // set per_channel to "DEFAULT_VALUE"
                "per_channel": "DEFAULT_VALUE",
                // set reduce_range to "SEARCHABLE_VALUES" value
                // redundant since disable_search is false
                "reduce_range": "SEARCHABLE_VALUES",
                // user defined value for weight_type
                "weight_type": "QUInt8"
            }

        .. note::
            :code:`type` is case insensitive.



    .. tab:: Python Class

        .. code-block:: python

            from olive.passes import OnnxQuantization
            from olive.passes.olive_pass import create_pass_from_dict

            onnx_quantization = create_pass_from_dict(OnnxQuantization,
                config={
                    "user_script": "./user_script.py",
                    "dataloader_func": "glue_calibration_reader",
                    # set per_channel to "DEFAULT_VALUE" value
                    "per_channel": "DEFAULT_VALUE",
                    # set reduce_range to "SEARCHABLE_VALUES"
                    # redundant since disable_search is false
                    "reduce_range": "SEARCHABLE_VALUES"
                    # user defined value for weight_type
                    "weight_type": "QUInt8"
                },
                disable_search=False
            )
