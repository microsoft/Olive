# How to configure a Workflow Pass

This document describes how to configure an Olive workflow Pass.

When configuring a Pass, the user can chose to set the values of parameters to their default value (no search), pre-defined search space
(search for the best value from the possible options) or a combination of the two (fix some parameters to a certain value, default or
user provided, and/or search for other parameters).

To fully configure a Pass, we require three things: `type`, `disable_search`, and `config`.

- `type`: This is the type of the Pass. Check out :ref:`passes` for the full list of supported Passes.
- `disable_search`: This decides whether to use the default value (`disable_search=True`) or the default searchable values,
  if any, (`disable_search=False`) for the optional parameters. This is :code:`False` by default.
- `config`: This is a dictionary of the config parameters and values. It must contain all required parameters. For optional parameters
  the default value or default searchable values (depending on whether `disable_search` is `True` or `False`) can be
  overridden by providing user defined values. You can also assign the value for a specific parameter as `DEFAULT_VALUE` to use the default
  value or `SEARCHABLE_VALUES` to use the default searchable values (if available).

Here is an example of how to configure the `OnnxQuantization` Pass:

```json
{
    "type": "OnnxQuantization",
    "disable_search": false,
    "data_config": "calib_data_config",
    // set per_channel to "DEFAULT_VALUE"
    "per_channel": "DEFAULT_VALUE",
    // set reduce_range to "SEARCHABLE_VALUES" value
    // redundant since disable_search is false
    "reduce_range": "SEARCHABLE_VALUES",
    // user defined value for weight_type
    "weight_type": "QUInt8"
}
```

```{Note}
`type` is case insensitive.
```
