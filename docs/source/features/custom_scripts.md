# Custom Scripts

Olive provides ability to write custom scripts for specific tasks such as loading a dataset using `user_script`.

## `user_script`
`user_script` can be added when you have your own classes and functions defined in a separate Python file. Olive will automatically use the classes and functions in your script.

`user_script` can be either a `str` path to your script file, or a `Path` object.

You can define your custom functions, and use them as attributes in configurations.

Olive supports following custom attributes in different configurations:

* `evaluate_func`: function name of your evaluate function.
* `metric_func`: function name of your metric function (`OpenVINOQuantization` pass only).



### Examples

You can create your own `my_script.py` with `dataloader` and `post_process`:
```python
# my_script.py

from olive.data.registry import Registry

class MyDataLoader:
    def __init__(self, dataset, batch_size):
        ...

    def __len__(self):
        ...

    def __getitem__(self):
        ...

@Registry.register_dataloader()
def my_dataloader(dataset, batch_size):
    return MyDataloader(dataset, batch_size)

@Registry.register_post_process()
def my_post_process(output):
    # your post processing logic here
    ...
```

Use `my_script.py` with Olive workflow configuration json file(sub_types name should be the returned dict key of your custom function):

```json
"data_configs": [
    {
        "name": "accuracy_data_config",
        "type": "HuggingfaceContainer",
        "user_script": "user_script.py",
        "load_dataset_config": { "type": "skip_dataset" },
        "pre_process_data_config": { "type": "skip_pre_process" },
        "post_process_data_config": { "type": "my_post_process" },
        "dataloader_config": { "type": "my_dataloader", "params": { "batch_size": 4 } }
    }
],
"metrics":[
    {
        "name": "accuracy",
        "type": "accuracy",
        "data_config": "accuracy_data_config",
        "sub_types": [
            {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
            {"name": "f1_score"},
            {"name": "auroc"}
        ]
    }
]
```


## `script_dir`
`script_dir` is the directory where you collect your own modules that will be used in Olive. Olive will append this directory to your `sys.path` for your Python interpreter.

`script_dir` can be either a `str` path to your script file, or a `Path` object.

### Examples

You can place your own Python modules, let's say `file.py`, in `my_modules` directory

```
- my_olive_project
  - my_modules
    - file.py
  - my_script.py

```

Your script can use this module directly when specifying `script_dir` in configuration:

```python
# my_script.py

import file

# You can use file module here
print(file.my_val)
...
```

Use `script_dir` and `my_script.py` with Olive workflow configuration json file:

```json
"data_configs": [
    {
        "name": "accuracy_data_config",
        "type": "HuggingfaceContainer",
        "user_script": "user_script.py",
        "script_dir": "my_modules",
        "load_dataset_config": { "type": "skip_dataset" },
        "pre_process_data_config": { "type": "skip_pre_process" },
        "post_process_data_config": { "type": "my_post_process" },
        "dataloader_config": { "type": "my_dataloader", "params": { "batch_size": 4 } }
    }
],
"metrics":[
    {
        "name": "accuracy",
        "type": "accuracy",
        "data_config": "accuracy_data_config",
        "sub_types": [
            {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
            {"name": "f1_score"},
            {"name": "auroc"}
        ]
    }
]
```
