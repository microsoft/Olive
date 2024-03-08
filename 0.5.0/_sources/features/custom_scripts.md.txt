# Custom Scripts

Olive provides ability to write custom scripts for specific tasks such as loading a dataset using `user_script`.

## `user_script`
`user_script` can be added when you have your own classes and functions defined in a separate Python file. Olive will automatically use the classes and functions in your script.

`user_script` can be either a `str` path to your script file, or a `Path` object.

You can define your custom functions, and use them as attributes in configurations.

Olive supports following custom attributes in different configurations:

* `dataloader_func`: function name of your dataloader function.
* `post_processing_func`: function name of your post processing function.
* `evaluate_func`: function name of your evaluate function.
* `metric_func`: function name of your metric function (`OpenVINOQuantization` pass only).



### Examples

You can create your own `my_script.py` with `dataloader_func` and `post_processing_func`:
```
# my_script.py

class MyDataLoader:
    def __init__(self, data_dir, batch_size):
        ...

    def __len__(self):
        ...

    def __getitem__(self):
        ...

def create_dataloader(data_dir, batch_size):
    return MyDataloader(data_dir, batch_size)

def post_process(output):
    # your post processing logic here
    ...
```

Use `my_script.py` with Olive workflow configuration json file(sub_types name should be the returned dict key of your custom function):

```
"metrics":[
    {
        "name": "accuracy",
        "type": "accuracy",
        "sub_types": [
            {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
            {"name": "f1_score"},
            {"name": "auroc"}
        ],
        "user_config":{
            "post_processing_func": "post_process",
            "user_script": "user_script.py",
            "dataloader_func": "create_dataloader",
            "batch_size": 4
        }
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

```
# my_script.py

import file

# You can use file module here
print(file.my_val)
...
```

Use `script_dir` and `my_script.py` with Olive workflow configuration json file:

```
"metrics":[
    {
        "name": "accuracy",
        "type": "accuracy",
        "sub_types": [
            {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
            {"name": "f1_score"},
            {"name": "auroc"}
        ]
        "user_config":{
            "post_processing_func": "post_process",
            "user_script": "user_script.py",
            "script_dir": "my_modules"
            "dataloader_func": "create_dataloader",
            "batch_size": 4
        }
    }
]
```
