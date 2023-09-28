import os
from pathlib import Path

if __name__ == "__main__":
    try:
        qnn_env_path = Path(os.environ["QNN_ENV_PATH"]).resolve().as_posix()
    except KeyError:
        raise ValueError("QNN_ENV_PATH environment variable is not set") from None
    try:
        qnn_lib_path = Path(os.environ["QNN_LIB_PATH"]).resolve().as_posix()
    except KeyError as e:
        raise ValueError("QNN_LIB_PATH environment variable is not set") from e

    template_config_path = Path(__file__).parent / "mobilenet_config_template.json"

    config = None
    with template_config_path.open() as f:
        config = f.read()
        config = config.replace("<python-environment-path>", qnn_env_path)
        config = config.replace("<qnn-lib-path>", qnn_lib_path)

    with open("mobilenet_config.json", "w") as f:  # noqa: PTH123
        f.write(config)
