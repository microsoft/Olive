# Examples

This folder contains actively maintained examples of use of Olive with different models, optimization tools and hardware.

Each example is a self-contained folder with a `README.md` file that explains how to run it.

## Important

To ensure that the latest versions of the examples can be run without issues, you have to install Olive from source. We also recommend using a new [conda](#conda-env) or [virtual environment](#virtual-env).

To install Olive from source, run the following command in a new conda or virtual environment:

```bash
git clone https://github.com/microsoft/Olive.git
cd Olive
python -m pip install .
```

Then cd into the desired example folder and follow the instructions in the `README.md` file.

For examples corresponding to a specific release of Olive, checkout the corresponding tag. For instance, to use the examples corresponding to the `v0.2.0` release, run the following command:

```bash
git checkout tags/v0.2.0
```

### Conda env
To create a new conda environment and activate it, run the following command:

```bash
conda create -n olive-env python=3.8
conda activate olive-env
```
You can replace `olive-env` with any name you want and `python=3.8` with the version of python you want to use.

Please refer to the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more information on how to create and manage conda environments.

### Virtual env
To create a new virtual environment and activate it, run the following command:

On Linux:
```bash
python -m venv olive-env
source olive-env/bin/activate
```

On Windows (CMD):
```cmd
python -m venv olive-env
olive-env\Scripts\activate.bat
```

On Windows (PowerShell):
```powershell
python -m venv olive-env
olive-env\Scripts\Activate.ps1
```

You can replace `olive-env` with any path you want. A new folder will be created at the specified path to contain the virtual environment.

Please refer to the [python documentation](https://docs.python.org/3/library/venv.html) for more information on how to create and manage virtual environments.
