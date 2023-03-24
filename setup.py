import os

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


# use techniques described at https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
# Don't use technique 6 since it need extra dependencies.
VERSION = get_version("olive/__init__.py")

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")) as req_file:
    requirements = req_file.read().splitlines()

EXTRAS = {
    "azureml": ["azure-ai-ml>=0.1.0b6", "azure-identity"],
    "docker": ["docker"],
    "cpu": ["onnxruntime"],
    "gpu": ["onnxruntime-gpu"],
    "openvino": ["openvino==2022.3.0", "openvino-dev[tensorflow,onnx]==2022.3.0"],
    "tf": ["tensorflow==1.15.0"],
}

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

setup(
    name="olive-ai",
    version=VERSION,
    description="A deep learning model optimization toolchain",
    long_description="",
    author="Microsoft Corporation",
    author_email="olivedevteam@microsoft.com",
    license="MIT License",
    classifiers=CLASSIFIERS,
    url="https://microsoft.github.io/Olive/",
    download_url="https://github.com/microsoft/Olive/tags",
    packages=find_packages(exclude=("test", "examples*")),
    install_requires=requirements,
    extras_require=EXTRAS,
    include_package_data=True,
    package_data={},
    data_files=[],
    entry_points={},
)
