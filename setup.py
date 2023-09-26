import json
import os

from setuptools import find_packages, setup

# ruff: noqa: PTH123


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


def get_extra_deps(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return json.load(fp)


# use techniques described at https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
# Don't use technique 6 since it needs extra dependencies.
VERSION = get_version("olive/__init__.py")
EXTRAS = get_extra_deps("olive/extra_dependencies.json")

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")) as req_file:
    requirements = req_file.read().splitlines()


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
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

long_description = (
    "Olive is an easy-to-use hardware-aware model optimization tool that composes industry-leading techniques across"
    " model compression, optimization, and compilation. Given a model and targeted hardware, Olive composes the best"
    " suitable optimization techniques to output the most efficient model(s) for inferencing on cloud or edge, while"
    " taking a set of constraints such as accuracy and latency into consideration."
)

description = long_description.split(".")[0] + "."

setup(
    name="olive-ai",
    version=VERSION,
    description=description,
    long_description=long_description,
    author="Microsoft Corporation",
    author_email="olivedevteam@microsoft.com",
    license="MIT License",
    classifiers=CLASSIFIERS,
    url="https://microsoft.github.io/Olive/",
    download_url="https://github.com/microsoft/Olive/tags",
    packages=find_packages(exclude=("test", "examples*")),
    python_requires=">=3.8.0",
    install_requires=requirements,
    extras_require=EXTRAS,
    include_package_data=True,
    package_data={},
    data_files=[],
    entry_points={
        "console_scripts": [
            "olive.scripts.manage_compute_instance = olive.scripts.manage_compute_instance:main",
        ],
    },
)
