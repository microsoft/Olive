import os

from setuptools import setup, find_packages


def get_install_requires():
    install_requires = ["numpy", "onnx", "psutil", "coloredlogs", "sympy", "docker==5.0.0", "six", "onnxconverter_common"]
    return install_requires


def get_package_data():
    package_data = {"olive": [os.path.join(r.replace('olive/', ''), file) for r,d,f in os.walk('olive/') for file in f]}
    return package_data


setup(
    name="onnxruntime-olive",
    version="0.1.0",
    description="ONNX model conversion and optimization techniques",
    author="OLive-Team",
    author_email="olive-team@microsoft.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
        ],
    packages=find_packages(),
    package_data=get_package_data(),
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": ["olive=olive.__main__:main"]},
    install_requires=get_install_requires())
