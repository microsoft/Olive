# Olive sample code instructions

## Prerequisites
Install the following:
* GCC 11.0 or higher for Linux
* Microsoft Visual Studio 2022 for Windows
* CMake

## Building sample code
Run the following commands in the sample code's directory.
```
mkdir build
cmake -S . -B build
cmake --build build
```

## Running the built binary
Run the following commands in the build directory.
```
./olive-genai-cpp-sample <Model's directory path>
```
