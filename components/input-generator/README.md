## Generate-Input Image

This folder contains source code for generate-input image. Generate-input image is used to generate randomized inputs for a specified model if there's no input files exist. After running this image, a ./test_data_set_0 folder will be created with input_*.pb files at the same directory of the input model file. 

```
model.onnx
./test_data_set_0
    input_0.pb
    input_1.pb
    ...
```

## How to Run
First build image with docker
```
docker build -t generate-input .
```
Upon success, you should see you can run the docker image with customized parameters. 

```
docker run generate-input --model [YOUR MODEL FILE]
```

For detailed description of all available parameters, refer to the following. 
## Run parameters

**model**: string
   
   Required. The path of the model that needs to be converted.