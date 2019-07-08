# ONNX Automation Pipeline with Kubernetes and Kubeflow

This guide shows how to deploy and use ONNX Automation pipeline with Kubernetes and Kubeflow. 

## Table of Contents
1. [Prerequisites](#prerequisites)
    - [Create a Kubernetes Cluster](#Create-a-Kubernetes-Cluster)
    - [Install and Deploy Kubeflow](#Install-And-Deploy-Kubeflow)
2. [Deploy Onnx Automation Pipeline](#Deploy-Onnx-Automation-Pipeline)
    - [Pipeline Storage](#Pipeline-Storage)
    - [Deploy ONNX Pipeline](#Deploy-ONNX-Pipeline)
3. [Run Onnx Automation Pipeline](#Run-Onnx-Automation-Pipeline)
    - [Run Parameters](#Run-Parameters)
    - [Components Source](#Components-Source)

## Prerequisites
Before using Kubeflow pipeline, you'll need a Kubernetes cluster. This quickstart provides instruction on how to set up 

### Create a Kubernetes Cluster

Follow https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough-portal to create Azure Kubernetes Cluster(AKS) from Azure portal. For the node size, choose Standard_NC6 for GPUs or Standard_D2_v2 if you just want CPUs. You can check a full list of NVIDIA GPUs (N-series) availability in [region availability documentation](https://azure.microsoft.com/en-us/global-infrastructure/services/?products=virtual-machines&regions=all).

#### Enable GPU (Optional)

If you provisioned GPU VM, install NVIDIA Device Plugin using:

For Kubernetes 1.10:

```bash
kubectl apply -f https://raw.githubusercontent.com/nvidia/k8s-device-plugin/v1.10/nvidia-device-plugin.yml
```

For Kubernetes 1.11 and above:

```bash
kubectl apply -f https://raw.githubusercontent.com/nvidia/k8s-device-plugin/v1.11/nvidia-device-plugin.yml
```

For AKS Engine, NVIDIA Device Plugin will automatically installed with N-Series GPU clusters.

Once you are done with the cluster creation, and downloaded the `kubeconfig` file, running the following command:

```console
kubectl get nodes
```

Should yield an output similar to this one:

```
NAME                       STATUS    ROLES     AGE       VERSION
aks-nodepool1-42640332-0   Ready     agent     1h        v1.11.1
aks-nodepool1-42640332-1   Ready     agent     1h        v1.11.1
aks-nodepool1-42640332-2   Ready     agent     1h        v1.11.1
```

If you provisioned GPU VM, describing one of the node should indicate the presence of GPU(s) on the node:

```console
> kubectl describe node <NODE_NAME>

[...]
Capacity:
 nvidia.com/gpu:     1
[...]
```

> Note: In some scenarios, you might not see GPU resources under Capacity. To resolve this, you must install a daemonset as described in the troubleshooting section here: https://docs.microsoft.com/en-us/azure/aks/gpu-cluster


### Install and Deploy Kubeflow

Follow the "Installation on existing Kubernetes" section in [Kubeflow official documentation](https://www.kubeflow.org/docs/started/getting-started/) to install Kubeflow. 

To validate your Kubeflow deployment,
``` 
kubectl get pods -n kubeflow
```

#### Access Kubeflow Pipeline Dashboard
To connect to Kubeflow pipeline dashboard over a public IP:

Update the default service created for Kubeflow pipeline to type LoadBalancer.

```
cd ks_app
ks param set ambassador ambassadorServiceType LoadBalancer
cd ..
${KUBEFLOW_SRC}/scripts/kfctl.sh apply k8s
```

Get the public IP address by running 
```
kubectl describe services -n kubeflow ambassador
```
which will produce output like this
```
Name:                     ambassador
Namespace:                kubeflow
Labels:                   app.kubernetes.io/
...
Selector:                 service=ambassador
Type:                     LoadBalancer
IP:                       10.0.32.215
LoadBalancer Ingress:     40.124.2.23
Port:                     ambassador  80/TCP
TargetPort:               80/TCP
NodePort:                 ambassador  32599/TCP
Endpoints:                10.244.0.12:80,10.244.1.5:80,10.244.2.4:80
...
```
The public IP address is listed right next to *LoadBalancer Ingress*. Open the IP address with a browser and navigate to Kubeflow Pipeline by clicking "Pipeline Dashboard". Kubeflow pipelines are ready to use. 

## Deploy ONNX Automation Pipeline

ONNX pipeline consists of two parts. 
        
1) **ONNX model converter** - Take a model from one of the supported frameworks and convert them to ONNX.
2) **ONNX performance tuning tool** - Run the converted model with ONNX Runtime, tune its performance by trying different execution providers and environment vairiables combinations, and output the best performance results. 

### Pipeline Storage
To pass any model files as inputs to the pipeline, you'll need to first upload them to a volume that the pipeline is able to access. Our pipeline recommends storing inputs from Azure File Share. Follow the steps below to persist data to ONNX automation pipeline using Azure files. 

*i.* Create a storage class
```
kubectl apply -f azure-files/azure-file-sc.yaml
```

*ii.* Create cluster role and binding
```
kubectl apply -f azure-files/azure-pvc-roles.yaml
```
*iii.* Create a persistent volume claim(PVC) named "azurefile"
```
kubectl apply -f azure-files/azure-file-pvc.yaml
```

To verify the creation of azurefile pvc, run
```
kubectl get pvc azurefile -n kubeflow
```
If succeeded, you will see something similar to this.

```
NAME        STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
azurefile   Bound    pvc-73bb032f-6075-11e9-8806-9a0060c93a61   5Gi        RWX            azurefile      22d
```

Then you can find the Azure files under your resource group and upload the files you need.

1) Click on the AKS resource group on your Azure portal (the resource group name is usually "MC_{YOUR AKS NAME}_{YOUR REGION})". 
2) Find and click on the storage account under the resource group. 
3) Click "Files"
4) Find the file share with name matches your "kubectl get pvc azurefile -n kubeflow" output. 
5) Upload/Create files in that file share.

Our pipeline will automatically search for input files in this directory. You can see the file share folder as the "root directory" for any input path. For example, if you want to use a file name `model.pb` located just under file share, you would type `model.pb` as input path in the pipeline parameter field.

For more details about configuring data volumes on AKS, refer to https://docs.microsoft.com/en-us/azure/aks/azure-files-dynamic-pv . Note that since our pipeline runs inside Kubeflow, the PVC should be created under the namespace "kubeflow". 

### Deploy ONNX pipeline to Kubeflow

The Kubeflow pipeline is built using [Kubeflow pipeline SDK](https://www.kubeflow.org/docs/pipelines/sdk/). For conceptual understanding, please refer to [Kubeflow Concepts](https://www.kubeflow.org/docs/pipelines/concepts/)

#### Requirements
Install Kubeflow pipeline SDK:
```
pip install https://storage.googleapis.com/ml-pipeline/release/0.1.14/kfp.tar.gz --upgrade
```

#### Compiling the pipeline template

```bash
python inference-pipeline.py
```

#### Upload the Built Pipeline

Open the Kubeflow pipelines UI. Click "Upload" on the top right corner, and then upload the compiled specification (`.tar.gz` file) as a new pipeline template.

Create a new pipeline by clicking on the uploaded pipeline, and follow the UI instructions. 

## Run Onnx Automation Pipeline
### Run parameters


`--model`: Required or specified in input json. The path of the model that needs to be converted.

`--output_onnx_path`: Required or specified in input json. The path to store the converted onnx model. Should end with ".onnx". e.g. "/newdir/output.onnx". A clean directory is recommended. 
   
`--output_perf_result_path`: The path to store the perf result text file. 

`--model_type`: Required or specified in input json. The name of original model framework. Available types are cntk, coreml, keras, scikit-learn, tensorflow and pytorch.

`--model_inputs_names`: Optional. The model's input names. Required for tensorflow frozen models and checkpoints.

`--model_outputs_names`: Optional. The model's output names. Required for tensorflow frozen models checkpoints.

`--model_params`: Optional. The params of the model if needed.

`--model_input_shapes`: Optional. List of tuples. The input shape(s) of the model. Each dimension separated by ','.

`--initial_types`: Optional. List of tuples. The initial types of model for onnxmltools

`--caffe_model_prototxt`: Optional. The path of the .prototxt file for caffe model.

`--target_opset`: Optional. Specifies the opset for ONNX, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.

### Components source
- ONNX Converter:

  [Source code](../components/onnx-converter)
  
  Container: TBD

- Perf Tuning:

  [Source code](../components/perf_test)

  Container: TBD