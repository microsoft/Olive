# Onnx Pipeline with Kubernetes and Kubeflow

This repository shows how to deploy and use Onnx pipeline with Kubernetes and Kubeflow. 

# Prerequisites

## Create a Kubernetes Cluster

Follow https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough-portal to create Azure Kubernetes Cluster(AKS) from Azure portal. For the node size, choose Standard_NC6 for GPUs or Standard_D2_v2 if you just want CPUs. You can check a full list of NVIDIA GPUs (N-series) availability in [region availability documentation](https://azure.microsoft.com/en-us/global-infrastructure/services/?products=virtual-machines&regions=all).

### Enable GPU (Optional)

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


## Deploy Kubeflow

Follow the "Installation on existing Kubernetes" section in [Kubeflow official documentation](https://www.kubeflow.org/docs/started/getting-started/) to install Kubeflow. 

To validate your Kubeflow deployment,
``` 
kubectl get pods -n kubeflow
```

## Access Kubeflow Pipeline Dashboard
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
The public IP address is listed right next to *LoadBalancer Ingress*. Open the IP address with a browser and navigate to Kubeflow Pipeline by clicking "Pipeline Dashboard".

# ONNX Pipeline

ONNX pipeline consists of two parts. 
        
1) ONNX model conversion - Take models from other frameworks and convert them to ONNX.
2) ONNX performance tool - Run the converted model with ONNX Runtime and output the performance. 

### Pipeline Storage
Our pipeline consumes input from azure files. To upload desired input files, first we need to persist data to the pipeline using Azure files. 

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

For more details about configuring data volumes on AKS, refer to https://docs.microsoft.com/en-us/azure/aks/azure-files-dynamic-pv . Note that since our pipeline runs inside Kubeflow, the PVC should be created under the namespace "kubeflow". 

### Deploy ONNX pipeline
See [Pipelines](pipelines/README.md)

Once you've upload the generated ".tar" file to pipeline, click on the pipeline. 


# How to Run the Pipeline
// TODO some pics. 

## Run parameters

**model**: string
   
   The path of the model that needs to be converted.

**output_onnx_path**: string
   
   The path to store the converted onnx model. Should end with ".onnx". e.g. output.onnx

**model_type**: string
   
   The name of original model framework. 
   
   Available types are caffe, cntk, coreml, keras, libsvm, mxnet, scikit-learn, tensorflow and pytorch.
   
**output_perf_result_path**: string
   
   The path to store the perf result text file. 

**model_inputs**: string
   
   Optional. The model's input names. Required for tensorflow frozen models and checkpoints.

**model_outputs**: string
   
   Optional. The model's output names. Required for tensorflow frozen models checkpoints.

**model_params**: string

Optional. The params of the model if needed.

**model_input_shapes**: list of tuple
   
   Optional. List of tuples. The input shape(s) of the model. Each dimension separated by ','.

**target_opset**: int

Optional. Specifies the opset for ONNX, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.

