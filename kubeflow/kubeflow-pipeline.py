# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import kfp.dsl as dsl
from kubernetes import client as k8s_client
from pathlib import PurePosixPath
class onnxConverterOp(dsl.ContainerOp):

  def __init__(self, name, 
  model,
  output_onnx_path, 
  model_type,
  model_inputs_names, 
  model_outputs_names,
  model_input_shapes,
  model_initial_types,
  caffe_model_prototxt,
  target_opset):

    super(onnxConverterOp, self).__init__(
      name=name,
      image='mcr.microsoft.com/onnxruntime/onnx-converter:latest',
      arguments=[
        '--model', str(PurePosixPath('/mnt', model)),
        '--output_onnx_path', str(PurePosixPath('/mnt', output_onnx_path)),
        '--model_type',model_type,
        '--model_inputs_names', model_inputs_names,
        '--model_outputs_names', model_outputs_names,
        '--model_input_shapes', model_input_shapes,
        '--initial_types', model_initial_types,
        '--caffe_model_prototxt', str(PurePosixPath('/mnt', caffe_model_prototxt)),
        '--target_opset', target_opset
      ],
    file_outputs={'output': '/output.txt'})

class perfTestOp(dsl.ContainerOp):

  def __init__(self, name,  
  model, output_perf_result_path, execution_providers):

    super(perfTestOp, self).__init__(
      name=name,
      image='mcr.microsoft.com/onnxruntime/perf-tuning:latest',
      arguments=[
        "--model", model, 
        "--result", str(PurePosixPath('/mnt', output_perf_result_path)),
        "-e", execution_providers
      ])

@dsl.pipeline(
  name='ONNX Ecosystem',
  description='A tool that allows ONNX model conversion and inference.'
)

# Create ONNX pipeline
# Parameters
# ----------
# model: string
#   The path of the model that needs to be converted
# output_onnx_path: string
#   The path to store the converted ONNX model. Should end with ".onnx". e.g. output.onnx
# model_type: string
#   The name of original model framework. 
#   Available types are caffe, cntk, coreml, keras, libsvm, mxnet, scikit-learn, tensorflow and pytorch.
# output_perf_result_path:
#   The path to store the perf result text file. 
# model_inputs_names: string
#   Optional. The model's input names. Required for tensorflow frozen models and checkpoints.
# model_outputs_names: string
#   Optional. The model's output names. Required for tensorflow frozen models checkpoints.
# model_input_shapes: list of tuple
#   Optional. List of tuples. The input shape(s) of the model. Each dimension separated by ','.
# target_opset: int
#   Optional. Specifies the opset for ONNX, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.    
def onnx_pipeline(
  model,
  output_onnx_path, 
  model_type,
  output_perf_result_path,
  execution_providers="",
  model_inputs_names="", 
  model_outputs_names="",
  model_input_shapes="",
  model_initial_types="",
  caffe_model_prototxt="",
  target_opset=7):

  # Create a component named "Convert To ONNX" and "ONNX Runtime Perf". Edit the V1PersistentVolumeClaimVolumeSource 
  # name to match the persistent volume claim you created if needed. By default the names match ../azure-files-sc.yaml 
  # and ../azure-files-pvc.yaml
  convert_op = onnxConverterOp('Convert To ONNX', 
    '%s' % model, 
    '%s' % output_onnx_path, 
    '%s' % model_type,
    '%s' % model_inputs_names, 
    '%s' % model_outputs_names,
    '%s' % model_input_shapes,
    '%s' % model_initial_types,
    '%s' % caffe_model_prototxt,
    '%s' % target_opset).add_volume(
        k8s_client.V1Volume(name='pipeline-nfs', persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
            claim_name='azurefile'))).add_volume_mount(k8s_client.V1VolumeMount(mount_path='/mnt', name='pipeline-nfs'))   

  perf_op = perfTestOp('ONNX Runtime Perf', 
    convert_op.output,
    '%s' % output_perf_result_path,
    '%s' % execution_providers,
    ).add_volume(
        k8s_client.V1Volume(name='pipeline-nfs', persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
            claim_name='azurefile'))).add_volume_mount(
    k8s_client.V1VolumeMount(mount_path='/mnt', name='pipeline-nfs')).set_gpu_limit(1)

  dsl.get_pipeline_conf().set_image_pull_secrets([k8s_client.V1ObjectReference(name="regcred")])
if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(onnx_pipeline, __file__ + '.tar.gz')