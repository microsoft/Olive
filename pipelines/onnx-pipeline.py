import kfp.dsl as dsl
from kubernetes import client as k8s_client
from pathlib import PurePosixPath
class onnxConverterOp(dsl.ContainerOp):

  def __init__(self, name, 
  model,
  output_onnx_path, 
  model_type,
  model_inputs, 
  model_outputs,
  model_params, 
  model_input_shapes,
  target_opset):

    super(onnxConverterOp, self).__init__(
      name=name,
      image='ziylregistry.azurecr.io/onnx-converter:latest',
      arguments=[
        '--model', str(PurePosixPath('/mnt', model)),
        '--output_onnx_path', str(PurePosixPath('/mnt', output_onnx_path)),
        '--model_type',model_type,
        '--model_inputs', model_inputs,
        '--model_outputs', model_outputs,
        '--model_params', model_params,
        '--model_input_shapes', model_input_shapes,
        '--target_opset', target_opset
      ],
    file_outputs={'output': '/output.txt'})

class perfTestOp(dsl.ContainerOp):

  def __init__(self, name, 
  model, output_perf_result_path):

    super(perfTestOp, self).__init__(
      name=name,
      image='ziylregistry.azurecr.io/perf_test:latest',
      arguments=[
        model, str(PurePosixPath('/mnt', output_perf_result_path))
      ])

@dsl.pipeline(
  name='ONNX Ecosystem',
  description='A tool that allows ONNX model conversion and inference.'
)

# Create Onnx pipeline
# Parameters
# ----------
# model: string
#   The path of the model that needs to be converted
# output_onnx_path: string
#   The path to store the converted onnx model. Should end with ".onnx". e.g. output.onnx
# model_type: string
#   The name of original model framework. 
#   Available types are caffe, cntk, coreml, keras, libsvm, mxnet, scikit-learn, tensorflow and pytorch.
# output_perf_result_path:
#   The path to store the perf result text file. 
# model_inputs: string
#   Optional. The model's input names. Required for tensorflow frozen models and checkpoints.
# model_outputs: string
#   Optional. The model's output names. Required for tensorflow frozen models checkpoints.
# model_params: string
#   Optional. The params of the model if needed.
# model_input_shapes: list of tuple
#   Optional. List of tuples. The input shape(s) of the model. Each dimension separated by ','.
# target_opset: int
#   Optional. Specifies the opset for ONNX, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.    
def onnx_pipeline(
  model,
  output_onnx_path, 
  model_type,
  output_perf_result_path,
  model_inputs, 
  model_outputs,
  model_params, 
  model_input_shapes,
  target_opset):

  # Create a component named "Convert To Onnx" and "ONNXRuntime Perf". Edit the V1PersistentVolumeClaimVolumeSource 
  # name to match the persistent volume claim you created if needed. By default the names match ../azure-files-sc.yaml 
  # and ../azure-files-pvc.yaml
  convert_op = onnxConverterOp('Convert To Onnx', 
    '%s' % model, 
    '%s' % output_onnx_path, 
    '%s' % model_type,
    '%s' % model_inputs, 
    '%s' % model_outputs,
    '%s' % model_params,
    '%s' % model_input_shapes,
    '%s' % target_opset).add_volume(
        k8s_client.V1Volume(name='pipeline-nfs', persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
            claim_name='azurefile'))).add_volume_mount(k8s_client.V1VolumeMount(mount_path='/mnt', name='pipeline-nfs'))

  inference_op = perfTestOp('ONNXRuntime Perf', 
    convert_op.output,
    '%s' % output_perf_result_path
    ).add_volume(
        k8s_client.V1Volume(name='pipeline-nfs', persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
            claim_name='azurefile'))).add_volume_mount(
    k8s_client.V1VolumeMount(mount_path='/mnt', name='pipeline-nfs')).set_gpu_limit(1)

  dsl.get_pipeline_conf().set_image_pull_secrets([k8s_client.V1ObjectReference(name="regcred")])
if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(onnx_pipeline, __file__ + '.tar.gz')