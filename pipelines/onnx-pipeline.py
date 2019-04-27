import kfp.dsl as dsl
from kubernetes import client as k8s_client
import os
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
      image='ziyl/onnx-converter:latest',
      arguments=[
        '--model', model,
        '--output_onnx_path', output_onnx_path,
        '--model_type',model_type,
        '--model_inputs', model_inputs,
        '--model_outputs', model_outputs,
        '--model_params', model_params,
        '--model_input_shapes', model_input_shapes,
        '--target_opset', target_opset
      ],
    file_outputs={'output': '/output.txt'})


@dsl.pipeline(
  name='ONNX Ecosystem',
  description='A tool that allows ONNX model conversion and inference.'
)


def onnx_converter_pipeline(
  model,
  output_onnx_path, 
  model_type,
  model_inputs, 
  model_outputs,
  model_params, 
  model_input_shapes,
  target_opset):

  convert_op = onnxConverterOp('Convert To Onnx', 
    '%s' % os.path.join('/mnt', model), 
    '%s' % os.path.join('/mnt', output_onnx_path), 
    '%s' % model_type,
    '%s' % model_inputs, 
    '%s' % model_outputs,
    '%s' % model_params,
    '%s' % model_input_shapes,
    '%s' % target_opset).add_volume(
        k8s_client.V1Volume(name='pipeline-nfs', persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
            claim_name='azurefile'))).add_volume_mount(k8s_client.V1VolumeMount(mount_path='/mnt', name='pipeline-nfs'))
#   inference_op = inferenceOp('ONNX Runtime Inference', 
#     '%s' % azure_storage_account_name, 
#     '%s' % azure_storage_account_key, 
#     '%s' % azure_storage_container,
#     convert_op.output).set_gpu_limit(1)
	
if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(onnx_converter_pipeline, __file__ + '.tar.gz')