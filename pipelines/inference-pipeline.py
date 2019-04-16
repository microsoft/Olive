import kfp.dsl as dsl
from kubernetes import client as k8s_client

class TF2ONNXOp(dsl.ContainerOp):

  def __init__(self, name, 
  azure_storage_account_name,
  azure_storage_account_key, 
  azure_storage_container,
  input, output):

    super(TF2ONNXOp, self).__init__(
      name=name,
      image='ziyl/tf2onnx-azure:latest',
      arguments=[
        '--azure_storage_account_name', azure_storage_account_name,
        '--azure_storage_account_key', azure_storage_account_key,
        '--azure_storage_container',azure_storage_container,
        '--saved-model', input,
        '--output', output,
      ],
    file_outputs={'output': '/output.txt'})

class inferenceOp(dsl.ContainerOp):

  def __init__(self, name, 
    azure_storage_account_name,
    azure_storage_account_key, 
    azure_storage_container,
    input_onnx_model):

    super(inferenceOp, self).__init__(
      name=name,
      image='ziyl/onnxruntime-inference-azure:latest',
      arguments=[
        '--azure_storage_account_name', azure_storage_account_name,
        '--azure_storage_account_key', azure_storage_account_key,
        '--azure_storage_container',azure_storage_container,
        '--input_onnx_model', input_onnx_model,
      ])

@dsl.pipeline(
  name='ONNX Ecosystem',
  description='A tool that allows ONNX model conversion and inference.'
)


def onnx_converter_pipeline(
  azure_storage_account_name,
  azure_storage_account_key, 
  azure_storage_container,
  input_model_path, 
  output_model_name):

  convert_op = TF2ONNXOp('Convert To Onnx', 
    '%s' % azure_storage_account_name, 
    '%s' % azure_storage_account_key, 
    '%s' % azure_storage_container,
    '%s' % input_model_path, 
    '%s' % output_model_name)
  inference_op = inferenceOp('ONNX Runtime Inference', 
    '%s' % azure_storage_account_name, 
    '%s' % azure_storage_account_key, 
    '%s' % azure_storage_container,
    convert_op.output).set_gpu_limit(1)
	
if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(onnx_converter_pipeline, __file__ + '.tar.gz')