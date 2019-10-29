// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
export const convert_form = {
  model: null,
  model_type: 'tensorflow',
  target_opset: '10',
  model_inputs_names: '',
  model_input_shapes: '',
  model_outputs_names: '',
  caffe_model_prototxt: '',
  initial_types: '',
  input_json: '',
  model_params: '',
};
export const perf_tuning_form = {
  model: '',
  config: 'RelWithDebInfo',
  test_mode: 'times',
  execution_provider: '',
  repeated_times: '20',
  duration_times: '10',
  parallel: true,
  intra_op_num_threads: '',
  num_threads: '',
  top_n: '3',
  optimization_level: '99',
};
