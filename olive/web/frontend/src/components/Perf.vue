<template>
    <div class="container" style="margin-top: 2%">
        <!-- perf test-->
        <div ref="perf_tuningModel"
                id="perf_tuning-model"
                title="Perf Test"
                hide-footer>

            <b-form @submit="perf_tuning" @reset="onReset_perf_tuning">
              <b-form-group id="form-model_type-group"
                        label="Performance Tuning Job Name:"
                        label-for="form-model_type-input"
                        label-class="font-weight-bold">
              <b-form-input v-model="job_name"></b-form-input>
              </b-form-group>

              <b-form-group id="form-onnxruntime_version-group"
                          label="onnxruntime version for optimization:"
                          label-for="form-onnxruntime_version-input"
                          label-class="font-weight-bold">
              <b-form-select v-model="perf_tuning_form.onnxruntime_version"
                          :options="options.onnxruntime_version"
                          label="onnxruntime version:"
                          class="mb-3">
                  <template slot="first">
                  </template>
              </b-form-select>
              </b-form-group>

              <b-form-checkbox
                    id="use_gpu"
                    v-model="perf_tuning_form.use_gpu"
                    name="use_gpu">
                    Use onnxruntime GPU package for optimization
              </b-form-checkbox>
              <hr/>
              <b-form-group id="form-optimization_option-group"
                          label="Optimization Option:"
                          label-for="form-optimization_option-input"
                          label-class="font-weight-bold">
              <b-form-select v-model="optimization_option">
                  <template slot="first"></template>
                  <option value=0>Run With Configuaration JSON File</option>
                  <option value=1>Run With Inline Arguments</option>
                </b-form-select>
              </b-form-group>
                <hr/>

              <div v-if="optimization_option == 0">
              <b-form-group id="form-optimization_config-group"
                        label="Optimization Configuration JSON File:"
                        label-for="form-optimization_config-input"
                        label-class="font-weight-bold"
                        v-if="optimization_option == 0">
                <b-form-file id="form-optimization_config-input"
                        v-model="optimization_config"
                        placeholder="Select your optimization configuration json file">
                </b-form-file>
              </b-form-group>
              </div>

              <div v-if="optimization_option == 1">

                <b-form-group id="form-model-group"
                        label="Model:"
                        label-for="form-model"
                        label-class="font-weight-bold">
                  <b-form-file id="form-model"
                                  v-model="customized_model"
                                  required
                                  placeholder="Choose a model...">
                  </b-form-file>
                </b-form-group>

                <b-row class="missing">
                    {{ model_missing }}
                </b-row>

                <div class="open_button" v-on:click="adv_setting = !adv_setting">
                    Advanced Settings
                </div>
                <br/>

              <div v-if="adv_setting">
                    <b-form-group id="form-input_names-group"
                              label="Model Inputs Names:"
                              label-for="form-input_names-input"
                              label-class="font-weight-bold">
                        <b-form-input id="form-input_names-input"
                            type="text"
                            v-model="perf_tuning_form.input_names"
                            value="20"
                            placeholder="Enter Model Inputs Names with Comma Seperated. e.g. input_1,input_2,input3">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-output_names-group"
                              label="Model Output Names:"
                              label-for="form-output_names-input"
                              label-class="font-weight-bold">
                        <b-form-input id="form-output_names-input"
                            type="text"
                            v-model="perf_tuning_form.output_names"
                            value="20"
                            placeholder="Enter Model Outputs Names with Comma Seperated. e.g. output_1,output_2">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-input_shapes-group"
                              label="Model Inputs Shapes:"
                              label-for="form-input_shapes-input"
                              label-class="font-weight-bold">
                        <b-form-input id="form-input_shapes-input"
                            type="text"
                            v-model="perf_tuning_form.input_shapes"
                            value="20"
                            placeholder="Enter List of Shapes of each Input Node. e.g. [[1,7],[1,7],[1,7]]">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-sample_input_data_path-group"
                            label="Model Sample Input Data Path:"
                            label-for="form-sample_input_data_path-input"
                            label-class="font-weight-bold">
                    <b-form-file id="form-sample_input_data_path-input"
                            v-model="sample_input_data_path"
                            placeholder="Select your sample input data">
                    </b-form-file>
                    </b-form-group>

                    <b-form-group id="form-providers_list-group"
                                label="Execution Providers (choose multiple):"
                                label-for="form-providers_list-input"
                                label-class="font-weight-bold">
                    <b-form-select multiple class="form-control" v-model="selected_eps"
                                    :options="options.providers_list"
                                    label="Execution Providers (choose multiple):"
                                    >
                        <template slot="first">
                        </template>
                        </b-form-select>
                    </b-form-group>

                    <b-form-checkbox
                    id="trt_fp16_enabled"
                    v-model="perf_tuning_form.trt_fp16_enabled"
                    name="trt_fp16_enabled">
                    Enable TensorRT FP16 mode optimization
                    </b-form-checkbox>

                    <b-form-checkbox
                    id="quantization_enabled"
                    v-model="perf_tuning_form.quantization_enabled"
                    name="quantization_enabled">
                    Enable quantization optimization
                    </b-form-checkbox>

                    <b-form-checkbox
                    id="transformer_enabled"
                    v-model="perf_tuning_form.transformer_enabled"
                    name="transformer_enabled">
                    Enable transformer optimization
                    </b-form-checkbox>

                    <b-form-group id="form-transformer_args-group"
                                label="Transformer Arguments:"
                                label-for="form-transformer_args-input"
                                label-class="font-weight-bold"
                                v-if="perf_tuning_form.transformer_enabled">
                        <b-form-input id="form-transformer_args-input"
                                    type="text"
                                    v-model="perf_tuning_form.transformer_args"
                                    placeholder="Enter transformer arguments">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-execution_mode_list-group"
                                label="Execution Modes (choose multiple):"
                                label-for="form-execution_mode_list-input"
                                label-class="font-weight-bold">
                    <b-form-select multiple class="form-control" v-model="selected_execution_modes"
                                    :options="options.execution_mode_list"
                                    label="Execution Modes (choose multiple):"
                                    >
                        <template slot="first">
                        </template>
                        </b-form-select>
                    </b-form-group>

                    <b-form-group id="form-intra_thread_num_list-group"
                                label="Number of Intra Threads:"
                                label-for="form-intra_thread_num_list-input"
                                label-class="font-weight-bold">
                        <b-form-input id="form-intra_thread_num_list-input"
                            type="text"
                            v-model="perf_tuning_form.intra_thread_num_list"
                            value="20"
                            placeholder="Enter intra_thread_num_list. e.g. 1,2,4">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-inter_thread_num_list-group"
                                label="Number of Inter Threads:"
                                label-for="form-inter_thread_num_list-input"
                                label-class="font-weight-bold">
                        <b-form-input id="form-inter_thread_num_list-input"
                            type="text"
                            v-model="perf_tuning_form.inter_thread_num_list"
                            value="20"
                            placeholder="Enter inter_thread_num_list. e.g. 1,2,4">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-omp_max_active_levels-group"
                                label="OMP_MAX_ACTIVE_LEVELS:"
                                label-for="form-omp_max_active_levels-input"
                                label-class="font-weight-bold">
                        <b-form-input id="form-omp_max_active_levels-input"
                            type="text"
                            v-model="perf_tuning_form.omp_max_active_levels"
                            value="20"
                            placeholder="Enter omp_max_active_levels.">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-omp_wait_policy_list-group"
                                label="OMP WAIT POLICY (choose multiple):"
                                label-for="form-omp_wait_policy_list-input"
                                label-class="font-weight-bold">
                    <b-form-select multiple class="form-control" v-model="selected_omp_wait_policy"
                                    :options="options.omp_wait_policy_list"
                                    label="OMP WAIT POLICY (choose multiple):"
                                    >
                        <template slot="first">
                        </template>
                        </b-form-select>
                    </b-form-group>

                    <b-form-group id="form-ort_opt_level_list-group"
                                label="Optimization Level (choose multiple):"
                                label-for="form-ort_opt_level_list-input"
                                label-class="font-weight-bold">
                    <b-form-select multiple class="form-control" v-model="selected_opt_levels"
                                    :options="options.ort_opt_level_list"
                                    label="Optimization Level (choose multiple):"
                                    >
                        <template slot="first">
                        </template>
                        </b-form-select>
                    </b-form-group>

                    <b-form-group id="form-concurrency_num-group"
                                label="Number of Concurrency:"
                                label-for="form-concurrency_num-input"
                                label-class="font-weight-bold">
                        <b-form-input id="form-concurrency_num-input"
                            type="text"
                            v-model="perf_tuning_form.concurrency_num"
                            value="20"
                            placeholder="Enter concurrency_num.
                            If leave blank, concurrency_num will be set to 1.">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-test_num-group"
                                label="Repeated times for test:"
                                label-for="form-test_num-input"
                                label-class="font-weight-bold">
                        <b-form-input id="form-test_num-input"
                                    type="text"
                                    v-model="perf_tuning_form.test_num">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-warmup_num-group"
                                label="Repeated times for warmup:"
                                label-for="form-warmup_num-input"
                                label-class="font-weight-bold">
                        <b-form-input id="form-warmup_num-input"
                                    type="text"
                                    v-model="perf_tuning_form.warmup_num">
                        </b-form-input>
                    </b-form-group>

                </div>
                </div>
                <hr/>
                <b-button type="submit"
                  :disabled="model_running"
                  variant="primary" class="button_right">Submit</b-button>
                <b-button type="reset" :disabled="model_running" variant="danger">Reset</b-button>
            </b-form>
        </div>
        <hr/>
        <alert :message=message :link=link :loading=model_running v-if="show_message"></alert>
    </div>
</template>

<script>
import axios from 'axios';
import Alert from './Alert.vue';
import { perf_tuning_form } from '../utils/const';

const origin_perf_tuning_form = Object.assign({}, perf_tuning_form);
export default {
  name: 'Perf',
  props: ['onnx_model'],
  data() {
    return {
      selected: -1,
      perf_tuning_form,
      selected_eps: [],
      optimization_option: 3,
      selected_execution_modes: [],
      selected_opt_levels: [],
      selected_omp_wait_policy: [],
      options: {
        onnxruntime_version: ['1.8.1', '1.9.0'],
        execution_mode_list: ['parallel', 'sequential'],
        providers_list: [
          { value: '', text: 'Run all EPs (default)' },
          { value: 'cpu', text: 'CPU' },
          { value: 'dnnl', text: 'DNNL (only can be selected when onnxruntime cpu version installed)' },
          { value: 'openvino', text: 'OPENVINO (only can be selected when onnxruntime cpu version installed)' },
          { value: 'cuda', text: 'CUDA (only can be selected when onnxruntime gpu version installed)' },
          { value: 'tensorrt', text: 'TensorRT (only can be selected when onnxruntime gpu version installed)' },
        ],
        ort_opt_level_list: ['disable', 'basic', 'extended', 'all'],
        omp_wait_policy_list: ['ACTIVE', 'PASSIVE'],
      },
      message: '',
      show_message: false,
      model_missing: '',
      adv_setting: false,
      model_running: false,
      host: `${window.location.protocol}//${window.location.host.split(':')[0]}`,
      link: '',
      job_id: '',
      job_name: `perf-tuning-${Date.now()}`,
      prev_job_list: [],
    };
  },

  components: {
    alert: Alert,
  },

  methods: {
    init_perf_tuning_form() {
      this.perf_tuning_form = Object.assign({}, origin_perf_tuning_form);
      this.perf_tuning_form.model = this.onnx_model;
      this.customized_model = null;
      this.selected_eps = [];
      this.selected_execution_modes = [];
      this.selected_opt_levels = [];
      this.selected_omp_wait_policy = [];
      this.sample_input_data_path = null;
      this.optimization_config = null;
      this.optimization_option = 3;
      this.onnxruntime_version = 0;
    },

    get_jobs() {
      axios.get(`${this.host}:5000/gettasks`)
        .then((res) => {
          this.prev_job_list = [];
          for (let i = 0; i < Object.keys(res.data).length; i++) {
            const t = res.data[Object.keys(res.data)[i]];
            const nameBrkIndex = t.name.indexOf('.');
            if (t.state == 'SUCCESS' && t.name.substring(0, nameBrkIndex) == 'convert') {
              if ((Date.now() / 1000 - t.timestamp) / 3600 < 24 * 7) {
                this.prev_job_list.push({
                  value: Object.keys(res.data)[i],
                  text: t.name.substring(nameBrkIndex + 1),
                  timestamp: t.timestamp,
                });
              }
            }
          }
          this.prev_job_list.sort((a, b) => a.timestamp - b.timestamp);
          if (this.prev_job_list.length > 0) {
            this.selected_converter_job = this.prev_job_list[this.prev_job_list.length - 1].value;
          }
        });
    },

    perf_tuning(evt) {
      this.close_all();
      this.model_running = true;
      this.show_message = true;
      this.message = `Submitting job ${this.job_name}`;
      evt.preventDefault();

      // run with selected execution providers
      if (this.selected_eps.length > 0) {
        this.perf_tuning_form.providers_list = '';
        for (let i = 0; i < this.selected_eps.length; i++) {
          this.perf_tuning_form.providers_list += this.selected_eps[i];
          if (i < this.selected_eps.length - 1) {
            this.perf_tuning_form.providers_list += ',';
          }
        }
      }

      // run with selected execution modes
      if (this.selected_execution_modes.length > 0) {
        this.perf_tuning_form.execution_mode_list = '';
        for (let i = 0; i < this.selected_execution_modes.length; i++) {
          this.perf_tuning_form.execution_mode_list += this.selected_execution_modes[i];
          if (i < this.selected_execution_modes.length - 1) {
            this.perf_tuning_form.execution_mode_list += ',';
          }
        }
      }

      // run with selected optimization levels
      if (this.selected_opt_levels.length > 0) {
        this.perf_tuning_form.ort_opt_level_list = '';
        for (let i = 0; i < this.selected_opt_levels.length; i++) {
          this.perf_tuning_form.ort_opt_level_list += this.selected_opt_levels[i];
          if (i < this.selected_opt_levels.length - 1) {
            this.perf_tuning_form.ort_opt_level_list += ',';
          }
        }
      }

      // run with selected omp wait policy
      if (this.selected_omp_wait_policy.length > 0) {
        this.perf_tuning_form.omp_wait_policy_list = '';
        for (let i = 0; i < this.selected_omp_wait_policy.length; i++) {
          this.perf_tuning_form.omp_wait_policy_list += this.selected_omp_wait_policy[i];
          if (i < this.selected_omp_wait_policy.length - 1) {
            this.perf_tuning_form.omp_wait_policy_list += ',';
          }
        }
      }

      const metadata = this.perf_tuning_form;
      const json = JSON.stringify(metadata);

      const blob = new Blob([json], {
        type: 'application/json',
      });
      const data = new FormData();
      data.append('metadata', blob);
      data.append('job_name', this.job_name);
      data.append('file', this.customized_model);
      data.append('sample_input_data_path', this.sample_input_data_path);
      data.append('optimization_config', this.optimization_config);

      axios.post(`${this.host}:5000/perf_tuning`, data)
        .then((res) => {
          this.link = `${this.host}:5000/perfresult/${res.data.job_id}`;
          this.model_running = false;
          this.show_message = true;
          this.message = 'Running job at ';
          this.update_result(res.data.job_id);
        })
        .catch((error) => {
          // eslint-disable-next-line
          this.message = error.toString();
          this.model_running = false;
        });
      this.job_name = `perf-tuning-${Date.now()}`;
    },

    update_result(location) {
      axios.get(`${this.host}:5000/perfstatus/${location}`)
        .then((res) => {
          if (res.data.state == 'SUCCESS') {
            this.$emit('update_model', res.data.optimized_model);
            this.message = 'Job succeeded. Result at ';
          } else if (res.data.state == 'FAILURE') {
            this.message = 'Job completed. Result at ';
            // this.show_logs = true;
          } else {
            // rerun in 10 seconds
            setTimeout(() => this.update_result(location), 10000);
          }
        })
        .catch((error) => {
          this.message = error.toString();
        });
    },
    onReset_perf_tuning(evt) {
      evt.preventDefault();
      this.init_perf_tuning_form();
    },
    close_all() {
      this.result = [];
      this.show_message = false;
    },
  },
};
</script>
