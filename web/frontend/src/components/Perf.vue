// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
<template>
    <div class="container">
        <!-- perf test-->
        <div ref="perf_tuningModal"
                id="perf_tuning-modal"
                title="Perf Test"
                hide-footer>

            <b-form-select v-model="run_option">
              <template slot="first"></template>
              <option value=0>Run With Model Converted From Previous Step</option>
              <option value=1>Run With Customized Model</option>
            </b-form-select>
            <b-form @submit="perf_tuning" @reset="onReset_perf_tuning" style="margin-top: 2%">
              <b-form-group id="form-model_type-group"
                        label="Job Name:"
                        label-for="form-model_type-input">
              <b-form-input v-model="job_name" placeholder="app.convert"></b-form-input>
            </b-form-group>
                <b-form-group id="form-model-group"
                              v-if="run_option == 0"
                              label="Model From Previous Step:"
                              label-for="form-model-input">
                    <b-form-input id="form-model-input"
                                    type="text"
                                    v-model="perf_tuning_form.model"
                                    readonly>
                    </b-form-input>
                </b-form-group>
                <b-form-group id="form-model-group"
                        v-if="run_option == 1"
                        label="Model:"
                        label-for="form-model-input">
                  <b-form-file id="form-model-input"
                                  v-model="customized_model"
                                  required
                                  placeholder="Choose a model...">
                  </b-form-file>
                </b-form-group>

                <b-form-group id="form-model-group"
                        label="Model Input Test Data Files:"
                        label-for="form-model-input">
                <b-form-file multiple id="form-model-input"
                                v-model="test_data"
                                placeholder="Select your input/output.pbs...">
                </b-form-file>
                </b-form-group>

                <b-row class="missing">
                    {{ model_missing }}
                </b-row>
                <b-row class="missing">
                    {{ test_data_missing }}
                </b-row>
                <div class="open_button" v-on:click="adv_setting = !adv_setting">
                    Advanced Settings
                </div>
                <br/>
                <div v-if="adv_setting">
                    <b-form-group id="form-execution_provider-group"
                                label="Execution Providers (choose multiple):"
                                label-for="form-execution_provider-input">
                    <b-form-select multiple class="form-control" v-model="selected_eps"
                                    required
                                    :options="options.execution_provider"
                                    label="Execution Providers (choose multiple):">
                        <template slot="first">
                        </template>
                        </b-form-select>
                    </b-form-group>

                    <b-form-group id="form-num_threads-group"
                                label="Number of Threads:"
                                label-for="form-num_threads-input">
                        <b-form-input id="form-num_threads-input"
                            type="text"
                            v-model="perf_tuning_form.num_threads"
                            value="20"
                            placeholder="Enter num_threads.
                            If leave blank, number of cores will be used.">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-top_n-group"
                                label="Top N results:"
                                label-for="form-top_n-input">
                        <b-form-input id="form-top_n-input"
                                    type="text"
                                    v-model="perf_tuning_form.top_n"
                                    value="20"
                                    placeholder="Enter top_n. ">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-optimization-level-group"
                                label="optimization_level:"
                                label-for="form-optimization_level-input">
                        <b-form-input id="form-optimization_level-input"
                                    type="text"
                                    v-model="perf_tuning_form.optimization_level">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-mode-group"
                                label="Mode:"
                                label-for="form-mode-input">
                    <b-form-select v-model="perf_tuning_form.mode"
                                    required
                                    :options="options.mode"
                                    label="Mode:"
                                    class="mb-3">
                        <template slot="first">
                        </template>
                        </b-form-select>
                    </b-form-group>

                    <b-form-group id="form-repeated_times-group"
                                label="Repeated times:"
                                label-for="form-repeated_times-input"
                                v-if="perf_tuning_form.mode == 'times'">
                        <b-form-input id="form-repeated_times-input"
                                    type="text"
                                    v-model="perf_tuning_form.repeated_times">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-duration_times-group"
                                label="Duration times:"
                                label-for="form-duration_times-input"
                                v-if="perf_tuning_form.mode == 'mode'">
                        <b-form-input id="form-duration_times-input"
                                    type="text"
                                    v-model="perf_tuning_form.duration_times"
                                    placeholder="Enter duration_times">
                        </b-form-input>
                    </b-form-group>

                    <b-form-checkbox
                    id="parallel"
                    v-model="perf_tuning_form.parallel"
                    name="parallel">
                    Use parallel executor
                    </b-form-checkbox>


                    <b-form-group v-if="perf_tuning_form.parallel"
                                id="form-threadpool_size-group"
                                label="Threadpool size:"
                                label-for="form-threadpool_size-input">
                        <b-form-input id="form-threadpool_size-input"
                          type="text"
                          v-model="perf_tuning_form.threadpool_size"
                          placeholder="Enter threadpool_size.
                            If leave blank, number of cores will be used.">
                        </b-form-input>
                    </b-form-group>

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
  props: ['converted_model'],
  data() {
    return {
      selected: -1,
      selected_profiling: -1,
      run_option: 1,
      perf_tuning_form,
      selected_eps: [],
      test_data: [],
      customized_model: null,
      options: {
        mode: ['duration', 'times'],
        execution_provider: ['', 'cpu', 'mklml', 'mkldnn', 'cuda', 'tensorrt', 'ngraph', 'cpu_openmp'],
      },
      message: '',
      show_message: false,
      model_missing: '',
      test_data_missing: '',
      adv_setting: false,
      fields: ['name', 'duration', 'op_name', 'tid'],
      model_running: false,
      host: `${window.location.protocol}//${window.location.host.split(':')[0]}`,
      link: '',
      job_id: '',
      job_name: 'app.perf_tuning',
    };
  },

  components: {
    alert: Alert,
  },

  watch: {
    converted_model() {
      this.perf_tuning_form.model = this.converted_model;
      if (this.converted_model.length > 0) {
        this.run_option = 0;
      }
    },
    run_option(newVal) {
      this.model_missing = '';
      this.test_data_missing = '';
      if (newVal == 0) {
        this.perf_tuning_form.model = this.converted_model;
      }
    },
  },
  methods: {
    init_perf_tuning_form() {
      this.perf_tuning_form = Object.assign({}, origin_perf_tuning_form);
      this.perf_tuning_form.model = this.converted_model;
      this.customized_model = null;
      this.selected_eps = [];
      this.test_data = [];
      this.run_option = 0;
    },

    perf_tuning(evt) {
      this.close_all();

      evt.preventDefault();
      if ((this.run_option == 0 && this.perf_tuning_form.model == '')
          || (this.run_option == 1 && this.customized_model == null)) {
        this.model_missing = 'Please provide an ONNX model to start performance tuning.';
        return;
      }
      if (this.run_option == 1 && this.test_data.length == 0) {
        this.test_data_missing = 'Please provide input data .pb files for customized models. ';
        return;
      }
      this.model_missing = '';
      this.test_data_missing = '';

      // run with selected execution providers
      if (this.selected_eps.length > 0) {
        this.perf_tuning_form.execution_provider = '';
        for (let i = 0; i < this.selected_eps.length; i++) {
          this.perf_tuning_form.execution_provider += this.selected_eps[i];
          if (i < this.selected_eps.length - 1) {
            this.perf_tuning_form.execution_provider += ',';
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
      if (this.customized_model && this.run_option == 1) {
        data.append('file', this.customized_model);
      } else {
        data.append('prev_model_path', perf_tuning_form.model);
      }
      for (let i = 0; i < this.test_data.length; i++) {
        data.append('test_data[]', this.test_data[i]);
      }

      axios.post(`${this.host}:5000/perf_tuning`, data)
        .then((res) => {
          this.link = `${this.host}:8000/perfresult/${res.data.job_id}`;
          this.show_message = true;
          this.message = 'Running job at ';
          this.update_result(res.data.job_id);
        })
        .catch((error) => {
          // eslint-disable-next-line
          this.message = error.toString();
          this.model_running = false;
        });
    },

    update_result(location) {
      axios.get(`${this.host}:5000/perfstatus/${location}`)
        .then((res) => {
          if (res.data.state == 'SUCCESS') {
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
