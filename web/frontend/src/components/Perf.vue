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
                <b-form-group id="form-model-group"
                              v-if="run_option == 0"
                              label="Model From Previous Step:"
                              label-for="form-model-input">
                    <b-form-input id="form-model-input"
                                    type="text"
                                    v-model="converted_model"
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
        <div v-if="Object.keys(result).length > 0">
        <ul class="list-group" v-for="(ep, index) in Object.keys(result)" :key="index">
            <li class="list-group-item" v-if="result[ep].length > 0">
                <h5>{{index+1}}. {{ep}} </h5>
                <table class="table-responsive-lg" style="table-layout: fixed">
                <thead>
                    <tr>
                    <th scope="col">name</th>
                    <th scope="col">avg (ms)</th>
                    <th scope="col">p90 (ms)</th>
                    <th scope="col">p95 (ms)</th>
                    <th scope="col">cpu (%)</th>
                    <th scope="col">gpu (%)</th>
                    <th scope="col">memory (%)</th>
                    <!--<th>code_snippet.execution_provider</th>-->
                    <th>code_snippet</th>
                    <th>profiling</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{{result[ep][0].name}}</td>
                        <td>{{result[ep][0].avg}}</td>
                        <td>{{result[ep][0].p90}}</td>
                        <td>{{result[ep][0].p95}}</td>
                        <td>{{result[ep][0].cpu_usage * 100}}%</td>
                        <td>{{result[ep][0].gpu_usage * 100}}%</td>
                        <td>{{result[ep][0].memory_util * 100}}%</td>
                        <td>
                        <div v-on:click="format_code_snippet(result[ep][0].code_snippet.code)"
                            v-bind:class="{open: !(selected == index)}"
                            class="before_open open_button"
                            v-b-modal.codeModal>details </div>
                        </td>
                        <!--profiling-->
                        <td>
                        <div
                            v-on:click="open_profiling(profiling[index].slice(0, PROFILING_MAX))"
                            v-bind:class="{open: !(selected_profiling == index)}"
                            class="before_open open_button" v-b-modal.opsModal>op </div>
                        </td>

                    </tr>
                </tbody>
                </table>
                <details style="margin: 10px">
                    <summary>
                        More options with good performance
                    </summary>
                    <div v-for="(item, index) in result[ep]" :key="index">
                        <p v-if="index > 0">{{item.name}}</p>
                    </div>
                </details>
            </li>
        </ul>
        </div>
        <div class="open_button" v-on:click="show_message = !show_message" v-if="show_logs">
            <hr/>Show logs
        </div>
        <alert :message=message :loading=model_running v-if="show_message"></alert>
        <br/>
        <b-modal ref="opsModal"
                id="opsModal"
                title="Top 5 ops"
                size="lg"
                hide-footer>
            <b-container fluid>
            <b-table class="table-responsive-lg" style="table-layout: fixed"
                striped hover :items="op_info" :fields="fields"></b-table>
            </b-container>
        </b-modal>
        <b-modal ref="codeModal"
                id="codeModal"
                title="Code details" style="width: 100%;"
                hide-footer>
            <div style="white-space: pre-wrap">{{code_details}}</div>
        </b-modal>
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
      PROFILING_MAX: 5,
      selected: -1,
      selected_profiling: -1,
      result: {},
      profiling: [],
      run_option: 1,
      perf_tuning_form,
      selected_eps: [],
      test_data: [],
      customized_model: null,
      options: {
        mode: ['duration', 'times'],
        execution_provider: ['', 'cpu', 'mklml', 'mkldnn', 'cuda', 'tensorrt', 'ngraph', 'cpu_openmp', 'mkldnn_openmp'],
      },
      message: '',
      show_message: false,
      show_logs: false,
      model_missing: '',
      test_data_missing: '',
      adv_setting: false,
      op_info: {},
      fields: ['name', 'duration', 'op_name', 'tid'],
      code_details: '',
      model_running: false,
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
    run_option() {
      this.model_missing = '';
      this.test_data_missing = '';
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

      // TODO cache model for model visualize
      const metadata = this.perf_tuning_form;

      const json = JSON.stringify(metadata);

      const blob = new Blob([json], {
        type: 'application/json',
      });
      const data = new FormData();
      data.append('metadata', blob);
      if (this.customized_model && this.run_option == 1) {
        data.append('file', this.customized_model);
      }
      for (let i = 0; i < this.test_data.length; i++) {
        data.append('test_data[]', this.test_data[i]);
      }

      this.show_message = true;
      this.message = 'Running...';
      this.model_running = true;
      const host = `${window.location.protocol}//${window.location.host.split(':')[0]}`;
      axios.post(`${host}:5000/perf_tuning`, data)
        .then((res) => {
          this.show_message = false;
          this.model_running = false;
          if (res.data.status === 'success') {
            const { logs } = res.data;
            this.show_logs = true;
            this.message = logs;
            this.result = JSON.parse(res.data.result);
            this.profiling = res.data.profiling;
          }
        })
        .catch((error) => {
          // eslint-disable-next-line
          this.message = error;
          this.model_running = false;
          console.log(error);
        });
    },
    format_code_snippet(code) {
      this.code_details = code.trim().replace(/\s\s+/g, '\n');
    },
    onReset_perf_tuning(evt) {
      evt.preventDefault();
      this.init_perf_tuning_form();
    },
    open_details(index) {
      if (this.selected == index) {
        this.selected = -1;
      } else {
        this.selected = index;
      }
    },
    open_profiling(ops) {
      this.op_info = [];
      for (let i = 0; i < ops.length; ++i) {
        this.op_info.push({
          name: ops[i].name,
          duration: ops[i].dur,
          op_name: ops[i].args.op_name,
          tid: ops[i].tid,
        });
      }
    },
    close_all() {
      this.result = [];
      this.show_message = false;
      this.show_logs = false;
    },
  },
};
</script>

<style scoped>
.before_open::after{
  content: "(-)";
  font-weight: bold;
}
.open::after{
  content: "(+)";
}
.op_table{
  padding: 10px;
  background: white;
}
.open_button{
  cursor: pointer;
  text-decoration: underline;
  color: #669;
}
</style>
