<template>
    <div class="container">
        <!-- perf test-->
        <div ref="perf_testModal"
                id="perf_test-modal"
                title="Perf Test"
                hide-footer>
            <b-form @submit="perf_test" @reset="onReset_perf_test" class="w-100">
                <b-form-group id="form-model-group"
                                label="model:"
                                label-for="form-model-input">
                    <b-form-input id="form-model-input"
                                    type="text"
                                    v-model="perf_testForm.model">
                    </b-form-input>
                </b-form-group>
                <b-row class="missing">
                    {{ model_missing }}
                </b-row>
                <div class="open_button" v-on:click="adv_setting = !adv_setting">
                    Advanced Settings
                </div>
                <br/>
                <div v-if="adv_setting">
                    <b-form-group id="form-execution_provider-group"
                                label="execution_provider:"
                                label-for="form-execution_provider-input">
                    <b-form-select multiple class="form-control" v-model="perf_testForm.execution_provider"
                                    required
                                    :options="options.execution_provider"
                                    label="execution_provider:">
                        <template slot="first">
                        </template>
                        </b-form-select>
                    </b-form-group>

                    <b-form-group id="form-num_threads-group"
                                label="num_threads:"
                                label-for="form-num_threads-input">
                        <b-form-input id="form-num_threads-input"
                                    type="text"
                                    v-model="perf_testForm.num_threads"
                                    value="20"
                                    placeholder="Enter num_threads">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-top_n-group"
                                label="top_n:"
                                label-for="form-top_n-input">
                        <b-form-input id="form-top_n-input"
                                    type="text"
                                    v-model="perf_testForm.top_n"
                                    value="20"
                                    placeholder="Enter top_n">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-optimization-level-group"
                                label="optimization_level:"
                                label-for="form-optimization_level-input">
                        <b-form-input id="form-optimization_level-input"
                                    type="text"
                                    v-model="perf_testForm.optimization_level">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-mode-group"
                                label="mode:"
                                label-for="form-mode-input">
                    <b-form-select v-model="perf_testForm.mode"
                                    required
                                    :options="options.mode"
                                    label="mode:"
                                    class="mb-3">
                        <template slot="first">
                        </template>
                        </b-form-select>
                    </b-form-group>

                    <b-form-group id="form-repeated_times-group"
                                label="repeated_times:"
                                label-for="form-repeated_times-input"
                                v-if="perf_testForm.mode == 'times'">
                        <b-form-input id="form-repeated_times-input"
                                    type="text"
                                    v-model="perf_testForm.repeated_times">
                        </b-form-input>
                    </b-form-group>

                    <b-form-group id="form-duration_times-group"
                                label="duration_times:"
                                label-for="form-duration_times-input"
                                v-if="perf_testForm.mode == 'mode'">
                        <b-form-input id="form-duration_times-input"
                                    type="text"
                                    v-model="perf_testForm.duration_times"
                                    placeholder="Enter duration_times">
                        </b-form-input>
                    </b-form-group>

                    <b-form-checkbox
                    id="parallel"
                    v-model="perf_testForm.parallel"
                    name="parallel">
                    parallel
                    </b-form-checkbox>


                    <b-form-group v-if="perf_testForm.parallel"
                                id="form-threadpool_size-group"
                                label="threadpool_size:"
                                label-for="form-threadpool_size-input">
                        <b-form-input id="form-threadpool_size-input"
                                    type="text"
                                    v-model="perf_testForm.threadpool_size"
                                    placeholder="Enter threadpool_size">
                        </b-form-input>
                    </b-form-group>

                </div>
                <b-button type="submit" variant="primary" class="button_right">Submit</b-button>
                <b-button type="reset" variant="danger">Reset</b-button>
            </b-form>
        </div>
        <hr/>
        <!-- <alert :message=message v-if="show_message"></alert> -->
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
                        <td>{{result[ep][0].cpu_usage}}</td>
                        <td>{{result[ep][0].gpu_usage}}</td>
                        <td>{{result[ep][0].memory_util}}</td>
                        <td>
                        <div v-on:click="code_details = result[ep][0].code_snippet.code"
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
        <alert :message=message v-if="show_message"></alert>
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
            <div >{{code_details}}</div>
        </b-modal>
    </div>
</template>

<script>
import axios from 'axios';
import Alert from './Alert.vue';
import { perf_testForm } from '../utils/const';

const origin_perf_testForm = Object.assign({}, perf_testForm);

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
      perf_testForm,
      options: {
        mode: ['duration', 'times'],
        execution_provider: ['', 'mklml', 'cpu_openmp', 'mkldnn', 'mkldnn_openmp', 'cpu', 'tensorrt', 'ngraph', 'cuda'],
      },
      message: '',
      show_message: false,
      show_logs: false,
      model_missing: '',
      adv_setting: false,
      op_info: {},
      fields: ['name', 'duration', 'op_name', 'tid'],
      code_details: '',
    };
  },

  components: {
    alert: Alert,
  },

  watch: {
    converted_model(newVal, oldVal) {
      this.perf_testForm.model = newVal;
    },
  },
  methods: {
    init_perf_testForm() {
      this.perf_testForm = Object.assign({}, origin_perf_testForm);
      this.perf_testForm.model = this.converted_model;
    },

    perf_test(evt) {
      this.close_all();
      evt.preventDefault();
      if (this.perf_testForm.model === '') {
        this.model_missing = 'You need to convert first.';
        return;
      }
      this.model_missing = '';

      // TODO cache model for model visualize
      const metadata = this.perf_testForm;

      const json = JSON.stringify(metadata);

      const blob = new Blob([json], {
        type: 'application/json',
      });
      const data = new FormData();
      data.append('metadata', blob);

      this.show_message = true;
      this.message = 'Running...';

      axios.post('http://localhost:5000/perf_test', data)
        .then((res) => {
          this.show_message = false;
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
          console.log(error);
        });
    },
    onReset_perf_test(evt) {
      evt.preventDefault();
      this.init_perf_testForm();
    },
    open_details(index) {
      if (this.selected === index) {
        this.selected = -1;
        console.log('if ', this.selected);
      } else {
        this.selected = index;
        console.log('else ', this.selected);
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
