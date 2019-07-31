<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-10">
        <h1>ONNX pipeline</h1>
        <hr>
        <button type="button" class="btn btn-success btn-sm button_right" v-b-modal.convert-modal>Convert</button>
        <button type="button" class="btn btn-info btn-sm button_right" v-b-modal.perf_test-modal>Pert Test</button>
        <button type="button" class="btn btn-primary btn-sm button_right" v-on:click="show_result">Result</button>
        <button type="button" class="btn btn-light btn-sm" v-b-modal.visualizeModal>Visualize</button>
        <hr/>
        <table v-if="showResult">
          <th>name</th>
          <th>avg</th>
          <th>p90</th>
          <th>p95</th>
          <th>cpu_usage</th>
          <th>gpu_usage</th>
          <th>memory_util</th>
          <!--<th>code_snippet.execution_provider</th>-->
          <th>code_snippet</th>
          <tr v-for="(r, index) in result">
              <td>{{r.name}}</td>
              <td>{{r.avg}}</td>
              <td>{{r.p90}}</td>
              <td>{{r.p95}}</td>
              <td>{{r.cpu_usage}}</td>
              <td>{{r.gpu_usage}}</td>
              <td>{{r.memory_util}}</td>
              <td>
                <table class="lit_table">
                  <tr class="lit_tr" v-for="(value, key) in r.code_snippet">
                    <div v-if="key === 'code'">
                      <td v-on:click="selected = index" v-bind:class="{open: !(selected == index)}" class="lit_td">{{key}}</td>
                      <td class="lit_td td2" v-bind:class="{ hide: !(selected == index)}" ref="code">{{value}}</td>
                    </div>
                    <div v-else-if="key === 'environment_variables'">
                      <table class="lit_table">
                        <tr class="lit_tr" v-for="(value, key) in r.code_snippet.environment_variables">
                          <td class="lit_td">{{key}}</td>
                          <td class="lit_td td2">{{value}}</td>
                        </tr>
                      </table>
                    </div>
                    <div v-else-if="key === 'execution_provider'">
                    </div>
                    <div v-else>
                      <td class="lit_td">{{key}}</td>
                      <td class="lit_td td2">{{value}}</td>
                    </div>
                  </tr>
                </table>  
              </td>
          </tr>
        </table>
        <alert :message=message v-if="showMessage"></alert>
        <br/>
        <div>
          <iframe src="http://localhost:8080" width="600" height="500" v-if="showVisualization"></iframe>
        </div>
      </div>
    </div>
    <b-modal ref="convertModal"
             id="convert-modal"
             title="Convert model"
             hide-footer>
      <b-form @submit="convert" @reset="onReset" class="w-100">
        <b-form-group id="form-model_type-group"
                      label="model_type:"
                      label-for="form-model_type-input">

        <b-form-select v-model="convertForm.model_type"
                      required
                      :options="convertForm.options"
                      label="model_type:"
                      class="mb-3">
            <template slot="first">
              <option :value="null" disabled>-- Please select an option --</option>
            </template>
          </b-form-select>
          </b-form-group>

      <b-form-group v-if="convertForm.model_type === 'tensorflow'"
                    id="form-model_inputs_names-group"
                    label="model_inputs_names:"
                    label-for="form-model_inputs_names-input">
          <b-form-input id="form-model_inputs_names-input"
                        type="text"
                        v-model="convertForm.model_inputs_names"
                        placeholder="Enter model_inputs_names">
          </b-form-input>
        </b-form-group>
      <b-form-group v-if="convertForm.model_type === 'tensorflow'"
                    id="form-model_outputs_names-group"
                    label="model_outputs_names:"
                    label-for="form-model_outputs_names-input">
          <b-form-input id="form-model_outputs_names-input"
                        type="text"
                        v-model="convertForm.model_outputs_names"
                        placeholder="Enter model_outputs_names">
          </b-form-input>
        </b-form-group>
      <b-form-group v-if="convertForm.model_type === 'mxnet'"
                    id="form-model_params-group"
                    label="model_params:"
                    label-for="form-model_params-input">
          <b-form-input id="form-model_params-input"
                        type="text"
                        v-model="convertForm.model_params"
                        placeholder="Enter model_params">
          </b-form-input>
        </b-form-group>

      <b-form-group v-if="convertForm.model_type === 'caffe'"
                    id="form-caffe_model_prototxt-group"
                    label="caffe_model_prototxt:"
                    label-for="form-caffe_model_prototxt-input">
          <b-form-input id="form-caffe_model_prototxt-input"
                        type="text"
                        v-model="convertForm.caffe_model_prototxt"
                        placeholder="Enter caffe_model_prototxt">
          </b-form-input>
        </b-form-group>

      <b-form-group v-if="convertForm.model_type === 'scikit-learn'"
                    id="form-initial_types-group"
                    label="initial_types:"
                    label-for="form-initial_types-input">
          <b-form-input id="form-initial_types-input"
                        type="text"
                        v-model="convertForm.initial_types"
                        placeholder="Enter initial_types">
          </b-form-input>
        </b-form-group>


      <b-form-group v-if="convertForm.model_type === 'pytorch'"
                    id="form-model_input_shapes-group"
                    label="model_input_shapes:"
                    label-for="form-model_input_shapes-input">
          <b-form-input id="form-model_input_shapes-input"
                        type="text"
                        v-model="convertForm.model_input_shapes"
                        placeholder="Enter model_input_shapes">
          </b-form-input>
        </b-form-group>
      <b-form-group id="form-model-group"
                    label="Model:"
                    label-for="form-model-input">
          <b-form-file id="form-model-input"
                        v-model="convertForm.model"
                        required
                        placeholder="Choose a model..."
                        drop-placeholder="Drop model here...">
          </b-form-file>
        </b-form-group>

      <b-form-group id="form-input_json-group"
                    label="input_json:"
                    label-for="form-input_json-input">
          <b-form-file id="form-input_json-input"
                        v-model="convertForm.input_json"
                        placeholder="Choose a json file for input..."
                        drop-placeholder="Drop json here...">
          </b-form-file>
        </b-form-group>

      <b-form-group id="form-target_opset-group"
                    label="target_opset:"
                    label-for="form-target_opset-input">
          <b-form-input id="form-target_opset-input"
                        type="text"
                        v-model="convertForm.target_opset"
                        placeholder="Enter target_opset">
          </b-form-input>
        </b-form-group>


        <b-button type="submit" variant="primary">Submit</b-button>
        <b-button type="reset" variant="danger">Reset</b-button>
      </b-form>
    </b-modal>
    <!-- perf test-->
    <b-modal ref="perf_testModal"
             id="perf_test-modal"
             title="Perf Test"
             hide-footer>
      <b-form @submit="perf_test" class="w-100">


      <b-form-group id="form-model-group"
                    label="model:"
                    label-for="form-model-input">
          <b-form-input id="form-model-input"
                        type="text"
                        v-model="perf_testForm.model"
                        readonly>
          </b-form-input>
        </b-form-group>
      <b-row class="missing">
        {{ model_missing }}
        </b-row>
      <b-form-group id="form-input_json-group"
                    label="input_json:"
                    label-for="form-input_json-input">
          <b-form-file id="form-input_json-input"
                        v-model="perf_testForm.input_json"
                        placeholder="Choose a json file for input..."
                        drop-placeholder="Drop json here...">
          </b-form-file>
        </b-form-group>

      <b-form-group id="form-config-group"
                    label="config:"
                    label-for="form-config-input">
        <b-form-select v-model="perf_testForm.config"
                      required
                      :options="perf_testForm.options_config"
                      label="config:"
                      class="mb-3">
            <template slot="first">
            </template>
          </b-form-select>
      </b-form-group>
      <b-form-group id="form-mode-group"
                    label="mode:"
                    label-for="form-mode-input">
        <b-form-select v-model="perf_testForm.mode"
                      required
                      :options="perf_testForm.options_mode"
                      label="mode:"
                      class="mb-3">
            <template slot="first">
            </template>
          </b-form-select>
      </b-form-group>

      <b-form-group id="form-execution_provider-group"
                    label="execution_provider:"
                    label-for="form-execution_provider-input">
        <b-form-select v-model="perf_testForm.execution_provider"
                      required
                      :options="perf_testForm.options_execution_provider"
                      label="execution_provider:"
                      class="mb-3">
            <template slot="first">
            </template>
          </b-form-select>
      </b-form-group>

      <b-form-group id="form-repeated_times-group"
                    label="repeated_times:"
                    label-for="form-repeated_times-input">
          <b-form-input id="form-repeated_times-input"
                        type="text"
                        v-model="perf_testForm.repeated_times">
          </b-form-input>
        </b-form-group>

      <b-form-group id="form-duration_times-group"
                    label="duration_times:"
                    label-for="form-duration_times-input">
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

      <b-form-checkbox
        id="runtime"
        v-model="perf_testForm.runtime"
        name="runtime">
        runtime
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

        <b-button type="submit" variant="primary">Submit</b-button>
        <b-button type="reset" variant="danger">Reset</b-button>
      </b-form>
    </b-modal>

  <b-modal ref="visualizeModal"
            id="visualizeModal"
            title="Visualization"
            hide-footer>
    <b-form @submit="visualize" class="w-100">
      <b-form-group id="visualize_model"
                    label="Model for visualization:"
                    label-for="form-visualize_model-input">
          <b-form-file id="form-visualize_model-input"
                        v-model="visualize_model"
                        required
                        placeholder="Choose a model..."
                        drop-placeholder="Drop model here...">
          </b-form-file>
        </b-form-group>
        <b-button type="submit" variant="primary">Submit</b-button>
      </b-form>
    </b-modal>
  </div>
</template>

<script>
import axios from 'axios';
import Alert from './Alert';

export default {
  data() {
    return {
      showResult: false,
      selected: -1,
      result: [],
      convertForm: {
        model_type: '',
        model_inputs_names: '',
        model_outputs_names: '',
        target_opset: '',
        model_input_shapes: '',
        caffe_model_prototxt: '',
        initial_types: '',
        input_json: '',
        model_params: '',
        options: [
          { value: 'pytorch', text: 'pytorch' },
          { value: 'tensorflow', text: 'tensorflow' },
          { value: 'onnx', text: 'onnx' },
          { value: 'keras', text: 'keras' },
          { value: 'caffe', text: 'caffe' },
          { value: 'scikit-learn', text: 'scikit-learn' }
        ],
        model: null
      },
      perf_testForm: {
        model: '',
        options_config: ["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        options_mode: ["duration", "times"],
        options_execution_provider: ["cpu", "cuda", "mkldnn", ""],
        config: 'RelWithDebInfo',
        mode: 'times',
        execution_provider: '',
        repeated_times: '20',
        duration_times: '10',
        parallel: false,
        threadpool_size: '',
        num_threads: '',
        top_n: '5',
        runtime: false,
        input_json: ''
      },
      visualize_model: null,
      message: '',
      showMessage: false,
      showVisualization: false,
      model_missing: ''
    };
  },
  components: {
    alert: Alert,
  },
  methods: {
    visualize(evt){
      evt.preventDefault();
      this.$refs.visualizeModal.hide();
      const path = 'http://localhost:5000/visualize';
      const data = new FormData();
      data.append("file", this.visualize_model);
      axios.post(path, data)
      .then((res) => {
          if(res.data['status'] == 'success'){
            this.showVisualization = true;
          }
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
        });
    },
    initForm() {
      this.convertForm.model_type = '';
      this.convertForm.model_inputs_names = '';
      this.convertForm.model_input_shapes = '';
      this.convertForm.model_outputs_names = '';
      this.convertForm.target_opset ='';
      this.convertForm.caffe_model_prototxt = '';
      this.convertForm.initial_types = '';
      this.convertForm.input_json ='';
      this.convertForm.model_params = '';
      this.convertForm.model = null;
    },
    convert(evt) {
      evt.preventDefault();
      this.$refs.convertModal.hide();
      const metadata = this.convertForm;
      const json = JSON.stringify(metadata);
      const blob = new Blob([json], {
        type: 'application/json'
      });

      const data = new FormData();
      data.append("metadata", blob);
      data.append("file", this.convertForm.model);

      this.showMessage = true;
      this.message = 'Running...';

      axios.post('http://localhost:5000/convert', data)
      .then((res) => {
          if(res.data['status'] == 'success'){
            var data = res.data['logs'];
            this.showMessage = true;
            this.message = data;
            this.perf_testForm.model = res.data['converted_model'];
          }
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
        });

      this.initForm();
    },
    perf_test(evt) {
      evt.preventDefault();
      if(this.perf_testForm.model === ""){
        this.model_missing = 'You need to convert first.';
        return;
      }
      else this.model_missing = '';
      this.$refs.perf_testModal.hide();
      const metadata = this.perf_testForm;

      const json = JSON.stringify(metadata);

      const blob = new Blob([json], {
        type: 'application/json'
      });
      const data = new FormData();
      data.append("metadata", blob);

      this.showMessage = true;
      this.message = 'Running...';

      axios.post('http://localhost:5000/perf_test', data)
      .then((res) => {
          if(res.data['status'] == 'success'){
            var data = res.data['logs'];
            this.showMessage = true;
            this.message = data;
            this.result = res.data['result'];
          }
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
        });

    },
    onReset(evt) {
      evt.preventDefault();
      this.$refs.convertModal.hide();
      this.initForm();
    },
    onResetUpdate(evt) {
      evt.preventDefault();
      this.$refs.editBookModal.hide();
      this.initForm();
    },
    show_result(evt) {
      this.showResult = !this.showResult;
    }
  },
  created() {
    //this.getBooks();
  },
};
</script>
<style>
.button_right{
  margin-right: 15px;
}
.missing{
  margin-left: 15px;
  color: red;
}
table {
  border-collapse: collapse;
  width: 100%;
}

th, td {
  text-align: left;
  padding: 8px;
}
.lit_table{
    margin: 4px 2px;
    background-color: #f0f0ff;
    border-collapse: collapse;
}
.lit_th, .lit_td {
    vertical-align: top;
    margin: 0px;
    padding: 2px 6px;
    border-width: 2px;
    border-color: #669;
    border-style: solid;
}
.lit_th {
    text-align: left;
    background-color: white;
}
.lit_td {
    text-align: left;
    background-color: #eef;
}
tr:nth-child(even) {background-color: #f2f2f2;}
.hide{
  display: none;
}
.open{
  cursor: pointer;
}
.open::after{ 
  content: "(+)";
  font-weight: bold;
}
.td2{
  background-color: white;
}
</style>