<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-10">
        <h1>ONNX pipeline</h1>
        <hr>
        <button type="button" class="btn btn-success btn-sm button_right" v-b-modal.convert-modal>Convert</button>
        <button type="button" class="btn btn-info btn-sm button_right" v-b-modal.perf_test-modal>Pert Test</button>
        <button type="button" class="btn btn-primary btn-sm" v-b-modal.visualizeModal>Visualize</button>
        <hr/>
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

      <b-form-select v-model="convertForm.model_type"
                     required
                     :options="convertForm.options"
                     label="model_type:"
                     class="mb-3">
          <template slot="first">
            <option :value="null" disabled>-- Please select an option --</option>
          </template>
        </b-form-select>

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

      <b-form-group v-if="perf_testForm.model_type === 'tensorflow'"
                    id="form-model_inputs_names-group"
                    label="model_inputs_names:"
                    label-for="form-model_inputs_names-input">
          <b-form-input id="form-model_inputs_names-input"
                        type="text"
                        v-model="perf_testForm.model_inputs_names"
                        placeholder="Enter model_inputs_names">
          </b-form-input>
        </b-form-group>
      <b-form-group id="form-model-group"
                    label="Model:"
                    label-for="form-model-input">
          <b-form-file id="form-model-input"
                        v-model="perf_testForm.model"
                        required
                        placeholder="Choose a model..."
                        drop-placeholder="Drop model here...">
          </b-form-file>
        </b-form-group>

      <b-form-group id="form-input_json-group"
                    label="input_json:"
                    label-for="form-input_json-input">
          <b-form-file id="form-input_json-input"
                        v-model="perf_testForm.input_json"
                        placeholder="Choose a json file for input..."
                        drop-placeholder="Drop json here...">
          </b-form-file>
        </b-form-group>

      <b-form-select v-model="perf_testForm.config"
                     required
                     :options="perf_testForm.options_config"
                     label="config:"
                     class="mb-3">
          <template slot="first">
            <option :value="null" disabled>-- Please select an option --</option>
          </template>
        </b-form-select>

      <b-form-group id="form-mode-group"
                    label="mode:"
                    label-for="form-mode-input">
          <b-form-input id="form-mode-input"
                        type="text"
                        v-model="perf_testForm.mode"
                        placeholder="Enter mode">
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
        options_execution_provider: ["cpu", "cuda", "mkldnn"],
        config: '',
        mode: '',
        execution_provider: '',
        repeated_times: '',
        duration_times: '',
        parallel: false,
        threadpool_size: '',
        num_threads: '',
        top_n: '',
        runtime: true,
        input_json: ''
      },
      visualize_model: null,
      message: '',
      showMessage: false,
      showVisualization: false
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
    submitForm(metadata, file, action) {
      console.log(action === 'perf_test');
      var path;
      if (action === 'convert') {
        path = 'http://localhost:5000/convert';
      }
      else if (action === 'perf_test') {
        path = 'http://localhost:5000/perf_test';
      }
      const json = JSON.stringify(metadata);

      const blob = new Blob([json], {
        type: 'application/json'
      });
      const data = new FormData();
      data.append("metadata", blob);
      data.append("file", file);

      axios.post(path, data)
      .then((res) => {
          if(res.data['status'] == 'success'){
            var data = res.data['logs'];
            this.showMessage = true;
            this.message = data;
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
      this.submitForm(metadata, this.convertForm.model, 'convert');
      this.initForm();
    },
    perf_test(evt) {
      evt.preventDefault();
      this.$refs.perf_testModal.hide();
      const metadata = this.perf_testForm;
      this.submitForm(metadata, this.perf_testForm.model, 'perf_test');
      //this.initForm();
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
</style>