<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-10">
        <h1>ONNX pipeline</h1>
        <hr><br><br>
        <alert :message=message v-if="showMessage"></alert>
        <button type="button" class="btn btn-success btn-sm" v-b-modal.book-modal>Convert</button>
        <br><br>
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
        <button v-on:click="visualize" type="button" class="btn btn-primary btn-sm">Visualize</button>
        <iframe src="http://localhost:8080" v-if="showVisualization"></iframe>
      </div>
    </div>
    <b-modal ref="convertModal"
             id="book-modal"
             title="Convert model"
             hide-footer>
      <b-form @submit="onSubmit" @reset="onReset" class="w-100">

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
        model: null,
        visualize_model: null
      },
      message: '',
      showMessage: false,
      showVisualization: false
    };
  },
  components: {
    alert: Alert,
  },
  methods: {
    visualize(){
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
    convert(metadata, file) {
      const path = 'http://localhost:5000/convert';
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
            this.message = res.data['logs'].replace("\n", "<br/>");
            this.showMessage = true;
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
    onSubmit(evt) {
      evt.preventDefault();
      this.$refs.convertModal.hide();
      const metadata = this.convertForm;
      /*
      const metadata = {
        model_type: this.convertForm.model_type,
        model_input_shapes: this.convertForm.model_input_shapes,
        model_inputs_names: this.convertForm.model_inputs_names,
      };*/

      this.convert(metadata, this.convertForm.model);
      this.initForm();
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
