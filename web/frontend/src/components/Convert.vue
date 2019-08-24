<template>
    <div class="container">
        <div ref="convertModal"
                id="convert-modal"
                title="Convert model"
                hide-footer>
        <b-form @submit="convert" @reset="onReset" class="w-100">
            <b-form-group id="form-model_type-group"
                        label="Model type:"
                        label-for="form-model_type-input">

            <b-form-select v-model="convertForm.model_type"
                        required
                        :options="options.model_type"
                        label="Model type:"
                        class="mb-3">
                <template slot="first">
                </template>
            </b-form-select>
            </b-form-group>

            <b-form-group v-if="convertForm.model_type == 'tensorflow'"
                        id="form-tf_model_type-group"
                        label="Tensorflow model type:"
                        label-for="form-tf_model_type-input">

            <b-form-select v-model="tf_model_type"
                        :options="options.tf_model_type"
                        label="Tensorflow model type:"
                        class="mb-3">
                <template slot="first">
                </template>
            </b-form-select>

            </b-form-group>

            <b-form-group id="form-model-group"
                        label="Model:"
                        label-for="form-model-input">
            <b-form-file id="form-model-input"
                            v-model="convertForm.model"
                            required
                            placeholder="Choose a model...">
            </b-form-file>
            </b-form-group>

            <b-form-group v-if="convertForm.model_type === 'tensorflow'
                          && tf_model_type === 'savedModel'"
                        id="form-model-group"
                        label="Tensorflow SavedModel Variable Files:"
                        label-for="form-model-input">
            <b-form-file id="form-model-input"
                        multiple
                        v-model="savedModel_vars"
                        required
                        placeholder="Select Tensorflow saved model variable files...">
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
        <b-form-group v-if="convertForm.model_type === 'tensorflow'
                        && tf_model_type != 'savedModel' && tf_model_type.length > 0"
                        id="form-model_inputs_names-group"
                        label="Model inputs names:"
                        label-for="form-model_inputs_names-input">
            <b-form-input id="form-model_inputs_names-input"
                            type="text"
                            v-model="convertForm.model_inputs_names"
                            placeholder="Enter model inputs names">
            </b-form-input>
            </b-form-group>
        <b-form-group v-if="convertForm.model_type === 'tensorflow'
                        && tf_model_type != 'savedModel' && tf_model_type.length > 0"
                        id="form-model_outputs_names-group"
                        label="Model outputs names:"
                        label-for="form-model_outputs_names-input">
            <b-form-input id="form-model_outputs_names-input"
                            type="text"
                            v-model="convertForm.model_outputs_names"
                            placeholder="Enter model_outputs_names">
            </b-form-input>
            </b-form-group>
        <b-form-group v-if="convertForm.model_type === 'mxnet'"
                        id="form-model_params-group"
                        label="Model params:"
                        label-for="form-model_params-input">
            <b-form-input id="form-model_params-input"
                            type="text"
                            v-model="convertForm.model_params"
                            placeholder="Enter model params">
            </b-form-input>
            </b-form-group>

        <b-form-group v-if="convertForm.model_type === 'caffe'"
                        id="form-caffe_model_prototxt-group"
                        label="Caffe model prototxt:"
                        label-for="form-caffe_model_prototxt-input">
            <b-form-input id="form-caffe_model_prototxt-input"
                            type="text"
                            v-model="convertForm.caffe_model_prototxt"
                            placeholder="Enter caffe model prototxt">
            </b-form-input>
            </b-form-group>

        <b-form-group v-if="convertForm.model_type === 'scikit-learn'"
                        id="form-initial_types-group"
                        label="Initial types:"
                        label-for="form-initial_types-input">
            <b-form-input id="form-initial_types-input"
                            type="text"
                            v-model="convertForm.initial_types"
                            placeholder="Enter initial types">
            </b-form-input>
            </b-form-group>

        <b-form-group v-if="convertForm.model_type === 'pytorch'"
                        id="form-model_input_shapes-group"
                        label="Model input shapes:"
                        label-for="form-model_input_shapes-input">
            <b-form-input id="form-model_input_shapes-input"
                            type="text"
                            v-model="convertForm.model_input_shapes"
                            placeholder="Enter model input shapes">
            </b-form-input>
            </b-form-group>


        <b-form-group id="form-target_opset-group"
                        label="Target Opset:"
                        label-for="form-target_opset-input">
            <b-form-input id="form-target_opset-input"
                            type="text"
                            v-model="convertForm.target_opset"
                            placeholder="Enter target opset">
            </b-form-input>
            </b-form-group>

            <b-button type="submit" variant="primary" class="button_right">Submit</b-button>
            <b-button type="reset" variant="danger">Reset</b-button>
        </b-form>
        </div>
        <hr/>
        <alert :message=message v-if="show_message"></alert>

        <div v-if="convert_result">
            <h5>Conversion Status:
            <b-badge variant="primary">
                {{convert_result['output_json']['conversion_status']}}
            </b-badge>
            </h5>
            <h5>Correctness Verified:
            <b-badge variant="primary">
                {{convert_result['output_json']['correctness_verified']}}
            </b-badge>
            </h5>
            <h5>Error:
            <b-badge variant="danger">{{convert_result['output_json']['error_message']}}</b-badge>
            </h5>
            <h5>Download</h5>
            <a :href="convert_result['input_path']" download>[input] </a>
            <a :href="convert_result['model_path']" download>[model]</a>
        </div>
    </div>
</template>

<script>
import axios from 'axios';
import Alert from './Alert.vue';
import { convertForm } from '../utils/const';

const origin_convertForm = Object.assign({}, convertForm);

export default {
  name: 'Convert',
  data() {
    return {
      result: {},
      convertForm,
      test_data: [],
      savedModel_vars: [],
      options: {
        model_type: [
          { value: 'pytorch', text: 'pytorch' },
          { value: 'tensorflow', text: 'tensorflow' },
          { value: 'onnx', text: 'onnx' },
          { value: 'keras', text: 'keras' },
          { value: 'caffe', text: 'caffe' },
          { value: 'scikit-learn', text: 'scikit-learn' },
        ],
        tf_model_type: [
          'savedModel',
          'frozen graph',
          'checkpoint',
        ],
      },
      tf_model_type: '',
      message: '',
      show_message: false,
      model_missing: '',
      convert_result: null,
      converted_model: '',
    };
  },
  components: {
    alert: Alert,
  },
  methods: {
    initForm() {
      this.convertForm = Object.assign({}, origin_convertForm);
    },
    onReset(evt) {
      evt.preventDefault();
      this.initForm();
      this.convert_result = null;
    },
    convert(evt) {
      this.close_all();
      this.$emit('update_model', '');
      evt.preventDefault();
      const metadata = this.convertForm;
      const json = JSON.stringify(metadata);
      const blob = new Blob([json], {
        type: 'application/json',
      });

      const data = new FormData();
      data.append('metadata', blob);
      data.append('file', this.convertForm.model);
      for (let i = 0; i < this.test_data.length; i++) {
        data.append('test_data[]', this.test_data[i]);
      }
      for (let i = 0; i < this.savedModel_vars.length; i++) {
        data.append('savedModel[]', this.savedModel_vars[i]);
      }
      this.show_message = true;
      this.message = 'Running...';
      const host = `${window.location.protocol}//${window.location.host.split(':')[0]}`;
      axios.post(`${host}:5000/convert`, data)
        .then((res) => {
          this.show_message = false;
          if (res.data.status === 'success') {
            this.message = res.data.logs;
            this.show_logs = true;
            this.convert_result = {
              output_json: res.data.output_json,
              input_path: `../static/${res.data.input_path}`,
              model_path: `../static/${res.data.model_path}`,
            };
            this.$emit('update_model', res.data.converted_model);
            // this.perf_testForm.model = res.data.converted_model;
            // TODO cache model for model visualize
          }
        })
        .catch((error) => {
          // eslint-disable-next-line]
          this.message = error;
          console.log(error);
        });
    },

    // upload() {
    //   const blob = new Blob([json], {
    //     type: 'application/json',
    //   });

    //   const data = new FormData();
    //   data.append('metadata', blob);


    // },
    close_all() {
      this.result = [];
      this.show_message = false;
      this.show_logs = false;
      this.convert_result = null;
    },
  },
};
</script>
