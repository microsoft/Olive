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

            <b-form-select v-model="convert_form.model_type"
                        required
                        :options="options.model_type"
                        label="Model type:"
                        class="mb-3">
                <template slot="first">
                </template>
            </b-form-select>
            </b-form-group>

            <b-form-group v-if="convert_form.model_type == 'tensorflow'"
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
                            v-model="convert_form.model"
                            required
                            placeholder="Choose a model...">
            </b-form-file>
            </b-form-group>

            <b-form-group v-if="convert_form.model_type === 'tensorflow'
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
        <b-form-group v-if="convert_form.model_type === 'tensorflow'
                        && tf_model_type != 'savedModel' && tf_model_type.length > 0"
                        id="form-model_inputs_names-group"
                        label="Model inputs names:"
                        label-for="form-model_inputs_names-input">
            <b-form-input id="form-model_inputs_names-input"
                            type="text"
                            v-model="convert_form.model_inputs_names"
                            placeholder="Enter model inputs names">
            </b-form-input>
            </b-form-group>
        <b-form-group v-if="convert_form.model_type === 'tensorflow'
                        && tf_model_type != 'savedModel' && tf_model_type.length > 0"
                        id="form-model_outputs_names-group"
                        label="Model outputs names:"
                        label-for="form-model_outputs_names-input">
            <b-form-input id="form-model_outputs_names-input"
                            type="text"
                            v-model="convert_form.model_outputs_names"
                            placeholder="Enter model_outputs_names">
            </b-form-input>
            </b-form-group>
        <b-form-group v-if="convert_form.model_type === 'mxnet'"
                        id="form-model_params-group"
                        label="Model params:"
                        label-for="form-model_params-input">
            <b-form-input id="form-model_params-input"
                            type="text"
                            v-model="convert_form.model_params"
                            placeholder="Enter model params">
            </b-form-input>
            </b-form-group>

        <b-form-group v-if="convert_form.model_type === 'caffe'"
                        id="form-caffe_model_prototxt-group"
                        label="Caffe model prototxt:"
                        label-for="form-caffe_model_prototxt-input">
            <b-form-input id="form-caffe_model_prototxt-input"
                            type="text"
                            v-model="convert_form.caffe_model_prototxt"
                            placeholder="Enter caffe model prototxt">
            </b-form-input>
            </b-form-group>

        <b-form-group v-if="convert_form.model_type === 'scikit-learn'"
                        id="form-initial_types-group"
                        label="Initial types:"
                        label-for="form-initial_types-input">
            <b-form-input id="form-initial_types-input"
                            type="text"
                            v-model="convert_form.initial_types"
                            placeholder="Enter initial types">
            </b-form-input>
            </b-form-group>

        <b-form-group v-if="convert_form.model_type === 'pytorch'"
                        id="form-model_input_shapes-group"
                        label="Model input shapes:"
                        label-for="form-model_input_shapes-input">
            <b-form-input id="form-model_input_shapes-input"
                            type="text"
                            v-model="convert_form.model_input_shapes"
                            placeholder="Enter model input shapes">
            </b-form-input>
            </b-form-group>


        <b-form-group id="form-target_opset-group"
                        label="Target Opset:"
                        label-for="form-target_opset-input">
            <b-form-input id="form-target_opset-input"
                            type="text"
                            v-model="convert_form.target_opset"
                            placeholder="Enter target opset">
            </b-form-input>
            </b-form-group>

            <b-button type="submit"
              :disabled="model_running" variant="primary" class="button_right">Submit</b-button>
            <b-button type="reset" :disabled="model_running" variant="danger">Reset</b-button>
        </b-form>
        </div>
        <hr/>
        <alert :message=message :loading=model_running v-if="show_message"></alert>
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
            <a :href="host + ':5000/download/input.tar.gz'" download>[input] </a>
            <a :href="host + ':5000/download/model.onnx'" download>[model]</a>
        </div>
    </div>
</template>

<script>
import axios from 'axios';
import Alert from './Alert.vue';
import { convert_form } from '../utils/const';

const origin_convert_form = Object.assign({}, convert_form);

export default {
  name: 'Convert',
  data() {
    return {
      result: {},
      convert_form,
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
      host: `${window.location.protocol}//${window.location.host.split(':')[0]}`,
      model_running: false,
    };
  },
  components: {
    alert: Alert,
  },
  methods: {
    initForm() {
      this.convert_form = Object.assign({}, origin_convert_form);
    },
    onReset(evt) {
      evt.preventDefault();
      this.initForm();
      this.convert_result = null;
    },
    convert(evt) {
      this.close_all();
      this.model_running = true;
      evt.preventDefault();
      const metadata = this.convert_form;
      const json = JSON.stringify(metadata);
      const blob = new Blob([json], {
        type: 'application/json',
      });

      const data = new FormData();
      data.append('metadata', blob);
      data.append('file', this.convert_form.model);
      for (let i = 0; i < this.test_data.length; i++) {
        data.append('test_data[]', this.test_data[i]);
      }
      for (let i = 0; i < this.savedModel_vars.length; i++) {
        data.append('savedModel[]', this.savedModel_vars[i]);
      }
      this.show_message = true;
      this.message = 'Running...';
      // const host = `${window.location.protocol}//${window.location.host.split(':')[0]}`;
      axios.post(`${this.host}:5000/convert`, data)
        .then((res) => {
          this.show_message = false;
          this.model_running = false;
          if (res.data.status === 'success') {
            this.message = res.data.logs;
            this.show_logs = true;
            this.convert_result = {
              output_json: res.data.output_json,
              input_path: `../../static/${res.data.input_path}`,
              model_path: `../../static/${res.data.model_path}`,
            };
            this.$emit('update_model', res.data.converted_model);
          }
        })
        .catch((error) => {
          // eslint-disable-next-line]
          this.model_running = false;
          this.message = error;
          console.log(error);
        });
    },

    close_all() {
      this.result = [];
      this.show_message = false;
      this.show_logs = false;
      this.convert_result = null;
    },
  },
};
</script>
