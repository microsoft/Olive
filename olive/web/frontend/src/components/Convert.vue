<template>
    <div class="container">
        <div ref="convertModel"
                id="convert-model"
                title="Convert model"
                hide-footer>
        <b-form @submit="convert" @reset="onReset" class="w-100">
            <b-form-group id="form-model_framework-group"
                        label="Model Conversion Job Name:"
                        label-for="form-model_framework-input"
                        label-class="font-weight-bold">
              <b-form-input v-model="job_name" placeholder="onnx-converter"></b-form-input>
            </b-form-group>

            <b-form-group id="form-model_framework-group"
                        label="Model Framework:"
                        label-for="form-model_framework-input"
                        label-class="font-weight-bold">

            <b-form-select v-model="convert_form.model_framework"
                        required
                        :options="options.model_framework"
                        label="Model Framework:"
                        class="mb-3">
                <template slot="first">
                </template>
            </b-form-select>
            </b-form-group>

            <div v-if="convert_form.model_framework === 'pytorch'">
            <b-form-group id="form-pytorch_version-group"
                          label="PyTorch version for conversion:"
                          label-for="form-pytorch_version-input"
                          label-class="font-weight-bold">
            <b-form-select v-model="convert_form.pytorch_version"
                          :options="options.pytorch_version"
                          label="PyTorch version:"
                          class="mb-3">
                  <template slot="first">
                  </template>
            </b-form-select>
            </b-form-group>
            </div>

            <div v-if="convert_form.model_framework === 'tensorflow'">
            <b-form-group id="form-tensorflow_version-group"
                          label="TensorFlow version for conversion:"
                          label-for="form-tensorflow_version-input"
                          label-class="font-weight-bold">
            <b-form-select v-model="convert_form.tensorflow_version"
                          :options="options.tensorflow_version"
                          label="TensorFlow version:"
                          class="mb-3">
                  <template slot="first">
                  </template>
            </b-form-select>
            </b-form-group>
            </div>

            <b-form-group id="form-conversion_option-group"
                          label="Conversion Option:"
                          label-for="form-conversion_option-input"
                          label-class="font-weight-bold">
            <b-form-select v-model="conversion_option">
                  <template slot="first"></template>
                  <option value=0>Run With Configuaration JSON File</option>
                  <option value=1>Run With Inline Arguments</option>
                </b-form-select>
            </b-form-group>
            <hr/>

            <div v-if="conversion_option == 0">
              <b-form-group id="form-conversion_config-group"
                        label="Conversion Configuration JSON File:"
                        label-for="form-conversion_config-input"
                        label-class="font-weight-bold">
                <b-form-file id="form-conversion_config-input"
                        v-model="conversion_config"
                        placeholder="Select your conversion configuration json file">
                </b-form-file>
              </b-form-group>
            </div>

            <div v-if="conversion_option == 1">

            <b-form-group id="form-model-group"
                        label="Model:"
                        label-for="form-model-input"
                        label-class="font-weight-bold">
            <b-form-file id="form-model-input"
                            v-model="convert_form.model"
                            required
                            placeholder="Choose a model...For TensorFlow saved model, please upload a zip or tar file">
            </b-form-file>
            </b-form-group>

            <b-form-group id="form-input_names-group"
                        label="Model Inputs Names:"
                        label-for="form-input_names-input"
                        label-class="font-weight-bold">
            <b-form-input id="form-input_names-input"
                    type="text"
                    v-model="convert_form.input_names"
                    placeholder="Enter Model Inputs Names with Comma Seperated. e.g. input_1,input_2,input3">
                </b-form-input>
            </b-form-group>

            <b-form-group id="form-output_names-group"
                        label="Model Outputs Names:"
                        label-for="form-output_names-input"
                        label-class="font-weight-bold">
            <b-form-input id="form-output_names-input"
                    type="text"
                    v-model="convert_form.output_names"
                    placeholder="Enter Model Outputs Names with Comma Seperated. e.g. output_1,output_2">
                </b-form-input>
            </b-form-group>

            <div class="open_button" v-on:click="adv_setting = !adv_setting">
                  Advanced Settings
            </div>
            <br/>
            <div v-if="adv_setting">
              <b-form-group id="form-input_shapes-group"
                          label="Model Inputs Shapes:"
                          label-for="form-input_shapes-input"
                          label-class="font-weight-bold">
              <b-form-input id="form-input_shapes-input"
                      type="text"
                      v-model="convert_form.input_shapes"
                      placeholder="Enter List of Shapes of each Input Node. e.g. [[1,7],[1,7],[1,7]]">
                  </b-form-input>
              </b-form-group>

              <b-form-group id="form-output_shapes-group"
                          label="Model Outputs Shapes:"
                          label-for="form-output_shapes-input"
                          label-class="font-weight-bold">
              <b-form-input id="form-output_shapes-input"
                      type="text"
                      v-model="convert_form.output_shapes"
                      placeholder="Enter List of Shapes of each Output Node. e.g. [[1,4],[1,4]]">
                  </b-form-input>
              </b-form-group>

              <b-form-group id="form-input_types-group"
                          label="Model Inputs Types:"
                          label-for="form-input_types-input"
                          label-class="font-weight-bold">
              <b-form-input id="form-input_types-input"
                      type="text"
                      v-model="convert_form.input_types"
                      placeholder="Enter Model Inputs Types with Comma Seperated. e.g. float32,float32,float32">
                  </b-form-input>
              </b-form-group>

              <b-form-group id="form-output_types-group"
                          label="Model Outputs Types:"
                          label-for="form-output_types-input"
                          label-class="font-weight-bold">
              <b-form-input id="form-output_types-input"
                      type="text"
                      v-model="convert_form.output_types"
                      placeholder="Enter Model Outputs Types with Comma Seperated. e.g. float32,float32">
                  </b-form-input>
              </b-form-group>

              <b-form-group id="form-sample_input_data_path-group"
                        label="Model Sample Input Data (.npz file):"
                        label-for="form-sample_input_data_path-input"
                        label-class="font-weight-bold">
              <b-form-file id="form-sample_input_data_path-input"
                        v-model="sample_input_data_path"
                        placeholder="Select your sample input data">
              </b-form-file>
              </b-form-group>

              <b-form-group id="form-onnx_opset-group"
                          label="ONNX Target Opset:"
                          label-for="form-onnx_opset-input"
                          label-class="font-weight-bold">
              <b-form-input id="form-onnx_opset-input"
                              type="text"
                              v-model="convert_form.onnx_opset"
                              placeholder="Enter ONNX target opset">
              </b-form-input>
              </b-form-group>

              <div v-if="convert_form.model_framework === 'pytorch'">
              <b-form-group id="form-model_root_path-group"
                          label="Model Root Path:"
                          label-for="form-model_root_path-input"
                          label-class="font-weight-bold">
              <b-form-input id="form-model_root_path-input"
                              type="text"
                              v-model="convert_form.model_root_path"
                              placeholder="Enter model root path">
              </b-form-input>
              </b-form-group>
              </div>
              </div>

            </div>
            <hr>
            <b-button type="submit"
              :disabled="model_running" variant="primary" class="button_right">Submit</b-button>
            <b-button type="reset" :disabled="model_running" variant="danger">Reset</b-button>
        </b-form>
        </div>
        <hr>
        <alert :message=message :link=link :loading=model_running v-if="show_message"></alert>
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
      convert_form,
      test_data: [],
      conversion_option: 3,
      savedModel_vars: [],
      options: {
        model_framework: [
          { value: 'pytorch', text: 'PyTorch' },
          { value: 'tensorflow', text: 'TensorFlow' },
        ],
        tensorflow_version: ['1.11', '1.12', '1.13', '1.14', '1.15'],
        pytorch_version: ['1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '1.10', '1.11'],
        tf_model_type: [
          'savedModel',
          'frozen graph',
          'checkpoint',
        ],
      },
      tf_model_type: '',
      message: '',
      adv_setting: false,
      show_message: false,
      model_missing: '',
      convert_result: null,
      host: `${window.location.protocol}//${window.location.host.split(':')[0]}`,
      model_running: false,
      link: '',
      job_id: '',
      job_name: `onnx-converter-${Date.now()}`,
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
      this.job_name = `onnx-converter-${Date.now()}`;
    },
    convert(evt) {
      this.close_all();
      this.model_running = true;
      this.show_message = true;
      this.message = `Submitting job ${this.job_name}`;
      evt.preventDefault();
      const metadata = this.convert_form;
      const json = JSON.stringify(metadata);
      const blob = new Blob([json], {
        type: 'application/json',
      });

      const data = new FormData();
      data.append('job_name', this.job_name);
      data.append('metadata', blob);
      data.append('file', this.convert_form.model);
      data.append('sample_input_data_path', this.sample_input_data_path);
      data.append('conversion_config', this.conversion_config);
      this.savedModel_vars = [];

      axios.post(`${this.host}:5000/convert`, data)
        .then((res) => {
          this.link = `${this.host}:5000/convertresult/${res.data.job_id}`;
          this.show_message = true;
          this.model_running = false;
          this.message = 'Running job at ';
          this.update_result(res.data.job_id);
        })
        .catch((error) => {
          // eslint-disable-next-line]
          this.model_running = false;
          this.message = error;
        });
      this.job_name = `onnx-converter-${Date.now()}`;
    },
    update_result(location) {
      axios.get(`${this.host}:5000/convertstatus/${location}`)
        .then((res) => {
          if (res.data.state == 'SUCCESS') {
            this.$emit('convert_model', res.data.converted_model);
            this.message = 'Job completed. See results at ';
          } else if (res.data.state == 'FAILURE') {
            this.message = 'Job failed. See results at ';
          } else {
            // rerun in 2 seconds
            setTimeout(() => this.update_result(location), 2000);
          }
        })
        .catch((error) => {
          this.model_running = false;
          this.message = error;
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
