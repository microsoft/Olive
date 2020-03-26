// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
<template>
    <div class="container">
        <h3>Job "{{job_name}}"</h3>
        <b-table style="table-layout: fixed"
            :items="args"
            :fields="fields"
            striped hover
            responsive="sm">
        </b-table>
        <div v-if=convert_result>
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
            <h5 v-if="convert_result['output_json']['error_message'].length > 0">Error:
            <b-badge variant="danger">{{convert_result['output_json']['error_message']}}</b-badge>
            </h5>
            <div v-if="convert_result['output_json']['conversion_status'] == 'SUCCESS'">
                <h5>Download: </h5>
                <a :href="host + ':5000/' + convert_result['input_path']" download>[input] </a>
                <a :href="host + ':5000/' + convert_result['model_path']" download>[model]</a>
            </div>                
            <br>
        </div>
        <alert :message=message :link=link v-if="show_message"></alert>
    </div>
</template>

<script>
import axios from 'axios';
import Alert from './Alert.vue';

export default {
  name: 'ConvertResult',
  data() {
    return {
      host: `${window.location.protocol}//${window.location.host.split(':')[0]}`,
      id: this.$route.params.id,
      fields: ['arg', 'value'],
      convert_result: null,
      args: [],
      job_name: '',
      message: '',
      show_message: false,
      link: '',
    };
  },
  components: {
    alert: Alert,
  },
  mounted() {
    this.update_result();
    this.get_args();
    this.get_job_name();
  },
  methods: {
    get_args() {
      axios.get(`${this.host}:5000/getargs/${this.id}`)
        .then((res) => {
          const args = Object.keys(res.data);
          for (let i = 0; i < args.length; i++) {
            if (res.data[args[i]].length > 0) {
              this.args.push({
                arg: args[i],
                value: res.data[args[i]],
              });
            }
          }
        }).catch((error) => {
          this.message = error.toString();
          this.show_message = true;
        });
    },
    get_job_name() {
      axios.get(`${this.host}:5000/getjobname/${this.id}`)
        .then((res) => {
          const nameBrkIndex = res.data.name.indexOf('.');
          this.job_name = res.data.name.substring(nameBrkIndex + 1);
        }).catch((error) => {
          this.message = error.toString();
        });
    },
    update_result() {
      this.link = '';
      axios.get(`${this.host}:5000/convertstatus/${this.id}`)
        .then((res) => {
          if (res.data.state == 'SUCCESS') {
            this.convert_result = {
              output_json: res.data.output_json,
              input_path: res.data.input_path,
              model_path: res.data.model_path,
            };
            this.$emit('update_model', res.data.converted_model);
            this.show_message = false;
          } else if (res.data.state == 'FAILURE') {
            // TODO
            this.show_message = true;
            this.message = `Job Failed. ${res.data.status}`;
          } else if (res.data.state == 'STARTED') {
            // rerun in 2 seconds
            this.show_message = true;
            this.message = 'Job running. Auto refreshing the page in 2 seconds. ';
            setTimeout(() => this.update_result(this.id), 2000);
          } else {
            // rerun in 2 seconds
            this.show_message = true;
            this.message = 'Job is pending or the job does not exist. Try refreshing the page or browse all available jobs at ';
            this.link = `${this.host}:8000/jobmonitor`;
          }
        })
        .catch((error) => {
          this.show_message = true;
          this.message = error.toString();
        });
    },
  },
};
</script>
