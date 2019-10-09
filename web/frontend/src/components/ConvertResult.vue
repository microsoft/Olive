// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
<template>
    <div class="container">
        <h3>Job {{job_name}}</h3>
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
            <h5>Error:
            <b-badge variant="danger">{{convert_result['output_json']['error_message']}}</b-badge>
            </h5>
            <h5>Download</h5>
            <a :href="host + ':5000/' + convert_result['input_path']" download>[input] </a>
            <a :href="host + ':5000/' + convert_result['model_path']" download>[model]</a>
        </div>
    </div>
</template>

<script>
import axios from 'axios';

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
    };
  },
  mounted() {
    this.update_result(this.id);
    this.get_args();
    this.get_job_name();
  },
  methods: {
    get_args() {
      axios.get(`${this.host}:5000/getargs/${this.id}`)
        .then((res) => {
          for (const i in res.data) {
            if (res.data[i].length > 0) {
              this.args.push({
                arg: i,
                value: res.data[i],
              });
            }
          }
        }).catch((error) => {
          this.message = error.toString();
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
    update_result(location) {
      axios.get(`${this.host}:5000/convertstatus/${location}`)
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
            this.message = 'Job Failed. See logs for more info. ';
          }
          // } else {
          //     // rerun in 2 seconds
          //     setTimeout(() => this.update_result(location), 2000);
          // }
        })
        .catch((error) => {
          this.message = error.toString();
        });
    },
  },
};
</script>
