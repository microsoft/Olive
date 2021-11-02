<template>
    <div class="container">
        <h3>Job "{{job_name}}"</h3>
        <div v-if="error_message">
          <b-alert show variant="danger">
            <h5>Error:</h5>
            {{error_message}}</b-alert>
        </div>
        <div v-else>
        <b-table style="table-layout: fixed"
            striped hover
            responsive="sm">
        </b-table>
        <div v-if=convert_result>
            <h5>Conversion Status:
            <b-badge variant="primary">
                {{convert_result['conversion_status']}}
            </b-badge>
            </h5>
            <div v-if="convert_result['conversion_status'] == 'SUCCESS'">
            <h5>Download: </h5>
            <a :href="host + ':5000/download/' + convert_result['converted_model']" download>[model]</a>
            </div>
            <br>
            <hr>
            <details style="margin: 10px">
              <table class="table-responsive-lg" style="table-layout: fixed">
              <thead>
                  <tr>
                  <th scope="col">logs</th>
                  </tr>
              </thead>
              <tbody>
                <p>
                  {{ convert_result['logs'] }}
                </p>
              </tbody>
              </table>
            </details>
        </div>
        <alert :message=message :link=link v-if="show_message"></alert>
        </div>
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
      error_message: '',
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
              logs: res.data.logs,
              converted_model: res.data.converted_model,
              conversion_status: res.data.conversion_status,
            };
            this.$emit('convert_model', res.data.converted_model);
            this.show_message = false;
          } else if (res.data.state == 'FAILURE') {
            this.error_message = res.data;
          } else if (res.data.state == 'STARTED') {
            // rerun in 2 seconds
            this.show_message = true;
            this.message = 'Job running. Auto refreshing the page in 2 seconds. ';
            setTimeout(() => this.update_result(this.id), 2000);
          } else {
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
