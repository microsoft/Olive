<template>
    <div class='container'>
        <h3>Job "{{job_name}}"</h3>
          <div v-if="error_message">
          <b-alert show variant="danger">
            <h5>Error:</h5>
            {{error_message}}</b-alert>
          </div>
        <div v-if="Object.keys(result).length > 0">
        <ul class="list-group">
            <li class="list-group-item">
                <table class="table-responsive-lg" style="table-layout: fixed">
                <thead>
                    <tr>
                    <th scope="col">pretuning avg (ms)</th>
                    <th scope="col">optimized avg (ms)</th>
                    <th scope="col">optimized p90 (ms)</th>
                    <th scope="col">optimized p95 (ms)</th>
                    <th scope="col">pretuning throughput</th>
                    <th scope="col">optimal throughput</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{{result.pretuning_avg}}</td>
                        <td>{{result.latency_ms_avg}}</td>
                        <td>{{result.latency_ms_p90}}</td>
                        <td>{{result.latency_ms_p95}}</td>
                        <td>{{result.pretuning_throughput}}</td>
                        <td>{{result.optimal_throughput}}</td>
                    </tr>
                </tbody>
                </table>
                <hr>
                <div>
                <h5>Download Optimized Model: </h5>
                <a :href="host + ':5000/download/' + result.optimized_model" download>[model]</a>
                </div>
                <hr>
                <details style="margin: 10px">
                  <table class="table-responsive-lg" style="table-layout: fixed">
                  <thead>
                      <tr>
                      <th scope="col">execution_provider</th>
                      <th scope="col">environment variables</th>
                      <th scope="col">session_options</th>
                      </tr>
                  </thead>
                  <tbody>
                      <tr>
                          <td>{{result.execution_provider}}</td>
                          <td>{{result.env_vars}}</td>
                          <td>{{result.session_options}}</td>
                      </tr>
                  </tbody>
                  </table>
                  <hr>
                  <table class="table-responsive-lg" style="table-layout: fixed">
                  <thead>
                      <tr>
                      <th scope="col">python sample script</th>
                      </tr>
                  </thead>
                  <tbody>
                    <p v-for="line in result.sample_script"  v-bind:key="line">
                      {{ line }}
                    </p>
                  </tbody>
                  </table>
                  <hr>
                  <table class="table-responsive-lg" style="table-layout: fixed">
                  <thead>
                      <tr>
                      <th scope="col">logs</th>
                      </tr>
                  </thead>
                  <tbody>
                    <p>
                      {{ this.message }}
                    </p>
                  </tbody>
                  </table>
                </details>
            </li>
        </ul>
        </div>
        <alert :message=message :link=link v-if="show_message"></alert>
    </div>
</template>

<script>
import axios from 'axios';
import Alert from './Alert.vue';

export default {
  name: 'PerfResult',
  data() {
    return {
      host: `${window.location.protocol}//${window.location.host.split(':')[0]}`,
      id: this.$route.params.id,
      arg_fields: ['arg', 'value'],
      args: [],
      sample_script: [],
      result: {},
      message: '',
      code_details: '',
      PROFILING_MAX: 5,
      selected: -1,
      show_message: false,
      job_name: '',
      link: '',
      error_message: '',
    };
  },
  components: {
    alert: Alert,
  },
  mounted() {
    try {
      this.get_args();
      this.get_job_name();
      this.update_result();
    } catch (e) {
      this.message = e.toString();
      this.show_message = true;
    }
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
      axios.get(`${this.host}:5000/perfstatus/${this.id}`)
        .then((res) => {
          if (res.data.state == 'SUCCESS') {
            const { logs } = res.data;
            this.message = logs;
            this.result = JSON.parse(res.data.result);
            this.$emit('update_model', this.result.optimized_model);
            this.sample_script = this.result.sample_script;
            this.show_message = false;
          } else if (res.data.state == 'FAILURE') {
            this.error_message = res.data;
          } else if (res.data.state == 'STARTED') {
            // rerun in 2 seconds
            this.show_message = true;
            this.message = 'Job running. Auto refreshing the page in 2 seconds. ';
            setTimeout(() => this.update_result(this.id), 2000);
          } else {
            // rerun in 2 seconds
            this.show_message = true;
            this.link = `${this.host}:5000/jobmonitor`;
            this.message = 'Job is pending or the job does not exist. Try refreshing the page or browse all available jobs at ';
          }
        })
        .catch((error) => {
          this.message = error.toString();
        });
    },
    format_code_snippet(code) {
      this.code_details = code.trim().replace(/\s\s+/g, '\n');
      this.$refs.codeModel.show();
    },
    open_details(index) {
      if (this.selected == index) {
        this.selected = -1;
      } else {
        this.selected = index;
      }
    },
  },
};
</script>
