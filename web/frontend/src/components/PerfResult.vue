// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
<template>
    <div class='container'>
        <h3>Job {{job_name}}</h3>
        <b-table style="table-layout: fixed"
            :items="args"
            :fields="arg_fields"
            striped hover
            responsive="sm">
        </b-table>
        <div v-if="Object.keys(result).length > 0">
        <ul class="list-group" v-for="(ep, index) in Object.keys(result)" :key="index">
            <li class="list-group-item" v-if="result[ep].length > 0">
                <h5>{{index+1}}. {{ep}} </h5>
                <table class="table-responsive-lg" style="table-layout: fixed">
                <thead>
                    <tr>
                    <th scope="col">name</th>
                    <th scope="col">avg (ms)</th>
                    <th scope="col">p90 (ms)</th>
                    <th scope="col">p95 (ms)</th>
                    <th scope="col">cpu (%)</th>
                    <th scope="col">gpu (%)</th>
                    <th scope="col">memory (%)</th>
                    <!--<th>code_snippet.execution_provider</th>-->
                    <th>code_snippet</th>
                    <th>profiling</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{{result[ep][0].name}}</td>
                        <td>{{result[ep][0].avg}}</td>
                        <td>{{result[ep][0].p90}}</td>
                        <td>{{result[ep][0].p95}}</td>
                        <td>{{result[ep][0].cpu_usage * 100}}%</td>
                        <td>{{result[ep][0].gpu_usage * 100}}%</td>
                        <td>{{result[ep][0].memory_util * 100}}%</td>
                        <td>
                        <div v-on:click="format_code_snippet(result[ep][0].code_snippet.code)"
                            v-bind:class="{open: !(selected == index)}"
                            class="before_open open_button"
                            v-b-modal.codeModal>details </div>
                        </td>
                        <!--profiling-->
                        <td>
                        <div
                            v-on:click="open_profiling(profiling[index].slice(0, PROFILING_MAX))"
                            v-bind:class="{open: !(selected_profiling == index)}"
                            class="before_open open_button" v-b-modal.opsModal>op </div>
                        </td>

                    </tr>
                </tbody>
                </table>
                <details style="margin: 10px">
                    <summary>
                        More options with good performance
                    </summary>
                    <div v-for="(item, index) in result[ep]" :key="index">
                        <p v-if="index > 0">{{item.name}}</p>
                    </div>
                </details>
            </li>
        </ul>
        </div>
        <div class="open_button" v-on:click="show_message = !show_message" v-if="show_logs">
            <hr/>Show logs
        </div>
        <alert :message=message :link=link v-if="show_message"></alert>
        <br/>
        <b-modal ref="opsModal"
                id="opsModal"
                title="Top 5 ops"
                size="lg"
                hide-footer>
            <b-container fluid>
            <b-table class="table-responsive-lg" style="table-layout: fixed"
                striped hover :items="op_info" :fields="fields"></b-table>
            </b-container>
        </b-modal>
        <b-modal ref="codeModal"
                id="codeModal"
                title="Code details" style="width: 100%;"
                hide-footer>
            <div style="white-space: pre-wrap">{{code_details}}</div>
        </b-modal>
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
      result: {},
      message: '',
      profiling: [],
      op_info: {},
      fields: ['name', 'duration', 'op_name', 'tid'],
      code_details: '',
      PROFILING_MAX: 5,
      selected: -1,
      selected_profiling: -1,
      show_logs: false,
      show_message: false,
      job_name: '',
      link: '',
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
    } catch(e) {
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
            this.show_logs = true;
            this.message = logs;
            this.result = JSON.parse(res.data.result);
            this.profiling = res.data.profiling;
            this.show_message = false;
          } else if (res.data.state == 'FAILURE') {
            this.message = res.data;
            this.show_logs = true;
          } else if (res.data.state == 'STARTED') {
            // rerun in 2 seconds
            this.show_message = true;
            this.message = 'Job running. Auto refreshing the page in 10 seconds. ';
            setTimeout(() => this.update_result(this.id), 10000);
          } else {
            // rerun in 2 seconds
            this.show_message = true;
            this.link = `${this.host}:8000/jobmonitor`;
            this.message = 'Job is pending or the job does not exist. Try refreshing the page or browse all available jobs at ';
          }
        })
        .catch((error) => {
          this.message = error.toString();
        });
    },
    format_code_snippet(code) {
      this.code_details = code.trim().replace(/\s\s+/g, '\n');
    },
    open_details(index) {
      if (this.selected == index) {
        this.selected = -1;
      } else {
        this.selected = index;
      }
    },
    open_profiling(ops) {
      this.op_info = [];
      for (let i = 0; i < ops.length; ++i) {
        this.op_info.push({
          name: ops[i].name,
          duration: ops[i].dur,
          op_name: ops[i].args.op_name,
          tid: ops[i].tid,
        });
      }
    },
  },
};
</script>
