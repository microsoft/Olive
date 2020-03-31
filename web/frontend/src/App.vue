// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
<template>
  <div class="container" id="app">
    <div class="row flex-xl-nowrap">
      <main class="col-20 col-md-12" role="main">
        <h1>OLive (ONNX Go Live)</h1>
        <hr>
        <router-link to="/convert">
          <button type="button" class="btn btn-success btn-md button_right" >
            Convert
          </button>
        </router-link>
        <router-link to="/perf" >
          <button type="button" class="btn btn-info btn-md button_right">
            Pert Test
          </button>
        </router-link>
        <router-link to="/visualize">
          <button type="button" class="btn btn-primary btn-md button_right">
            Model Visualize
          </button>
        </router-link>
        <router-link to="/jobmonitor">
          <button type="button" class="btn btn-warning btn-md">
            Job Monitor
          </button>
        </router-link>
        <hr/>
        <keep-alive>
          <router-view
            v-on:update_model="updateModelHandler"
            :converted_model="converted_model"/>
        </keep-alive>
      </main>
      <footer></footer>
    </div>
  </div>
</template>

<script>
export default {
  name: 'App',
  data() {
    return {
      converted_model: '',
    };
  },
  methods: {
    updateModelHandler(value) {
      this.converted_model = value;
    },
  },

};
</script>

<style>
#app {
  margin-top: 60px
}

.button_right{
  margin-right: 15px;
}
.missing{
  margin-left: 15px;
  color: red;
}
table {
  border-collapse: collapse;
  width: 100%;
}

th, td {
  text-align: left;
  padding: 8px;
  word-wrap: break-word;
}
.before_open::after{
  content: "(-)";
  font-weight: bold;
}

.open::after{
  content: "(+)";
}
.op_table{
  padding: 10px;
  background: white;
}
.open_button{
  cursor: pointer;
  text-decoration: underline;
  color: #669;
}

</style>
