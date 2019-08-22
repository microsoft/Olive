<template>
  <div class="container" id="app">
    <div class="row flex-xl-nowrap">
      <main class="col-18 col-md-12" role="main">
        <h1>OLive (ONNX Live)</h1>
        <hr>
        <router-link to="/convert">
          <button type="button" class="btn btn-success btn-sm button_right" >
            Convert
          </button>
        </router-link>
        <router-link to="/perf" >
          <button type="button" class="btn btn-info btn-sm button_right">
            Pert Test
          </button>
        </router-link>
        <router-link to="/visualize">
          <button type="button" class="btn btn-primary btn-sm">
            Model Visualize
          </button>
        </router-link>
        <hr/>
        <keep-alive >
          <router-view v-on:update_model="updateModelHandler" :converted_model="converted_model"/>
        </keep-alive>
      </main>
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
</style>
