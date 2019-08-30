<template>
  <div style="white-space: pre-line;">
    <b-alert variant="success" show>{{ message }}</b-alert>
    <b-progress 
      :v-if=model_running
      :value="100" variant="success" striped class="mt-2"></b-progress>
    <br>
  </div>
</template>

<script>
export default {
  props: ['message', 'model_running'],
};
</script>
