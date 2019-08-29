<template>
  <div style="white-space: pre-line;">
    <b-alert variant="success" show>{{ message }}</b-alert>
    <b-progress :value="100" variant="success" striped class="mt-2"></b-progress>
    <br>
  </div>
</template>

<script>
export default {
  props: ['message'],
};
</script>
