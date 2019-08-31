<template>
    <div>
        <div ref="visualizeModal"
            id="visualizeModal"
            title="Visualization"
            hide-footer>
            <b-form @submit="visualize" class="w-100">
            <b-form-group id="visualize_model"
                            label="Model for visualization:"
                            label-for="form-visualize_model-input">
                <b-form-file id="form-visualize_model-input"
                                v-model="visualize_model"
                                required
                                placeholder="Choose a model..."
                                drop-placeholder="Drop model here...">
                </b-form-file>
                </b-form-group>
                <b-button type="submit" :disabled="model_running" variant="primary">Submit</b-button>
            </b-form>
        </div>
        <hr/>
        <iframe :src="this.host + ':8080'" width="100%" height="800" v-if="show_visualization">

        </iframe>
        <hr/>
    </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      visualize_model: null,
      show_visualization: false,
      host: `${window.location.protocol}//${window.location.host.split(':')[0]}`,
      model_running: false,
    };
  },
  methods: {
    visualize(evt) {
      evt.preventDefault();
      this.model_running = true;
      const path = `${this.host}:5000/visualize`;
      const data = new FormData();
      data.append('file', this.visualize_model);
      axios.post(path, data)
        .then((res) => {
          if (res.data.status === 'success') {
            this.show_visualization = true;
            this.model_running = false;
          }
        })
        .catch((error) => {
          // eslint-disable-next-line
          this.model_running = false;
            console.log(error);
        });
    },
  },

};
</script>
