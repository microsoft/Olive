import Vue from 'vue';
import Router from 'vue-router';
import Pipeline from '@/components/Pipeline';

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Pipeline',
      component: Pipeline
    }
  ],
  mode: 'history'
});
