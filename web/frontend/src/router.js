// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
import Vue from 'vue';
import Router from 'vue-router';
import Convert from '@/components/Convert.vue';
import Perf from '@/components/Perf.vue';
import Visualize from '@/components/Visualize.vue';
import JobMonitor from '@/components/JobMonitor.vue';
import ConvertResult from '@/components/ConvertResult.vue';
import PerfResult from '@/components/PerfResult.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      name: 'Pipeline',
    },
    {
      path: '/convert',
      name: 'Convert',
      component: Convert,
    },
    {
      path: '/perf',
      name: 'Perf',
      component: Perf,
    },
    {
      path: '/visualize',
      name: 'Visualize',
      component: Visualize,
    },
    {
      path: '/jobmonitor',
      name: 'JobMonitor',
      component: JobMonitor,
    },
    {
      path: '/convertresult/:id',
      name: 'ConvertResult',
      component: ConvertResult,
    },
    {
      path: '/perfresult/:id',
      name: 'PerfResult',
      component: PerfResult,
    },
  ],
});
