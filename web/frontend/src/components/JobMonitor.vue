// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
<template>
    <div class="container">
        <b-button variant="outline-primary" v-on:click=get_jobs>Refresh</b-button>
        <b-table style="table-layout: fixed"
            :items="tasks"
            :fields="fields"
            :sort-by.sync="sortBy"
            :sort-desc.sync="sortDesc"
            striped hover
            @row-clicked="get_details"
            responsive="sm">
        </b-table>
    </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'JobMonitor',
  data() {
    return {
      sortBy: 'started',
      sortDesc: true,
      fields: [
        { key: 'name', sortable: true },
        { key: 'state', sortable: true },
        { key: 'type', sortable: true },
        { key: 'received', sortable: true },
        { key: 'started', sortable: true },
      ],
      host: `${window.location.protocol}//${window.location.host.split(':')[0]}`,
      tasks: [],
    };
  },
  mounted() {
    this.get_jobs();
  },
  methods: {
    get_jobs() {
      axios.get(`${this.host}:5000/gettasks`)
        .then((res) => {
          this.tasks = [];
          for (let i = 0; i < Object.keys(res.data).length; i++) {
            const t = res.data[Object.keys(res.data)[i]];
            const nameBrkIndex = t.name.indexOf('.');
            this.tasks.push({
              name: t.name.substring(nameBrkIndex + 1),
              state: t.state,
              type: t.name.substring(0, nameBrkIndex),
              received: t.received,
              started: t.started,
              id: Object.keys(res.data)[i],
            });
          }
        });
    },
    get_details(row) {
      if (row.type == 'convert') {
        window.open(`${this.host}:8000/convertresult/${row.id}`, '_blank');
      } else if (row.type == 'perf_tuning') {
        window.open(`${this.host}:8000/perfresult/${row.id}`, '_blank');
      }
    },
  },
};
</script>
