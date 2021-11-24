import logging
import threading
import time
import array

import mlperf_loadgen as lg
import numpy as np

from ..constants import QUERY_COUNT, NANO_SEC, MILLI_SEC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ServerRunner():
    def __init__(self, session, ds, optimization_config, onnx_output_names):

        self.session = session
        self.threads = optimization_config.threads_num
        self.max_batchsize = optimization_config.dynamic_batching_size
        self.ds = ds
        self.onnx_output_names = onnx_output_names
        self.guess = None

        self.cv = threading.Condition()
        self.done = False
        self.q_idx = []
        self.q_query_id = []
        self.workers = []

        self.settings = lg.TestSettings()
        self.settings.scenario = lg.TestScenario.Server
        self.settings.mode = lg.TestMode.FindPeakPerformance

        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = optimization_config.result_path
        log_output_settings.copy_summary_to_stdout = False
        self.log_settings = lg.LogSettings()
        self.log_settings.enable_trace = False
        self.log_settings.log_output = log_output_settings

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        self.qsl = lg.ConstructQSL(QUERY_COUNT, QUERY_COUNT, ds.load_query_samples, ds.unload_query_samples)

        self.settings.server_coalesce_queries = True
        self.settings.server_target_latency_ns = int(optimization_config.max_latency * NANO_SEC)
        self.settings.server_target_latency_percentile = optimization_config.max_latency_percentile
        self.settings.min_duration_ms = optimization_config.min_duration_sec * MILLI_SEC

        # start all threads
        for _ in range(self.threads):
            worker = threading.Thread(target=self.handle_tasks, args=(self.cv,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()
        time.sleep(1)

    def issue_queries(self, query_samples):
        self.enqueue(query_samples)

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ms):
        pass

    def handle_tasks(self, cv):
        """Worker thread."""
        max_batchsize = self.max_batchsize
        stats = [0] * (max_batchsize + 1)
        while True:
            with cv:
                # wait for something to do
                while len(self.q_idx) == 0 and not self.done:
                    cv.wait()
                idx = self.q_idx
                query_id = self.q_query_id
                if len(idx) > max_batchsize:
                    # only take max_batchsize
                    self.q_idx = idx[max_batchsize:]
                    self.q_query_id = query_id[max_batchsize:]
                    idx = idx[:max_batchsize]
                    query_id = query_id[:max_batchsize]
                    # wake up somebody to take care of it
                    cv.notify()
                else:
                    # swap the entire queue
                    self.q_idx = []
                    self.q_query_id = []
            if self.done:
                # parent wants us to exit
                break
            # run inference, lock is released
            feed = self.ds.make_batch(idx)
            self.run_one_item((query_id, idx, feed))

            # count stats
            stats[len(idx)] += 1

    def run_one_item(self, qitem):
        # run the prediction
        processed_results = []
        query_id, content_id, feed = qitem
        results = self.session.run(self.onnx_output_names, feed)
        processed_results = [[]] * len(query_id)
        response_array_refs = []
        response = []
        for idx, qid in enumerate(query_id):
            response_array = array.array("B", np.array(processed_results[idx], np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        with self.cv:
            scheduled = len(self.q_idx)
            # add new items to the queue
            self.q_idx.extend(idx)
            self.q_query_id.extend(query_id)
            # notify only if queue was empty
            if scheduled == 0:
                self.cv.notify()

    def finish(self):
        # exit all threads
        self.done = True
        for worker in self.workers:
            with self.cv:
                self.cv.notify()
        for worker in self.workers:
            worker.join()

    def start_run(self):
        lg.StartTestWithLogSettings(self.sut, self.qsl, self.settings, self.log_settings)
    
    def warmup(self, warmup_num):
        self.ds.load_query_samples([0])

        start = time.time()
        for _ in range(warmup_num):
            feed = self.ds.make_batch([0])
            _ = self.session.run(self.onnx_output_names, feed)
        self.guess = (time.time() - start) / warmup_num
        self.settings.server_target_qps = int(1 / self.guess / 3)

        self.ds.unload_query_samples(None)