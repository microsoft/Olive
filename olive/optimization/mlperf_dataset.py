import logging

import numpy as np

from ..constants import QUERY_COUNT, ONNX_TO_NP_TYPE_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Dataset():
    def __init__(self, session, inputs_spec):

        self.data = []
        self.odd_shape = inputs_spec

        inputs = session.get_inputs()
        input_types = []
        for i in range(0, len(inputs)):
            if inputs[i].type in ONNX_TO_NP_TYPE_MAP.keys():
                input_types.append(ONNX_TO_NP_TYPE_MAP[inputs[i].type])
        
        data_shapes = [[i for i in dims if i != -1] for dims in inputs_spec.values()]
        data = []
        for i in range(len(data_shapes)):
            vals = np.random.random_sample(data_shapes[i]).astype(input_types[i])
            data.append(vals)

        for i in range(QUERY_COUNT):
            self.data.append(data)
        self.data_inmemory = {}
        self.session = session

    def get_item_count(self):
        return len(self.data)

    def load_query_samples(self, sample_list):
        self.data_inmemory = {}
        for sample in sample_list:
            self.data_inmemory[sample] = self.get_item(sample)

    def get_item(self, num):
        return self.data[num]

    def unload_query_samples(self, sample_list):
        self.data_inmemory = {}

    def make_batch(self, id_list):
        feed = {}
        for i, name in enumerate([meta.name for meta in self.session.get_inputs()]):
            feed[name] = np.array([self.data_inmemory[id][i] for id in id_list])
            if len(self.odd_shape[name]) != len(feed[name].shape):
                feed[name] = np.squeeze(feed[name], axis=0)
        return feed
