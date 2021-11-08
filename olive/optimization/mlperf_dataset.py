import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Dataset():
    def __init__(self, session):
        self.data = []
        self.data_inmemory = {}
        self.session = session
        self.odd_shape = self.get_input_shapes()

    def get_item_count(self):
        return len(self.data)

    def load_query_samples(self, sample_list):
        self.data_inmemory = {}
        for sample in sample_list:
            self.data_inmemory[sample] = self.get_item(sample)

    def unload_query_samples(self, sample_list):
        self.data_inmemory = {}

    def make_batch(self, id_list):
        feed = {}
        for i, name in enumerate([meta.name for meta in self.session.get_inputs()]):
            feed[name] = np.array([self.data_inmemory[id][i] for id in id_list])
            if len(self.odd_shape[name]) != len(feed[name].shape):
                feed[name] = np.squeeze(feed[name], axis=0)
        return feed

    def get_input_shapes(self):
        shapes = {meta.name: meta.shape for meta in self.session.get_inputs()}
        for k, v in shapes.items():
            for i, d in enumerate(v):
                if isinstance(d, str):
                    v[i] = -1
        return shapes

