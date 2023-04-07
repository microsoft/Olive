from datasets import load_dataset

from olive.dataset import DatasetPipeline


class HuggingfaceDatasetPipeline(DatasetPipeline):
    def load_dataset(self, name, **hf_kwargs):
        return load_dataset(name, **hf_kwargs)

    def pre_process(self):
        # tokenizer
        pass

    def post_process(self):
        pass
