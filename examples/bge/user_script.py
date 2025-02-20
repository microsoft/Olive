from olive.model import OliveModelHandler, HfModelHandler
from olive.constants import Framework
from olive.workflows import run as olive_run
import mteb
from typing import List
from transformers import AutoTokenizer, BertModel
import numpy as np
import json
import torch
from pathlib import Path
from olive.data.registry import Registry

class OliveEncoder:
    def __init__(self, model, session):
        self.model = model
        self.session = session
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
        self.total = 0
        self.max_len = 0

    def encode(self, corpus: List, **kwargs):
        if self.model.framework == Framework.ONNX:
            encoded_input = self.tokenizer(corpus, padding="max_length", max_length=128, truncation=True, return_tensors='np')
            # batch_size is 1 for static model
            model_outputs = []
            for i in range(len(corpus)):
                model_inputs = {
                    "input_ids": encoded_input.input_ids[i:i+1,:].astype(np.int64),
                    "attention_mask": encoded_input.attention_mask[i:i+1,:].astype(np.int64),
                    "token_type_ids": encoded_input.token_type_ids[i:i+1,:].astype(np.int64)
                }
                model_output = self.model.run_session(self.session, model_inputs)[0]
                model_outputs.append(model_output[0])
            model_output = np.array(model_outputs)
        elif self.model.framework == Framework.PYTORCH:
            encoded_input = self.tokenizer(corpus, padding=True, truncation=True, return_tensors='pt')
            model_inputs = {
                "input_ids": encoded_input.input_ids,
                "attention_mask": encoded_input.attention_mask,
                "token_type_ids": encoded_input.token_type_ids
            }
            self.max_len = max(self.max_len, model_inputs["input_ids"].shape[1])
            print(self.max_len)
            with torch.no_grad():
                model_output = self.model.run_session(self.session, model_inputs)
            model_output = model_output.last_hidden_state.numpy()
        # select the last hidden state of the first token (i.e., [CLS]) as the sentence embedding.
        model_output = model_output[:, 0, :]
        self.total += len(corpus)
        print(self.total)
        return model_output


def eval_accuracy(model: OliveModelHandler, device, execution_providers, tasks):
    sess = model.prepare_session(inference_settings=None, device=device, execution_providers=execution_providers)

    evaluation = mteb.MTEB(tasks=tasks)
    oliveEncoder = OliveEncoder(model, sess)
    results = evaluation.run(oliveEncoder, output_folder=None)
    return results[0].scores["test"][0]["main_score"]


class DataLoader:
    def __init__(self, data):
        self.input_ids = torch.from_numpy(data["input_ids"])
        self.attention_mask = torch.from_numpy(data["attention_mask"])
        self.token_type_ids = torch.from_numpy(data["token_type_ids"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        data = {"input_ids": self.input_ids[idx],"attention_mask": self.attention_mask[idx],"token_type_ids": self.token_type_ids[idx]}
        return data


@Registry.register_pre_process()
def dataset_pre_process(output_data, **kwargs):
    cache_key = kwargs.get("cache_key")
    cache_file = None
    if cache_key:
        cache_file = Path(f"{cache_key}.npz")
        if cache_file.exists():
            with np.load(Path(cache_file)) as data:
                return DataLoader(data)
    
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
    input_ids = []
    attention_mask = []
    token_type_ids = []
    for item in output_data:
        encoded_input = tokenizer(item['text'], padding="max_length", max_length=128, truncation=True, return_tensors='np')
        input_ids.append(encoded_input.input_ids[0].astype(np.int64))
        attention_mask.append(encoded_input.attention_mask[0].astype(np.int64))
        token_type_ids.append(encoded_input.token_type_ids[0].astype(np.int64))
    
    data = {"input_ids": np.array(input_ids), "attention_mask": np.array(attention_mask), "token_type_ids": np.array(token_type_ids)}
    result_data = DataLoader(data)

    if cache_file:
        cache_file.parent.resolve().mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, **data)

    return result_data


if __name__ == "__main__":
    with Path("bge-small-en-v1.5.json").open() as fin:
        olive_config = json.load(fin)
    olive_run(olive_config)