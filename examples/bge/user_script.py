from olive.model import OliveModelHandler, HfModelHandler
from olive.constants import Framework
from olive.workflows import run as olive_run
import mteb
from typing import List
from transformers import AutoTokenizer, BertModel
import numpy as np
import json
from pathlib import Path

class OliveEncoder:
    def __init__(self, model, session):
        self.model = model
        self.session = session
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')

    def encode(self, corpus: List, **kwargs):
        if self.model.framework == Framework.ONNX:
            encoded_input = self.tokenizer(corpus, padding=True, truncation=True, return_tensors='np')
            model_inputs = {
                "input_ids": encoded_input.input_ids.astype(np.int64),
                "attention_mask": encoded_input.attention_mask.astype(np.int64),
                "token_type_ids": encoded_input.token_type_ids.astype(np.int64)
            }
            model_output = self.model.run_session(self.session, model_inputs)[0]
        elif self.model.framework == Framework.PYTORCH:
            model: HfModelHandler = self.model
            session: BertModel = self.session # BertLMHeadModel
            encoded_input = self.tokenizer(corpus, padding=True, truncation=True, return_tensors='pt')
            model_inputs = {
                "input_ids": encoded_input.input_ids,
                "attention_mask": encoded_input.attention_mask,
                "token_type_ids": encoded_input.token_type_ids
            }
            model_output = model.run_session(session, model_inputs)
            model_output = model_output.last_hidden_state.detach().numpy()
        # select the last hidden state of the first token (i.e., [CLS]) as the sentence embedding.
        model_output = model_output[:, 0, :]
        return model_output


def eval_accuracy(model: OliveModelHandler, device, execution_providers, tasks):
    sess = model.prepare_session(inference_settings=None, device=device, execution_providers=execution_providers)

    evaluation = mteb.MTEB(tasks=tasks)
    oliveEncoder = OliveEncoder(model, sess)
    results = evaluation.run(oliveEncoder, output_folder=None)
    return results[0].scores.test[0].main_score


if __name__ == "__main__":
    with Path("bge-small-en-v1.5.json").open() as fin:
        olive_config = json.load(fin)
    olive_run(olive_config)