import onnxruntime
import numpy as np
import time
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

dataset = load_dataset("sentence-transformers/stsb", split="test")
sentences1 = dataset["sentence1"]
sentences2 = dataset["sentence2"]

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

options = onnxruntime.SessionOptions()
session = onnxruntime.InferenceSession(
    r"path_to_model.onnx",
    sess_options=options,
    providers=["QNNExecutionProvider"],
    provider_options=[{"backend_path": "QnnHtp.dll"}],
)

def mean_pooling(embeddings, attention_mask):
    mask = np.expand_dims(attention_mask, axis=-1)
    sum_embeddings = np.sum(embeddings * mask, axis=1)
    sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def encode_onnx(session, sentence):
    tokens = tokenizer(
        sentence,
        padding="max_length",
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    inputs = {
        "input_ids": input_ids.int().cpu().numpy(),
        "attention_mask": attention_mask.float().cpu().numpy(),
    }
    outputs = session.run(None, inputs)
    embedding = mean_pooling(np.array(outputs[0]), attention_mask)
    return embedding.squeeze(0)

cosine_similarities = []
inference_times = []

for s1, s2 in zip(sentences1, sentences2):
    start_time = time.time()
    emb1 = encode_onnx(session, s1)
    inference_times.append(time.time() - start_time)

    start_time = time.time()
    emb2 = encode_onnx(session, s2)
    inference_times.append(time.time() - start_time)

    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    cosine_similarities.append(cosine_sim)

cosine_similarities = np.array(cosine_similarities)
gold_scores = np.array(dataset["score"])

pearson_corr, _ = pearsonr(gold_scores, cosine_similarities)
spearman_corr, _ = spearmanr(gold_scores, cosine_similarities)

print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")
print(f"Average Inference Time: {np.mean(inference_times):.4f} seconds")
