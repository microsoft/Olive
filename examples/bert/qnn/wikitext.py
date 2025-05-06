from random import Random
from typing import Optional

import nltk
import pandas as pd
from datasets import Dataset, load_dataset
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")


def extract_sentences(text, min_length=20):
    if text.startswith("=") or text.strip() == "":
        return []

    # Filter short sentences (<20 chars)
    sentences = sent_tokenize(text.strip())
    return [s for s in sentences if len(s) >= min_length]


def create_nsp_dataset(
    dataset: str,
    sent_cols: list[str] = None,
    label_col: str = "label",
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    **kwargs,
):
    random = Random(seed)

    if sent_cols is None:
        sent_cols = ["sentence1", "sentence2"]

    name = kwargs.get("name")
    split = kwargs.get("split", "train")
    min_length = kwargs.get("min_length", 20)

    d_s = load_dataset(dataset, name=name, split=split)

    # TODO(zhengte): refactor with d_s.map
    all_sentences = []
    for sample in d_s:
        sentences = extract_sentences(sample["text"], min_length=min_length)
        if sentences:
            all_sentences.append(sentences)
    if shuffle:
        random.shuffle(all_sentences)
    if max_samples is not None:
        all_sentences = all_sentences[:max_samples]

    # Create positive and negative sentence pairs
    pairs = []
    labels = []
    for doc_idx, doc in enumerate(all_sentences):
        if len(doc) < 2:
            continue

        # Positive example: consecutive sentences
        sent_idx = random.randrange(len(doc) - 1)
        sent1, sent2 = doc[sent_idx], doc[sent_idx + 1]
        pairs.append((sent1, sent2))
        labels.append(0)

        # Negative example: random sentence from different doc
        while True:
            rand_doc_idx = random.randrange(len(all_sentences))
            if rand_doc_idx == doc_idx:
                continue
            rand_doc = all_sentences[rand_doc_idx]
            rand_sent_idx = random.randrange(len(rand_doc))
            pairs.append((sent1, rand_doc[rand_sent_idx]))
            labels.append(1)
            break

    data = {
        sent_cols[0]: [pair[0] for pair in pairs],
        sent_cols[1]: [pair[1] for pair in pairs],
        label_col: labels,
    }

    return Dataset.from_pandas(pd.DataFrame(data))
