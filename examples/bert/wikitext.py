import logging
from random import Random
from typing import Optional

import nltk
import pandas as pd
from datasets import Dataset, load_dataset
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")

logger = logging.getLogger("Evaluate")


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


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-csv",
        type=str,
        default="nsp_wikitext_pairs.csv",
        help="Output CSV file name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name to load.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset name to load.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Number of sentence pairs to generate.",
    )
    parser.add_argument(
        "--sentence-cols",
        type=list[str],
        default=["sentence1", "sentence2"],
        help="Column names for the sentences.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Column name for the labels.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum length of sentences to include.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create NSP dataset
    ds = create_nsp_dataset(
        args.dataset,
        name=args.name,
        sent_cols=args.sentence_cols,
        label_col=args.label_col,
        max_samples=args.max_samples,
        min_length=args.min_length,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    # Save the dataset to CSV
    data_set = ds.to_pandas()
    # print(f"Dataset size: {len(data_set)}")
    # print(data_set.head())

    data_set.to_csv(args.output_csv, index=False)
    # print(f"Dataset saved to {args.output_csv}")
