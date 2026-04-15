# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""MTEB-compatible wrappers for ONNX Runtime embedding models.

Mirrors the pattern in ``lmeval_ort.py``: thin adapters that bridge an
exported ONNX (or ORT-GenAI) model into the interface expected by the
evaluation library — in this case MTEB's ``EncoderProtocol``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from olive.common.onnx_io import get_io_config

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

logger = logging.getLogger(__name__)


class MTEBOnnxBase(ABC):
    """Base class for MTEB-compatible ONNX embedding model wrappers.

    Subclasses must implement :meth:`_encode_batch` which takes tokenised
    inputs and returns a numpy embedding matrix.
    """

    def __init__(self, tokenizer_path: str, batch_size: int = 32, max_length: int | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.batch_size = batch_size
        self._max_length = max_length

    @property
    def max_length(self) -> int:
        if self._max_length:
            return self._max_length
        if hasattr(self.tokenizer, "model_max_length") and self.tokenizer.model_max_length < 1_000_000:
            return self.tokenizer.model_max_length
        return 512

    # ------------------------------------------------------------------
    # MTEB EncoderProtocol interface
    # ------------------------------------------------------------------

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        """Encode sentences into embeddings — MTEB ``EncoderProtocol.encode``."""
        # Handle string input (single sentence)
        if isinstance(inputs, str):
            inputs = [inputs]

        # Flatten DataLoader batches into a plain list of strings
        sentences: list[str] = []
        for batch in inputs:
            if isinstance(batch, dict) and "text" in batch:
                sentences.extend(batch["text"])
            elif isinstance(batch, (list, tuple)):
                sentences.extend(batch)
            elif isinstance(batch, str):
                sentences.append(batch)
            else:
                sentences.extend(list(batch))

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            embeddings = self._encode_batch(encoded)
            all_embeddings.append(embeddings)

        result = np.concatenate(all_embeddings, axis=0)
        # L2 normalize (matches SentenceTransformer Normalize module)
        # Compute norms in float32 for numerical stability when ORT returns
        # low-precision outputs such as float16/bfloat16, then cast back.
        result_dtype = result.dtype
        result_fp32 = result.astype(np.float32, copy=False)
        norms = np.linalg.norm(result_fp32, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized = result_fp32 / norms
        return normalized.astype(result_dtype, copy=False)

    @staticmethod
    def similarity(embeddings1, embeddings2):
        """Cosine similarity — default for MTEB."""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        embeddings1 = torch.nn.functional.normalize(embeddings1.float(), p=2, dim=-1)
        embeddings2 = torch.nn.functional.normalize(embeddings2.float(), p=2, dim=-1)
        return embeddings1 @ embeddings2.T

    @staticmethod
    def similarity_pairwise(embeddings1, embeddings2):
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        embeddings1 = torch.nn.functional.normalize(embeddings1.float(), p=2, dim=-1)
        embeddings2 = torch.nn.functional.normalize(embeddings2.float(), p=2, dim=-1)
        return (embeddings1 * embeddings2).sum(dim=-1)

    @property
    def mteb_model_meta(self):
        from mteb.models.model_meta import ModelMeta

        return ModelMeta.create_empty()

    @abstractmethod
    def _encode_batch(self, encoded_input: dict) -> np.ndarray:
        """Run model inference on a tokenised batch and return embeddings.

        Args:
            encoded_input: Dictionary with at least ``input_ids`` and
                ``attention_mask`` as numpy arrays of shape ``[batch, seqlen]``.

        Returns:
            Embeddings array of shape ``[batch, embed_dim]``.

        """
        raise NotImplementedError


# ------------------------------------------------------------------
# ORT (plain ONNX) variant
# ------------------------------------------------------------------


class MTEBORTEvaluator(MTEBOnnxBase):
    """MTEB wrapper for a plain ONNX embedding model run via ORT."""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        max_length: int | None = None,
        ep: str | None = None,
        ep_options: dict | None = None,
    ):
        import onnxruntime as ort

        model_dir = str(Path(model_path).parent)
        super().__init__(tokenizer_path=model_dir, batch_size=batch_size, max_length=max_length)

        providers = []
        if ep:
            providers.append((ep, ep_options or {}))
        providers.append(("CPUExecutionProvider", {}))

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.io_config = get_io_config(model_path)
        self._output_names = self.io_config["output_names"]

    def _encode_batch(self, encoded_input: dict) -> np.ndarray:
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        feeds = {"input_ids": input_ids.astype(np.int64), "attention_mask": attention_mask.astype(np.int64)}
        # Some models also accept token_type_ids
        if "token_type_ids" in list(self.io_config["input_names"]):
            feeds["token_type_ids"] = encoded_input.get("token_type_ids", np.zeros_like(input_ids, dtype=np.int64))

        outputs = self.session.run(None, feeds)

        # Determine which output contains the embeddings.
        # Common patterns: "last_hidden_state" (index 0) or a dedicated "sentence_embedding" output.
        if len(outputs) == 1:
            hidden_states = outputs[0]
        else:
            # Try to find a sentence-level output first
            for i, name in enumerate(self._output_names):
                if "sentence" in name.lower() or "pooler" in name.lower() or "embedding" in name.lower():
                    return outputs[i]  # Already pooled
            # Fall back to the first output (last_hidden_state)
            hidden_states = outputs[0]

        # Last-token pooling (matches Qwen3-Embedding pooling_mode_lasttoken=True)
        return self._last_token_pool(hidden_states, attention_mask)

    @staticmethod
    def _last_token_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Last-token pooling: take the hidden state at the last non-padding token position."""
        token_counts = attention_mask.sum(axis=1).astype(int)
        if np.any(token_counts <= 0):
            raise ValueError("attention_mask contains a zero-length sequence; cannot perform last-token pooling.")
        sequence_lengths = token_counts - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[np.arange(batch_size), sequence_lengths]


# ------------------------------------------------------------------
# ORT-GenAI variant  (for models built with ModelBuilder)
# ------------------------------------------------------------------


class MTEBORTGenAIEvaluator(MTEBOnnxBase):
    """MTEB wrapper for an ORT-GenAI embedding model (ModelBuilder output)."""

    def __init__(
        self,
        pretrained: str,
        batch_size: int = 32,
        max_length: int | None = None,
        ep: str = "follow_config",
        ep_options: dict | None = None,
    ):
        if og is None:
            raise ImportError("onnxruntime-genai is not installed.")

        super().__init__(tokenizer_path=pretrained, batch_size=batch_size, max_length=max_length)

        self.config = og.Config(pretrained)
        if ep != "follow_config":
            ep_clean = ep.lower().replace("executionprovider", "")
            self.config.clear_providers()
            if ep_clean != "cpu":
                self.config.append_provider(ep_clean)
            for key, value in (ep_options or {}).items():
                self.config.set_provider_option(ep_clean, key, value)

        self.model = og.Model(self.config)
        self.og_tokenizer = og.Tokenizer(self.model)
        self.pretrained = pretrained

    def _encode_batch(self, encoded_input: dict) -> np.ndarray:
        input_ids = encoded_input["input_ids"].astype(np.int64)
        attention_mask = encoded_input["attention_mask"].astype(np.int64)
        batch_size, seq_len = input_ids.shape

        # GenAI Generator does not accept attention_mask, so padding tokens
        # contaminate hidden states via self-attention. To avoid this, group
        # sequences by real length and process each group as a single batch
        # (no padding needed within a group of equal-length sequences).
        real_lengths = attention_mask.sum(axis=1).astype(int)

        # Group sample indices by their real (non-padding) token count
        length_to_indices: dict[int, list[int]] = {}
        for i in range(batch_size):
            length_to_indices.setdefault(int(real_lengths[i]), []).append(i)

        all_embeddings = [None] * batch_size
        for real_len, indices in length_to_indices.items():
            # Stack all sequences of the same length into a single batch
            ids_batch = np.stack([input_ids[i, :real_len] for i in indices], axis=0)
            group_batch_size = len(indices)

            params = og.GeneratorParams(self.model)
            params.set_search_options(
                max_length=real_len + 1, past_present_share_buffer=False, batch_size=group_batch_size
            )

            generator = og.Generator(self.model, params)
            generator.append_tokens(ids_batch.tolist())

            try:
                hidden_states = generator.get_output("hidden_states")
            except Exception as e:
                raise RuntimeError(
                    "Failed to retrieve hidden_states from the ORT-GenAI model. "
                    "Ensure the model was built with include_hidden_states=1."
                ) from e
            hidden_states = np.array(hidden_states, copy=False)
            if hidden_states.ndim == 2:
                embed_dim = hidden_states.shape[-1]
                hidden_states = hidden_states.reshape(group_batch_size, real_len, embed_dim)

            # Last-token pooling: take the final token for each sequence
            for j, idx in enumerate(indices):
                all_embeddings[idx] = hidden_states[j, -1, :]

        return np.stack(all_embeddings, axis=0)
