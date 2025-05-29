import logging
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import onnxruntime_genai as og
import torch
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from tqdm import tqdm

logger = logging.getLogger(__name__)

LogLikelihoodInputs = tuple[tuple[str, str], list[int], list[int]]


@register_model("onnx", "ONNX")
class LMEvalOnnxModelEvaluator(TemplateLM):
    def __init__(
        self,
        pretrained: og.Model,
        tokenizer: og.Tokenizer,
        max_length: Optional[int] = 256,
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        self._model = pretrained
        self._tokenizer = tokenizer
        self._max_length = max_length or 256
        self._device = device
        self._batch_size = batch_size or 1

        self._params = og.GeneratorParams(self._model)

        search_options = {"max_length": self._max_length}
        self._params.set_search_options(**search_options)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self._tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        """Tokenize a string using the model's tokenizer and return a list of token IDs."""
        return self._tokenizer.encode(string).tolist()

    def _model_call(self, input_ids: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        generator = og.Generator(self._model, self._params)
        generator.append_tokens(input_ids)

        count = input_ids.shape[0]
        with torch.no_grad():
            while not generator.is_done() and count < self._max_length:
                generator.generate_next_token()
                count += 1

            logits = generator.get_logits().squeeze().squeeze().tolist()
            tokens = generator.get_sequence(0)

        log_probs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=-1).numpy()
        return logits, log_probs, tokens

    def _loglikelihood_tokens(self, requests: list[LogLikelihoodInputs], **kwargs) -> list[tuple[float, bool]]:
        def _collate(req: LogLikelihoodInputs):
            """Define the key for the sorted method."""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        disable_tqdm = kwargs.get("disable_tqdm") or False

        result = []
        re_ord = Collator(requests, sort_fn=_collate, group_by=None)
        pbar = tqdm(desc="Running loglikelihood requests", total=len(requests), disable=disable_tqdm)
        for chunk in re_ord.get_batched(n=self._batch_size):
            _, context_enc, continuation_enc = next(iter(chunk))

            input_ids = (context_enc + continuation_enc)[-(self._max_length + 1) :][:-1]
            ctx_len = len(input_ids)

            if len(context_enc) + len(continuation_enc) > (self._max_length + 1):
                logger.warning(
                    "Context length (%d) + continuation length (%d) > max_length (%d). Left truncating context.",
                    len(context_enc),
                    len(continuation_enc),
                    self._max_length,
                )

            input_ids = np.asarray(input_ids)
            _, log_probs, output_tokens = self._model_call(input_ids)

            cont_len = len(continuation_enc)
            cont_tokens = np.asarray(continuation_enc)
            greedy_tokens = output_tokens[ctx_len - cont_len : ctx_len]

            is_greedy = (cont_tokens == greedy_tokens).all()
            log_probs = np.take(log_probs, cont_tokens, 0)

            answer = (float(log_probs.sum()), bool(is_greedy))
            result.append(answer)

            pbar.update(1)

        pbar.close()
        return re_ord.get_original(result)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        raise NotImplementedError("Yet to be implemented!")

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("Yet to be implemented!")
