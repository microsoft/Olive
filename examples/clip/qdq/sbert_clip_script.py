import torch
import torch.nn.functional as F
from transformers import DistilBertModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import cached_file


class SDistilBertTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distilbert = DistilBertModel.from_pretrained(model_name).eval()
        self.dense_weights = torch.load(cached_file(model_name, "2_Dense/pytorch_model.bin"))["linear.weight"]

    def _get_boolean_mask(self, attn_mask):
        return attn_mask > -1.0

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask):
        model_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Mean Pooling - Take attention mask into account for correct averaging
        last_hidden_state = model_output[0]
        input_mask_expanded = (
            self._get_boolean_mask(attention_mask).float().unsqueeze(dim=-1).expand(last_hidden_state.size())
        )
        sum_state = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        pooled_output = F.normalize(sum_state, p=2, dim=1)

        # Calculate embedding
        text_embeds = torch.matmul(pooled_output, self.dense_weights.T)

        return ModelOutput(
            embeds=text_embeds,
            last_hidden_state=last_hidden_state,
        )


class SimpleDistilBert(torch.nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = model.config
        self.embeddings = model.embeddings
        self.transformer = model.transformer

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask):
        embedding_output = self.embeddings(input_ids)
        head_mask = [None] * self.config.num_hidden_layers
        sequence_output = self.transformer(
            x=embedding_output,
            attn_mask=attention_mask,
            head_mask=head_mask,
        )
        return ModelOutput(
            last_hidden_state=sequence_output[0],
        )


class SimpleSDistilBertTextEncoder(SDistilBertTextEncoder):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.distilbert = SimpleDistilBert(self.distilbert)  # codeql: disable=py/overwritten-inherited-attribute

    def _get_boolean_mask(self, attn_mask):
        return attn_mask[:, 0, 0, :] > -1.0


def load_sdistilbert_text_encoder(model_name: str):
    return SimpleSDistilBertTextEncoder(model_name).eval()
