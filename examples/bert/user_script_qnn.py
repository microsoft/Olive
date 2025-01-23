# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import types

from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


def new_bert_forward(self, *args, **kwargs):
    """Forward method for BertModel that accepts a 4D extented attention mask."""
    embeddings_output = self.embeddings(
        input_ids=args[0],
        token_type_ids=kwargs["token_type_ids"],
    )
    sequence_output = self.encoder(embeddings_output, attention_mask=kwargs["attention_mask"])[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
    )


def load_bert(model_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.bert.forward = types.MethodType(new_bert_forward, model.bert)
    return model
