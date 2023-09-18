# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch


class UpdateEmbeddings(torch.nn.Module):
    def __init__(self, embedding_file: str, vocab_size: int, hidden_size: int):
        super(UpdateEmbeddings, self).__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, hidden_size, dtype=torch.float16)
        self.embedding_layer.load_state_dict(torch.load(embedding_file))

    def forward(self, tokens):
        embeddings = self.embedding_layer(tokens)
        embeddings = torch.unsqueeze(embeddings, 0)
        return embeddings
