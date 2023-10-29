import os
from pathlib import Path
from typing import List

import torch
from argmax_sampling_model import ArgmaxSampling
from decoder_model_test import DecoderModel
from sentencepiece import SentencePieceProcessor
from update_embeddings_model import UpdateEmbeddings


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


max_seq_len = 50

data_type = torch.float16

n_layers = 32
n_heads = 32
hidden_size = 4096

prompt = "What data were you trained on?"

tokenizer = Tokenizer(model_path="models/optimized/llama_v2/tokenizer.model")
tokens = tokenizer.encode(prompt, bos=True, eos=False)
tokens = torch.tensor(tokens, dtype=torch.int64)
seq_len = tokens.shape[0]

attn_mask = torch.nn.functional.pad(
    torch.ones((1, seq_len), dtype=torch.int32, device=torch.device("cuda")), ((max_seq_len - seq_len, 0))
)

head_dim = int(hidden_size / n_heads)

cache_shape = (1, n_heads, max_seq_len, head_dim)
initial_cache = torch.zeros(cache_shape, dtype=data_type, device=torch.device("cuda"))

k_caches = []
v_caches = []

for layer_idx in range(n_layers):
    k_caches.append(initial_cache)
    v_caches.append(initial_cache)

max_gen_len = 20

decoder_model = DecoderModel(n_layers, tokenizer.n_words, hidden_size, n_heads, "SquareRootHeadDim")
decoder_model.set_use_cache(False)
decoder_model.eval()
script_dir = Path(__file__).resolve().parent
weights_path = script_dir / "raw_model_data" / "7B-chat" / "llama-2-7b-chat.pth"

state_dict = torch.load(weights_path, map_location=torch.device("cuda"))


# permute for sliced rotary
def permute(weight):
    return (
        weight.view(n_heads, hidden_size // n_heads // 2, 2, hidden_size)
        .transpose(1, 2)
        .reshape(hidden_size, hidden_size)
    )


for layer_idx in range(n_layers):
    state_dict[f"layers.{layer_idx}.attention.wq.weight"] = permute(
        state_dict[f"layers.{layer_idx}.attention.wq.weight"]
    )
    state_dict[f"layers.{layer_idx}.attention.wk.weight"] = permute(
        state_dict[f"layers.{layer_idx}.attention.wk.weight"]
    )

del state_dict["rope.freqs"]
decoder_model.load_state_dict(state_dict)

update_embeddings_model = UpdateEmbeddings(
    r"C:\Users\pavignol\projects\Llama-2-Onnx\7B_FT_float16\embeddings.pth", tokenizer.n_words, hidden_size
)
update_embeddings_model.eval()

argmax_sampling_model = ArgmaxSampling()
argmax_sampling_model.eval()

caches = []
for layer_idx in range(n_layers):
    caches.append(
        {
            "key": k_caches[layer_idx],
            "value": v_caches[layer_idx],
        }
    )

position_ids = torch.arange(seq_len, dtype=torch.int64, device=torch.device("cuda")).reshape((1, seq_len))

output_tokens = []
for idx in range(max_gen_len):
    if idx > 0:
        decoder_model.set_use_cache(True)
        position_ids = torch.arange(seq_len, seq_len + 1, dtype=torch.int64, device=torch.device("cuda")).reshape(
            (1, 1)
        )

    print(tokens)
    x = update_embeddings_model(tokens).to("cuda")

    print("LALALA")
    outputs = decoder_model(x, position_ids, attn_mask, caches)

    logits = outputs[0]
    attn_mask = outputs[1].to("cuda")

    for layer_idx in range(n_layers):
        caches[layer_idx] = {
            "key": outputs[layer_idx * 2 + 2],
            "value": outputs[layer_idx * 2 + 3],
        }

    next_token = argmax_sampling_model(logits).to("cpu")

    output_tokens.append(next_token.cpu().numpy().item())

    if output_tokens[-1] == tokenizer.eos_id:
        break

    tokens = next_token

    seq_len += 1

output_str = tokenizer.decode(output_tokens)
print(output_str)
