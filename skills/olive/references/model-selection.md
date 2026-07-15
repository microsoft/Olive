# Model and Hardware Selection

Use these as starting recommendations only when the user has not selected a model. Preserve the user's
model choice, and do not claim that an example is the latest or best without current evidence.

| Use case | Starting models |
| --- | --- |
| Text chat or general LLM | `microsoft/Phi-4-mini-instruct` for smaller hardware; `microsoft/Phi-4` for more capacity |
| Code generation | `microsoft/Phi-4-mini-instruct` |
| Image generation | `runwayml/stable-diffusion-v1-5` or `stabilityai/stable-diffusion-xl-base-1.0` |
| Embedding or retrieval | `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2` |
| Vision and language | `microsoft/Phi-4-multimodal-instruct` |

After selecting a candidate, use `detect_hardware` before choosing optimization settings.

## Hardware heuristics

- NVIDIA GPU with at least 8 GB VRAM: fp16 optimization and LoRA fine-tuning are reasonable starting
  points.
- NVIDIA GPU with 4-8 GB VRAM: prefer int4 or int8; use QLoRA rather than LoRA for text fine-tuning.
- No detected NVIDIA GPU: use CPU int4 or int8. On Windows, DirectML may still be available for an AMD,
  Intel, or NVIDIA DirectX 12 GPU.
- Less than 8 GB system RAM: prefer a smaller model such as Phi-4 Mini and int4 quantization.

These are planning heuristics, not guarantees. Model architecture, sequence length, batch size, calibration,
and training settings can materially change memory requirements.
