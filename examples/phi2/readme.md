## Prerequisites
* einops

## Usage
```bash
python -m olive.workflows.run --config phi2_optimize.json
```

## Limitations
when https://github.com/huggingface/optimum/issues/1642 is fixed, we need turn on the post_process by changing `no_post_process` to False in the config file.
