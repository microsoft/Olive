import argparse
from OnnxAnimateDiffPipeline import OnnxAnimateDiffPipeline
from pathlib import Path
import numpy as np
import json
from olive.workflows import run as olive_run
import os
import shutil

parser = argparse.ArgumentParser("Common arguments")
parser.add_argument("--steps", default=2, type=int, help="Number of steps. Should match model")
parser.add_argument(
    "--prompt",
    default=(
        "a cat smiling"
    ),
    type=str,
)
parser.add_argument(
    "--input",
    default=(
        "models/stable-diffusion-v1-5"
    ),
    type=str,
)
parser.add_argument(
    "--output",
    default=(
        "animation.gif"
    ),
    type=str,
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="The seed to give to the generator to generate deterministic results.",
)
parser.add_argument(
    "--guidance_scale",
    default=1,
    type=float,
    help="Guidance scale as defined in Classifier-Free Diffusion Guidance",
)
parser.add_argument("--save_data", action="store_true")
parser.add_argument("--data_dir", default="quantize_data", type=str)
parser.add_argument("--split", action="store_true")
parser.add_argument("--quantize", action="store_true")


def split(input_dir: Path):
    # Split model
    with Path("config_unet_split.json").open() as file:
        olive_config = json.load(file)
    run_res = olive_run(olive_config)

    # Move model
    i = 0
    while True:
        f = input_dir / 'unet_split' / 'output_model' / f'split_{i}_model'
        if not os.path.exists(f): break
        move_to = input_dir / f'unet_{i}'
        os.makedirs(move_to, exist_ok=True)
        shutil.move(f / f'split_{i}.onnx', move_to / 'model.onnx')
        shutil.move(f / f'split_{i}.onnx.data', move_to / f'split_{i}.onnx.data')
        i += 1

    # Update config
    update_model_index(input_dir, i)


def quantize(input_dir: Path, output: Path):
    with Path("config_unet_qnn_template.json").open() as file:
        olive_config = json.load(file)

    i = 0
    while True:
        f = input_dir / f'unet_{i}' / "model.onnx"
        if not os.path.exists(f): break

        olive_config["input_model"]["model_path"] = f.as_posix()
        output_f = output / f"unet_qnn_{i}"
        olive_config["output_dir"] = output_f.as_posix()
        olive_config["data_configs"][0]["dataloader_config"]["id"] = i

        print(f"Start optimize {f}")
        run_res = olive_run(olive_config)

        move_to = output / f"unet_{i}"
        os.makedirs(move_to, exist_ok=True)
        shutil.move(output_f / "output_model" / 'model.onnx', move_to / 'model.onnx')
        i += 1
        if i == 1: break

    update_model_index(output, 5)
    

def update_model_index(path: Path, total: int):
    with (path / "model_index.json").open() as file:
        config = json.load(file)
    config.pop("unet")
    for i in range(total):
        config[f"unet_{i}"] = [
            "diffusers",
            "OnnxRuntimeModel"
        ]
    with (path / "model_index.json").open("w") as file:
        json.dump(config, file, indent=4)


def main(raw_args=None):
    args = parser.parse_args(raw_args)

    if args.split:
        split(Path(args.input))
        return
    
    if args.quantize:
        quantize(Path(args.input), Path(args.output))
        return

    from diffusers import EulerDiscreteScheduler
    from diffusers.utils import export_to_gif

    pipe = OnnxAnimateDiffPipeline.from_pretrained(
        args.input, provider="CPUExecutionProvider"
    )

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

    generator = None if args.seed is None else np.random.RandomState(seed=args.seed)
    save_data_dir = None
    if args.save_data:
        import os
        save_data_dir = Path(args.data_dir) / args.prompt
        os.makedirs(save_data_dir, exist_ok=True)
    output = pipe(prompt=args.prompt, guidance_scale=args.guidance_scale, num_inference_steps=args.steps, decode_chunk_size=1, generator=generator, save_data_dir=save_data_dir)
    export_to_gif(output.frames[0], args.output)


if __name__ == "__main__":
    main()