from pathlib import Path

from olive.common.utils import run_subprocess

models = {
    "Llama-2-7B": "meta-llama/Llama-2-7b-hf",
    "Llama-2-13B": "meta-llama/Llama-2-13b-hf",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "Phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}


def main():
    asset_dir = Path(__file__).parent.parent / "assets" / "cost_models"
    asset_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_id in models.items():
        run_subprocess(
            ["olive", "generate-cost-model", "-m", model_id, "-o", str(asset_dir / f"{model_name}.csv")], check=True
        )


if __name__ == "__main__":
    main()
