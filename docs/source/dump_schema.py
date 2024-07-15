import argparse
from pathlib import Path

from olive.workflows.run.config import RunConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump workflow schema")

    parser.add_argument("--output", type=str, default="schema.json", help="Output file")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(RunConfig.schema_json(indent=2))
