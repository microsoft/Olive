import argparse
import pickle
from pathlib import Path

import onnxruntime as ort

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # get available execution providers
    available_eps = ort.get_available_providers()

    # save to pickle
    output_path = Path(args.output_path)
    pickle.dump(available_eps, output_path.open("wb"))
