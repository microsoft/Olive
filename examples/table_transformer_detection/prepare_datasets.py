# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201
# pylint: disable=unsubscriptable-object

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

# Path to Detection\annotations\tablebank_latex_val.json
tablebank_json_path = Path(r"path_to_tablebank_latex_val.json")
# Path to Detection\images
tablebank_image_dir = Path(r"path_to_images")

output_root_dir = Path("tablebank_test")
output_json_path = output_root_dir / "tablebank_latex_val_small.json"
output_image_dir = output_root_dir / "tablebank_latex_val_small_images"

with tablebank_json_path.open("r", encoding="utf-8") as f:
    data: Dict = json.load(f)
images: List[Dict] = data["images"]
annotations: List[Dict] = data["annotations"]

# Randomly select 256 images
selected_images = random.sample(images, 256)
selected_image_ids = {img["id"] for img in selected_images}
selected_annotations = [ann for ann in annotations if ann["image_id"] in selected_image_ids]
small_data = {
    "images": selected_images,
    "annotations": selected_annotations,
    "categories": data["categories"],
}

with output_json_path.open("w", encoding="utf-8") as f:
    json.dump(small_data, f, indent=4)

output_image_dir.mkdir(parents=True, exist_ok=True)
for img in selected_images:
    src_path = tablebank_image_dir / img["file_name"]
    dst_path = output_image_dir / img["file_name"]
    if src_path.exists():
        shutil.copy2(src_path, dst_path)

print(f"Small dataset generated: {output_json_path}")
print(f"Images saved to: {output_image_dir}")
