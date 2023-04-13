# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import requests
from tqdm import tqdm

subdir = "data"
if not os.path.exists(subdir):
    os.makedirs(subdir)
subdir = subdir.replace("\\", "/")

for ds in ["small-117M", "webtext"]:
    for split in ["valid"]:
        filename = ds + "." + split + ".jsonl"
        r = requests.get("https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/" + filename, stream=True)

        with open(os.path.join(subdir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as progress_bar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    progress_bar.update(chunk_size)
