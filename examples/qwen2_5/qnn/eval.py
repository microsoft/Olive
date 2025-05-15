# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import numpy as np
import onnxruntime_genai as og
import torch
import time

model_dir = "models/model"
# max_length = 128
max_length = 1024

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model = og.Model(model_dir)
tokenizer = og.Tokenizer(model)
params = og.GeneratorParams(model)
params.set_search_options(do_sample=False, max_length=max_length, batch_size=1)

input_ids = [ 1528,   974,   528,   292,   280, 28744, 29901,   319,   767,
            338, 16246,   373,   263, 17526, 29889,   940,   338,   773,
            12244,   304, 12244,   263,  5101,   310,  2071,   275]
input_ids = np.asarray(input_ids)

start_time = time.time()
num_iterations = 10
time1 = 0
time2 = 0
time3 = 0
time4 = 0
for _ in range(num_iterations):
    t0 = time.time()
    generator = og.Generator(model, params)
    time1 += time.time() - t0

    t0 = time.time()
    generator.append_tokens(input_ids)
    time2 += time.time() - t0

    # logits = []
    count = input_ids.shape[0]
    with torch.no_grad():
        while count < max_length:
            t0 = time.time()
            if generator.is_done():
                break
            time3 += time.time() - t0

            t0 = time.time()
            generator.generate_next_token()
            time4 += time.time() - t0

            count += 1
            # print("Stepping: ", count)

    #     logits = generator.get_logits().squeeze().squeeze().tolist()
    #     sequence = generator.get_sequence(0)

    # sequence = np.asarray(sequence)
    # logits = np.asarray(logits)

total_time = time.time() - start_time

print(f"total_time: {total_time}")
print(f"average: {total_time / num_iterations}")

print(f"time1: {time1 / total_time * 100} %")
print(f"time2: {time2 / total_time * 100} %")
print(f"time3: {time3 / total_time * 100} %")
print(f"time4: {time4 / total_time * 100} %")


# print("tokens.shape: ", sequence.shape)
# print("logits.shape: ", logits.shape)
