# DLRM Model Quantization Using Quark

This document provides examples of quantizing and exporting the DLRM models using Quark. Please refer to [DLRM](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch) for more details about the model.

## Preparation

### Envrionment

Run the script to prepare the environment at first

```bash

pip install scikit-learn pybind11 iopath==0.1.10 pyre_extensions==0.0.30
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install fbgemm-gpu==1.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install torchrec==0.7.0
pip install torchsnapshot==0.1.0

```

### Third-Party Dependencies

The example relies on some code from [inference_results_v3.1](https://github.com/mlcommons/inference_results_v3.1/tree/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python) repo.

```bash
git clone https://github.com/mlcommons/inference_results_v3.1.git
cd inference_results_v3.1
git checkout 951b4a7686692d1a0d9b9067a36a7fc26d72ada5
cp -r inference_results_v3.1/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/* /path/to/Quark/examples/torch/rm/utils
export PYTHONPATH=$PYTHONPATH:/path/to/Quark/examples/torch/rm/utils
pip install intel-extension-for-pytorch==2.5.0
pip install transformers==4.45.0
```

#### Install mlperf loadgen

```bash
git clone https://github.com/mlcommons/inference.git
cd inference
git checkout v4.1
git submodule update --init --recursive
cd loadgen
CFLAGS="-std=c++14" python setup.py install
cd ../..
```

There is a small bug in [this Line](https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/multihot_criteo.py#L418). When you copy the code to utils, remember to change that line as follows:

```python3

# offsets.append(torch.arange(0, batchsize*multi_hot_size, multi_hot_size))
offsets.append(torch.arange(0, (batchsize + 1)*multi_hot_size, multi_hot_size))

```

A change is required in the following file at [line](https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/model/dlrm_model.py#L204). When you copy the code to utils, replace the forward() function with the following function:

```python3
    def forward(
        self, index: List[torch.Tensor], offset: List[torch.Tensor], dense
    ) -> torch.Tensor:
        B = offset[0].numel() - 1

        res = []  # removed list comprehension
        for idx in range(len(self.embedding_bags)):
            e, i, o = self.embedding_bags[idx], index[idx], offset[idx]
            res.append(e(i, o))
        res = [dense] + res
        data = torch.cat(res, dim=1).reshape(
            B, (self._num_embeddings + 1) * self._embedding_dim
        )
        return data
```

### Model weights

For DLRM model, refer to the [README.md](https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v2/pytorch/README.md) to download the model weights and datasets for calibration. The model weights have multiple files, you need to pack them in a single pt file. Please run the script to get the single pt file

```bash
python utils/dump_torch_model.py \
    --model-path=/path/to/model_dir \
    --dataset-path=/path/to/data/Criteo1TBMultiHotPreprocessed
```

## Quantization & Export Scripts

Run the python script as follows

```python3
# The compressed quantized model
python quark_dlrm.py \
    --max-batchsize=64000 \
    --model-path=/path/to/dlrm-multihot-pytorch.pt \
    --int8-model-dir /dir/to/dlrm_quark \
    --int8-model-name DLRM_INT
    --dataset-path=/path/to/data/Criteo1TBMultiHotPreprocessed \
    --calibration \
    --compressed

# The QDQ model
python quark_dlrm.py \
    --max-batchsize=64000 \
    --model-path=/path/to/dlrm-multihot-pytorch.pt \
    --int8-model-dir /dir/to/dlrm_quark \
    --int8-model-name DLRM_INT
    --dataset-path=/path/to/data/Criteo1TBMultiHotPreprocessed \
    --calibration
```

## Evaluation

Quark currently uses Area Under the Curve(AUC) as the evaluation metric of dlrm for accuracy loss before and after quantization.The specific AUC algorithm can be referenced [roc_auc_score](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.roc_auc_score.html).

Run the python script as follows to calculate the AUC(Area Under Curve) for the quantized model

```python3
# evaluate the compressed quantized model
python quark_dlrm_eva.py \
    --max-batchsize=64000 \
    --model-path=/path/to/dlrm-multihot-pytorch.pt \
    --int8-model-dir /dir/to/dlrm_quark \
    --int8-model-name DLRM_INT
    --dataset-path=/path/to/data/Criteo1TBMultiHotPreprocessed \
    --calibration \
    --compressed

# evaluate the QDQ model
python quark_dlrm_eva.py \
    --max-batchsize=64000 \
    --model-path=/path/to/dlrm-multihot-pytorch.pt \
    --int8-model-dir /dir/to/dlrm_quark \
    --int8-model-name DLRM_INT
    --dataset-path=/path/to/data/Criteo1TBMultiHotPreprocessed \
    --calibration
```

The quantization evaluation results are conducted in pseudo-quantization mode, which may slightly differ from the actual quantized inference accuracy. These results are provided for reference only.

### Evaluation scores

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>dlrm </strong>
   </td>
   <td><strong>dlrm-embeddingbag-uint4-weight-int8(this model)</strong>
   </td>
  </tr>
  <tr>
   <td>AUC-MultihotCriteo
   </td>
   <td>0.8031
   </td>
   <td>0.8027
   </td>
  </tr>
</table>

<!--
## License
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
-->
