#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libomp.so:$LD_PRELOAD"


numactl -c 0 -m 0 python quark_dlrm_eva.py --max-batchsize=64000 --model-path=/workspace/zhiqchen/model/dlrm-multihot-pytorch.pt --int8-model-dir /workspace/zhiqchen/model/dlrm_quark --int8-model-name DLRM_INT_tt --dataset-path=/workspace/zhiqchen/data/Criteo1TBMultiHotPreprocessed --calibration
