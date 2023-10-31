# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import ClassVar

from olive.systems.common import SystemType


class AzureMLSystemAlias:
    system_type = SystemType.AzureML
    accelerators: ClassVar[list] = ["GPU"]


class AzureND12SSystem(AzureMLSystemAlias):
    sku = "STANDARD_ND12S"
    num_cpus = 12
    num_gpus = 2
    # TODO(myguo): add other attributes when needed from
    # https://learn.microsoft.com/en-us/azure/virtual-machines/nd-series


class AzureND24RSSystem(AzureMLSystemAlias):
    sku = "STANDARD_ND24RS"
    num_cpus = 24
    num_gpus = 4


class AzureND24SSystem(AzureMLSystemAlias):
    sku = "STANDARD_ND24S"
    num_cpus = 24
    num_gpus = 4


class AzureNDV2System(AzureMLSystemAlias):
    sku = "STANDARD_ND40RS_V2"
    num_cpus = 40
    num_gpus = 8
    # add other attributes when needed from https://learn.microsoft.com/en-us/azure/virtual-machines/ndv2-series


class AzureND6SSystem(AzureMLSystemAlias):
    sku = "STANDARD_ND6S"
    num_cpus = 6
    num_gpus = 1
    # add other attributes when needed from https://learn.microsoft.com/en-us/azure/virtual-machines/nd-series


class AzureND96A100System(AzureMLSystemAlias):
    sku = "STANDARD_ND96AMSR_A100_V4"
    num_cpus = 96
    num_gpus = 8
    # add other attributes when needed from https://learn.microsoft.com/en-us/azure/virtual-machines/ndm-a100-v4-series


class AzureND96ASystem(AzureMLSystemAlias):
    sku = "STANDARD_ND96ASR_V4"
    num_cpus = 96
    num_gpus = 8
    # add other attributes when needed from https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series


# TODO(myguo): add the following alias system
# STANDARD_DS2_V2
# STANDARD_DS3_V2
# STANDARD_NC12
# STANDARD_NC12S_V2
# STANDARD_NC12S_V3
# STANDARD_NC16AS_T4_V3
# STANDARD_NC24
# STANDARD_NC24ADS_A100_V4
# STANDARD_NC24R
# STANDARD_NC24RS_V2
# STANDARD_NC24RS_V3
# STANDARD_NC24S_V2
# STANDARD_NC24S_V3
# STANDARD_NC48ADS_A100_V4
# STANDARD_NC4AS_T4_V3
# STANDARD_NC6
# STANDARD_NC64AS_T4_V3
# STANDARD_NC6S_V2
# STANDARD_NC6S_V3
# STANDARD_NC8AS_T4_V3
# STANDARD_NC96ADS_A100_V4
# STANDARD_NV12
# STANDARD_NV12ADS_A10_V5
# STANDARD_NV12S_V3
# STANDARD_NV16AS_V4
# STANDARD_NV18ADS_A10_V5
# STANDARD_NV24
# STANDARD_NV24S_V3
# STANDARD_NV32AS_V4
# STANDARD_NV36ADMS_A10_V5
# STANDARD_NV36ADS_A10_V5
# STANDARD_NV48S_V3
# STANDARD_NV4AS_V4
# STANDARD_NV6
# STANDARD_NV6ADS_A10_V5
# STANDARD_NV72ADS_A10_V5
# STANDARD_NV8AS_V4


# Please add surface readymade system alias from https://learn.microsoft.com/en-us/surface/surface-system-sku-reference
class SurfaceSystemAlias:
    system_type = SystemType.Local
    accelerators: ClassVar[list] = ["GPU"]


class SurfaceProSystem1796(SurfaceSystemAlias):
    sku = "Surface_Pro_1796"
    # we could add the num_cpus and num_gpus if we don't want query locally and
    # could find the appropriated spec definition.


class SurfaceProSystem1807(SurfaceSystemAlias):
    sku = "Surface_Pro_1807"
