#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import logging

import numpy as np

from olive.passes.onnx.vitis_ai.quant_utils import pos2scale, scale2pos

refine_op_type = ["DequantizeLinear", "QuantizeLinear"]
postfix = "_Output"
logger = logging.getLogger(__name__)

# pylint: skip-file
# ruff: noqa


class QuantPosManager(object):
    def __init__(self, model):
        self.model = model

    def get_scale(self, node):
        for i in self.model.model.graph.initializer:
            if i.name == node.input[1]:
                return i.float_data[0]
        raise ValueError("DequantizeLinear and QuantizeLinear do not have scale.")

    def get_pos(self, node):
        if node.op_type in refine_op_type:
            return scale2pos(self.get_scale(node))
        return None

    def set_scale(self, node, new_scale):
        for i in self.model.model.graph.initializer:
            if i.name == node.input[1]:
                if i.float_data[0] != new_scale:
                    i.float_data[0] = new_scale

    def set_pos(self, node, new_pos):
        if node.op_type == "QuantizeLinear":
            new_scale = pos2scale(new_pos)
            self.set_scale(node, new_scale)

            if node.output:
                for n in self.model.model.graph.node:
                    if n.name == node.output[0].strip(postfix) and n.op_type == "DequantizeLinear":
                        self.set_scale(node, new_scale)

        elif node.op_type == "DequantizeLinear":
            new_scale = pos2scale(new_pos)
            self.set_scale(node, new_scale)
            for n in self.model.model.graph.node:
                if n.name == node.input[0].strip(postfix) and n.op_type == "QuantizeLinear":
                    self.set_scale(node, new_scale)

    def find_node_name(self, name):
        for node in self.model.model.graph.node:
            if node.output[0] == name:
                return node.name

        return None

    def get_ipos_name(self, node, input_id=None):
        if len(node.input) > 0:
            i_name = node.input[0]
            return self.find_node_name(i_name)
        else:
            return None

    def get_ipos_name_by_id(self, node, input_id=0):
        if len(node.input) > input_id:
            i_name = node.input[input_id]
            return self.find_node_name(i_name)
        else:
            return None

    def get_node_by_name(self, node_name):
        for node in self.model.model.graph.node:
            if node.name == node_name:
                return node
        return None

    def get_pos_by_name(self, name):
        for node in self.model.model.graph.node:
            if node.op_type in refine_op_type and node.name == name:
                return self.get_pos(node), node

        return None, None

    def get_opos_by_name(self, name):
        for node in self.model.model.graph.node:
            if node.name == name:
                if node.op_type in refine_op_type:
                    return self.get_pos(node), node

        return None, None

    def find_o_name(self, o_name):
        opos_name = o_name + "_QuantizeLinear"
        for node in self.model.model.graph.node:
            if node.name == opos_name and node.op_type in "QuantizeLinear":
                return opos_name
        return None

    def get_opos_name(self, node, input_id=None):
        o_name = node.output[0]
        opos_name = self.find_o_name(o_name)
        if opos_name:
            return opos_name
        else:
            for node in self.model.model.graph.node:
                if len(node.input) == 0:
                    continue
                elif node.input[0] == o_name:
                    if node.op_type in refine_op_type:
                        return node.name
                    else:
                        o_name = node.output[0]
                        opos_name = self.find_o_name(o_name)
                        if opos_name:
                            return opos_name
        return None

    def get_wpos_name(self, node):
        if len(node.input) > 1:
            w_name = node.input[1]
            return self.find_node_name(w_name)
        else:
            return None

    def get_bpos_name(self, node):
        if len(node.input) > 2:
            b_name = node.input[2]
            return self.find_node_name(b_name)
        else:
            return None

    def adjust_shift_cut(self):
        """Adjust the shift cut of nodes.

        shift_cut = wpos + ipos - opos

        DPU compiler constraints of shift_cut:
        1. 0 <= shift_cut <= 16
        """
        for i, node in enumerate(self.model.model.graph.node):
            if node.op_type not in ["Conv"]:
                continue
            ipos_name = self.get_ipos_name(node)
            ipos, _ = self.get_pos_by_name(ipos_name)

            opos_name = self.get_opos_name(node)
            opos, _ = self.get_pos_by_name(opos_name)
            wpos_name = self.get_wpos_name(node)
            wpos, wpos_node = self.get_pos_by_name(wpos_name)

            # Adjust shift_cut
            min_sc = 0
            max_sc = 16
            if wpos is None or ipos is None or opos is None:
                logger.warning(
                    "Found a pos that is None. Shift cut of layer {} has not taken effect.".format(node.name)
                )
                continue
            sc = wpos + ipos - opos
            new_sc = None
            if sc < min_sc:
                new_sc = min_sc
            elif sc > max_sc:
                new_sc = max_sc

            if new_sc is not None:
                new_wpos = new_sc + opos - ipos
                self.set_pos(wpos_node, new_wpos)
                logger.info(
                    "Shift cut of layer {} is {}. It exceeds range [{}, {}]. "
                    "Modify wpos from {} to {}.".format(
                        node.input[1], int(sc), int(min_sc), int(max_sc), int(wpos), int(new_wpos)
                    )
                )

    def adjust_shift_bias(self):
        """Adjust the shift bias of node.

        shift_bias = wpos + ipos - bpos

        DPU compiler constraints of shift_bias:
        1. min(0, -(24 - (8 + shift_cut))) <= shfit_bias <= 16
        """
        for i, node in enumerate(self.model.model.graph.node):
            if node.op_type not in ["Conv"]:
                continue
            ipos_name = self.get_ipos_name(node)
            ipos, _ = self.get_pos_by_name(ipos_name)

            opos_name = self.get_opos_name(node)
            opos, _ = self.get_pos_by_name(opos_name)
            wpos_name = self.get_wpos_name(node)
            wpos, wpos_node = self.get_pos_by_name(wpos_name)
            bpos_name = self.get_bpos_name(node)
            if bpos_name:
                bpos, bpos_node = self.get_pos_by_name(bpos_name)
                # Adjust shift_bias
                if wpos is None or ipos is None or opos is None or bpos is None:
                    logger.warning(
                        "Found a pos that is None. Shift bias of layer {} has not taken effect.".format(node.name)
                    )
                    continue
                shift_cut = wpos + ipos - opos
                min_sb = min(0, -(24 - (8 + shift_cut)))
                max_sb = 16
                shift_bias = wpos + ipos - bpos

                new_sb = None
                if shift_bias < min_sb:
                    new_sb = min_sb
                elif shift_bias > max_sb:
                    new_sb = max_sb

                if new_sb is not None:
                    new_bpos = wpos + ipos - new_sb
                    self.set_pos(self.get_node_by_name(node.input[2].strip(postfix)), new_bpos)
                    logger.info(
                        "Shift bias of layer {} is {}. It exceeds range [{}, {}]. "
                        "Modify bpos from {} to {}.".format(
                            node.input[2], int(shift_bias), int(min_sb), int(max_sb), int(bpos), int(new_bpos)
                        )
                    )

    def adjust_vitis_sigmoid(self):
        """Adjust quantize info of VitisSigmoid nodes.

        DPU compiler constraints for VitisSigmoid:
        1. input pos of VitisSigmoid >= 0 && <= 15
        2. output pos of VitisSigmoid >= 7
        3. shift_sigmoid >= 0 && shift_sigmoid <= 31 where
            shift_sigmoid = 14 + 'input pos' - ' output pos'
        """
        for i, node in enumerate(self.model.model.graph.node):
            if node.op_type not in ["Sigmoid"]:
                continue
            ipos_name = self.get_ipos_name(node)
            ipos, _ = self.get_pos_by_name(ipos_name)

            opos_name = self.get_opos_name(node)
            opos, _ = self.get_pos_by_name(opos_name)

            if ipos is None or opos is None:
                logger.warning(
                    "Found a pos that is None. Adjust quantize info of VitisSigmoid "
                    "nodes of layer {} has not taken effect.".format(node.name)
                )
                continue

            new_ipos = ipos if ipos > 0 else 0
            new_ipos = new_ipos if new_ipos <= 15 else 15

            new_opos = opos if opos > 7 else 7
            shift_sigmoid = 14 + new_ipos - new_opos  # will not bigger than 31 now
            new_opos = new_opos if shift_sigmoid > 0 else 14 + new_ipos

            if new_ipos != ipos:
                self.set_pos(self.get_node_by_name(node.input[0].strip(postfix)), new_ipos)
            logger.info(
                "Input quantize pos of VitisSimoid layer {} is {}, modify it to {} "
                "to meet the DPU constraints.".format(node.input[0], int(ipos), int(new_ipos))
            )

            if new_opos != opos:
                self.set_pos(self.get_node_by_name(self.find_o_name(node.output[0])), new_opos)
            logger.info(
                "Output quantize pos of VitisSimoid layer {} is {}, modify it to {} "
                "to meet the DPU constraints.".format(node.output[0], int(opos), int(new_opos))
            )

    def adjust_shift_read(self):
        """Adjust the shift bias of node.

        shift_read = max(ipos) - min(ipos)

        DPU compiler constraints of shift_bias:
        1. 0 <= shift_read <= 15
        """
        for index, node in enumerate(self.model.model.graph.node):
            if node.op_type not in ["Add"] or node.op_type not in ["Mul"]:
                continue
            ipos_layers = []
            iposes = []
            skip = False

            for i in range(len(node.input)):
                ipos_name = self.get_ipos_name_by_id(node, i)
                ipos_layers.append(ipos_name)
            for i in ipos_layers:
                ipos, _ = self.get_pos_by_name(i)
                if ipos is None:
                    logger.info(
                        "Fail to get quantize position for layer {}, "
                        "skip adjust_shift_read for it.".format(ipos_layers[i])
                    )
                    skip = True
                iposes.append(ipos)

            if skip:
                continue

            id_max = np.argmax(iposes)
            id_min = np.argmin(iposes)
            sr = iposes[id_max] - iposes[id_min]
            min_sr, max_sr = 0, 15

            new_sr = None
            if sr > max_sr:
                new_sr = max_sr

            if new_sr is not None:
                new_ipos_max = iposes[id_min] + new_sr
                self.set_pos(self.get_node_by_name(ipos_layers[id_max].strip(postfix)), new_ipos_max)
                logger.info(
                    "Shift read of layer {} is {}({}-{}). It exceeds range [{}, {}]. "
                    "Modify ipos from {} to {}.".format(
                        node.name,
                        int(sr),
                        int(iposes[id_max]),
                        int(iposes[id_min]),
                        int(min_sr),
                        int(max_sr),
                        int(iposes[id_max]),
                        int(new_ipos_max),
                    )
                )

    def adjust_shift_write(self):
        """Adjust the shift write of node.

        shift_write = min(ipos) - opos

        DPU compiler constraints of shift_write:
        1. -15 <= shift_write <= 15
        """
        for node in self.model.model.graph.node:
            if node.op_type not in ["Add"] or node.op_type not in ["Mul"]:
                continue
            ipos_layers = []
            iposes = []
            skip = False

            for input_id in range(len(node.input)):
                ipos_name = self.get_ipos_name_by_id(node, input_id)
                ipos_layers.append(ipos_name)
            for layer_id in ipos_layers:
                ipos, _ = self.get_pos_by_name(layer_id)
                if ipos is None:
                    logger.info(
                        "Fail to get quantize position for layer {}(input:{}) (output of layer {}), "
                        "skip adjust_shift_read for it.".format(ipos_layers[layer_id], layer_id, ipos_layers[layer_id])
                    )
                    skip = True
                iposes.append(ipos)
            opos_name = self.get_opos_name(node)
            opos, _ = self.get_pos_by_name(opos_name)
            if opos is None:
                logger.info(
                    "Fail to get quantize position for layer {}(output:0), "
                    "skip adjust_shift_write for it.".format(node.name)
                )
            if skip:
                continue

            id_min = np.argmin(iposes)
            sw = iposes[id_min] - opos
            min_sw, max_sw = -15, 15

            new_sw = None
            if sw > max_sw:
                new_sw = max_sw
            elif sw < min_sw:
                new_sw = min_sw

            if new_sw is not None:
                new_opos = iposes[id_min] - new_sw
                self.set_pos(self.get_node_by_name(self.find_o_name(node.output[0])), new_opos)
                logger.info(
                    "Shift write of layer {} is {}({}-{}). It exceeds range [{}, {}]. "
                    "Modify opos from {} to {}.".format(
                        node.name,
                        int(sw),
                        int(iposes[id_min]),
                        int(opos),
                        int(min_sw),
                        int(max_sw),
                        int(opos),
                        int(new_opos),
                    )
                )

    def align_concat(self):
        """Align concat op's inputs and output pos."""
        for node in self.model.model.graph.node:
            if node.op_type not in ["Concat"]:
                continue
            input_node_num = len(node.input)
            opos_name = self.get_opos_name(node)
            opos, _ = self.get_pos_by_name(opos_name)
            min_pos = opos
            ipos_layers = []

            for input_id in range(input_node_num):
                ipos_name = self.get_ipos_name_by_id(node, input_id)
                ipos_layers.append(ipos_name)
            for name in ipos_layers:
                ipos, _ = self.get_pos_by_name(name)
                if ipos is not None:
                    min_pos = min(ipos, min_pos)
            if opos != min_pos:
                self.set_pos(self.get_node_by_name(self.find_o_name(node.output[0])), min_pos)
                logger.info(
                    (
                        "Output pos of concat node {} is {}, min_pos is {}. "
                        "Modify opos from {} to {}.".format(node.name, int(opos), int(min_pos), int(opos), int(min_pos))
                    )
                )
            for name in ipos_layers:
                ipos, ipos_node = self.get_pos_by_name(name)
                if ipos is not None and ipos != min_pos:
                    self.set_pos(ipos_node, min_pos)
                    logger.info(
                        "Input pos of concat node {} is {}, min_pos is {}. "
                        "Modify ipos from {} to {}.".format(node.name, int(ipos), int(min_pos), int(ipos), int(min_pos))
                    )

    def align_pool(self):
        """Align max/avg pooling input and output pos."""
        for i, node in enumerate(self.model.model.graph.node):
            if (
                node.op_type not in ["MaxPool"]
                or node.op_type not in ["AveragePool"]
                or node.op_type not in ["GlobalAveragePool"]
            ):
                continue
            ipos_name = self.get_ipos_name(node)
            ipos, ipos_layer = self.get_pos_by_name(ipos_name)

            opos_name = self.get_opos_name(node)
            opos, opos_layer = self.get_pos_by_name(opos_name)
            if ipos is not None and opos is not None and opos > ipos:
                self.set_pos(opos_layer, ipos)
                logger.info(
                    "Input pos of pooling layer {} is {}. Output pos of pooling layer {} is {}."
                    "Modify opos from {} to {}.".format(
                        node.name, int(ipos), node.name, int(opos), int(opos), int(ipos)
                    )
                )
            elif ipos is not None and opos is not None and opos < ipos:
                self.set_pos(ipos_layer, opos)
                logger.info(
                    "Input pos of pooling layer {} is {}. Output pos of pooling layer {} is {}."
                    "Modify ipos from {} to {}.".format(
                        node.name, int(ipos), node.name, int(opos), int(ipos), int(opos)
                    )
                )

    def check_scale(self):
        """checking whether a number of scale is a power of 2"""
        for node in self.model.model.graph.node:
            if node.op_type in refine_op_type:
                scale = self.get_scale(node)
                new_scale = pos2scale(scale2pos(scale))
                self.set_scale(node, new_scale)


def adjust_quantize_info(
    model,
    adjust_vitis_sigmoid=True,
    adjust_shift_cut=True,
    adjust_shift_bias=True,
    adjust_shift_read=True,
    adjust_shift_write=True,
    align_concat=True,
    align_pool=True,
):
    """Adjust the quantize info to meet the compiler constraints."""

    manager = QuantPosManager(model)

    if adjust_vitis_sigmoid:
        manager.adjust_vitis_sigmoid()

    if adjust_shift_read:
        manager.adjust_shift_read()

    if adjust_shift_write:
        manager.adjust_shift_write()

    if adjust_shift_cut:
        manager.adjust_shift_cut()

    if adjust_shift_bias:
        manager.adjust_shift_bias()

    if align_concat:
        manager.align_concat()

    if align_pool:
        manager.align_pool()

    manager.check_scale()
    return manager.model
