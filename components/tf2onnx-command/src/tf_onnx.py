# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
python -m tf2onnx.convert : tool to convert a frozen tensorflow graph to onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from onnx import helper
import tensorflow as tf
from tf2onnx import utils
from tf2onnx import loader
from tf2onnx.graph import GraphUtil
from tf2onnx.tfonnx import process_tf_graph, tf_optimize, DEFAULT_TARGET, POSSIBLE_TARGETS
from azure.storage.blob import BlockBlobService
import os

_TENSORFLOW_DOMAIN = "ai.onnx.converters.tensorflow"
STORE_PATH = "./input"

# pylint: disable=unused-argument


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--azure_storage_account_name", 
        required=True,
        help="Azure storage account that uses to store input/output files.")
    # parser.add_argument(
    #     "--azure_storage_account_key", 
    #     required=True,
    #     help="Azure storage account key.")
    parser.add_argument(
        "--azure_storage_container", 
        required=True,
        help="Azure storage container to hold your output.")
    parser.add_argument("--input", 
        help="Tensorflow model input in graphdef format. If this option is elected, \"--input\" and \"--output\" should be provided as well. ")
    parser.add_argument("--graphdef", 
        help="Tensorflow model input in graphdef format. If this option is elected, \"--input\" and \"--output\" should be provided as well. ")
    parser.add_argument("--saved-model", 
        help="Directory of Tensorflow model outputed from tf.save_model.")
    parser.add_argument("--checkpoint", 
        help="Directory of .meta format Tensorflow checkpoint model. ")
    parser.add_argument("--output", 
        help="Output path for the converted model file")
    parser.add_argument("--inputs", 
        help="Model input_names")
    parser.add_argument("--outputs", 
        help="Model output_names")
    parser.add_argument("--opset", 
        type=int, default=None, 
        help="ONNX opset to use. By default use the newest opset 7.")
    parser.add_argument("--custom-ops", 
        help="list of custom ops")
    parser.add_argument("--target", 
        default=",".join(DEFAULT_TARGET), 
        choices=POSSIBLE_TARGETS, 
        help="target platform")
    parser.add_argument("--continue_on_error", 
        help="continue_on_error", 
        action="store_true")
    parser.add_argument("--verbose", 
        help="verbose output", 
        action="store_true")
    parser.add_argument("--fold_const", 
        help="enable tf constant_folding transformation before conversion",
        action="store_true")
    # experimental
    parser.add_argument("--inputs-as-nchw", help="transpose inputs as from nhwc to nchw")
    # depreciated, going to be removed some time in the future
    parser.add_argument("--unknown-dim", type=int, default=-1, help="default for unknown dimensions")
    args = parser.parse_args()

    args.shape_override = None
    if args.input:
        # for backward compativility
        args.graphdef = args.input
    if args.graphdef or args.checkpoint:
        if not args.input and not args.outputs:
            raise ValueError("graphdef and checkpoint models need to provide inputs and outputs")
    if not any([args.graphdef, args.checkpoint, args.saved_model]):
        raise ValueError("need input as graphdef, checkpoint or saved_model")
    if args.inputs:
        args.inputs, args.shape_override = utils.split_nodename_and_shape(args.inputs)
    if args.outputs:
        args.outputs = args.outputs.split(",")
    if args.inputs_as_nchw:
        args.inputs_as_nchw = args.inputs_as_nchw.split(",")
    if args.target:
        args.target = args.target.split(",")

    return args


def default_custom_op_handler(ctx, node, name, args):
    node.domain = _TENSORFLOW_DOMAIN
    return node

def download_from_path(block_blob_service, azure_storage_container, download_path):

    generator = block_blob_service.list_blobs(azure_storage_container)
    
    for blob in generator.items:
        if (blob.name.find(os.path.normpath(download_path)) == 0):
            download_file(block_blob_service, azure_storage_container, blob.name)

def download_file(block_blob_service, azure_storage_container, file_name):
    print("downloading file " + file_name)
    
    if "/" in "{}".format(file_name):
        print("There is a path in the download file. ")
        head, tail = os.path.split("{}".format(file_name))
        if (os.path.isdir(os.path.join(STORE_PATH, head)) == False):
            print("Directory doesn't exist, create it. ")
            os.makedirs(os.path.join(STORE_PATH, head), exist_ok = True)
        block_blob_service.get_blob_to_path(azure_storage_container, file_name, STORE_PATH + "/" + head + "/" + tail)
    else:
        block_blob_service.get_blob_to_path(azure_storage_container, file_name, STORE_PATH + "/" + file_name)

def main():
    args = get_args()
    # Create the BlockBlockService that is used to call the Blob service for the storage account
    block_blob_service = BlockBlobService(account_name=args.azure_storage_account_name)
    # Generate the store path
    if (os.path.isdir(os.getcwd()+ "/" + STORE_PATH) == False):
         #create the diretcory and download the file to it
        print("store path doesn't exist, creating it now")
        os.makedirs(os.getcwd()+ "/" + STORE_PATH, exist_ok=True)
        print("store path created, download initiated")

    # override unknown dimensions from -1 to 1 (aka batchsize 1) since not every runtime does
    # support unknown dimensions.
    utils.ONNX_UNKNOWN_DIMENSION = args.unknown_dim

    if args.custom_ops:
        # default custom ops for tensorflow-onnx are in the "tf" namespace
        custom_ops = {op: (default_custom_op_handler, []) for op in args.custom_ops.split(",")}
        extra_opset = [helper.make_opsetid(_TENSORFLOW_DOMAIN, 1)]
    else:
        custom_ops = {}
        extra_opset = None

    # get the frozen tensorflow model from graphdef, checkpoint or saved_model.
    if args.graphdef:
        download_file(block_blob_service, args.azure_storage_container, args.graphdef)
        graph_def, inputs, outputs = loader.from_graphdef(os.path.join(STORE_PATH, args.graphdef), args.inputs, args.outputs)
        model_path = args.graphdef
    if args.checkpoint:
        download_file(block_blob_service, args.azure_storage_container, args.checkpoint)
        graph_def, inputs, outputs = loader.from_checkpoint(os.path.join(STORE_PATH, args.checkpoint), args.inputs, args.outputs)
        model_path = args.checkpoint
    if args.saved_model:
        download_from_path(block_blob_service, args.azure_storage_container, args.saved_model)
        graph_def, inputs, outputs = loader.from_saved_model(os.path.join(STORE_PATH, args.saved_model), args.inputs, args.outputs)
        model_path = args.saved_model

    # todo: consider to enable const folding by default?
    graph_def = tf_optimize(inputs, outputs, graph_def, args.fold_const)

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')
    with tf.Session(graph=tf_graph):
        g = process_tf_graph(tf_graph,
                             continue_on_error=args.continue_on_error,
                             verbose=args.verbose,
                             target=args.target,
                             opset=args.opset,
                             custom_op_handlers=custom_ops,
                             extra_opset=extra_opset,
                             shape_override=args.shape_override,
                             input_names=inputs,
                             output_names=outputs,
                             inputs_as_nchw=args.inputs_as_nchw)

    model_proto = g.make_model("converted from {}".format(model_path))

    new_model_proto = GraphUtil.optimize_model_proto(model_proto)
    if new_model_proto:
        model_proto = new_model_proto
    else:
        print("NON-CRITICAL, optimizers are not applied successfully")

    # write onnx graph
    if args.output:
        utils.save_protobuf(args.output, model_proto)
        print("\nComplete successfully, the onnx model is generated at " + args.output)
    
    # upload model to azure blob
    block_blob_service.create_blob_from_path(args.azure_storage_container, args.output, args.output)
    with open('/output.txt', 'w') as f:
        f.write(args.output)


if __name__ == "__main__":
    main()