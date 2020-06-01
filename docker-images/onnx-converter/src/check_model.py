# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import onnxruntime
import onnx
import json
import numpy as np
from onnx import helper, numpy_helper
import os
import tensorflow as tf
from pathlib import Path
import coremltools
import torch

def caffeRunner(model_path, inputs_path):
    # TODO: Install pycafe
    return

def cntkRunner(model_path, inputs_path):
    import cntk as C
    model = C.load_model(model_path, device=C.device.cpu())
    input_dict = gen_io_dict(inputs_path, model.arguments, True)
    output = model.eval(input_dict)
    return output

def coremlRunner(model_path, inputs_path):
    return 

def kerasRunner(model_path, inputs_path):
    import keras
    # Load your Keras model
    keras_model = keras.models.load_model(model_path)
    input_list = gen_input_list(inputs_path)
    output = keras_model.predict(input_list)
    return output

def mxnetRunner(model_path, inputs_path):
    return

def sklearnRunner(model_path, inputs_path):
    from sklearn.externals import joblib
    # Load your sklearn model
    skl_model = joblib.load(model_path)
    input_list = gen_input_list(inputs_path)
    output = skl_model.predict(input_list)
    if type(output) is not list:
        output = [output]
    try:
        proba = skl_model.predict_proba(input_list)    
        output.append(proba)
    except:
        print("Current sklearn model .predict_proba() is not available. ")
    return output

def pytorchRunner(model_path, inputs_path):
    # load the PyTorch model
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    output_pytorch = model(gen_input_list(inputs_path, True))
    if output_pytorch is list:
        output_ndarray = [o.detach().numpy() for o in output_pytorch]
    else:
        output_ndarray = output_pytorch.detach().numpy()
    return output_ndarray

def tfRunner(model_path, inputs_path, inputs, outputs):
    input_dict = {}
    output_dict = {}
    # load tensorflow model from file in three formats
    if get_extension(model_path) == "pb":
        # tensorflow frozen model
        input_names = inputs.split(",")
        output_names = outputs.split(",")
        g = tf.Graph()
        with tf.Session(graph=g) as sess:
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            input_dict = gen_io_dict(inputs_path, input_names, True)
            for out in output_names:
                output_dict[out] = sess.run(out, input_dict)
        return output_dict
    elif get_extension(model_path) == "meta":
        # tensorflow checkpoint model
        input_names = inputs.split(",")
        output_names = outputs.split(",")
        g = tf.Graph()
        with tf.Session(graph=g) as sess:        
            saver = tf.train.import_meta_graph(model_path, clear_devices=True)
            # restore from model_path minus the ".meta"
            saver.restore(sess, model_path[:-5])
            input_dict = gen_io_dict(inputs_path, input_names, True)
            for out in output_names:
                output_dict[out] = sess.run(out, input_dict)
        return output_dict
    else:
        # saved model
        # Get input and output keys
        with tf.Session(graph=tf.Graph()) as sess:
            metagraph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
            inputs_mapping = dict(metagraph.signature_def['serving_default'].inputs)
            outputs_mapping = dict(metagraph.signature_def['serving_default'].outputs)
            input_names = [inputs_mapping[i].name for i in inputs_mapping.keys()]
            output_names = [outputs_mapping[i].name for i in outputs_mapping.keys()]
            input_dict = gen_io_dict(inputs_path, input_names, True)
            for out in output_names:
                output_dict[out] = sess.run(out, input_dict)
            sess.close()
        return output_dict

def xgboostRunner(model, inputs):
    return

def onnxRunner(model, inputs_path):
    sess = onnxruntime.InferenceSession(model)
    outputs = {}
    input_names = [sess.get_inputs()[i].name for i in range(len(sess.get_inputs()))]
    for i in sess.get_outputs():
        outputs[i.name] = sess.run([i.name], gen_io_dict(inputs_path, input_names, True))[0]
    return outputs

def readInputFromFile(full_path):
    t = onnx.TensorProto()
    with open(full_path, 'rb') as f:
        t.ParseFromString(f.read())
    return t

# Generate a {input/output_name: input/output_arr} dictionary
def gen_io_dict(input_path, names=None, isInput=True):
    io_dict = {}
    i = 0
    filePrefix = "input" if isInput else "output"
    full_path = os.path.join(input_path, filePrefix + "_%s.pb" % i)
    while os.path.isfile(full_path):
        tensorProto = readInputFromFile(full_path)        
        name = names[i] if names != None else tensorProto.name
        io_dict[name] = numpy_helper.to_array(tensorProto)
        i += 1        
        full_path = os.path.join(input_path, filePrefix + "_%s.pb" % i)
    return io_dict

# Generate input list from input_0.pb, input_1.pb ...
def gen_input_list(input_path, isPytorch=False):
    inputs = []
    i = 0    
    print(input_path)
    full_path = os.path.join(input_path, "input_%s.pb" % i)
    while os.path.isfile(full_path):
        if isPytorch:
            inputs.append(torch.tensor(numpy_helper.to_array(readInputFromFile(full_path))))
        else:
            inputs.append(numpy_helper.to_array(readInputFromFile(full_path)))
        i += 1
        full_path = os.path.join(input_path, "input_%s.pb" % i)
    if len(inputs) == 1:
        return inputs[0]
    else:
        return inputs

runner = {
    # "caffe": caffeRunner,
    "cntk": cntkRunner,
    # "coreml": coremlRunner,
    "keras": kerasRunner,
    # "libsvm": libsvmRunner,
    # "lightgbm": lightgbm2onnx,
    # "mxnet": mxnetRunner,    
    "scikit-learn": sklearnRunner,
    "pytorch": pytorchRunner,
    "tensorflow": tfRunner,
    # "xgboost": xgboostRunner
}

def get_extension(path):
    return Path(path).suffix[1:].lower()

def check_model(original_model_path, onnx_model_path, inputs_path, model_type, input_names, output_names):
    # Check if your ONNX model is valid
    model = onnx.load(onnx_model_path)
    print("\nCheck the ONNX model for validity ")
    onnx.checker.check_model(model)
    print('The ONNX model is valid.\n')
    modelPredictor = runner.get(model_type)
    if model_type == "onnx" or get_extension(original_model_path) == "onnx":
        # Skip model correctness conversion for onnx models
        print("The original model is already onnx. Skipping correctness test. ")
        return "SKIPPED"    

    print("Check ONNX model for correctness. ")
    # Check if expected output file exists
    expected_outputs = gen_io_dict(inputs_path, None, False)
    if len(expected_outputs) > 0:
        print("Using output files provided for correctness test. ")
        print("...")
    else: 
        if modelPredictor == None:
            print("No correctness verification method for %s model type" % model_type)
            return "UNSUPPORTED"
        print("Running inference on original model with specified or random inputs. ")    
        print("...")
        try:
            if model_type == "tensorflow":
                expected_outputs = tfRunner(original_model_path, inputs_path, input_names, output_names)
            else:
                expected_outputs = modelPredictor(original_model_path, inputs_path)
        except Exception as e:
            print(e)
            print("\nCannot run original model under current context. Skipping correctness verification.")
            return "UNSUPPORTED"

    print("Running inference on the converted model with the same inputs")
    print("...\n")
    onnx_outputs = onnxRunner(onnx_model_path, inputs_path)

    # Compare two outputs 
    print("Comparing the outputs from two models. ")
    expected_decimal = 5
    if type(expected_outputs) is list:
        # If no output name information can be extracted from original model, infer the names from onnx
        sess = onnxruntime.InferenceSession(onnx_model_path)
        output_names = sess.get_outputs()
        for i in range(len(expected_outputs)):
            np.testing.assert_almost_equal(expected_outputs[i], onnx_outputs.get(output_names[i].name), decimal=expected_decimal)
    elif type(expected_outputs) is not dict:    
        # If only one output, just compare them
        for onnx_output in onnx_outputs.values():
            np.testing.assert_almost_equal(expected_outputs, onnx_output, decimal=expected_decimal)
    else:
        for output_name in onnx_outputs.keys():
            try:
                expected_outputs.get(output_name)
            except Exception as e:
                print(e)
                print("Output names in output.pb does not align with the names in model. Please fix your output.pb files and run again.")
                return "UNSUPPORTED"
            np.testing.assert_almost_equal(expected_outputs.get(output_name), onnx_outputs.get(output_name), decimal=expected_decimal)
    print("The converted model achieves {}-decimal precision compared to the original model.".format(expected_decimal))
    print("MODEL CONVERSION SUCCESS. ")
    return "SUCCESS"