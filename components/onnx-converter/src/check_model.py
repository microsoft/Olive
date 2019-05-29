import onnxruntime
import onnx
import json
import numpy as np
from onnx import helper, numpy_helper
import os
import tensorflow as tf
from pathlib import Path
import cntk as C
import coremltools
import torch

def caffeRunner(model_path, inputs_path):
    # TODO: Install pycafe
    return

def cntkRunner(model_path, inputs_path):
    model = C.load_model(model_path)
    print("model ", model)
    print("inputs? ", model.arguments)
    input_dict = gen_input_dict(model.arguments, inputs_path)
    output = model.eval(input_dict)
    # print("outputs? ", output)
    return output

def coremlRunner(model_path, inputs_path):
    # # Load a Core ML model
    # coreml_model = coremltools.models.MLModel(model_path)
    # for i in coreml_model.input_description:
    #     print(i)
    # print("input names = ", coreml_model.input_description)
    # print("output names = ", coreml_model.output_description)
    # input_dict = gen_input_dict(coreml_model.input_description, inputs_path)
    # output_dict = coreml_model.predict(input_dict)
    # print("output = ", output_dict)
    return 

def kerasRunner(model_path, inputs_path):
    return

def mxnetRunner(model_path, inputs_path):
    return

def sklearnRunner(model_path, inputs_path):
    return

def pytorchRunner(model_path, inputs_path):
    # load the PyTorch model
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    output_pytorch = model(gen_input_list(inputs_path))
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
            input_dict = gen_input_dict(input_names, inputs_path)
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
            input_dict = gen_input_dict(input_names, inputs_path)
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
            input_dict = gen_input_dict(input_names, inputs_path)
            for out in output_names:
                output_dict[out] = sess.run(out, input_dict)
            sess.close()
        # # Construct input command string
        # input_dict = gen_input_tf_str(inputs_mapping.keys(), inputs_path)

        # # Run inference on saved model with specified inputs
        # subprocess.check_call(["saved_model_cli", "run", "--dir", model_path, "--tag_set", [tf.saved_model.tag_constants.SERVING], 
        #     "--signature_def", "serving_default", "--inputs", input_dict, "--overwrite", "--outdir", output_dir], stdout=open(os.devnull, 'w'))

        # # Read generated outputs     
        # outputs = {}
        # for key, output in outputs_mapping.items():
        #     o = np.load(os.path.join(output_dir, key + ".npy"))
        #     outputs[output.name] = o
        print(input_names)
        print(output_names)
        return output_dict

def xgboostRunner(model, inputs):
    return

def onnxRunner(model, inputs_path):
    sess = onnxruntime.InferenceSession(model)
    outputs = {}
    input_names = [sess.get_inputs()[i].name for i in range(len(sess.get_inputs()))]
    print("onnx inputs ", input_names)
    for i in sess.get_outputs():
        outputs[i.name] = sess.run([i.name], gen_input_dict(input_names, inputs_path))[0]
    # print("onnx outputs: ", outputs)
    return outputs

def readInputFromFile(full_path):
    t = onnx.TensorProto()
    with open(full_path, 'rb') as f:
        t.ParseFromString(f.read())
    data = numpy_helper.to_array(t)
    return data

# Generate a {input_name: input_arr} dictionary
def gen_input_dict(input_names, input_path):
    input_dict = {}
    i = 0
    for name in input_names:
        full_path = os.path.join(input_path, "input_%s.pb" % i)
        input_dict[name] = readInputFromFile(full_path)
        i += 1
    return input_dict

def gen_input_list(input_path):
    inputs = []
    i = 0    
    print(input_path)
    full_path = os.path.join(input_path, "input_%s.pb" % i)
    while os.path.isfile(full_path):
        inputs.append(torch.tensor(readInputFromFile(full_path)))
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
    # "keras": kerasRunner,
    # "libsvm": libsvmRunner,
    # "lightgbm": lightgbm2onnx,
    # "mxnet": mxnetRunner,    
    # "scikit-learn": sklearnRunner,
    "pytorch": pytorchRunner,
    "tensorflow": tfRunner,
    # "xgboost": xgboostRunner
}

def get_extension(path):
    return Path(path).suffix[1:].lower()

def check_model(original_model_path, onnx_model_path, inputs, model_type, input_names, output_names):
    # Check if your ONNX model is valid
    model = onnx.load(onnx_model_path)
    print("\nCheck the ONNX model for validity ")
    onnx.checker.check_model(model)
    print('The ONNX model is valid.\n')

    print("Check ONNX model for correctness. ")
    print("Running inference on original model with specified or random inputs. ")    
    print("...\n")
    if (model_type == "tensorflow"):
        expected_outputs = tfRunner(original_model_path, inputs, input_names, output_names)
    else:
        modelPredictor = runner.get(model_type)
        if (modelPredictor == None):
            return
        expected_outputs = modelPredictor(original_model_path, inputs)

    print("Running inference on the converted model with the same inputs")
    print("...\n")
    onnx_outputs = onnxRunner(onnx_model_path, inputs)
    
    # print("onnx_outputs: %s\n" % onnx_outputs)
    # print("expected_outputs: %s\n" % expected_outputs)

    # Compare two outputs 
    print("Comparing the outputs from two models. ")
    expected_decimal = 5
    # If only one output, just compare them
    if type(expected_outputs) is not dict:
        print("compare not dict outputs")
        for onnx_output in onnx_outputs.values():
            np.testing.assert_almost_equal(expected_outputs, onnx_output, decimal=expected_decimal)
    else:
        for output_name in onnx_outputs.keys():
            np.testing.assert_almost_equal(expected_outputs.get(output_name), onnx_outputs.get(output_name), decimal=expected_decimal)
    print("The converted model achieves {}-decimal precision compared to the original model.".format(expected_decimal))
    print("MODEL CONVERSION SUCCESS. ")
    