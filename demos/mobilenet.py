from PIL import Image
import numpy as np
import onnxruntime
import torch

image_width = 224#300
image_height = 224#168

def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

# Read the categories
with open("./imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

if True:
    options = onnxruntime.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
    #options.log_severity_level = 0

    session = onnxruntime.InferenceSession("./mobilenet-qnn.onnx",
                                        sess_options=options,
                                        providers=["QNNExecutionProvider"],
                                        provider_options=[{"backend_path": "QnnHtp.dll"}])
else:
    session = onnxruntime.InferenceSession("../examples/mobilenet/models/mobilenetv2-12.onnx")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_sample(session, image_file, categories):
    output = session.run([], {'input':preprocess_image(image_file, image_height, image_width)})[0]
    output = output.flatten()
    output = softmax(output) # this is optional
    top5_catid = np.argsort(-output)[:5]
    for catid in top5_catid:
        print(categories[catid], output[catid])

run_sample(session, 'cat.jpg', categories)