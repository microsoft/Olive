def get_package_name_from_ep(execution_provider):
    PROVIDER_PACKAGE_MAPPING = {
        "CPUExecutionProvider": ("onnxruntime", "ort-nightly"),
        "CUDAExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
        "TensorrtExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
        "RocmExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
        "OpenVINOExecutionProvider": ("onnxruntime-openvino", None), 
        "DmlExecutionProvider": ("onnxruntime-directml", "ort-nightly-directml"),
    }
    return PROVIDER_PACKAGE_MAPPING.get(execution_provider, ("onnxruntime", "ort-nightly"))