
import onnxruntime
import onnx
import numpy as np
import argparse
import logging
from azure.storage.blob import BlockBlobService
import os

STORE_PATH = "./input"
def inference_with_onnxruntime(input_onnx_model):
	# Check if your ONNX model is valid

	model = onnx.load(input_onnx_model)
	onnx.checker.check_model(model)

	print('The ONNX model is checked!')

	# Start an ONNX Runtime inference session
	sess = onnxruntime.InferenceSession(input_onnx_model)
	input_name = sess.get_inputs()[0].name
	print("Input name  :", input_name)
	input_shape = process_input_shape(sess.get_inputs()[0].shape)
	print("Input shape :", input_shape)
	input_type = sess.get_inputs()[0].type
	print("Input type  :", input_type)
	output_name = sess.get_outputs()[0].name
	print("Output name  :", output_name)  
	output_shape = sess.get_outputs()[0].shape
	print("Output shape :", output_shape)
	output_type = sess.get_outputs()[0].type
	print("Output type  :", output_type)

	# replace the values of X with test input for your model
	X = np.random.random(input_shape).astype(np.float32)
	pred_onnx = sess.run(None, {input_name: X})
	# with open('/output.txt', 'w') as f:
	# 	f.write(pred_onnx)
	print(pred_onnx)

def process_input_shape(input_shape):
	for i in range(len(input_shape)):
		if input_shape[i] is None:
			input_shape[i] = 1
	return input_shape

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
	
def parse_arguments():
	"""Parse command line arguments."""

	parser = argparse.ArgumentParser()
	parser.add_argument(
        "--azure_storage_account_name", 
        required=True,
        help="Azure storage account that uses to store input/output files.")
	parser.add_argument(
        "--azure_storage_account_key", 
        required=True,
        help="Azure storage account key.")
	parser.add_argument(
        "--azure_storage_container", 
        required=True,
        help="Azure storage container to hold your output.")
	parser.add_argument(
	  '--input_onnx_model',
	  type=str,
	  required=True,
	  default='model.onnx',
	  help='The converted ONNX model name.')

	args = parser.parse_args()
	return args
  
def main():
	args = parse_arguments()
	# Create the BlockBlockService that is used to call the Blob service for the storage account
	block_blob_service = BlockBlobService(account_name=args.azure_storage_account_name, account_key=args.azure_storage_account_key)
    # Generate the store path
	if (os.path.isdir(os.getcwd()+ "/" + STORE_PATH) == False):
		#create the diretcory and download the file to it
		print("store path doesn't exist, creating it now")
		os.makedirs(os.getcwd()+ "/" + STORE_PATH, exist_ok=True)
		print("store path created, download initiated")

	download_file(block_blob_service, args.azure_storage_container, args.input_onnx_model)
	logging.getLogger().setLevel(logging.INFO)
	inference_with_onnxruntime(os.path.join(STORE_PATH, args.input_onnx_model))


if __name__ == '__main__':
	main()