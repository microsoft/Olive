import json
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cvt_config = "conversion_config.json"
cmd = "olive convert --conversion_config {} --model_framework tensorflow".format(os.path.join(os.path.dirname(__file__), cvt_config))
subprocess.run(cmd, shell=True, check=True)
logger.info("CLI test succeeded for configuration {}".format(cvt_config))

input_names = "title_lengths:0,title_encoder:0,ratings:0,query_lengths:0,passage_lengths:0,features:0,encoder:0,decoder:0,Placeholder:0"
output_names = "output_identity:0,loss_identity:0"

cvt_args_list = ["--model_path {}".format(os.path.join("ci_tests", "tests", "conversion", "models", "full_doran_frozen.pb")),
                 "--model_framework tensorflow ",
                 "--input_names {} --output_names {}".format(input_names, output_names),
                 "--sample_input_data_path {}".format(os.path.join("ci_tests", "tests", "conversion", "doran.npz")),
                 "--onnx_opset 11"]

cmd_list = ["olive convert {}".format(" ".join(cvt_args_list[:4])),
            "olive convert {}".format(" ".join(cvt_args_list[:5]))]

for cmd in cmd_list:
    logger.info("CLI test with args {}".format(cmd))
    subprocess.run(cmd, shell=True, check=True)
    logger.info("CLI test succeeded")
