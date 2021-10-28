import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config_files = ["optimization_config_1.json", "optimization_config_2.json"]
for opt_config in config_files:
    cmd = "olive optimize --optimization_config {}".format(os.path.join(os.path.dirname(__file__),
                                                                                  "opt_configs", opt_config))
    subprocess.run(cmd, shell=True, check=True)
    logger.info("CLI test succeeded for configuration {}".format(opt_config))

opt_args_list = ["--model_path {}".format(os.path.join("ci_tests", "tests", "optimization", "onnx_mnist", "model.onnx")),
                 "--providers_list {}".format("cpu"),
                 "--concurrency_num {}".format("2"),
                 "--kmp_affinity {}".format("respect,none"),
                 "--omp_max_active_levels {}".format("1,2")]

cmd_list = ["olive optimize {}".format(" ".join(opt_args_list[:1])),
            "olive optimize {}".format(" ".join(opt_args_list[:2])),
            "olive optimize {}".format(" ".join(opt_args_list[:3])),
            "olive optimize {}".format(" ".join(opt_args_list[:4])),
            "olive optimize {}".format(" ".join(opt_args_list[:5]))]

for cmd in cmd_list:
    logger.info("CLI test with args {}".format(cmd))
    subprocess.run(cmd, shell=True, check=True)
    logger.info("CLI test succeeded")
