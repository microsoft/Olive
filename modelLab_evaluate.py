import argparse
from pathlib import Path
from typing import cast
#from modelLab import logger

from typing import List, Union
import time
from olive.evaluator.registry import Registry
from olive.evaluator import (
    Metric,
    MetricResult,
    OliveEvaluator,
    flatten_metric_result,
)
from olive.evaluator.metric import MetricType
from olive.hardware import Device
from olive.model.handler import OliveModelHandler

@Registry.register("MyCustomEvaluator")
class MyCustomEvaluator(OliveEvaluator):
    def evaluate(
        self,
        model: OliveModelHandler,
        metrics: List[Metric],
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        metrics_res = {}
        for original_metric in metrics:
            # use model io_config if user does not specify input_names and input_shapes
            metric = OliveEvaluator.generate_metric_user_config_with_model_io(original_metric, model)
            dataloader, eval_func, post_func = OliveEvaluator.get_user_config(model.framework, metric)
            if metric.type == MetricType.LATENCY:
                self._evaluate_latency_all(
                    model, metric, dataloader, metrics_res, post_func, device, execution_providers
                )
            else:
                raise TypeError(f"{metric.type} is not a supported metric type")
        return flatten_metric_result(metrics_res)
    
    def _evaluate_latency_all(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        metrics_res,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, list[str]] = None,
    ):
        latencies = self._evaluate_raw_latency(model, metric, dataloader, post_func, device, execution_providers)
        flatten_latencies = [item for sublist in latencies for item in sublist]
        metrics_res["latency"] = OliveEvaluator.compute_latency(metric, flatten_latencies)
        metrics_res["throughput"] = OliveEvaluator.compute_throughput(metric, flatten_latencies)
        first_latencies = [latency[0] for latency in latencies if latency]
        metrics_res["FTL"] = OliveEvaluator.compute_latency(metric, first_latencies)

    def _evaluate_raw_latency(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, list[str]] = None,
    ) -> list[list[float]]:
        # import onnxruntime_genai as og
        # model = og.Model(model_folder)
        # tokenizer = og.Tokenizer(model)
        # params = og.GeneratorParams(model)
        # search_options = {}
        # search_options["max_length"] = metric.user_config.max_length
        # params.set_search_options(**search_options)

        result = []
        i = 0
        for input_data, _ in dataloader:
            # chat_template = "<｜User｜>{input}<｜Assistant｜><think>"
            # prompt = f"{chat_template.format(input=input_data)}"
            # input_tokens = tokenizer.encode(prompt)

            # generator = og.Generator(model, params)
            # generator.append_tokens(input_tokens)

            latencies = [ 0 ]
            # while not generator.is_done():
            #     t = time.perf_counter()
            #     generator.generate_next_token()
            #     latencies.append(time.perf_counter() - t)
            result.append(latencies)
            i += 1
            if i >= metric.user_config.max_samples:
                break
        return result

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test.json", help="path to input config file")
    parser.add_argument("--model_config", default="D:\\Downloads\\test-final\\huggingface_Intel_bert-base-uncased-mrpc_v1\\history\\no quant\\model_config.json", help="path to input model config file")
    return parser.parse_args()

def main():
    args = parse_arguments()

    p = Path(args.model_config)
    if not p.exists():
        raise FileNotFoundError(f"Model config file {p} does not exist.")

    from olive.evaluator.metric_result import MetricResult
    from olive.model.config import ModelConfig
    from olive.resource_path import create_resource_path, LocalFile
    from olive.systems.accelerator_creator import create_accelerators
    from olive.systems.olive_system import OliveSystem
    from olive.workflows.run.config import RunConfig

    #logger.info("Loading model and configuration ...")

    run_config = cast(RunConfig, RunConfig.parse_file_or_obj(args.config))

    engine = run_config.engine.create_engine(
        olive_config=run_config,
        azureml_client_config=None,
        workflow_id=run_config.workflow_id,
    )
    engine.initialize()

    accelerator_specs = create_accelerators(
        engine.target_config,
        skip_supported_eps_check=True,
        is_ep_required=True,
    )

    target: OliveSystem = engine.target_config.create_system()

    model_config_file: LocalFile = cast(LocalFile, create_resource_path(p))
    model_config = cast(
        ModelConfig,
        ModelConfig.parse_file_or_obj(model_config_file.get_path()),
    )

    #logger.info("Evaluating model ...")
    result: MetricResult = target.evaluate_model(
        model_config=model_config,
        evaluator_config=engine.evaluator_config,
        accelerator=accelerator_specs[0],
    )

    output_file = Path(args.config).parent / "metrics.json"
    resultStr = str(result)
    with open(output_file, 'w') as file:
        file.write(resultStr)
    #logger.info("Model lab succeeded for evaluation.\n%s", resultStr)

if __name__ == "__main__":
    main()
