# Performance Monitoring in Azure DevOps with Olive

Contains Python scripts and Azure DevOps YAML files for performance monitoring of Olive models. The scripts and YAML files help you compare the performance of different models and ensure no regression occurs over time.

## Contents

### Azure DevOps YAML Files

The YAML files define Azure DevOps pipelines for automated testing and performance monitoring.

### -olive-perf-monitoring-template.yaml

This YAML file defines a pipeline template for performance monitoring of the models. It uses Python 3.8, installs Olive, runs the performance monitoring script, detects and registers components, publishes test results, and cleans up.

### -perfmonitoring-ci.yaml

This YAML file defines a CI pipeline triggered on changes to the main branch, excluding changes only to documentation and README files. It runs the performance monitoring template for several models on both Windows and Linux environments.

### Python Scripts
The Python scripts utils.py and test_perf_monitoring.py perform the main tasks of model performance comparison. They load model configurations, run the models, and compare the performance metrics.

The utils.py script contains several utility functions, including functions to:

-Patch the model configuration JSON file

-Extract the best metrics from the model's performance footprint

-Compare the metrics against previous best metrics

-Assert whether performance has not regressed

-The test_perf_monitoring.py script uses pytest to set up a testing environment, run the model, and check the results.

### -Bash and Batch Scripts
Two additional scripts, perf_monitoring.sh for Unix-like systems and perf_monitoring.bat for Windows, are used for the performance monitoring tasks. They perform the following tasks:

-Set environment variables based on pipeline parameters

-Activate the Python virtual environment if in a pipeline

-Install pytest

-Install the necessary Python packages for performance monitoring

-Run the performance monitoring Python script with pytest and capture the results


## Models

The performance monitoring setup utilizes the following models, each configured via their respective JSON files:

1. **BERT** (`bert_cpu_config.json`): BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art transformer model for a wide range of NLP tasks.

2. **RoBERTa** (`roberta_cpu_config.json`): RoBERTa is a variant of BERT that uses a different training approach for improved performance.

3. **DeBERTa** (`microsoft-deberta_cpu_config.json`): DeBERTa (Decoding-enhanced BERT with disentangled attention) improves the BERT and RoBERTa models through a two-step disentangled attention mechanism.

4. **CamemBERT** (`camembert_cpu_config.json`): A BERT-based model specifically trained for French language tasks.

5. **DistilBERT** (`distilbert_cpu_config.json`): DistilBERT is a smaller, faster, cheaper, lighter version of BERT, trained using knowledge distillation.

6. **BERTweet** (`bertweet_cpu_config.json`): BERTweet is a BERT-based model specifically fine-tuned for English Twitter sentiment analysis tasks.

Each JSON file contains configurations for the input model, evaluators, passes, and the engine.

The **input_model** section specifies the type of model (PyTorchModel in these cases), and the model's configuration, including the Hugging Face (hf) configuration for the model name, task, and dataset.

The **evaluators** section defines the metrics to be used for evaluation, such as accuracy and latency.

The **passes** section includes the type and configuration of optimization and conversion processes to be performed on the model.

The **engine** section specifies the search strategy for performance tuning, the evaluator to use, the execution providers, and the directories for caching and output.


## USAGE (Locally and on CI)

**On CI Pipeline**

Set up your Python environment with Python 3.8 and the necessary packages.

Define the necessary environment variables, pools, and connection strings in your Azure DevOps environment.

Adjust the paths and model names in the Python scripts and YAML files to match your specific requirements.

Manually start the pipelines, or push a change to your repository to trigger them automatically.

Check the results in the Azure DevOps portal.

**Running the Scripts Locally**

After setting up your environment and familiarizing yourself with the configuration files, you can run the performance monitoring script:

```bash
python -m pytest -v -s test_perf_monitoring_models_cpu.py
```

This command starts pytest, which runs all the test cases in the test_perf_monitoring_models_cpu.py script.


## Comparison Metrics and Process
Performance comparison takes into account the following metrics:

**Accuracy** - Calculated as the number of correct predictions divided by the total number of predictions.

**Latency** - Measured in seconds, calculated as the time taken to run the model on a specific test set.

The metrics from each model run are compared with a set of reference metrics stored in a **best_metrics.json** file using the **compared_metrics** function in **utils.py**. This function calculates the percentage change in each metric relative to the stored "best" metrics.

If the function detects a regression (increase in latency or decrease in accuracy) exceeding a predefined threshold, it raises an error, causing the Azure pipeline to fail.

The **best_metrics.json** file is updated with the new metrics if the function does not detect a regression.
