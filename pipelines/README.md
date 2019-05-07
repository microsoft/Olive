## Overview


## The requirements
Install Kubeflow pipeline SDK:
```
pip install https://storage.googleapis.com/ml-pipeline/release/0.1.14/kfp.tar.gz --upgrade
```

## Compiling the pipeline template

```bash
python inference-pipeline.py
```

## Deploying the pipeline

Open the Kubeflow pipelines UI. Click "Upload" on the top right corner, and then upload the compiled specification (`.tar.gz` file) as a new pipeline template.


## Components source


Converting:
  [source code](), 
  [container]()