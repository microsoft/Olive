## Prerequisites
* transformers>=4.34.99
* optimum
* ort-nightly
* git+https://github.com/intel/neural-compressor.git

## Installation
```bash
conda create -n olive python=3.8 -y
conda activate olive
git clone https://github.com/microsoft/Olive.git
cd Olive
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install -e .
cd examples/mistral
pip install -r requirements.txt
# manually install the nightly ORT
pip install ort-nightly==1.17.0.dev20231225002 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

In above steps, please run the following command in Administrator command prompt if you hit "Filename too long" when installing the packages.
```bash
git config --system core.longpaths true
```

## Usage
```bash
python -m olive.workflows.run --config mistral_optimize.json
```
