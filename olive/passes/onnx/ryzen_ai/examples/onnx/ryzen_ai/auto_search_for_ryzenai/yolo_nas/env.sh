#!/bin/bash

pip install super-gradients==3.7.1
pip install numpy==1.26.0
pip install onnx==1.17.0
pip install onnxruntime==1.19.2
pip install onnxruntime_extensions==0.12.0
pip install opencv-python==3.4.18.65

package_name="super_gradients"
package_location=$(pip show "$package_name" | grep "Location" | cut -d' ' -f2)
checkpoint_utils_path="$package_location/$package_name/training/utils/checkpoint_utils.py"
echo "$checkpoint_utils_path"
sed -i 's/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/g' $checkpoint_utils_path
pretrained_models_path="$package_location/$package_name/training/pretrained_models.py"
sed -i 's/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/g' $pretrained_models_path
