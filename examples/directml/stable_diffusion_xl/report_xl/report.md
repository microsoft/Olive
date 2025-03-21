# Data and Config

- Model: stabilityai/sdxl-turbo
- Num steps: 1
- Train Prompts: 50 from phiyodr/coco2017
- Test Prompts: train prompts + another 50 from phiyodr/coco2017

# Quantization steps

For all text encoder, unet and vae decoder model, I use the following passes from Olive

- https://microsoft.github.io/Olive/reference/pass.html#qnnpreprocess
    + with fuse_layernorm = true
- https://microsoft.github.io/Olive/reference/pass.html#onnxstaticquantization
    + with quant_preprocess = true, prepare_qnn_config = true, activation_type = QUInt16 and weight_type = QUInt8
    + by default, it uses QDQ quant_format and MinMax as calibrate_method

For text encoder, an additional ReplaceAttentionMaskValue pass is used.

# Result

[Quantized Models](https://github.com/xieofxie/Olive/releases/tag/sdxl-turbo)

|model|clip|fid|mse with original outputs|
|-|-|-|-|
|original|31.58|174.41|N/A|
|quantized|31.70|176.45|425|
|quantized two steps|31.70|168.21|N/A|

Some selected results for test prompts

A few people working on various computers in an office

|original|quantized 1 step|quantized 2 steps|
|-|-|-|
|![o](./unoptimized/A%20few%20people%20working%20on%20various%20computers%20in%20an%20office.png)|![s1](./optimized_1step/A%20few%20people%20working%20on%20various%20computers%20in%20an%20office.png)|![s2](./optimized_2steps/A%20few%20people%20working%20on%20various%20computers%20in%20an%20office.png)|

A purple motorcycle parked in front of a red brick building

|original|quantized 1 step|quantized 2 steps|
|-|-|-|
|![o](./unoptimized/A%20purple%20motorcycle%20parked%20in%20front%20of%20a%20red%20brick%20building.png)|![s1](./optimized_1step/A%20purple%20motorcycle%20parked%20in%20front%20of%20a%20red%20brick%20building.png)|![s2](./optimized_2steps/A%20purple%20motorcycle%20parked%20in%20front%20of%20a%20red%20brick%20building.png)|

A small white toilet sitting next to a metal trash can

|original|quantized 1 step|quantized 2 steps|
|-|-|-|
|![o](./unoptimized/A%20small%20white%20toilet%20sitting%20next%20to%20a%20metal%20trash%20can.png)|![s1](./optimized_1step/A%20small%20white%20toilet%20sitting%20next%20to%20a%20metal%20trash%20can.png)|![s2](./optimized_2steps/A%20small%20white%20toilet%20sitting%20next%20to%20a%20metal%20trash%20can.png)|

# Conclusion

From quantitative evaluation , the scores are similar.

From qualitative evaluation, the 1 step results are blurry but the 2 steps results are having higher quality.
