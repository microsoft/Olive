# Static Quantize Stable Diffusion via Olive Report

## Data and Config

These are the parameters I used to generate the quantization data and evaluate the original and generated model

- Model: stabilityai/stable-diffusion-2-1
- Num steps: 10
- Guidance Scale: 7.5
- Prompts: Use 10 captions from https://huggingface.co/datasets/laion/relaion2B-en-research-safe, 8 for training and 2 for testing

## Quantization steps

For all text encoder, unet and vae decoder model, I use the following passes from Olive

- https://microsoft.github.io/Olive/reference/pass.html#onnxpeepholeoptimizer
- https://microsoft.github.io/Olive/reference/pass.html#qnnpreprocess
    + with fuse_layernorm = true
- https://microsoft.github.io/Olive/reference/pass.html#onnxstaticquantization
    + with quant_preprocess = true, prepare_qnn_config = true, activation_type = QUInt16 and weight_type = QUInt8
    + by default, it uses QDQ quant_format and MinMax as calibrate_method

For text encoder, I didn't quantize Add and Softmax nodes. For unet and vae decoder, all nodes are quantized.

## Result

For a clear comparison, only one model uses quantized version each time.

The images for original model are in [unoptimized](./unoptimized).

### text encoder is quantized

The images are in [optimized_text_encoder](./optimized_text_encoder).

| Prompt | MSE |
|-|-|
| Arroyo Hondo Preserve Wedding | 54.031979 |
| Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo | 218.897995 |
| Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow | 942.312500 |
| Lovely Anthodium N Roses Arrangement with Cute Teddy | 1964.954590 |
| Everyone can join and learn how to cook delicious dishes with us. | 1232.637695 |
| Budget-Friendly Thanksgiving Table Decor Ideas | 766.385925 |
| Image result for youth worker superhero | 1346.230835 |
| Road improvements coming along in west Gulfport | 1002.286438 |
| Butcher storefront and a companion work, Louis Hayet, Click for value | 1519.580078 |
| folding electric bike | 114.862068 |

Average train error 940.967285
Average test error 817.221069

### unet is quantized

The images are in [optimized_unet](./optimized_unet).

| Prompt | MSE |
|-|-|
Images passed the safety checker.
| Arroyo Hondo Preserve Wedding | 285.498596 |
| Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo | 428.798737 |
| Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow | 883.842224 |
| Lovely Anthodium N Roses Arrangement with Cute Teddy | 2322.932617 |
| Everyone can join and learn how to cook delicious dishes with us. | 2533.020996 |
| Budget-Friendly Thanksgiving Table Decor Ideas | 1352.172119 |
| Image result for youth worker superhero | 1874.321899 |
| Road improvements coming along in west Gulfport | 1208.272095 |
| Butcher storefront and a companion work, Louis Hayet, Click for value | 596.398438 |
| folding electric bike | 470.758514 |

Average train error 1361.107422
Average test error 533.578491

### vae decoder is quantized

The images are in [optimized_vae_decoder](./optimized_vae_decoder).

| Prompt | MSE |
|-|-|
| Arroyo Hondo Preserve Wedding | 1.923393 |
| Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo | 1.801051 |
| Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow | 1.089745 |
| Lovely Anthodium N Roses Arrangement with Cute Teddy | 1.847665 |
| Everyone can join and learn how to cook delicious dishes with us. | 1.615799 |
| Budget-Friendly Thanksgiving Table Decor Ideas | 2.106870 |
| Image result for youth worker superhero | 0.873670 |
| Road improvements coming along in west Gulfport | 1.345603 |
| Butcher storefront and a companion work, Louis Hayet, Click for value | 1.832780 |
| folding electric bike | 1.767643 |

Average train error 1.575474
Average test error 1.800212

## Summary and Question

The current pipeline works for vae decoder, but not for text encoder and unet. They generate similar images but with low quality.

Compared with [ASG](https://perceptiveshell.azurewebsites.net/2.9.7/pg_models.html), it looks like text encoder and unet models are devided into multiple parts.

So questions are:

- For the text encoder and unet model, how the nodes are processed? i.e. please share some insights about what nodes should be processed specially to keep accuracy.
