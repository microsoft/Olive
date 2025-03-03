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

For text encoder, an additional ReplaceAttentionMaskValue pass is used.

## Result

For a clear comparison, only one model uses quantized version each time.

The images for original model are in [unoptimized](./unoptimized).

### text encoder is quantized

The images are in [optimized_text_encoder](./optimized_text_encoder).

| Prompt | MSE |
|-|-|
| Arroyo Hondo Preserve Wedding | 89.942192 |
| Budget-Friendly Thanksgiving Table Decor Ideas | 806.491272 |
| Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo | 221.874329 |
| Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow | 809.603455 |
| Lovely Anthodium N Roses Arrangement with Cute Teddy | 1888.589355 |
| Everyone can join and learn how to cook delicious dishes with us. | 855.126892 |
| Image result for youth worker superhero | 1048.559082 |
| Road improvements coming along in west Gulfport | 1126.492798 |
| Butcher storefront and a companion work, Louis Hayet, Click for value | 1524.231934 |
| folding electric bike | 99.657501 |

Average train error 855.834900
Average test error 811.944702

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

## Data and Config (50 iterations)

- Model: stabilityai/stable-diffusion-2-1
- Num steps: 50
- Guidance Scale: 7.5
- Prompts: Same

### Text encoder is quantized

| Prompt | MSE |
|-|-|
| Arroyo Hondo Preserve Wedding | 378.618042 |
| Budget-Friendly Thanksgiving Table Decor Ideas | 1592.694946 |
| Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo | 175.496658 |
| Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow | 2615.081299 |
| Lovely Anthodium N Roses Arrangement with Cute Teddy | 820.896179 |
| Everyone can join and learn how to cook delicious dishes with us. | 1852.791504 |
| Image result for youth worker superhero | 1056.542114 |
| Road improvements coming along in west Gulfport | 689.256897 |
| Butcher storefront and a companion work, Louis Hayet, Click for value | 310.129669 |
| folding electric bike | 357.625122 |

Average train error 1147.672119
Average test error 333.877380

### Unet is quantized

| Prompt | MSE |
|-|-|
| Arroyo Hondo Preserve Wedding | 309.682770 |
| Budget-Friendly Thanksgiving Table Decor Ideas | 3247.779541 |
| Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo | 209.519699 |
| Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow | 1361.687744 |
| Lovely Anthodium N Roses Arrangement with Cute Teddy | 897.913757 |
| Everyone can join and learn how to cook delicious dishes with us. | 1341.339966 |
| Image result for youth worker superhero | 811.186218 |
| Road improvements coming along in west Gulfport | 364.869293 |
| Butcher storefront and a companion work, Louis Hayet, Click for value | 395.311188 |
| folding electric bike | 474.218903 |

Average train error 1067.997314
Average test error 434.765045

## Data and Config (turbo)

- Model: stabilityai/sd-turbo
- Num steps: 1
- Guidance Scale: 0

## Unet is quantized

| Prompt | MSE |
|-|-|
| Arroyo Hondo Preserve Wedding | 206.009277 |
| Budget-Friendly Thanksgiving Table Decor Ideas | 357.284088 |
| Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo | 235.289551 |
| Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow | 661.799622 |
| Lovely Anthodium N Roses Arrangement with Cute Teddy | 760.237061 |
| Everyone can join and learn how to cook delicious dishes with us. | 701.614258 |
| Image result for youth worker superhero | 1077.782349 |
| Road improvements coming along in west Gulfport | 272.503571 |
| Butcher storefront and a companion work, Louis Hayet, Click for value | 172.381332 |
| folding electric bike | 333.455811 |

Average train error 534.064941
Average test error 252.918579

## All are quantized

| Prompt | MSE |
|-|-|
| Arroyo Hondo Preserve Wedding | 246.735092 |
| Budget-Friendly Thanksgiving Table Decor Ideas | 364.980377 |
| Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo | 242.039566 |
| Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow | 900.216370 |
| Lovely Anthodium N Roses Arrangement with Cute Teddy | 801.109192 |
| Everyone can join and learn how to cook delicious dishes with us. | 683.701721 |
| Image result for youth worker superhero | 1287.780029 |
| Road improvements coming along in west Gulfport | 292.771942 |
| Butcher storefront and a companion work, Louis Hayet, Click for value | 275.939545 |
| folding electric bike | 326.379730 |
Average train error 602.416748
Average test error 301.159637
