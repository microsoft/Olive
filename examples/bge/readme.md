## Precision

| Quantized Ops | precision |
|-|-|
| None (original model) | 0.8574675324675324 |
| All ("Mul", "Transpose", "Unsqueeze", "Add", "Softmax", "Gelu", "LayerNormalization", "Gather", "MatMul", "Sub", "Where", "Expand", "Gemm", "Tanh", "Reshape") | 0.5315909090909091 |
| "MatMul", "LayerNormalization", "Gemm", "Gelu" | 0.8506818181818183 |
| "Mul", "MatMul", "LayerNormalization", "Gemm", "Gelu" | 0.850487012987013 |
| "Mul", "Transpose", "MatMul", "LayerNormalization", "Gemm", "Gelu" | 0.8504870129870131 |
| 'Mul', 'Transpose', 'MatMul', 'LayerNormalization', 'Gemm', 'Gelu', 'Unsqueeze' | 0.8504870129870131 |
| 'Mul', 'Transpose', 'MatMul', 'LayerNormalization', 'Gemm', 'Gelu', 'Unsqueeze', 'Add' | 0.5317207792207792 |
| 'Mul', 'Transpose', 'MatMul', 'LayerNormalization', 'Gemm', 'Gelu', 'Unsqueeze', 'Softmax' | 0.5313961038961039 |
