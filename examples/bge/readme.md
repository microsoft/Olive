# original model accuracy

"accuracy-accuracy_custom": 0.8574675324675324

"op_types_to_quantize": ["Mul", "Transpose", "Unsqueeze", "Add", "Softmax", "Gelu", "LayerNormalization", "Gather", "MatMul", "Sub", "Where", "Expand", "Gemm", "Tanh", "Reshape"]

# QDQ

All: 0.5315909090909091

[ "MatMul", "LayerNormalization", "Gemm", "Gelu" ]: "accuracy-accuracy_custom": 0.8506818181818183

[ "Mul", "MatMul", "LayerNormalization", "Gemm", "Gelu" ]: 0.850487012987013

[ "Mul", "Transpose", "MatMul", "LayerNormalization", "Gemm", "Gelu" ]: 0.8504870129870131
