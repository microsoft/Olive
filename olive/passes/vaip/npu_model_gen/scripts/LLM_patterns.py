patterns = {
  "SSMLP": [
    [
      {
        "name": "/model/layers.0/post_attention_layernorm/SkipLayerNorm",
        "op": "AMDSkipSimplifiedLayerNormalization",
        "attrs": [],
        "inport": [],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/gate_proj/MatMulNBits_cast_fp32_",
            0
          ],
          [
            0,
            "/model/layers.0/mlp/up_proj/MatMulNBits_cast_fp32_",
            0
          ],
          [
            1,
            "/model/layers.1/input_layernorm/SkipLayerNorm",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/gate_proj/MatMulNBits_cast_fp32_",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/post_attention_layernorm/SkipLayerNorm",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/gate_proj/MatMulNBits",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/gate_proj/MatMulNBits",
        "op": "MatMulNBits",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/gate_proj/MatMulNBits_cast_fp32_",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/gate_proj/MatMulNBits_cast_bf16",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/gate_proj/MatMulNBits_cast_bf16",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/gate_proj/MatMulNBits",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Sigmoid_cast_fp32_0",
            0
          ],
          [
            0,
            "/model/layers.0/mlp/act_fn/Mul_cast_fp32_x",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/act_fn/Sigmoid_cast_fp32_0",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/gate_proj/MatMulNBits_cast_bf16",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Sigmoid",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/act_fn/Mul_cast_fp32_x",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/gate_proj/MatMulNBits_cast_bf16",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Mul",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/act_fn/Sigmoid",
        "op": "Sigmoid",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Sigmoid_cast_fp32_0",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Sigmoid_cast_bf16_0",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/act_fn/Sigmoid_cast_bf16_0",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Sigmoid",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Mul_cast_fp32_y",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/act_fn/Mul_cast_fp32_y",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Sigmoid_cast_bf16_0",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Mul",
            1
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/act_fn/Mul",
        "op": "Mul",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Mul_cast_fp32_x",
            0
          ],
          [
            1,
            "/model/layers.0/mlp/act_fn/Mul_cast_fp32_y",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Mul_cast_bf16_",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/act_fn/Mul_cast_bf16_",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Mul",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/Mul_cast_fp32_x",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/Mul_cast_fp32_x",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/act_fn/Mul_cast_bf16_",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/Mul",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/up_proj/MatMulNBits_cast_fp32_",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/post_attention_layernorm/SkipLayerNorm",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/up_proj/MatMulNBits",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/up_proj/MatMulNBits",
        "op": "MatMulNBits",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/up_proj/MatMulNBits_cast_fp32_",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/up_proj/MatMulNBits_cast_bf16",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/up_proj/MatMulNBits_cast_bf16",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/up_proj/MatMulNBits",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/Mul_cast_fp32_y",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/Mul_cast_fp32_y",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/up_proj/MatMulNBits_cast_bf16",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/Mul",
            1
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/Mul",
        "op": "Mul",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/Mul_cast_fp32_x",
            0
          ],
          [
            1,
            "/model/layers.0/mlp/Mul_cast_fp32_y",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/Mul_cast_bf16_",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/Mul_cast_bf16_",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/Mul",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/down_proj/MatMulNBits_cast_fp32_",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/down_proj/MatMulNBits_cast_fp32_",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/Mul_cast_bf16_",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/down_proj/MatMulNBits",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/down_proj/MatMulNBits",
        "op": "MatMulNBits",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/down_proj/MatMulNBits_cast_fp32_",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/mlp/down_proj/MatMulNBits_cast_bf16",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/mlp/down_proj/MatMulNBits_cast_bf16",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/mlp/down_proj/MatMulNBits",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.1/input_layernorm/SkipLayerNorm",
            1
          ]
        ]
      },
      {
        "name": "/model/layers.1/input_layernorm/SkipLayerNorm",
        "op": "AMDSkipSimplifiedLayerNormalization",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/post_attention_layernorm/SkipLayerNorm",
            1
          ],
          [
            1,
            "/model/layers.0/mlp/down_proj/MatMulNBits_cast_bf16",
            0
          ]
        ],
        "outport": []
      }
    ]
  ],
  "GQO": [
    [
      {
        "name": "/model/layers.0/attn/GroupQueryAttention",
        "op": "AMDGroupQueryAttention",
        "attrs": [],
        "inport": [],
        "outport": [
          [
            0,
            "/model/layers.0/attn/o_proj/MatMulNBits_cast_fp32_",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/attn/o_proj/MatMulNBits_cast_fp32_",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/attn/GroupQueryAttention",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/attn/o_proj/MatMulNBits",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/attn/o_proj/MatMulNBits",
        "op": "MatMulNBits",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/attn/o_proj/MatMulNBits_cast_fp32_",
            0
          ]
        ],
        "outport": [
          [
            0,
            "/model/layers.0/attn/o_proj/MatMulNBits_cast_bf16",
            0
          ]
        ]
      },
      {
        "name": "/model/layers.0/attn/o_proj/MatMulNBits_cast_bf16",
        "op": "Cast",
        "attrs": [],
        "inport": [
          [
            0,
            "/model/layers.0/attn/o_proj/MatMulNBits",
            0
          ]
        ],
        "outport": []
      }
    ]
  ]
}