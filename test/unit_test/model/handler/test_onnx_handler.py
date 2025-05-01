import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from olive.model import ONNXModelHandler
from olive.model.handler.mixin.onnx_graph import clean_initializers_with_dag
from olive.passes.onnx.onnx_dag import OnnxDAG


class TestONNXHandler:
    def test_topological_sort_with_onnxdag(self, tmp_path):
        """测试使用OnnxDAG实现的拓扑排序功能。"""
        # 创建一个简单的ONNX模型用于测试
        # A -> B -> C
        #  \-> D -/
        
        # 创建节点
        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
        
        # 创建四个节点，形成依赖关系
        node_a = helper.make_node("Identity", ["input"], ["a_out"], name="A")
        node_b = helper.make_node("Identity", ["a_out"], ["b_out"], name="B")
        node_c = helper.make_node("Add", ["b_out", "d_out"], ["output"], name="C")
        node_d = helper.make_node("Identity", ["a_out"], ["d_out"], name="D")
        
        # 以乱序添加节点
        nodes = [node_c, node_a, node_d, node_b]
        
        # 创建图和模型
        graph = helper.make_graph(
            nodes=nodes,
            name="test_graph",
            inputs=[input_info],
            outputs=[output_info],
            initializer=[]
        )
        
        model = helper.make_model(graph, producer_name="test")
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))
        
        # 使用ONNXModelHandler加载模型并执行拓扑排序
        handler = ONNXModelHandler(model_path=str(model_path))
        sorted_model = handler.topological_sort()
        
        # 验证节点顺序是否正确
        # A 应该在 B 和 D 之前
        # B 和 D 应该在 C 之前
        node_order = {}
        for i, node in enumerate(sorted_model.graph.node):
            node_order[node.name] = i
        
        assert node_order["A"] < node_order["B"], "节点 A 应该在 B 之前"
        assert node_order["A"] < node_order["D"], "节点 A 应该在 D 之前"
        assert node_order["B"] < node_order["C"], "节点 B 应该在 C 之前"
        assert node_order["D"] < node_order["C"], "节点 D 应该在 C 之前"
    
    def test_clean_initializers_with_dag(self, tmp_path):
        """测试使用DAG清理未使用的初始化器。"""
        # 创建一个带有未使用初始化器的模型
        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
        
        # 创建一些初始化器
        weight1 = helper.make_tensor(
            name="weight1",
            data_type=TensorProto.FLOAT,
            dims=[3, 3],
            vals=np.random.rand(9).astype(np.float32).tolist()
        )
        
        weight2 = helper.make_tensor(
            name="weight2",  # 这个权重将被使用
            data_type=TensorProto.FLOAT,
            dims=[3, 3],
            vals=np.random.rand(9).astype(np.float32).tolist()
        )
        
        unused_weight = helper.make_tensor(
            name="unused_weight",  # 这个权重不会被使用
            data_type=TensorProto.FLOAT,
            dims=[3, 3],
            vals=np.random.rand(9).astype(np.float32).tolist()
        )
        
        # 创建节点
        node1 = helper.make_node("MatMul", ["input", "weight1"], ["inter1"], name="MatMul1")
        node2 = helper.make_node("MatMul", ["inter1", "weight2"], ["output"], name="MatMul2")
        
        # 创建图和模型
        graph = helper.make_graph(
            nodes=[node1, node2],
            name="test_graph",
            inputs=[input_info],
            outputs=[output_info],
            initializer=[weight1, weight2, unused_weight]
        )
        
        model = helper.make_model(graph, producer_name="test")
        model_path = tmp_path / "model_with_unused_initializer.onnx"
        onnx.save(model, str(model_path))
        
        # 使用DAG清理初始化器
        dag = OnnxDAG(model)
        
        # 验证初始清理前有3个初始化器
        assert len(dag.get_initializer_names()) == 3
        assert "unused_weight" in dag.get_initializer_names()
        
        # 清理未使用的初始化器
        dag = clean_initializers_with_dag(dag)
        
        # 验证清理后只剩下2个初始化器
        assert len(dag.get_initializer_names()) == 2
        assert "weight1" in dag.get_initializer_names()
        assert "weight2" in dag.get_initializer_names()
        assert "unused_weight" not in dag.get_initializer_names()
        
        # 更新模型并保存
        dag.update()
        cleaned_model_path = tmp_path / "cleaned_model.onnx"
        onnx.save(dag.model, str(cleaned_model_path))
        
        # 加载清理后的模型并验证
        cleaned_model = onnx.load(str(cleaned_model_path))
        assert len(cleaned_model.graph.initializer) == 2
        initializer_names = [init.name for init in cleaned_model.graph.initializer]
        assert "weight1" in initializer_names
        assert "weight2" in initializer_names
        assert "unused_weight" not in initializer_names
    
    def test_onnx_dag_clean_unused_initializers(self, tmp_path):
        """测试OnnxDAG的clean_unused_initializers方法。"""
        # 创建一个带有未使用初始化器的模型
        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
        
        # 创建一些初始化器
        weight1 = helper.make_tensor(
            name="weight1",
            data_type=TensorProto.FLOAT,
            dims=[3, 3],
            vals=np.random.rand(9).astype(np.float32).tolist()
        )
        
        weight2 = helper.make_tensor(
            name="weight2",  # 这个权重将被使用
            data_type=TensorProto.FLOAT,
            dims=[3, 3],
            vals=np.random.rand(9).astype(np.float32).tolist()
        )
        
        unused_weight = helper.make_tensor(
            name="unused_weight",  # 这个权重不会被使用
            data_type=TensorProto.FLOAT,
            dims=[3, 3],
            vals=np.random.rand(9).astype(np.float32).tolist()
        )
        
        # 创建节点
        node1 = helper.make_node("MatMul", ["input", "weight1"], ["inter1"], name="MatMul1")
        node2 = helper.make_node("MatMul", ["inter1", "weight2"], ["output"], name="MatMul2")
        
        # 创建图和模型
        graph = helper.make_graph(
            nodes=[node1, node2],
            name="test_graph",
            inputs=[input_info],
            outputs=[output_info],
            initializer=[weight1, weight2, unused_weight]
        )
        
        model = helper.make_model(graph, producer_name="test")
        model_path = tmp_path / "model_with_unused_initializer.onnx"
        onnx.save(model, str(model_path))
        
        # 创建OnnxDAG并清理未使用的初始化器
        dag = OnnxDAG(model)
        
        # 验证初始状态有3个初始化器
        assert len(dag.get_initializer_names()) == 3
        assert "unused_weight" in dag.get_initializer_names()
        
        # 使用新方法清理未使用的初始化器
        dag.clean_unused_initializers()
        
        # 验证清理后只剩下2个初始化器
        assert len(dag.get_initializer_names()) == 2
        assert "weight1" in dag.get_initializer_names()
        assert "weight2" in dag.get_initializer_names()
        assert "unused_weight" not in dag.get_initializer_names()
        
        # 更新模型并保存
        dag.update()
        cleaned_model_path = tmp_path / "cleaned_model.onnx"
        onnx.save(dag.model, str(cleaned_model_path))
        
        # 加载清理后的模型并验证
        cleaned_model = onnx.load(str(cleaned_model_path))
        assert len(cleaned_model.graph.initializer) == 2
        initializer_names = [init.name for init in cleaned_model.graph.initializer]
        assert "weight1" in initializer_names
        assert "weight2" in initializer_names
        assert "unused_weight" not in initializer_names 