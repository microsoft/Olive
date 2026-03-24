# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from onnxscript import ir

from olive.passes.onnx.layer_annotation import _flatten_annotations, annotate_ir_model, annotate_proto_model


class TestFlattenAnnotations:
    """Test the _flatten_annotations helper function."""

    def test_flatten_single_layer_single_substring(self):
        """Test flattening with a single layer and single substring."""
        annotations = {"layer1": ["sub1"]}
        result = _flatten_annotations(annotations)
        assert result == [("sub1", "layer1")]

    def test_flatten_single_layer_multiple_substrings(self):
        """Test flattening with a single layer and multiple substrings."""
        annotations = {"layer1": ["sub1", "sub2", "sub3"]}
        result = _flatten_annotations(annotations)
        assert result == [("sub1", "layer1"), ("sub2", "layer1"), ("sub3", "layer1")]

    def test_flatten_multiple_layers(self):
        """Test flattening with multiple layers."""
        annotations = {"layer1": ["sub1", "sub2"], "layer2": ["sub3"]}
        result = _flatten_annotations(annotations)
        assert len(result) == 3
        assert ("sub1", "layer1") in result
        assert ("sub2", "layer1") in result
        assert ("sub3", "layer2") in result

    def test_flatten_empty(self):
        """Test flattening with empty annotations."""
        annotations = {}
        result = _flatten_annotations(annotations)
        assert result == []


class TestAnnotateIrModel:
    """Test the annotate_ir_model function."""

    def test_annotate_simple_graph(self):
        """Test annotating a simple graph with no subgraphs."""
        node1 = ir.Node("", "Add", inputs=[], name="linear_1")
        node2 = ir.Node("", "Mul", inputs=[], name="sigmoid_2")
        graph = ir.Graph([], [], nodes=[node1, node2], name="test_graph", opset_imports={"": 11})
        model = ir.Model(graph, ir_version=8)

        # Annotate with layer annotations
        layer_annotations = {
            "layer_linear": ["linear"],
            "layer_sigmoid": ["sigmoid"],
        }
        annotate_ir_model(model, layer_annotations)

        # Check annotations were applied
        assert model.graph[0].metadata_props["layer_ann"] == "layer_linear"
        assert model.graph[1].metadata_props["layer_ann"] == "layer_sigmoid"

    def test_annotate_no_matching_substrings(self):
        """Test that nodes without matching substrings are not annotated."""
        node1 = ir.Node("", "Add", inputs=[], name="add_1")
        graph = ir.Graph([], [], nodes=[node1], name="test_graph", opset_imports={"": 11})
        model = ir.Model(graph, ir_version=8)

        layer_annotations = {"layer_other": ["other"]}
        annotate_ir_model(model, layer_annotations)

        # Node should not have layer_ann metadata
        assert "layer_ann" not in model.graph[0].metadata_props

    def test_annotate_first_match_wins(self):
        """Test that the first matching substring wins when multiple match."""
        node = ir.Node("", "Add", inputs=[], name="linear_sigmoid_1")
        graph = ir.Graph([], [], nodes=[node], name="test_graph", opset_imports={"": 11})
        model = ir.Model(graph, ir_version=8)

        # both "linear" and "sigmoid" are substrings; "linear" comes first
        layer_annotations = {
            "layer_linear": ["linear"],
            "layer_sigmoid": ["sigmoid"],
        }
        annotate_ir_model(model, layer_annotations)

        assert model.graph[0].metadata_props["layer_ann"] == "layer_linear"

    def test_annotate_node_with_none_name(self):
        """Test that nodes without a name are skipped."""
        node1 = ir.Node("", "Add", inputs=[])
        node1.name = None
        node2 = ir.Node("", "Mul", inputs=[], name="linear_1")
        graph = ir.Graph([], [], nodes=[node1, node2], name="test_graph", opset_imports={"": 11})
        model = ir.Model(graph, ir_version=8)

        layer_annotations = {"layer_linear": ["linear"]}
        annotate_ir_model(model, layer_annotations)

        # First node should not have annotation (no name)
        assert "layer_ann" not in model.graph[0].metadata_props
        # Second node should have annotation
        assert model.graph[1].metadata_props["layer_ann"] == "layer_linear"

    def test_annotate_empty_annotations(self):
        """Test annotating with empty layer_annotations."""
        node = ir.Node("", "Add", inputs=[], name="linear_1")
        graph = ir.Graph([], [], nodes=[node], name="test_graph", opset_imports={"": 11})
        model = ir.Model(graph, ir_version=8)

        annotate_ir_model(model, {})

        # Node should not have annotation
        assert "layer_ann" not in model.graph[0].metadata_props

    def test_annotate_empty_graph(self):
        """Test annotating an empty graph."""
        graph = ir.Graph([], [], nodes=[], name="test_graph", opset_imports={"": 11})
        model = ir.Model(graph, ir_version=8)

        layer_annotations = {"layer_linear": ["linear"]}
        # Should not raise any errors
        annotate_ir_model(model, layer_annotations)


class TestAnnotateProtoModel:
    """Test the annotate_proto_model function (onnx.ModelProto path)."""

    @staticmethod
    def _make_proto_model(node_names):
        """Build a minimal onnx.ModelProto with nodes having the given names."""
        from onnx import TensorProto, helper

        nodes = [
            helper.make_node("Relu", inputs=["x"], outputs=[f"y_{i}"], name=name) for i, name in enumerate(node_names)
        ]
        graph = helper.make_graph(nodes, "test", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])], [])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

    def test_annotate_proto_simple(self):
        model_proto = self._make_proto_model(["linear_1", "sigmoid_2"])
        annotate_proto_model(model_proto, {"layer_linear": ["linear"], "layer_sigmoid": ["sigmoid"]})

        props = {n.name: {p.key: p.value for p in n.metadata_props} for n in model_proto.graph.node}
        assert props["linear_1"]["layer_ann"] == "layer_linear"
        assert props["sigmoid_2"]["layer_ann"] == "layer_sigmoid"

    def test_annotate_proto_no_match(self):
        model_proto = self._make_proto_model(["add_1"])
        annotate_proto_model(model_proto, {"layer_other": ["other"]})

        assert len(model_proto.graph.node[0].metadata_props) == 0

    def test_annotate_proto_skips_unnamed_nodes(self):
        model_proto = self._make_proto_model(["", "linear_1"])
        annotate_proto_model(model_proto, {"layer_linear": ["linear"]})

        assert len(model_proto.graph.node[0].metadata_props) == 0
        props = {p.key: p.value for p in model_proto.graph.node[1].metadata_props}
        assert props["layer_ann"] == "layer_linear"
