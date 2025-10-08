# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

from olive.common.utils import WeightsFileFormat, save_weights
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    DORA_NAME_PATTERNS,
    LOHA_NAME_PATTERNS,
    LORA_NAME_PATTERNS,
    AdapterType,
    get_adapter_name,
    get_external_data_config,
    model_proto_to_olive_model,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ExtractAdapters(Pass):
    """Extract adapter weights from ONNX model and save them as external weights file.

    If make_inputs is False, model proto is invalid after this pass as the adapter weights point to non-existent
    external files. Inference session must be created by first loading the adapter weights using
    SessionOptions.add_external_initializers.

    If make_inputs is True, the adapter weights are inputs to the model and must be provided during inference.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "adapter_type": PassConfigParam(
                type_=AdapterType,
                default_value=AdapterType.LORA,
                description=f"Type of adapter to extract. Valid values are {AdapterType.__members__.values()}.",
            ),
            "make_inputs": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Convert adapter weights to inputs. If false, the adapter weights will be set as initializers with"
                    " external data."
                ),
            ),
            "dynamic_lora_r": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Whether the model uses dynamic shape for lora_r. Only used if make_inputs is True. Valid only for"
                    " float modules."
                ),
            ),
            "optional_inputs": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Create default initializers (empty tensor with lora_r dimension set to 0) for the adapter weights,"
                    " if inputs not provided during inference. Only used if make_inputs is True. Valid only for float"
                    " modules."
                ),
            ),
            "save_format": PassConfigParam(
                type_=WeightsFileFormat,
                default_value=WeightsFileFormat.ONNX_ADAPTER,
                description="Format to save the weights in.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        print("=== ExtractAdapters Pass 开始执行 ===")
        print(f"输入模型路径: {model.model_path}")
        print(f"输出模型路径: {output_model_path}")
        print(f"适配器类型: {config.adapter_type}")
        print(f"make_inputs: {config.make_inputs}")
        print(f"save_format: {config.save_format}")
        logger.warning("=== ExtractAdapters Pass 开始执行 ===")
        logger.warning(f"输入模型路径: {model.model_path}")
        logger.warning(f"输出模型路径: {output_model_path}")
        logger.warning(f"适配器类型: {config.adapter_type}")
        logger.warning(f"make_inputs: {config.make_inputs}")
        logger.warning(f"save_format: {config.save_format}")
        
        # 验证输入模型
        if model is None:
            print("❌ 错误：输入模型为 None！")
            logger.error("输入模型为 None！")
            return None
        
        if not hasattr(model, 'model_path') or not model.model_path:
            print("❌ 错误：输入模型没有有效的 model_path！")
            logger.error("输入模型没有有效的 model_path！")
            return None
        
        logger.warning(f"输入模型类型: {type(model)}")
        logger.warning(f"输入模型属性: {getattr(model, 'model_attributes', 'None')}")
        
        try:
            output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
            logger.warning(f"解析后的输出模型路径: {output_model_path}")
        except Exception as e:
            logger.error(f"解析输出模型路径时出错: {e}")
            return None

        try:
            logger.warning("正在加载 IR 模型...")
            ir_model = model.load_ir_model()
            logger.warning(f"IR 模型加载成功，类型: {type(ir_model)}")
            
            logger.warning("正在加载外部数据到模型...")
            ir.external_data.load_to_model(ir_model)
            logger.warning("外部数据加载完成")
            
            # 记录模型基本信息
            if hasattr(ir_model, 'graph') and ir_model.graph:
                logger.warning(f"模型图中的初始化器数量: {len(ir_model.graph.initializers)}")
                logger.warning(f"模型图中的输入数量: {len(ir_model.graph.inputs)}")
                logger.warning(f"模型图中的输出数量: {len(ir_model.graph.outputs)}")
                
                # 记录前几个初始化器的名称
                init_names = list(ir_model.graph.initializers.keys())[:10]
                logger.warning(f"前10个初始化器名称: {init_names}")
            else:
                logger.error("IR 模型没有有效的图结构！")
                return None
                
        except Exception as e:
            logger.error(f"加载 IR 模型时出错: {e}")
            logger.error(f"错误类型: {type(e)}")
            import traceback
            logger.error(f"完整错误堆栈: {traceback.format_exc()}")
            return None

        # dictionary to store adapter weights
        weights = {}

        try:
            print(f"🔍 开始提取 {config.adapter_type} 适配器权重...")
            logger.warning(f"开始提取 {config.adapter_type} 适配器权重...")
            if config.adapter_type in [AdapterType.LORA, AdapterType.DORA, AdapterType.LOHA]:
                weights = self._extract_adapter(ir_model, adapter_type=config.adapter_type)
                print(f"📊 提取到的权重数量: {len(weights)}")
                logger.warning(f"提取到的权重数量: {len(weights)}")
                if weights:
                    print(f"📝 权重名称: {list(weights.keys())}")
                    logger.warning(f"权重名称: {list(weights.keys())}")
                    # 记录权重的形状信息
                    for name, weight in weights.items():
                        print(f"  权重 {name}: shape={weight.shape}, dtype={weight.dtype}")
                        logger.warning(f"权重 {name}: shape={weight.shape}, dtype={weight.dtype}")
                else:
                    print("⚠️ 没有提取到任何权重！")
                    logger.warning("没有提取到任何权重！")
            else:
                print(f"❌ 错误：不支持的适配器类型: {config.adapter_type}")
                logger.error(f"不支持的适配器类型: {config.adapter_type}")
                raise ValueError(f"Unsupported adapter type: {config.adapter_type}")
        except Exception as e:
            print(f"❌ 提取适配器权重时出错: {e}")
            print(f"错误类型: {type(e)}")
            logger.error(f"提取适配器权重时出错: {e}")
            logger.error(f"错误类型: {type(e)}")
            import traceback
            print(f"完整错误堆栈: {traceback.format_exc()}")
            logger.error(f"完整错误堆栈: {traceback.format_exc()}")
            return None

        if not weights:
            print(f"⚠️ 模型中没有找到 {config.adapter_type} 模块，返回原始模型")
            print("=== ExtractAdapters Pass 结束（返回原始模型）===")
            logger.warning("No %s modules found in the model. Returning the original model.", config.adapter_type)
            logger.warning("=== ExtractAdapters Pass 结束（返回原始模型）===")
            return model

        try:
            if config.make_inputs:
                logger.info("开始将权重转换为输入...")
                # create inputs for the weights
                for weight_name in weights:
                    logger.info(f"正在处理权重: {weight_name}")
                    self._convert_initializer_to_input(ir_model, weight_name)
                    self._make_dynamic_optional(ir_model, weights, weight_name, config)
                logger.info("权重转换为输入完成")
        except Exception as e:
            logger.error(f"转换权重为输入时出错: {e}")
            logger.error(f"错误类型: {type(e)}")
            import traceback
            logger.error(f"完整错误堆栈: {traceback.format_exc()}")
            return None

        try:
            print("💾 开始保存权重文件...")
            print(f"权重数量: {len(weights)}")
            print(f"输出模型路径: {output_model_path}")
            print(f"输出模型父目录: {Path(output_model_path).parent}")
            print(f"保存格式: {config.save_format}")
            logger.info("开始保存权重文件...")
            weights_path = save_weights(weights, Path(output_model_path).parent / "adapter_weights", config.save_format)
            print(f"✅ 权重文件保存成功")
            print(f"  完整路径: {weights_path}")
            print(f"  文件名: {weights_path.name if hasattr(weights_path, 'name') else Path(weights_path).name}")
            print(f"  文件是否存在: {Path(weights_path).exists()}")
            logger.info(f"权重文件保存成功: {weights_path}")
        except Exception as e:
            print(f"❌ 保存权重文件时出错: {e}")
            logger.error(f"保存权重文件时出错: {e}")
            logger.error(f"错误类型: {type(e)}")
            import traceback
            print(f"完整错误堆栈: {traceback.format_exc()}")
            logger.error(f"完整错误堆栈: {traceback.format_exc()}")
            return None

        try:
            print("🏗️ 开始保存模型...")
            weights_file_name = weights_path.name if hasattr(weights_path, 'name') else Path(weights_path).name
            print(f"config.make_inputs = {config.make_inputs}")
            print(f"weights_file_name = {weights_file_name}")
            external_init_file = weights_file_name if not config.make_inputs else None
            constant_inputs_file = weights_file_name if config.make_inputs else None
            print(f"external_initializers_file_name = {external_init_file}")
            print(f"constant_inputs_file_name = {constant_inputs_file}")
            logger.info("开始保存模型...")
            # save the model
            output_model = model_proto_to_olive_model(
                ir.to_proto(ir_model),
                output_model_path,
                config,
                external_initializers_file_name=external_init_file,
                constant_inputs_file_name=constant_inputs_file,
            )
            
            if output_model is None:
                print("❌ 致命错误：model_proto_to_olive_model 返回了 None！")
                logger.error("model_proto_to_olive_model 返回了 None！")
                return None
            
            print(f"✅ 输出模型创建成功，类型: {type(output_model)}")
            print(f"📁 输出模型路径: {getattr(output_model, 'model_path', 'None')}")
            print(f"📁 constant_inputs_file_name: {getattr(output_model, 'constant_inputs_file_name', 'None')}")
            print(f"📁 constant_inputs_path: {getattr(output_model, 'constant_inputs_path', 'None')}")
            logger.info(f"输出模型创建成功，类型: {type(output_model)}")
            logger.info(f"输出模型路径: {getattr(output_model, 'model_path', 'None')}")
            
        except Exception as e:
            print(f"❌ 创建输出模型时出错: {e}")
            print(f"错误详情: {str(e)}")
            logger.error(f"创建输出模型时出错: {e}")
            logger.error(f"错误类型: {type(e)}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"完整错误堆栈:\n{traceback_str}")
            logger.error(f"完整错误堆栈: {traceback_str}")
            return None

        try:
            logger.info("开始设置模型属性...")
            output_model.model_attributes = deepcopy(model.model_attributes) or {}
            logger.info(f"复制的模型属性: {output_model.model_attributes}")
            
            # add adapter weights to the model attributes
            output_model.model_attributes["additional_files"] = additional_files = output_model.model_attributes.get(
                "additional_files", []
            )
            additional_files.append(str(weights_path))
            logger.info(f"添加的附加文件: {additional_files}")
            
            # save information about the weights in the model attributes
            weights_info = {name: [list(value.shape), str(value.dtype)] for name, value in weights.items()}
            logger.info(f"权重信息: {weights_info}")
            
            if not config.make_inputs:
                output_model.model_attributes["external_initializers"] = weights_info
                logger.info("权重信息保存为 external_initializers")
            else:
                output_model.model_attributes["constant_inputs"] = weights_info
                logger.info("权重信息保存为 constant_inputs")
                
            logger.info(f"最终模型属性: {output_model.model_attributes}")
            
        except Exception as e:
            logger.error(f"设置模型属性时出错: {e}")
            logger.error(f"错误类型: {type(e)}")
            import traceback
            logger.error(f"完整错误堆栈: {traceback.format_exc()}")
            return None
        
        print("🎉 === ExtractAdapters Pass 成功完成 ===")
        print(f"📦 返回的模型类型: {type(output_model)}")
        print(f"📁 返回的模型路径: {getattr(output_model, 'model_path', 'None')}")
        print(f"📁 constant_inputs_path: {getattr(output_model, 'constant_inputs_path', 'None')}")
        print(f"📁 external_initializers_path: {getattr(output_model, 'external_initializers_path', 'None')}")
        logger.warning("=== ExtractAdapters Pass 成功完成 ===")
        logger.warning(f"返回的模型类型: {type(output_model)}")
        logger.warning(f"返回的模型路径: {getattr(output_model, 'model_path', 'None')}")
        return output_model

    def _convert_initializer_to_input(self, model: ir.Model, initializer_name: str):
        """Convert a specific initializer to an input."""
        logger.info(f"将初始化器转换为输入: {initializer_name}")
        
        graph = model.graph

        # Check if the initializer exists
        if initializer_name not in graph.initializers:
            logger.error(f"初始化器 '{initializer_name}' 在图中不存在！")
            logger.info(f"可用的初始化器: {list(graph.initializers.keys())[:10]}...")
            raise ValueError(f"Initializer '{initializer_name}' not found in graph")

        # Get the initializer
        initializer = graph.initializers[initializer_name]
        logger.info(f"找到初始化器: {initializer_name}, 类型: {type(initializer)}")

        # Check if it's already an input
        if initializer in graph.inputs:
            logger.info(f"初始化器 {initializer_name} 已经是输入，跳过")
            return  # Already an input

        # Add to inputs
        graph.inputs.append(initializer)
        logger.info(f"成功将 {initializer_name} 添加到输入列表，当前输入数量: {len(graph.inputs)}")

    def _decompose_gemm(self, ir_model: ir.Model):
        """Decompose Gemm nodes into MatMul and Add nodes."""
        from onnxscript import rewriter
        from onnxscript.rewriter.rules.common import gemm_to_matmul_add_rule

        return rewriter.rewrite(ir_model, pattern_rewrite_rules=[gemm_to_matmul_add_rule])

    def _extract_adapter(self, ir_model: ir.Model, adapter_type: AdapterType = AdapterType.LORA):
        """Extract adapter weights for LoRA, DoRA, or LoHa from an ONNX model.

        LoRA:
        lora_A -> MatMul -> ...
        lora_B -> MatMul -> ...

        DoRA:
        Besides LoRA A and LoRA B, DoRA also has a learnable magnitude vector M (dora_M):
                         W' = mV + dV = mV + mAB
        AB + dora_M -> Div -> ...

        LoHa:
        hada_w1_a + hada_w1_b -> MatMul -> ...
        hada_w2_a + hada_w2_b -> MatMul -> ...
        """
        logger.warning(f"=== 开始提取 {adapter_type} 适配器权重 ===")
        
        if adapter_type == AdapterType.DORA:
            logger.warning("DoRA 类型，需要先分解 Gemm 节点...")
            try:
                ir_model = self._decompose_gemm(ir_model)
                logger.warning("Gemm 节点分解完成")
            except Exception as e:
                logger.error(f"分解 Gemm 节点时出错: {e}")
                raise

        # dictionary to store adapter weights
        weights = {}

        # Get the appropriate patterns for the adapter type
        patterns = None
        if adapter_type == AdapterType.LORA:
            patterns = LORA_NAME_PATTERNS
            logger.warning(f"使用 LoRA 模式: {patterns}")
        elif adapter_type == AdapterType.DORA:
            patterns = DORA_NAME_PATTERNS
            logger.warning(f"使用 DoRA 模式: {patterns}")
        elif adapter_type == AdapterType.LOHA:
            patterns = LOHA_NAME_PATTERNS
            logger.warning(f"使用 LoHa 模式: {patterns}")
        else:
            logger.error(f"不支持的适配器类型: {adapter_type}")
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

        logger.warning(f"开始扫描模型中的初始化器，总数: {len(ir_model.graph.initializers)}")
        
        to_rename = []
        matched_count = 0
        for i, initializer in enumerate(ir_model.graph.initializers.values()):
            if i < 10:  # 只记录前10个的详细信息
                logger.warning(f"检查初始化器 [{i}]: {initializer.name}")
            
            adapter_weight = get_adapter_name(initializer, patterns)
            if adapter_weight is None:
                if i < 10:
                    logger.warning(f"  -> 不匹配任何适配器模式")
                continue

            logger.warning(f"找到匹配的适配器权重: {initializer.name} -> {adapter_weight}")
            to_rename.append((initializer, adapter_weight))
            matched_count += 1

        logger.warning(f"总共找到 {matched_count} 个匹配的适配器权重")
        
        if not to_rename:
            print("⚠️ 没有找到任何匹配的适配器权重！")
            print("可能的原因:")
            print("1. 模型中没有适配器权重")
            print("2. 适配器权重的命名模式与预期不符")
            print("3. 适配器类型选择错误")
            
            # 记录一些初始化器名称供调试
            init_names = list(ir_model.graph.initializers.keys())[:20]
            print(f"🔍 模型中的前20个初始化器名称: {init_names}")
            logger.warning("没有找到任何匹配的适配器权重！")
            logger.warning("可能的原因:")
            logger.warning("1. 模型中没有适配器权重")
            logger.warning("2. 适配器权重的命名模式与预期不符")
            logger.warning("3. 适配器类型选择错误")
            logger.warning(f"模型中的前20个初始化器名称: {init_names}")
            return weights

        logger.warning("开始处理匹配的适配器权重...")
        for i, (initializer, adapter_weight) in enumerate(to_rename):
            logger.info(f"处理权重 [{i+1}/{len(to_rename)}]: {initializer.name} -> {adapter_weight}")
            
            try:
                old_name = initializer.name

                # Store the weight data
                if hasattr(initializer, 'const_value') and initializer.const_value is not None:
                    weight_data = initializer.const_value.numpy()
                    weights[adapter_weight] = weight_data
                    logger.info(f"  权重数据提取成功: shape={weight_data.shape}, dtype={weight_data.dtype}")
                else:
                    logger.error(f"  初始化器 {old_name} 没有有效的 const_value")
                    continue

                # Rename the initializer
                initializer.name = adapter_weight

                # Update the initializers dictionary
                if old_name in ir_model.graph.initializers:
                    ir_model.graph.initializers.pop(old_name)
                    ir_model.graph.initializers[adapter_weight] = initializer
                    logger.info(f"  初始化器字典更新成功")
                else:
                    logger.warning(f"  初始化器 {old_name} 不在字典中")

                # Create external tensor
                external_tensor = ir.ExternalTensor(
                    location="dummy-location.bin",
                    offset=None,
                    length=None,
                    dtype=initializer.const_value.dtype,
                    shape=initializer.const_value.shape,
                    name=adapter_weight,
                    base_dir="",
                )

                initializer.const_value = external_tensor
                logger.info(f"  外部张量创建成功")
                
            except Exception as e:
                logger.error(f"处理权重 {initializer.name} 时出错: {e}")
                logger.error(f"错误类型: {type(e)}")
                import traceback
                logger.error(f"完整错误堆栈: {traceback.format_exc()}")
                continue

        logger.warning(f"=== 适配器权重提取完成，成功提取 {len(weights)} 个权重 ===")
        return weights

    def _make_dynamic_optional(
        self, model: ir.Model, weights: dict[str, "NDArray"], name: str, config: type[BasePassConfig]
    ):
        """Make the input dynamic and optional."""
        logger.info(f"处理动态可选输入: {name}")
        
        if "lora_magnitude_vector" in name:
            # magnitude vector's shape is independent of lora_r, so we do nothing
            logger.info(f"跳过 magnitude vector: {name}")
            return

        # Determine which dimension should be made dynamic based on pattern in name
        dim_idx = 1
        if "lora_A" in name:
            dim_idx = 1
            logger.info(f"LoRA A 权重，使用维度索引: {dim_idx}")
        elif "lora_B" in name:
            dim_idx = 0
            logger.info(f"LoRA B 权重，使用维度索引: {dim_idx}")
        elif "hada_w1_a" in name or "hada_w2_a" in name:
            dim_idx = 0  # For the first matrix in Hadamard products
            logger.info(f"LoHa w1_a/w2_a 权重，使用维度索引: {dim_idx}")
        elif "hada_w1_b" in name or "hada_w2_b" in name:
            dim_idx = 1  # For the second matrix in Hadamard products
            logger.info(f"LoHa w1_b/w2_b 权重，使用维度索引: {dim_idx}")

        # make the input dynamic
        if config.dynamic_lora_r:
            logger.info(f"设置动态 lora_r 维度: {name}, dim_idx={dim_idx}")
            try:
                self._make_input_dim_dynamic(model, name, dim_idx, "lora_r")
                logger.info(f"成功设置动态维度: {name}")
            except Exception as e:
                logger.error(f"设置动态维度时出错: {e}")
                raise
        else:
            logger.info(f"跳过动态维度设置 (dynamic_lora_r=False): {name}")

        # create default initializer with the lora_r dimension set to 0
        if config.optional_inputs:
            logger.info(f"创建可选输入的默认初始化器: {name}")
            try:
                shape = list(weights[name].shape)
                original_shape = shape.copy()
                shape[dim_idx] = 0
                logger.info(f"原始形状: {original_shape}, 新形状: {shape}")
                
                zero_array = np.zeros(shape, dtype=weights[name].dtype)
                logger.info(f"创建零数组: shape={zero_array.shape}, dtype={zero_array.dtype}")
                
                initializer_value = model.graph.initializers[name]
                initializer_value.const_value = ir.Tensor(zero_array)
                model.graph.inputs.append(initializer_value)
                logger.info(f"成功创建可选输入: {name}")
            except Exception as e:
                logger.error(f"创建可选输入时出错: {e}")
                raise
        else:
            logger.info(f"跳过可选输入创建 (optional_inputs=False): {name}")

    def _make_input_dim_dynamic(self, model: ir.Model, input_name: str, dim_idx: int, dim_param: str):
        """Make a dimension of an input dynamic."""
        logger.info(f"设置输入维度为动态: {input_name}, dim_idx={dim_idx}, dim_param={dim_param}")
        
        # Find the input value
        input_value = None
        for inp in model.graph.inputs:
            if inp.name == input_name:
                input_value = inp
                break

        if input_value is None:
            logger.error(f"{input_name} 不是一个输入！")
            logger.info(f"当前输入列表: {[inp.name for inp in model.graph.inputs]}")
            raise ValueError(f"{input_name} is not an input.")

        logger.info(f"找到输入: {input_name}, 类型: {type(input_value)}")

        if input_value.shape is None:
            logger.error(f"输入 {input_name} 没有形状信息！")
            raise ValueError(f"Input {input_name} does not have shape information.")

        logger.info(f"输入形状: {input_value.shape}, 长度: {len(input_value.shape)}")

        if dim_idx >= len(input_value.shape):
            logger.error(f"输入 {input_name} 的维度数为 {len(input_value.shape)}，但尝试访问维度 {dim_idx}")
            raise ValueError(
                f"Input {input_name} has rank {len(input_value.shape.dims)} but trying to access dim {dim_idx}."
            )

        # Create new shape with symbolic dimension
        new_dims = list(input_value.shape)
        logger.info(f"原始维度: {new_dims}")
        
        if isinstance(new_dims[dim_idx], ir.SymbolicDim) and new_dims[dim_idx].value is not None:
            logger.error(f"无法替换现有的动态维度 {new_dims[dim_idx].value} 为 {dim_param}")
            raise ValueError(f"Can't replace existing dynamic dim {new_dims[dim_idx].value} with {dim_param}")

        new_dims[dim_idx] = ir.SymbolicDim(dim_param)
        input_value.shape = ir.Shape(new_dims)
        logger.info(f"新维度: {new_dims}")
        logger.info(f"成功设置动态维度: {input_name}[{dim_idx}] = {dim_param}")
