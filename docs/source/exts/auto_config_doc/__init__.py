# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
from importlib import import_module
from typing import ClassVar, Dict, Union

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import stringify_annotation

from olive.common.auto_config import AutoConfigClass
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.package_config import OlivePackageConfig
from olive.passes import Pass

# pylint: skip-file


def import_class(class_name: str, package_config: OlivePackageConfig):
    module_path, module_name = class_name.rsplit(".", 1)
    if module_name.lower() in package_config.passes:
        return package_config.import_pass_module(module_name)

    module = import_module(module_path)
    return getattr(module, module_name)


class AutoConfigDirective(Directive):
    has_content = True
    required_arguments = 1
    option_spec: ClassVar[Dict] = {}

    def make_doc(self, auto_config_class: Union[AutoConfigClass, Pass]):
        class_doc = auto_config_class.__doc__
        lines = []
        if class_doc is not None:
            lines += [class_doc]
        if issubclass(auto_config_class, Pass):
            run_signature = inspect.signature(auto_config_class._run_for_config)
            input_model_type = run_signature.parameters["model"].annotation
            input_model_type = stringify_annotation(input_model_type).replace("olive.model.", "")
            lines += ["", f"**Input:** {stringify_annotation(input_model_type)}"]

            output_model_type = run_signature.return_annotation
            output_model_type = stringify_annotation(output_model_type).replace("olive.model.", "")
            lines += ["", f"**Output:** {stringify_annotation(output_model_type)}"]

        if issubclass(auto_config_class, Pass):
            default_config = auto_config_class.default_config(DEFAULT_CPU_ACCELERATOR)
        else:
            default_config = auto_config_class.default_config()
        for key in default_config:
            param = default_config[key]
            lines += ["", f".. option:: {key}"]
            if param.description is not None:
                lines += ["", f"   {param.description}"]
            lines += ["", f"   **type:** {stringify_annotation(param.type_)}"]
            if param.required:
                lines += ["", "   **required:** True"]
            else:
                lines += ["", f"   **default_value:** {param.default_value}"]
                if hasattr(param, "search_defaults"):
                    lines += ["", f"   **search_defaults:** {param.search_defaults}"]

        return lines

    def run(self):
        (class_name,) = self.arguments
        package_config = OlivePackageConfig.load_default_config()
        auto_config_class = import_class(class_name, package_config)
        assert issubclass(auto_config_class, AutoConfigClass) or issubclass(
            auto_config_class, Pass
        ), f"{class_name} is not a subclass of AutoConfigClass or Pass"

        node = nodes.section()
        node.document = self.state.document
        result = StringList()
        doc = self.make_doc(auto_config_class)
        for line in doc:
            result.append(line, "<autoconfigclass>")
        nested_parse_with_titles(self.state, result, node)
        return node.children


def setup(app: Sphinx):
    app.add_directive("autoconfigclass", AutoConfigDirective)
    return {"version": "0.2", "parallel_read_safe": True}
