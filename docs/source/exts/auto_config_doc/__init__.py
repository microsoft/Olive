import inspect
from importlib import import_module

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import stringify_annotation

from olive.common.auto_config import AutoConfigClass
from olive.passes import Pass


def import_class(class_name: str):
    module_name = ".".join(class_name.split(".")[:-1])
    class_name = class_name.split(".")[-1]
    module = import_module(module_name)
    return getattr(module, class_name)


class AutoConfigDirective(Directive):
    has_content = True
    required_arguments = 1
    option_spec = {}

    def make_doc(self, auto_config_class: AutoConfigClass):
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
                lines += ["", f"   **default:** {param.default}"]
                if hasattr(param, "searchable_values"):
                    lines += ["", f"   **searchable_values:** {param.searchable_values}"]

        return lines

    def run(self):
        (class_name,) = self.arguments
        auto_config_class = import_class(class_name)
        assert issubclass(auto_config_class, AutoConfigClass), f"{class_name} is not a subclass of AutoConfigClass"

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
