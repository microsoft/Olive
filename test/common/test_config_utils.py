# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path

import pytest
from pydantic import Field

from olive.common.config_utils import (
    ConfigBase,
    ConfigDictBase,
    ConfigListBase,
    ConfigParam,
    NestedConfig,
    ParamCategory,
    config_json_dumps,
    config_json_loads,
    convert_configs_to_dicts,
    create_config_class,
    load_config_file,
    serialize_function,
    serialize_object,
    serialize_to_json,
    validate_config,
    validate_enum,
    validate_lowercase,
)


class TestSerializeFunction:
    def test_serialize_function_returns_dict(self):
        # setup
        def my_func(x):
            return x

        # execute
        result = serialize_function(my_func)

        # assert
        assert result["olive_parameter_type"] == "Function"
        assert result["name"] == "my_func"
        assert "signature" in result
        assert "sourcecode_hash" in result


class TestSerializeObject:
    def test_serialize_object_returns_dict(self):
        # setup
        obj = {"key": "value"}

        # execute
        result = serialize_object(obj)

        # assert
        assert result["olive_parameter_type"] == "Object"
        assert result["type"] == "dict"
        assert "hash" in result


class TestConfigJsonDumps:
    def test_basic_dict(self):
        # setup
        data = {"key": "value", "num": 42}

        # execute
        result = config_json_dumps(data)
        parsed = json.loads(result)

        # assert
        assert parsed == data

    def test_path_serialization_absolute(self):
        # setup
        data = {"path": Path("/some/path")}

        # execute
        result = config_json_dumps(data, make_absolute=True)
        parsed = json.loads(result)

        # assert
        assert isinstance(parsed["path"], str)

    def test_path_serialization_relative(self):
        # setup
        data = {"path": Path("relative/path")}

        # execute
        result = config_json_dumps(data, make_absolute=False)
        parsed = json.loads(result)

        # assert
        assert parsed["path"] == "relative/path"

    def test_function_serialization(self):
        # setup
        def sample_func():
            pass

        data = {"func": sample_func}

        # execute
        result = config_json_dumps(data)
        parsed = json.loads(result)

        # assert
        assert parsed["func"]["olive_parameter_type"] == "Function"


class TestConfigJsonLoads:
    def test_basic_json(self):
        # setup
        data = '{"key": "value"}'

        # execute
        result = config_json_loads(data)

        # assert
        assert result == {"key": "value"}

    def test_function_object_raises_error(self):
        # setup
        data = json.dumps({"olive_parameter_type": "Function", "name": "my_func"})

        # execute & assert
        with pytest.raises(ValueError, match="Cannot load"):
            config_json_loads(data)

    def test_custom_object_hook(self):
        # setup
        data = '{"key": "value"}'

        # execute
        result = config_json_loads(data, object_hook=lambda obj: obj)

        # assert
        assert result == {"key": "value"}


class TestSerializeToJson:
    def test_dict_input(self):
        # setup
        data = {"key": "value"}

        # execute
        result = serialize_to_json(data)

        # assert
        assert result == data

    def test_config_base_input(self):
        # setup
        config = ConfigBase()

        # execute
        result = serialize_to_json(config)

        # assert
        assert isinstance(result, dict)

    def test_check_object_with_function_raises(self):
        # setup
        def my_func():
            pass

        # execute & assert
        with pytest.raises(ValueError, match="Cannot serialize"):
            serialize_to_json({"func": my_func}, check_object=True)


class TestLoadConfigFile:
    def test_load_json_file(self, tmp_path):
        # setup
        config_file = tmp_path / "config.json"
        config_file.write_text('{"key": "value"}')

        # execute
        result = load_config_file(config_file)

        # assert
        assert result == {"key": "value"}

    def test_load_yaml_file(self, tmp_path):
        # setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value\n")

        # execute
        result = load_config_file(config_file)

        # assert
        assert result == {"key": "value"}

    def test_load_yml_file(self, tmp_path):
        # setup
        config_file = tmp_path / "config.yml"
        config_file.write_text("key: value\n")

        # execute
        result = load_config_file(config_file)

        # assert
        assert result == {"key": "value"}

    def test_unsupported_file_type_raises(self, tmp_path):
        # setup
        config_file = tmp_path / "config.txt"
        config_file.write_text("key=value")

        # execute & assert
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_config_file(config_file)


class TestConfigBase:
    def test_to_json(self):
        # setup
        config = ConfigBase()

        # execute
        result = config.to_json()

        # assert
        assert isinstance(result, dict)

    def test_from_json(self):
        # execute
        config = ConfigBase.from_json({})

        # assert
        assert isinstance(config, ConfigBase)

    def test_parse_file_or_obj_dict(self):
        # execute
        config = ConfigBase.parse_file_or_obj({})

        # assert
        assert isinstance(config, ConfigBase)

    def test_parse_file_or_obj_json_file(self, tmp_path):
        # setup
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        # execute
        config = ConfigBase.parse_file_or_obj(config_file)

        # assert
        assert isinstance(config, ConfigBase)


class TestConfigListBase:
    def test_iter(self):
        # setup
        config = ConfigListBase.model_validate([1, 2, 3])

        # execute
        result = list(config)

        # assert
        assert result == [1, 2, 3]

    def test_getitem(self):
        # setup
        config = ConfigListBase.model_validate([10, 20, 30])

        # execute
        first = config[0]
        last = config[2]

        # assert
        assert first == 10
        assert last == 30

    def test_len(self):
        # setup
        config = ConfigListBase.model_validate([1, 2, 3])

        # execute
        result = len(config)

        # assert
        assert result == 3

    def test_to_json(self):
        # setup
        config = ConfigListBase.model_validate([1, 2, 3])

        # execute
        result = config.to_json()

        # assert
        assert isinstance(result, list)


class TestConfigDictBase:
    def test_iter(self):
        # setup
        config = ConfigDictBase.model_validate({"a": 1, "b": 2})

        # execute
        result = set(config)

        # assert
        assert result == {"a", "b"}

    def test_keys(self):
        # setup
        config = ConfigDictBase.model_validate({"a": 1, "b": 2})

        # execute
        result = set(config.keys())

        # assert
        assert result == {"a", "b"}

    def test_values(self):
        # setup
        config = ConfigDictBase.model_validate({"a": 1, "b": 2})

        # execute
        result = set(config.values())

        # assert
        assert result == {1, 2}

    def test_items(self):
        # setup
        config = ConfigDictBase.model_validate({"a": 1, "b": 2})

        # execute
        result = dict(config.items())

        # assert
        assert result == {"a": 1, "b": 2}

    def test_getitem(self):
        # setup
        config = ConfigDictBase.model_validate({"key": "value"})

        # execute
        result = config["key"]

        # assert
        assert result == "value"

    def test_len(self):
        # setup
        config = ConfigDictBase.model_validate({"a": 1, "b": 2})

        # execute
        result = len(config)

        # assert
        assert result == 2

    def test_len_empty(self):
        # setup
        config = ConfigDictBase.model_validate({})

        # execute
        result = len(config)

        # assert
        assert result == 0


class TestNestedConfig:
    def test_gather_nested_field_basic(self):
        # setup
        class MyConfig(NestedConfig):
            type: str
            config: dict = Field(default_factory=dict)

        # execute
        c = MyConfig(type="test", key1="val1")

        # assert
        assert c.config == {"key1": "val1"}

    def test_gather_nested_field_none_values(self):
        # setup
        class MyConfig(NestedConfig):
            type: str = "default"
            config: dict = Field(default_factory=dict)

        # execute
        c = MyConfig.model_validate(None)

        # assert
        assert c.type == "default"

    def test_gather_nested_field_explicit_config(self):
        # setup
        class MyConfig(NestedConfig):
            type: str
            config: dict = Field(default_factory=dict)

        # execute
        c = MyConfig(type="test", config={"inner_key": "inner_val"})

        # assert
        assert c.config == {"inner_key": "inner_val"}


class TestCaseInsensitiveEnum:
    def test_case_insensitive_creation(self):
        # execute
        lower = ParamCategory("none")
        upper = ParamCategory("NONE")
        mixed = ParamCategory("None")

        # assert
        assert lower == ParamCategory.NONE
        assert upper == ParamCategory.NONE
        assert mixed == ParamCategory.NONE

    def test_invalid_value_returns_none(self):
        # execute
        result = ParamCategory._missing_("nonexistent")

        # assert
        assert result is None


class TestConfigParam:
    def test_config_param_defaults(self):
        # execute
        param = ConfigParam(type_=str)

        # assert
        assert param.required is False
        assert param.default_value is None
        assert param.category == ParamCategory.NONE

    def test_config_param_required(self):
        # execute
        param = ConfigParam(type_=str, required=True)

        # assert
        assert param.required is True

    def test_config_param_repr(self):
        # setup
        param = ConfigParam(type_=str, required=True, description="A test param")

        # execute
        repr_str = repr(param)

        # assert
        assert "required=True" in repr_str
        assert "description=" in repr_str


class TestValidateEnum:
    def test_valid_enum_value(self):
        # execute
        result = validate_enum(ParamCategory, "none")

        # assert
        assert result == ParamCategory.NONE

    def test_invalid_enum_value_raises(self):
        # execute & assert
        with pytest.raises(ValueError, match="Invalid value"):
            validate_enum(ParamCategory, "invalid_value")


class TestValidateLowercase:
    def test_string_lowercased(self):
        # execute
        result = validate_lowercase("HELLO")

        # assert
        assert result == "hello"

    def test_non_string_unchanged(self):
        # execute
        result_int = validate_lowercase(42)
        result_none = validate_lowercase(None)

        # assert
        assert result_int == 42
        assert result_none is None


class TestCreateConfigClass:
    def test_create_basic_config_class(self):
        # setup
        config = {
            "name": ConfigParam(type_=str, required=True),
            "value": ConfigParam(type_=int, default_value=10),
        }

        # execute
        cls = create_config_class("TestConfig", config)
        instance = cls(name="test")

        # assert
        assert instance.name == "test"
        assert instance.value == 10

    def test_create_config_class_with_optional_field(self):
        # setup
        config = {
            "name": ConfigParam(type_=str, default_value=None),
        }

        # execute
        cls = create_config_class("OptionalConfig", config)
        instance = cls()

        # assert
        assert instance.name is None


class TestValidateConfig:
    def test_validate_dict_config(self):
        # setup
        config = {"name": "test"}

        class MyConfig(ConfigBase):
            name: str

        # execute
        result = validate_config(config, MyConfig)

        # assert
        assert isinstance(result, MyConfig)
        assert result.name == "test"

    def test_validate_none_config(self):
        # setup
        class MyConfig(ConfigBase):
            name: str = "default"

        # execute
        result = validate_config(None, MyConfig)

        # assert
        assert result.name == "default"

    def test_validate_config_instance(self):
        # setup
        class MyConfig(ConfigBase):
            name: str

        config = MyConfig(name="test")

        # execute
        result = validate_config(config, MyConfig)

        # assert
        assert result.name == "test"

    def test_validate_config_wrong_class_raises(self):
        # setup
        class MyConfig(ConfigBase):
            name: str

        class OtherConfig(ConfigBase):
            value: int

        config = OtherConfig(value=42)

        # execute & assert
        with pytest.raises(ValueError, match="Invalid config class"):
            validate_config(config, MyConfig)


class TestConvertConfigsToDicts:
    def test_config_base_to_dict(self):
        # setup
        config = ConfigBase()

        # execute
        result = convert_configs_to_dicts(config)

        # assert
        assert isinstance(result, dict)

    def test_nested_dict_conversion(self):
        # setup
        data = {"key": "value"}

        # execute
        result = convert_configs_to_dicts(data)

        # assert
        assert result == {"key": "value"}

    def test_list_conversion(self):
        # setup
        data = ["a", "b"]

        # execute
        result = convert_configs_to_dicts(data)

        # assert
        assert result == ["a", "b"]

    def test_plain_value_passthrough(self):
        # execute
        result_int = convert_configs_to_dicts(42)
        result_str = convert_configs_to_dicts("hello")

        # assert
        assert result_int == 42
        assert result_str == "hello"
