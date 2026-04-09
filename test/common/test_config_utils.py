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
        def my_func(x):
            return x

        result = serialize_function(my_func)
        assert result["olive_parameter_type"] == "Function"
        assert result["name"] == "my_func"
        assert "signature" in result
        assert "sourcecode_hash" in result


class TestSerializeObject:
    def test_serialize_object_returns_dict(self):
        obj = {"key": "value"}
        result = serialize_object(obj)
        assert result["olive_parameter_type"] == "Object"
        assert result["type"] == "dict"
        assert "hash" in result


class TestConfigJsonDumps:
    def test_basic_dict(self):
        data = {"key": "value", "num": 42}
        result = config_json_dumps(data)
        parsed = json.loads(result)
        assert parsed == data

    def test_path_serialization_absolute(self):
        data = {"path": Path("/some/path")}
        result = config_json_dumps(data, make_absolute=True)
        parsed = json.loads(result)
        assert isinstance(parsed["path"], str)

    def test_path_serialization_relative(self):
        data = {"path": Path("relative/path")}
        result = config_json_dumps(data, make_absolute=False)
        parsed = json.loads(result)
        assert parsed["path"] == "relative/path"

    def test_function_serialization(self):
        def sample_func():
            pass

        data = {"func": sample_func}
        result = config_json_dumps(data)
        parsed = json.loads(result)
        assert parsed["func"]["olive_parameter_type"] == "Function"


class TestConfigJsonLoads:
    def test_basic_json(self):
        data = '{"key": "value"}'
        result = config_json_loads(data)
        assert result == {"key": "value"}

    def test_function_object_raises_error(self):
        data = json.dumps({"olive_parameter_type": "Function", "name": "my_func"})
        with pytest.raises(ValueError, match="Cannot load"):
            config_json_loads(data)

    def test_custom_object_hook(self):
        data = '{"key": "value"}'
        result = config_json_loads(data, object_hook=lambda obj: obj)
        assert result == {"key": "value"}


class TestSerializeToJson:
    def test_dict_input(self):
        data = {"key": "value"}
        result = serialize_to_json(data)
        assert result == data

    def test_config_base_input(self):
        config = ConfigBase()
        result = serialize_to_json(config)
        assert isinstance(result, dict)

    def test_check_object_with_function_raises(self):
        def my_func():
            pass

        with pytest.raises(ValueError, match="Cannot serialize"):
            serialize_to_json({"func": my_func}, check_object=True)


class TestLoadConfigFile:
    def test_load_json_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text('{"key": "value"}')
        result = load_config_file(config_file)
        assert result == {"key": "value"}

    def test_load_yaml_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value\n")
        result = load_config_file(config_file)
        assert result == {"key": "value"}

    def test_load_yml_file(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("key: value\n")
        result = load_config_file(config_file)
        assert result == {"key": "value"}

    def test_unsupported_file_type_raises(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text("key=value")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_config_file(config_file)


class TestConfigBase:
    def test_to_json(self):
        config = ConfigBase()
        result = config.to_json()
        assert isinstance(result, dict)

    def test_from_json(self):
        config = ConfigBase.from_json({})
        assert isinstance(config, ConfigBase)

    def test_parse_file_or_obj_dict(self):
        config = ConfigBase.parse_file_or_obj({})
        assert isinstance(config, ConfigBase)

    def test_parse_file_or_obj_json_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        config = ConfigBase.parse_file_or_obj(config_file)
        assert isinstance(config, ConfigBase)


class TestConfigListBase:
    def test_iter(self):
        config = ConfigListBase.model_validate([1, 2, 3])
        assert list(config) == [1, 2, 3]

    def test_getitem(self):
        config = ConfigListBase.model_validate([10, 20, 30])
        assert config[0] == 10
        assert config[2] == 30

    def test_len(self):
        config = ConfigListBase.model_validate([1, 2, 3])
        assert len(config) == 3

    def test_to_json(self):
        config = ConfigListBase.model_validate([1, 2, 3])
        result = config.to_json()
        assert isinstance(result, list)


class TestConfigDictBase:
    def test_iter(self):
        config = ConfigDictBase.model_validate({"a": 1, "b": 2})
        assert set(config) == {"a", "b"}

    def test_keys(self):
        config = ConfigDictBase.model_validate({"a": 1, "b": 2})
        assert set(config.keys()) == {"a", "b"}

    def test_values(self):
        config = ConfigDictBase.model_validate({"a": 1, "b": 2})
        assert set(config.values()) == {1, 2}

    def test_items(self):
        config = ConfigDictBase.model_validate({"a": 1, "b": 2})
        assert dict(config.items()) == {"a": 1, "b": 2}

    def test_getitem(self):
        config = ConfigDictBase.model_validate({"key": "value"})
        assert config["key"] == "value"

    def test_len(self):
        config = ConfigDictBase.model_validate({"a": 1, "b": 2})
        assert len(config) == 2

    def test_len_empty(self):
        config = ConfigDictBase.model_validate({})
        assert len(config) == 0


class TestNestedConfig:
    def test_gather_nested_field_basic(self):
        class MyConfig(NestedConfig):
            type: str
            config: dict = Field(default_factory=dict)

        c = MyConfig(type="test", key1="val1")
        assert c.config == {"key1": "val1"}

    def test_gather_nested_field_none_values(self):
        class MyConfig(NestedConfig):
            type: str = "default"
            config: dict = Field(default_factory=dict)

        c = MyConfig.model_validate(None)
        assert c.type == "default"

    def test_gather_nested_field_explicit_config(self):
        class MyConfig(NestedConfig):
            type: str
            config: dict = Field(default_factory=dict)

        c = MyConfig(type="test", config={"inner_key": "inner_val"})
        assert c.config == {"inner_key": "inner_val"}


class TestCaseInsensitiveEnum:
    def test_case_insensitive_creation(self):
        assert ParamCategory("none") == ParamCategory.NONE
        assert ParamCategory("NONE") == ParamCategory.NONE
        assert ParamCategory("None") == ParamCategory.NONE

    def test_invalid_value_returns_none(self):
        result = ParamCategory._missing_("nonexistent")
        assert result is None


class TestConfigParam:
    def test_config_param_defaults(self):
        param = ConfigParam(type_=str)
        assert param.required is False
        assert param.default_value is None
        assert param.category == ParamCategory.NONE

    def test_config_param_required(self):
        param = ConfigParam(type_=str, required=True)
        assert param.required is True

    def test_config_param_repr(self):
        param = ConfigParam(type_=str, required=True, description="A test param")
        repr_str = repr(param)
        assert "required=True" in repr_str
        assert "description=" in repr_str


class TestValidateEnum:
    def test_valid_enum_value(self):
        result = validate_enum(ParamCategory, "none")
        assert result == ParamCategory.NONE

    def test_invalid_enum_value_raises(self):
        with pytest.raises(ValueError, match="Invalid value"):
            validate_enum(ParamCategory, "invalid_value")


class TestValidateLowercase:
    def test_string_lowercased(self):
        assert validate_lowercase("HELLO") == "hello"

    def test_non_string_unchanged(self):
        assert validate_lowercase(42) == 42
        assert validate_lowercase(None) is None


class TestCreateConfigClass:
    def test_create_basic_config_class(self):
        config = {
            "name": ConfigParam(type_=str, required=True),
            "value": ConfigParam(type_=int, default_value=10),
        }
        cls = create_config_class("TestConfig", config)
        instance = cls(name="test")
        assert instance.name == "test"
        assert instance.value == 10

    def test_create_config_class_with_optional_field(self):
        config = {
            "name": ConfigParam(type_=str, default_value=None),
        }
        cls = create_config_class("OptionalConfig", config)
        instance = cls()
        assert instance.name is None


class TestValidateConfig:
    def test_validate_dict_config(self):
        config = {"name": "test"}

        class MyConfig(ConfigBase):
            name: str

        result = validate_config(config, MyConfig)
        assert isinstance(result, MyConfig)
        assert result.name == "test"

    def test_validate_none_config(self):
        class MyConfig(ConfigBase):
            name: str = "default"

        result = validate_config(None, MyConfig)
        assert result.name == "default"

    def test_validate_config_instance(self):
        class MyConfig(ConfigBase):
            name: str

        config = MyConfig(name="test")
        result = validate_config(config, MyConfig)
        assert result.name == "test"

    def test_validate_config_wrong_class_raises(self):
        class MyConfig(ConfigBase):
            name: str

        class OtherConfig(ConfigBase):
            value: int

        config = OtherConfig(value=42)
        with pytest.raises(ValueError, match="Invalid config class"):
            validate_config(config, MyConfig)


class TestConvertConfigsToDicts:
    def test_config_base_to_dict(self):
        config = ConfigBase()
        result = convert_configs_to_dicts(config)
        assert isinstance(result, dict)

    def test_nested_dict_conversion(self):
        result = convert_configs_to_dicts({"key": "value"})
        assert result == {"key": "value"}

    def test_list_conversion(self):
        result = convert_configs_to_dicts(["a", "b"])
        assert result == ["a", "b"]

    def test_plain_value_passthrough(self):
        assert convert_configs_to_dicts(42) == 42
        assert convert_configs_to_dicts("hello") == "hello"
