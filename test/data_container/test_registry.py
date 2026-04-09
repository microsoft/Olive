# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.data.constants import (
    DataComponentType,
    DefaultDataContainer,
)
from olive.data.registry import Registry


class TestRegistryRegister:
    def test_register_dataset_component(self):
        @Registry.register(DataComponentType.LOAD_DATASET, name="test_dataset_reg")
        def my_dataset():
            return "dataset"

        result = Registry.get_load_dataset_component("test_dataset_reg")
        assert result is my_dataset

    def test_register_pre_process_component(self):
        @Registry.register_pre_process(name="test_pre_process_reg")
        def my_pre_process(data):
            return data

        result = Registry.get_pre_process_component("test_pre_process_reg")
        assert result is my_pre_process

    def test_register_post_process_component(self):
        @Registry.register_post_process(name="test_post_process_reg")
        def my_post_process(data):
            return data

        result = Registry.get_post_process_component("test_post_process_reg")
        assert result is my_post_process

    def test_register_dataloader_component(self):
        @Registry.register_dataloader(name="test_dataloader_reg")
        def my_dataloader(data):
            return data

        result = Registry.get_dataloader_component("test_dataloader_reg")
        assert result is my_dataloader

    def test_register_case_insensitive(self):
        @Registry.register(DataComponentType.LOAD_DATASET, name="CaseSensitiveTest_Reg")
        def my_func():
            pass

        result = Registry.get_load_dataset_component("casesensitivetest_reg")
        assert result is my_func

    def test_register_uses_class_name_when_no_name(self):
        @Registry.register(DataComponentType.LOAD_DATASET)
        def unique_named_test_func_reg():
            pass

        result = Registry.get_load_dataset_component("unique_named_test_func_reg")
        assert result is unique_named_test_func_reg


class TestRegistryGet:
    def test_get_component(self):
        @Registry.register(DataComponentType.LOAD_DATASET, name="test_get_comp_reg")
        def my_func():
            pass

        result = Registry.get_component(DataComponentType.LOAD_DATASET.value, "test_get_comp_reg")
        assert result is my_func

    def test_get_by_subtype(self):
        @Registry.register(DataComponentType.LOAD_DATASET, name="test_get_subtype_reg")
        def my_func():
            pass

        result = Registry.get(DataComponentType.LOAD_DATASET.value, "test_get_subtype_reg")
        assert result is my_func


class TestRegistryDefaultComponents:
    def test_get_default_load_dataset(self):
        result = Registry.get_default_load_dataset_component()
        assert result is not None

    def test_get_default_pre_process(self):
        result = Registry.get_default_pre_process_component()
        assert result is not None

    def test_get_default_post_process(self):
        result = Registry.get_default_post_process_component()
        assert result is not None

    def test_get_default_dataloader(self):
        result = Registry.get_default_dataloader_component()
        assert result is not None


class TestRegistryContainer:
    def test_get_container_default(self):
        result = Registry.get_container(None)
        assert result is not None

    def test_get_container_by_name(self):
        result = Registry.get_container(DefaultDataContainer.DATA_CONTAINER.value)
        assert result is not None
